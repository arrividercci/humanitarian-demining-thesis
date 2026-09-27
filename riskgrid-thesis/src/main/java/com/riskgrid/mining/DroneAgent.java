package com.riskgrid.mining;

import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Collections;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Виконує повний цикл роботи одного дрона:
 *   selectNextTarget -> tryClaim -> (політ) -> detector.getSuspicionScore(image)
 *   -> riskGrid.update -> markVisited -> release
 *
 * Дрон працює, поки CoverageTracker повідомляє про непокриті клітинки.
 * Якщо tryClaim не вдався (гонка з іншим дроном за ту саму ціль) - одразу
 * повторюємо пошук без паузи.
 *
 * RoutePlanner.selectNextTarget природно уникає вже відвіданих клітинок
 * (їхня uncertainty низька -> низька корисність), тож excludedCells
 * потрібен лише для клітинок, застовплених ЗАРАЗ іншими дронами.
 *
 * selectNextTarget кидає IllegalStateException, якщо навіть після 6
 * подвоєнь радіуса не знайшлося жодного не-застовпленого кандидата
 * (рідкісний випадок під кінець покриття). У цьому випадку дрон
 * "перелітає" до довільної ще не відвіданої клітинки й пробує ще раз.
 *
 * Розширення для експериментів (Options):
 *   - claimingEnabled = false вимикає механізм захоплення клітинок
 *     (абляція: показує, що відбувається без розв'язання конфлікту рішень);
 *   - travelDelayMsPerUnit > 0 моделює політ: дрон "спить" відстань * цей
 *     коефіцієнт (для cellSize = 1.0 це мс на клітинку);
 *   - monitor рахує одночасні дублі (ConflictMonitor);
 *   - minInformationGain > 0 не дає дрону застрягати на повторних обробках
 *     вичерпаних клітинок (див. RoutePlanner.selectNextTarget);
 *   - maxObservationsPerCell > 0 обмежує кількість спостережень на клітинку
 *     (гарантує скінченність обстеження; строго діє при увімкненому захопленні);
 *   - slowdownFactor > 1 моделює повільніший дрон: після кожного кроку він додатково
 *     "спить" (slowdownFactor - 1) * (час кроку), тож працює в slowdownFactor разів повільніше.
 * Також збирається затримка рішення (вибір цілі + захоплення) на кожен крок.
 */
public final class DroneAgent implements Runnable {

    /** Один запис у журналі обходу - для подальшої візуалізації в Python. */
    public record VisitRecord(int droneId, int i, int j, double z, long elapsedNanos) {}

    /** Параметри експерименту. Типові значення відтворюють початкову поведінку. */
    public record Options(boolean claimingEnabled, double travelDelayMsPerUnit, ConflictMonitor monitor,
                          double minInformationGain, int maxObservationsPerCell, double slowdownFactor) {
        public Options(boolean claimingEnabled, double travelDelayMsPerUnit, ConflictMonitor monitor,
                       double minInformationGain, int maxObservationsPerCell) {
            this(claimingEnabled, travelDelayMsPerUnit, monitor, minInformationGain, maxObservationsPerCell, 1.0);
        }

        public Options(boolean claimingEnabled, double travelDelayMsPerUnit, ConflictMonitor monitor,
                       double minInformationGain) {
            this(claimingEnabled, travelDelayMsPerUnit, monitor, minInformationGain, 0, 1.0);
        }

        public Options(boolean claimingEnabled, double travelDelayMsPerUnit, ConflictMonitor monitor) {
            this(claimingEnabled, travelDelayMsPerUnit, monitor, 0.0, 0, 1.0);
        }

        public static Options defaults() {
            return new Options(true, 0.0, null, 0.0, 0, 1.0);
        }
    }

    private final int id;
    private final RiskGrid riskGrid;
    private final RoutePlanner routePlanner;
    private final DroneCoordinator coordinator;
    private final CoverageTracker coverage;
    private final TerrainSource terrain;
    private final MineDetectorOnnx detector;
    private final double radius;
    private final double lambda;
    private final double sensorWeight;
    private final Queue<VisitRecord> visitLog; // спільна, потокобезпечна черга (може бути null)
    private final long runStartNanos;
    private final AtomicBoolean stopFlag;      // може бути null
    private final Options opts;

    private volatile RoutePlanner.Point currentPosition;

    private final AtomicInteger stepsCompleted = new AtomicInteger(0);
    private final AtomicInteger detectionsAboveThreshold = new AtomicInteger(0);

    // Лічильники нижче пише лише потік цього дрона; читати їх безпечно
    // після завершення потоку (Future.get / awaitTermination).
    private volatile int revisits = 0;      // обробка вже відвіданої клітинки (markVisited == false)
    private volatile int staleRevisits = 0; // з них: клітинка була НЕвідвіданою на момент вибору (застарілий вибір)
    private volatile int claimFailures = 0; // програні гонки за клітинку
    private volatile int capRejections = 0; // відмови після захоплення: клітинка вже досягла ліміту спостережень
    private volatile int fallbackJumps = 0; // перельоти до довільної непокритої клітинки
    private long[] decisionNanos = new long[1024];
    private int decisionCount = 0;

    // Статичний сектор (якщо планувальник обмежений областю): дрон працює, поки в секторі є
    // невідвідані клітинки, і завершується, коли сектор покрито. Клітинки сектора обробляє
    // лише цей дрон, тож лічильник не потребує синхронізації.
    private final RoutePlanner.Region sector; // null - вся карта (динамічний режим)
    private int sectorUnvisited = 0;
    private volatile long finishedAtNanos = -1L; // момент завершення роботи дрона відносно старту запуску

    /** Початкова сигнатура - поведінка без змін (захоплення увімкнене, без затримки польоту). */
    public DroneAgent(int id,
                       RiskGrid riskGrid,
                       RoutePlanner routePlanner,
                       DroneCoordinator coordinator,
                       CoverageTracker coverage,
                       TerrainSource terrain,
                       MineDetectorOnnx detector,
                       RoutePlanner.Point startPosition,
                       double radius, double lambda, double sensorWeight,
                       Queue<VisitRecord> visitLog, long runStartNanos,
                       AtomicBoolean stopFlag) {
        this(id, riskGrid, routePlanner, coordinator, coverage, terrain, detector,
                startPosition, radius, lambda, sensorWeight, visitLog, runStartNanos,
                stopFlag, Options.defaults());
    }

    public DroneAgent(int id,
                       RiskGrid riskGrid,
                       RoutePlanner routePlanner,
                       DroneCoordinator coordinator,
                       CoverageTracker coverage,
                       TerrainSource terrain,
                       MineDetectorOnnx detector,
                       RoutePlanner.Point startPosition,
                       double radius, double lambda, double sensorWeight,
                       Queue<VisitRecord> visitLog, long runStartNanos,
                       AtomicBoolean stopFlag,
                       Options opts) {
        this.id = id;
        this.riskGrid = riskGrid;
        this.routePlanner = routePlanner;
        this.coordinator = coordinator;
        this.coverage = coverage;
        this.terrain = terrain;
        this.detector = detector;
        this.currentPosition = startPosition;
        this.radius = radius;
        this.lambda = lambda;
        this.sensorWeight = sensorWeight;
        this.visitLog = visitLog;
        this.runStartNanos = runStartNanos;
        this.stopFlag = stopFlag;
        this.opts = opts;
        this.sector = routePlanner.getRegion();
    }

    @Override
    public void run() {
        final boolean claiming = opts.claimingEnabled();
        final ConflictMonitor monitor = opts.monitor();

        if (sector != null) {
            sectorUnvisited = countUnvisitedInSector();
        }

        while (!coverage.isComplete()
                && !(sector != null && sectorUnvisited <= 0)
                && (stopFlag == null || !stopFlag.get())
                && !Thread.currentThread().isInterrupted()) {

            long decisionStart = System.nanoTime();

            RoutePlanner.TargetResult target;
            Set<Long> excluded = claiming ? coordinator.getClaimedCells() : Collections.emptySet();
            try {
                target = routePlanner.selectNextTarget(currentPosition, radius, lambda, excluded,
                        opts.minInformationGain(), opts.maxObservationsPerCell());
            } catch (IllegalStateException e) {
                // Всі кандидати в межах максимального радіуса зараз застовплені
                // іншими дронами - рідкісний випадок ближче до кінця покриття.
                int[] fallback = findFallbackCell();
                if (fallback == null) {
                    break; // все вже покрито
                }
                fallbackJumps++;
                RoutePlanner.Point jumpTo = routePlanner.cellCenter(fallback[0], fallback[1]);
                if (!travelTo(jumpTo)) {
                    break;
                }
                currentPosition = jumpTo;
                continue;
            }

            final boolean unvisitedAtSelection = !coverage.isVisited(target.i(), target.j());

            if (claiming && !coordinator.tryClaim(target.i(), target.j())) {
                claimFailures++;
                continue; // хтось встиг застовпити раніше - одразу повторюємо пошук
            }

            // Строга перевірка обмеження: між вибором і захопленням інший дрон міг встигнути
            // обробити клітинку. Тепер клітинка застовплена нами, тож лічильник спостережень
            // не зміниться, поки ми її не звільнимо - перевірка остаточна.
            if (claiming && opts.maxObservationsPerCell() > 0
                    && riskGrid.getObservationCount(target.i(), target.j()) >= opts.maxObservationsPerCell()) {
                coordinator.release(target.i(), target.j());
                capRejections++;
                continue;
            }

            // Затримка рішення: від початку вибору до отримання ціли (з повторами після програних гонок).
            recordDecisionLatency(System.nanoTime() - decisionStart);

            if (monitor != null) {
                monitor.enter(target.i(), target.j());
            }
            try {
                final long stepStartNanos = System.nanoTime();
                RoutePlanner.Point targetPoint = routePlanner.cellCenter(target.i(), target.j());
                if (!travelTo(targetPoint)) {
                    break;
                }
                currentPosition = targetPoint;

                BufferedImage image = terrain.getImageAt(target.i(), target.j());
                float rawZ = (image != null) ? detector.getSuspicionScore(image) : 0f;
                double z = clampZ(rawZ);

                // Повільніший дрон: додатковий "сон" пропорційно часу кроку (політ + інференс).
                if (!slowDown(System.nanoTime() - stepStartNanos)) {
                    break;
                }

                riskGrid.update(target.i(), target.j(), z, sensorWeight, System.nanoTime());
                if (coverage.markVisited(target.i(), target.j())) {
                    sectorUnvisited--; // значення має сенс лише в секторному режимі
                } else {
                    revisits++;
                    if (unvisitedAtSelection) {
                        staleRevisits++;
                    }
                }

                if (visitLog != null) {
                    visitLog.add(new VisitRecord(id, target.i(), target.j(), rawZ,
                            System.nanoTime() - runStartNanos));
                }

                stepsCompleted.incrementAndGet();
                if (rawZ >= 0.5f) {
                    detectionsAboveThreshold.incrementAndGet();
                }
            } catch (Exception e) {
                System.err.println("Дрон " + id + ": помилка обробки клітинки ("
                        + target.i() + "," + target.j() + "): " + e.getMessage());
                if (coverage.markVisited(target.i(), target.j())) {
                    sectorUnvisited--;
                }
            } finally {
                if (monitor != null) {
                    monitor.exit(target.i(), target.j());
                }
                if (claiming) {
                    coordinator.release(target.i(), target.j());
                }
            }
        }
        finishedAtNanos = System.nanoTime() - runStartNanos;
    }

    /** Кількість невідвіданих клітинок у власному секторі (викликається лише в секторному режимі). */
    private int countUnvisitedInSector() {
        int count = 0;
        for (int i = sector.iMin(); i <= sector.iMax(); i++) {
            for (int j = sector.jMin(); j <= sector.jMax(); j++) {
                if (!coverage.isVisited(i, j)) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * Клітинка для "перельоту", коли планувальник не знайшов кандидата: у динамічному
     * режимі - будь-яка непокрита клітинка карти, у секторному - найближча непокрита
     * клітинка власного сектора (null, якщо сектор уже покрито).
     */
    private int[] findFallbackCell() {
        if (sector == null) {
            return coverage.findAnyUncovered();
        }
        int[] best = null;
        double bestDistance = Double.MAX_VALUE;
        for (int i = sector.iMin(); i <= sector.iMax(); i++) {
            for (int j = sector.jMin(); j <= sector.jMax(); j++) {
                if (coverage.isVisited(i, j)) {
                    continue;
                }
                double d = routePlanner.dist(routePlanner.cellCenter(i, j), currentPosition);
                if (d < bestDistance) {
                    bestDistance = d;
                    best = new int[]{i, j};
                }
            }
        }
        return best;
    }

    /**
     * Моделює повільніший дрон: додатково "спить" (slowdownFactor - 1) * (тривалість кроку),
     * тож крок займає slowdownFactor разів більше часу. Повертає false, якщо потік перервано.
     */
    private boolean slowDown(long stepNanos) {
        double factor = opts.slowdownFactor();
        if (factor <= 1.0) {
            return true;
        }
        long extra = (long) ((factor - 1.0) * stepNanos);
        if (extra <= 0L) {
            return true;
        }
        try {
            TimeUnit.NANOSECONDS.sleep(extra);
            return true;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    /** Імітує політ до точки. Повертає false, якщо потік перервано (треба зупинитись). */
    private boolean travelTo(RoutePlanner.Point target) {
        double msPerUnit = opts.travelDelayMsPerUnit();
        if (msPerUnit <= 0.0) {
            return true;
        }
        long nanos = (long) (routePlanner.dist(currentPosition, target) * msPerUnit * 1_000_000.0);
        if (nanos <= 0L) {
            return true;
        }
        try {
            TimeUnit.NANOSECONDS.sleep(nanos);
            return true;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    private void recordDecisionLatency(long nanos) {
        if (decisionCount == decisionNanos.length) {
            decisionNanos = Arrays.copyOf(decisionNanos, decisionNanos.length * 2);
        }
        decisionNanos[decisionCount++] = nanos;
    }

    public int getId() {
        return id;
    }

    /** Поточна фізична позиція дрона - для живого відображення на карті в GUI. */
    public RoutePlanner.Point getCurrentPosition() {
        return currentPosition;
    }

    public int getStepsCompleted() {
        return stepsCompleted.get();
    }

    public int getDetectionsAboveThreshold() {
        return detectionsAboveThreshold.get();
    }

    public int getRevisits() {
        return revisits;
    }

    /**
     * Повторні обробки, спричинені застарілим вибором: на момент вибору цілі
     * клітинка була невідвіданою, але до початку обробки її вже обробив інший
     * дрон. Захоплення (tryClaim) цього не запобігає, якщо той дрон встиг
     * завершити й звільнити клітинку між вибором і захопленням.
     */
    public int getStaleRevisits() {
        return staleRevisits;
    }

    /** Момент завершення роботи дрона (нс від старту запуску); -1, якщо ще працює. */
    public long getFinishedAtNanos() {
        return finishedAtNanos;
    }

    public int getCapRejections() {
        return capRejections;
    }

    public int getClaimFailures() {
        return claimFailures;
    }

    public int getFallbackJumps() {
        return fallbackJumps;
    }

    /** Затримки рішень (нс) - викликати лише після завершення потоку дрона. */
    public long[] getDecisionLatenciesNanos() {
        return Arrays.copyOf(decisionNanos, decisionCount);
    }

    /**
     * RiskGrid.update() вимагає z у ВІДКРИТОМУ інтервалі (0,1), бо
     * log-odds ln(z/(1-z)) не визначений на межах.
     *
     * Відсутність детекції (raw z = 0.0) НЕ клемпиться до довільного EPS = 1e-6:
     * таке значення трактувало б "детектор нічого не побачив" як практично
     * достовірне підтвердження відсутності міни (Δl ≈ -11 при w = 0.8 - одна
     * така відсутність нівелює кілька впевнених позитивних спостережень).
     * Замість цього нижня межа Z_FLOOR калібрується через виміряну надійність
     * детектора: Z_FLOOR = 1 - recall, де recall = 0.933 виміряно на тестовій
     * вибірці (ваги PyTorch, розділ 4.2). Це означає, що "немає детекції"
     * трактується як залишкова ймовірність пропуску міни (~6.7 %), а не як
     * майже певна її відсутність - оцінка z лишається агрегованим показником
     * ризику, відкалиброваним надійністю сенсора, а не каліброваною
     * ймовірністю в строгому байєсівському сенсі.
     *
     * Верхня межа Z_CEIL - суто числовий зажим (уникнути log(0)), а не
     * модельне припущення: конфідентності детектора з даних (>= τ_det = 0.25)
     * і так рідко сягають рівно 1.0.
     */
    private static final double Z_FLOOR = 0.067; // 1 - recall (0.933), тестова вибірка, ваги PyTorch
    private static final double Z_CEIL = 1.0 - 1e-6; // лише числова стабільність

    private static double clampZ(double z) {
        if (z <= 0.0) return Z_FLOOR;
        if (z >= 1.0) return Z_CEIL;
        return z;
    }
}