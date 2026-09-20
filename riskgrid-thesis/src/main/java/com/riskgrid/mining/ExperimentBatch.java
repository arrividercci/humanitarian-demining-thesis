package com.riskgrid.mining;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Пакетний експеримент для статті. Для кожного seed (окрема територія) і
 * кожної конфігурації (кількість дронів, захоплення клітинок увімк./вимк.,
 * затримка польоту) виконує ОДИН повний запуск до повного покриття та
 * дописує рядок у CSV.
 *
 * Що вимірюється:
 *   1. Масштабування: elapsed_ms для 1/2/4/8 дронів (прискорення рахує Python).
 *   2. Абляція: claiming=0 vs claiming=1 - concurrent_duplicates, revisits, elapsed_ms.
 *      revisits - усі повторні обробки клітинок; stale_revisits - ті з них, де клітинка
 *      була невідвіданою на момент вибору (застарілий вибір), решта - навмисні повтори.
 *   3. Затримка рішення: lat_mean/p50/p95/p99/max (вибір цілі + захоплення), мс.
 *   4. Коректність: precision/recall/F1 та risk_mae_vs_seq - середня абсолютна
 *      різниця фінальної карти ризику від еталонного запуску з 1 дроном
 *      (та сама територія).
 *
 * Методика: прогрів JIT/ONNX перед вимірюваннями; один потік ONNX на дрон;
 * однакові стартові позиції (перші N з однієї послідовності) для всіх
 * конфігурацій одного seed; запис у CSV після завершення КОЖНОГО seed
 * (атомарно), тож перервану серію можна продовжити - завершені seed
 * пропускаються.
 *
 * Запуск (приклад):
 *   mvn exec:java "-Dexec.mainClass=com.riskgrid.mining.ExperimentBatch" "-Dexec.args=--seeds 3 --delays 0"
 *
 * Параметри (усі необов'язкові):
 *   --seeds N            кількість територій (типово 20)
 *   --seed-start S       перший seed (типово 1)
 *   --drones 1,2,4,8     кількості дронів (має містити 1; сортується)
 *   --delays 0,20        затримка польоту, мс на одиницю відстані (клітинку)
 *   --rows R --cols C    розмір сітки (типово 30x30)
 *   --clusters K --cluster-radius X
 *   --search-radius R --lambda L --weight W --threshold T
 *   --min-gain G         поріг інформаційного виграшу (0 = вимкнено; див. RoutePlanner)
 *   --with-sectors       режим порівняння: для кожної конфігурації (n > 1) запускаються
 *                        динамічний метод (захоплення клітинок) і базовий метод - статичні
 *                        сектори (без захоплення); режим "без захоплення" не запускається.
 *                        У CSV: сектори мають partition = 1 і claiming = 0
 *   --max-obs N          максимум спостережень на клітинку (0 = вимкнено); гарантує
 *                        скінченність: кроків не більше N * rows * cols (із захопленням)
 *   --timeout-min M      ліміт часу на один запуск (типово 15)
 *   --max-steps-factor F ліміт кроків = F * rows * cols (типово 4); запуск, що його перевищив
 *                        (дрони не завершують покриття), позначається complete=0 і зупиняється
 *   --no-warmup          пропустити прогрів
 *   --out FILE           файл результатів (типово batch_results.csv)
 *   --mine-dir DIR --bg-dir DIR --onnx FILE
 */
public final class ExperimentBatch {

    private static final double CELL_SIZE = 1.0;
    private static final double ORIGIN = 0.0;

    private static final String HEADER = String.join(",",
            "seed", "rows", "cols", "drones", "claiming", "travel_delay_ms", "elapsed_ms", "complete",
            "steps", "unique_cells", "revisits", "stale_revisits", "claim_failures", "concurrent_duplicates", "fallback_jumps",
            "decisions", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "lat_max_ms",
            "total_mines", "accuracy", "precision", "recall", "f1", "risk_mae_vs_seq", "threshold", "min_gain", "max_obs", "cap_rejections", "partition", "idle_fraction");

    /** Результат одного запуску. */
    private record RunOutcome(double elapsedMs, boolean complete, int steps, int uniqueCells,
                              int revisits, int staleRevisits, int claimFailures, int capRejections,
                              double idleFraction, long conflicts, int fallbackJumps,
                              long[] latenciesNanos, double[] risk, DetectionMetrics.Result metrics) {}

    /** Налаштування з командного рядка. */
    private static final class Config {
        int rows = 30, cols = 30, seeds = 20;
        long seedStart = 1;
        int[] droneCounts = {1, 2, 4, 8};
        double[] delays = {0.0, 20.0};
        int clusters = 3;
        double clusterRadius = 4.0;
        double searchRadius = 5.0, lambda = 0.1, weight = 0.8,
                threshold = 0.5; // ризик >= 0.5: "імовірніше міна, ніж ні"; не підбирається під дані
        double minGain = 0.0; // 0 = початкова поведінка планувальника
        int maxObs = 0;       // 0 = без обмеження спостережень на клітинку
        boolean withSectors = false; // true = додатково запускати базовий метод: статичні сектори
        int timeoutMin = 15;
        double maxStepsFactor = 4.0; // ліміт кроків = factor * rows * cols; далі запуск вважається незавершеним
        boolean warmup = true;
        String out = "batch_results.csv";
        Path mineDir = Paths.get("terrain-images", "mine");
        Path bgDir = Paths.get("terrain-images", "background");
        String onnx = "model/best.onnx";

        static Config parse(String[] a) {
            Config c = new Config();
            for (int k = 0; k < a.length; k++) {
                String key = a[k];
                if (key.equals("--no-warmup")) { c.warmup = false; continue; }
                if (key.equals("--with-sectors")) { c.withSectors = true; continue; }
                if (k + 1 >= a.length) throw new IllegalArgumentException("Немає значення для " + key);
                String v = a[++k];
                switch (key) {
                    case "--seeds" -> c.seeds = Integer.parseInt(v);
                    case "--seed-start" -> c.seedStart = Long.parseLong(v);
                    case "--drones" -> c.droneCounts = Arrays.stream(v.split(",")).mapToInt(Integer::parseInt).sorted().toArray();
                    case "--delays" -> c.delays = Arrays.stream(v.split(",")).mapToDouble(Double::parseDouble).toArray();
                    case "--rows" -> c.rows = Integer.parseInt(v);
                    case "--cols" -> c.cols = Integer.parseInt(v);
                    case "--clusters" -> c.clusters = Integer.parseInt(v);
                    case "--cluster-radius" -> c.clusterRadius = Double.parseDouble(v);
                    case "--search-radius" -> c.searchRadius = Double.parseDouble(v);
                    case "--lambda" -> c.lambda = Double.parseDouble(v);
                    case "--weight" -> c.weight = Double.parseDouble(v);
                    case "--threshold" -> c.threshold = Double.parseDouble(v);
                    case "--min-gain" -> c.minGain = Double.parseDouble(v);
                    case "--max-obs" -> c.maxObs = Integer.parseInt(v);
                    case "--timeout-min" -> c.timeoutMin = Integer.parseInt(v);
                    case "--max-steps-factor" -> c.maxStepsFactor = Double.parseDouble(v);
                    case "--out" -> c.out = v;
                    case "--mine-dir" -> c.mineDir = Paths.get(v);
                    case "--bg-dir" -> c.bgDir = Paths.get(v);
                    case "--onnx" -> c.onnx = v;
                    default -> throw new IllegalArgumentException("Невідомий параметр: " + key);
                }
            }
            if (c.droneCounts.length == 0 || c.droneCounts[0] != 1) {
                throw new IllegalArgumentException("--drones має містити 1 (еталон для прискорення й коректності)");
            }
            return c;
        }
    }

    public static void main(String[] args) throws Exception {
        System.setOut(new java.io.PrintStream(System.out, true, StandardCharsets.UTF_8));
        System.setErr(new java.io.PrintStream(System.err, true, StandardCharsets.UTF_8));
        Config cfg = Config.parse(args);

        writeMeta(cfg, args);
        Set<Long> doneSeeds = readDoneSeeds(cfg);

        try (MineDetectorOnnx detector = new MineDetectorOnnx(cfg.onnx, 1)) {
            if (cfg.warmup) {
                warmup(cfg, detector);
            }

            long batchStart = System.nanoTime();
            for (int s = 0; s < cfg.seeds; s++) {
                long seed = cfg.seedStart + s;
                if (doneSeeds.contains(seed)) {
                    System.out.printf("Seed %d: вже є в %s - пропускаю%n", seed, cfg.out);
                    continue;
                }
                long seedStart = System.nanoTime();
                List<String> mapLines = new ArrayList<>();
                List<String> truthLines = new ArrayList<>();
                List<String> rows = runSeed(cfg, detector, seed, mapLines, truthLines);
                // Карти й ground truth пишемо ПЕРЕД основним CSV: якщо запуск перервуть
                // між записами, seed просто повториться цілком.
                appendRows(sideFile(cfg, "_maps"), mapHeader(cfg), mapLines);
                appendRows(sideFile(cfg, "_truth"), truthHeader(cfg), truthLines);
                appendRows(cfg.out, HEADER, rows);
                System.out.printf(Locale.US, "Seed %d готово (%d рядків, %.1f хв). Усього минуло %.1f хв.%n",
                        seed, rows.size(), (System.nanoTime() - seedStart) / 6e10,
                        (System.nanoTime() - batchStart) / 6e10);
            }
        }
        System.out.println("Готово. Результати: " + Paths.get(cfg.out).toAbsolutePath());
    }

    // ---------------------------------------------------------------------

    /** Усі конфігурації для одного seed на одній території. Повертає готові CSV-рядки. */
    private static List<String> runSeed(Config c, MineDetectorOnnx detector, long seed,
                                         List<String> mapLines, List<String> truthLines) throws Exception {
        SimulatedTerrain terrain = new SimulatedTerrain(c.rows, c.cols, c.mineDir, c.bgDir,
                c.clusters, c.clusterRadius, seed);
        int maxDrones = c.droneCounts[c.droneCounts.length - 1];
        int[][] starts = generateStartCells(maxDrones, c.rows, c.cols, seed * 1_000_003L + 999L);
        int totalMines = countMines(terrain, c.rows, c.cols);
        truthLines.add(truthRow(terrain, seed, c));

        List<String> out = new ArrayList<>();
        double[] referenceRisk = null; // фінальна карта еталонного запуску з 1 дроном

        for (double delay : c.delays) {
            for (int n : c.droneCounts) {
                // Для 1 дрона конфлікти неможливі, тож режим "без захоплення" не відрізняється - пропускаємо.
                // Варіанти: 0 = динамічний метод із захопленням, 1 = статичні сектори
                // (захоплення не використовується - сектори не перетинаються), 2 = без захоплення.
                // Для 1 дрона всі варіанти збігаються, тож запускається лише варіант 0.
                int[] variants = (n == 1) ? new int[]{0}
                        : (c.withSectors ? new int[]{0, 1} : new int[]{0, 2});
                for (int variant : variants) {
                    boolean sectorMode = (variant == 1);
                    boolean claiming = (variant == 0);
                    System.out.printf(Locale.US, "  seed=%d  дронів=%d  захоплення=%s  політ=%.0f мс/кл ... ",
                            seed, n, sectorMode ? "сектори" : (claiming ? "так" : "ні"), delay);
                    RunOutcome o = runOnce(c, detector, terrain, n, starts, claiming, sectorMode, delay);
                    System.out.printf(Locale.US, "%.1f с, кроків=%d, дублів=%d%n",
                            o.elapsedMs / 1000.0, o.steps, o.conflicts);

                    if (n == 1 && referenceRisk == null && o.complete) {
                        referenceRisk = o.risk;
                    }
                    double mae = (referenceRisk == null) ? Double.NaN : meanAbsDiff(o.risk, referenceRisk);
                    out.add(csvRow(c, seed, n, claiming, sectorMode, delay, totalMines, o, mae));
                    mapLines.add(mapRow(seed, n, claiming, delay, o));
                }
            }
        }
        return out;
    }

    private static RunOutcome runOnce(Config c, MineDetectorOnnx detector, SimulatedTerrain terrain,
                                       int droneCount, int[][] starts,
                                       boolean claiming, boolean sectorMode,
                                       double delayMsPerUnit) throws Exception {
        return runOnce(c.rows, c.cols, c, detector, terrain, droneCount, starts, claiming, sectorMode,
                delayMsPerUnit);
    }

    private static RunOutcome runOnce(int rows, int cols, Config c, MineDetectorOnnx detector,
                                       SimulatedTerrain terrain, int droneCount, int[][] starts,
                                       boolean claiming, boolean sectorMode,
                                       double delayMsPerUnit) throws Exception {
        RiskGrid grid = new RiskGrid(rows, cols, CELL_SIZE, ORIGIN, ORIGIN);
        RoutePlanner planner = new RoutePlanner(grid, CELL_SIZE, ORIGIN, ORIGIN);
        DroneCoordinator coordinator = new DroneCoordinator();
        CoverageTracker coverage = new CoverageTracker(rows, cols);
        ConflictMonitor monitor = new ConflictMonitor();
        AtomicBoolean stop = new AtomicBoolean(false);
        final boolean useSectors = sectorMode;
        DroneAgent.Options opts = new DroneAgent.Options(useSectors ? false : claiming, delayMsPerUnit,
                monitor, c.minGain, c.maxObs);

        // Секторний режим: кожен дрон отримує власний планувальник, обмежений його сектором.
        List<RoutePlanner.Region> sectors = null;
        int[] sectorOf = null;
        if (useSectors) {
            sectors = SectorPartition.partition(rows, cols, droneCount);
            sectorOf = SectorPartition.assign(sectors, starts, droneCount);
        }

        long t0 = System.nanoTime();
        List<DroneAgent> agents = new ArrayList<>();
        for (int d = 0; d < droneCount; d++) {
            RoutePlanner.Point start = planner.cellCenter(starts[d][0], starts[d][1]);
            RoutePlanner dronePlanner = useSectors
                    ? new RoutePlanner(grid, CELL_SIZE, ORIGIN, ORIGIN, sectors.get(sectorOf[d]))
                    : planner;
            agents.add(new DroneAgent(d, grid, dronePlanner, coordinator, coverage, terrain, detector,
                    start, c.searchRadius, c.lambda, c.weight, null, t0, stop, opts));
        }

        ExecutorService executor = Executors.newFixedThreadPool(droneCount);
        List<Future<?>> futures = new ArrayList<>();
        for (DroneAgent a : agents) {
            futures.add(executor.submit(a));
        }
        executor.shutdown();

        // Чекаємо завершення, але припиняємо запуск, якщо він не збігається: перевищено
        // бюджет кроків (дрони крутяться на повторних обробках) або ліміт часу.
        long stepBudget = (long) (c.maxStepsFactor * rows * cols);
        long deadline = t0 + TimeUnit.MINUTES.toNanos(c.timeoutMin);
        boolean finished = false;
        String abortReason = null;
        while (true) {
            if (executor.awaitTermination(50, TimeUnit.MILLISECONDS)) {
                finished = true;
                break;
            }
            long total = 0;
            for (DroneAgent a : agents) total += a.getStepsCompleted();
            if (total > stepBudget) { abortReason = "перевищено бюджет кроків (" + stepBudget + ")"; break; }
            if (System.nanoTime() > deadline) { abortReason = "перевищено ліміт часу (" + c.timeoutMin + " хв)"; break; }
        }
        double elapsedMs = (System.nanoTime() - t0) / 1_000_000.0;
        if (!finished) {
            System.err.println("УВАГА: запуск не завершився - " + abortReason + "; зупиняю");
            stop.set(true);
            executor.shutdownNow();
            executor.awaitTermination(30, TimeUnit.SECONDS);
        }
        for (Future<?> f : futures) {
            try {
                f.get(1, TimeUnit.SECONDS);
            } catch (ExecutionException e) {
                throw new IllegalStateException("Потік дрона впав: " + e.getCause(), e.getCause());
            }
        }

        int steps = 0, revisits = 0, staleRevisits = 0, claimFailures = 0, capRejections = 0,
                fallbacks = 0, latTotal = 0;
        for (DroneAgent a : agents) {
            steps += a.getStepsCompleted();
            revisits += a.getRevisits();
            staleRevisits += a.getStaleRevisits();
            claimFailures += a.getClaimFailures();
            capRejections += a.getCapRejections();
            fallbacks += a.getFallbackJumps();
            latTotal += a.getDecisionLatenciesNanos().length;
        }
        // Частка простою: 1 - (сума часів роботи дронів) / (кількість дронів * час найповільнішого).
        // 0 = усі дрони працювали до кінця; більше = нерівномірне навантаження.
        double maxFinish = 0.0, sumFinish = 0.0;
        for (DroneAgent a : agents) {
            long f = a.getFinishedAtNanos();
            double finish = (f >= 0) ? f : elapsedMs * 1_000_000.0; // незавершений дрон рахуємо до кінця запуску
            maxFinish = Math.max(maxFinish, finish);
            sumFinish += finish;
        }
        double idleFraction = (maxFinish > 0.0) ? 1.0 - sumFinish / (agents.size() * maxFinish) : 0.0;

        long[] lat = new long[latTotal];
        int pos = 0;
        for (DroneAgent a : agents) {
            long[] l = a.getDecisionLatenciesNanos();
            System.arraycopy(l, 0, lat, pos, l.length);
            pos += l.length;
        }
        Arrays.sort(lat);

        double[] risk = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                risk[i * cols + j] = grid.getRisk(i, j);
            }
        }
        DetectionMetrics.Result metrics = DetectionMetrics.compute(grid, terrain, coverage, rows, cols, c.threshold);

        return new RunOutcome(elapsedMs, finished && coverage.isComplete(), steps,
                rows * cols - coverage.remainingCount(), revisits, staleRevisits, claimFailures, capRejections,
                idleFraction,
                monitor.conflictCount(), fallbacks, lat, risk, metrics);
    }

    /** Прогрів JIT і ONNX Runtime на малій сітці для всіх кількостей дронів; результати відкидаються. */
    private static void warmup(Config c, MineDetectorOnnx detector) throws Exception {
        System.out.println("Прогрів (результати не записуються)...");
        int wr = 12, wc = 12;
        SimulatedTerrain t = new SimulatedTerrain(wr, wc, c.mineDir, c.bgDir, 1, 2.0, 0L);
        int[][] starts = generateStartCells(c.droneCounts[c.droneCounts.length - 1], wr, wc, 12345L);
        for (int pass = 0; pass < 2; pass++) {
            for (int n : c.droneCounts) {
                runOnce(wr, wc, c, detector, t, n, starts, true, false, 0.0);
                if (c.withSectors && n > 1) {
                    runOnce(wr, wc, c, detector, t, n, starts, false, true, 0.0);
                }
            }
        }
    }

    // ---------------------------------------------------------------------

    private static String csvRow(Config c, long seed, int drones, boolean claiming, boolean sectorMode,
                                  double delay, int totalMines, RunOutcome o, double mae) {
        long[] lat = o.latenciesNanos;
        double mean = lat.length == 0 ? Double.NaN : Arrays.stream(lat).average().orElse(Double.NaN) / 1e6;
        return String.format(Locale.US,
                "%d,%d,%d,%d,%d,%.1f,%.3f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.5f,%.5f,%.5f,%.5f,%.5f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f,%.6f,%d,%d,%d,%.5f",
                seed, c.rows, c.cols, drones, claiming ? 1 : 0, delay, o.elapsedMs, o.complete ? 1 : 0,
                o.steps, o.uniqueCells, o.revisits, o.staleRevisits, o.claimFailures, o.conflicts, o.fallbackJumps,
                lat.length, mean, pct(lat, 0.50), pct(lat, 0.95), pct(lat, 0.99),
                lat.length == 0 ? Double.NaN : lat[lat.length - 1] / 1e6,
                totalMines, o.metrics.accuracy(), o.metrics.precision(), o.metrics.recall(), o.metrics.f1(),
                mae, c.threshold, c.minGain, c.maxObs, o.capRejections, sectorMode ? 1 : 0, o.idleFraction);
    }

    /** Перцентиль (метод найближчого рангу) у мс; масив уже відсортований. */
    private static double pct(long[] sortedNanos, double p) {
        if (sortedNanos.length == 0) return Double.NaN;
        int idx = (int) Math.ceil(p * sortedNanos.length) - 1;
        return sortedNanos[Math.max(0, Math.min(idx, sortedNanos.length - 1))] / 1e6;
    }

    private static double meanAbsDiff(double[] a, double[] b) {
        double s = 0;
        for (int k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]);
        return s / a.length;
    }

    private static int countMines(SimulatedTerrain t, int rows, int cols) {
        int n = 0;
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) if (t.isMineCell(i, j)) n++;
        return n;
    }

    /** До maxCount унікальних стартових клітинок; перші N однакові для будь-якої кількості дронів. */
    private static int[][] generateStartCells(int maxCount, int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        Set<Long> used = new HashSet<>();
        int[][] result = new int[maxCount][2];
        int filled = 0;
        while (filled < maxCount) {
            int i = rnd.nextInt(rows);
            int j = rnd.nextInt(cols);
            long key = ((long) i << 32) | (j & 0xffffffffL);
            if (used.add(key)) {
                result[filled][0] = i;
                result[filled][1] = j;
                filled++;
            }
        }
        return result;
    }

    // ---- карти ризику й ground truth (для метрик, які захочеться порахувати пізніше) ----

    /** Фінальна карта ризику одного запуску, розгорнута по рядках (індекс = i * cols + j). */
    private static String mapRow(long seed, int drones, boolean claiming, double delay, RunOutcome o) {
        StringBuilder sb = new StringBuilder();
        sb.append(seed).append(',').append(drones).append(',').append(claiming ? 1 : 0).append(',')
                .append(String.format(Locale.US, "%.1f", delay)).append(',').append(o.complete ? 1 : 0);
        for (double v : o.risk) {
            sb.append(',').append(String.format(Locale.US, "%.6f", v));
        }
        return sb.toString();
    }

    /** Розкладка мін території (1 = клітинка мінного кластера), в тому ж порядку, що й карта ризику. */
    private static String truthRow(SimulatedTerrain t, long seed, Config c) {
        StringBuilder sb = new StringBuilder();
        sb.append(seed);
        for (int i = 0; i < c.rows; i++) {
            for (int j = 0; j < c.cols; j++) {
                sb.append(',').append(t.isMineCell(i, j) ? 1 : 0);
            }
        }
        return sb.toString();
    }

    private static String mapHeader(Config c) {
        StringBuilder sb = new StringBuilder("seed,drones,claiming,travel_delay_ms,complete");
        for (int k = 0; k < c.rows * c.cols; k++) sb.append(",r").append(k);
        return sb.toString();
    }

    private static String truthHeader(Config c) {
        StringBuilder sb = new StringBuilder("seed");
        for (int k = 0; k < c.rows * c.cols; k++) sb.append(",m").append(k);
        return sb.toString();
    }

    /** batch_mg0.csv -> batch_mg0_maps.csv / batch_mg0_truth.csv */
    private static String sideFile(Config c, String suffix) {
        return c.out.replaceAll("\\.csv$", "") + suffix + ".csv";
    }

    // ---- файли -----------------------------------------------------------

    private static synchronized void appendRows(String file, String header, List<String> rows) throws IOException {
        Path path = Paths.get(file);
        boolean exists = Files.exists(path) && Files.size(path) > 0;
        try (BufferedWriter w = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            if (!exists) {
                w.write(header);
                w.newLine();
            }
            for (String r : rows) {
                w.write(r);
                w.newLine();
            }
        }
    }

    /** Завершені seed для тієї самої сітки - їх при повторному запуску пропускаємо. */
    private static Set<Long> readDoneSeeds(Config c) throws IOException {
        Set<Long> done = new HashSet<>();
        Path path = Paths.get(c.out);
        if (!Files.exists(path)) return done;
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        for (int k = 1; k < lines.size(); k++) {
            String[] p = lines.get(k).split(",");
            if (p.length > 2 && Integer.parseInt(p[1]) == c.rows && Integer.parseInt(p[2]) == c.cols) {
                done.add(Long.parseLong(p[0]));
            }
        }
        return done;
    }

    /** Середовище запуску - для розділу "Методика" у статті. */
    private static void writeMeta(Config c, String[] args) throws IOException {
        Path path = Paths.get(c.out.replaceAll("\\.csv$", "") + "_meta.txt");
        String text = "=== " + Instant.now() + " ===\n"
                + "args: " + String.join(" ", args) + "\n"
                + "cpu_logical_cores: " + Runtime.getRuntime().availableProcessors() + "\n"
                + "java: " + System.getProperty("java.version") + " (" + System.getProperty("java.vm.name") + ")\n"
                + "os: " + System.getProperty("os.name") + " " + System.getProperty("os.version")
                + " " + System.getProperty("os.arch") + "\n"
                + "max_heap_mb: " + Runtime.getRuntime().maxMemory() / (1024 * 1024) + "\n"
                + "onnx_intra_op_threads: 1\n\n";
        Files.writeString(path, text, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND);
    }
}