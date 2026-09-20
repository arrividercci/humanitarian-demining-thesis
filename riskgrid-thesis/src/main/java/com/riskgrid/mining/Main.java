package com.riskgrid.mining;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.Random;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Головний клас симуляції. Автоматично порівнює конфігурації з 1, 2, 4 і 8
 * дронами на ОДНІЙ І ТІЙ САМІЙ детермінованій території (SimulatedTerrain
 * будується один раз із фіксованим seed) - це гарантує, що різниця в часі
 * покриття пояснюється саме паралелізмом, а не різною складністю території.
 *
 * Для кожної конфігурації RiskGrid/DroneCoordinator/CoverageTracker
 * створюються заново (щоб не переносити стан між запусками), а
 * SimulatedTerrain і MineDetectorOnnx (дорогі в ініціалізації) - спільні
 * для всіх запусків.
 */
public final class Main {

    // ---- Конфігурація - підправ під свою реальну структуру папок ----
    private static final int ROWS = 30;
    private static final int COLS = 30;
    private static final double CELL_SIZE = 1.0;
    private static final double ORIGIN_X = 0.0;
    private static final double ORIGIN_Y = 0.0;

    private static final double SEARCH_RADIUS = 5.0;
    private static final double LAMBDA = 0.1;
    private static final double SENSOR_WEIGHT = 0.8;

    private static final int CLUSTER_COUNT = 3;
    private static final double CLUSTER_RADIUS_CELLS = 4.0;
    private static final long TERRAIN_SEED = 42L;
    private static final long START_POSITIONS_SEED = 999L;

    private static final Path MINE_IMAGES_DIR = Paths.get("terrain-images", "mine");
    private static final Path BACKGROUND_IMAGES_DIR = Paths.get("terrain-images", "background");
    private static final String ONNX_MODEL_PATH = "model/best.onnx";

    private static final int[] DRONE_COUNTS = {1, 2, 4, 8};
    private static final int MAX_DRONES = 8;
    private static final double DETECTION_THRESHOLD = 0.422; // F2-оптимальний поріг (evaluate_detection.py): Precision=1.000, Recall=0.841
    // -------------------------------------------------------------------

    public static void main(String[] args) throws Exception {
        System.setOut(new java.io.PrintStream(System.out, true, java.nio.charset.StandardCharsets.UTF_8));

        SimulatedTerrain terrain = new SimulatedTerrain(
                ROWS, COLS, MINE_IMAGES_DIR, BACKGROUND_IMAGES_DIR,
                CLUSTER_COUNT, CLUSTER_RADIUS_CELLS, TERRAIN_SEED);

        // Розкладка території однакова для всіх запусків (фіксований seed) -
        // достатньо зберегти один раз, потім Python-скрипт розфарбує фон карти.
        writeTerrainLayout(terrain);

        int[][] startCells = generateStartCells(MAX_DRONES, ROWS, COLS, START_POSITIONS_SEED);

        try (MineDetectorOnnx detector = new MineDetectorOnnx(ONNX_MODEL_PATH)) {

            List<RunResult> results = new ArrayList<>();
            for (int droneCount : DRONE_COUNTS) {
                System.out.println("=== Запуск з " + droneCount + " дрон(ами) ===");
                RunResult result = runSimulation(droneCount, terrain, detector, startCells);
                results.add(result);
                System.out.printf("  Час покриття: %.1f мс, кроків усього: %d, детекцій: %d%n",
                        result.elapsedMillis, result.totalSteps, result.totalDetections);
            }

            printComparisonTable(results);
        }
    }

    private static RunResult runSimulation(int droneCount,
                                            SimulatedTerrain terrain,
                                            MineDetectorOnnx detector,
                                            int[][] startCells) throws InterruptedException, IOException {

        RiskGrid riskGrid = new RiskGrid(ROWS, COLS, CELL_SIZE, ORIGIN_X, ORIGIN_Y);
        RoutePlanner routePlanner = new RoutePlanner(riskGrid, CELL_SIZE, ORIGIN_X, ORIGIN_Y);
        DroneCoordinator coordinator = new DroneCoordinator();
        CoverageTracker coverage = new CoverageTracker(ROWS, COLS);
        ConcurrentLinkedQueue<DroneAgent.VisitRecord> visitLog = new ConcurrentLinkedQueue<>();

        long startTime = System.nanoTime();

        List<DroneAgent> agents = new ArrayList<>();
        for (int d = 0; d < droneCount; d++) {
            int startI = startCells[d][0];
            int startJ = startCells[d][1];
            RoutePlanner.Point startPoint = routePlanner.cellCenter(startI, startJ);

            agents.add(new DroneAgent(
                    d, riskGrid, routePlanner, coordinator, coverage, terrain, detector,
                    startPoint, SEARCH_RADIUS, LAMBDA, SENSOR_WEIGHT,
                    visitLog, startTime, null)); // null: пакетний режим ніколи не зупиняється достроково
        }

        ExecutorService executor = Executors.newFixedThreadPool(droneCount);
        for (DroneAgent agent : agents) {
            executor.submit(agent);
        }
        executor.shutdown();
        boolean finished = executor.awaitTermination(10, TimeUnit.MINUTES);
        long elapsedNanos = System.nanoTime() - startTime;

        if (!finished) {
            System.err.println("УВАГА: симуляція з " + droneCount + " дронами не завершилась за 10 хвилин");
        }

        int totalSteps = agents.stream().mapToInt(DroneAgent::getStepsCompleted).sum();
        int totalDetections = agents.stream().mapToInt(DroneAgent::getDetectionsAboveThreshold).sum();

        writeVisitLog(droneCount, visitLog);
        writeFinalRiskGrid(droneCount, riskGrid, terrain);

        DetectionMetrics.Result metrics = DetectionMetrics.compute(
                riskGrid, terrain, coverage, ROWS, COLS, DETECTION_THRESHOLD);
        int totalMines = countMineCells(terrain);
        int uniqueCellsCovered = ROWS * COLS - coverage.remainingCount();
        ExperimentRecorder.record("batch", ROWS, COLS, droneCount, TERRAIN_SEED, DETECTION_THRESHOLD,
                CLUSTER_COUNT, CLUSTER_RADIUS_CELLS, elapsedNanos / 1_000_000.0, totalSteps,
                uniqueCellsCovered, totalMines, totalDetections, metrics);

        return new RunResult(droneCount, elapsedNanos / 1_000_000.0, totalSteps, totalDetections);
    }

    /**
     * Генерує до maxCount унікальних стартових клітинок, детерміновано за seed.
     * Однакові перші N позицій використовуються для будь-якої конфігурації
     * (наприклад, дрон #0 завжди стартує з тієї самої клітинки незалежно
     * від того, скільки всього дронів у поточному запуску).
     */
    private static int[][] generateStartCells(int maxCount, int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        java.util.Set<Long> used = new java.util.HashSet<>();
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

    /** Одноразово зберігає розкладку території (яка клітинка - міна) у CSV для Python-скрипта візуалізації. */
    private static void writeTerrainLayout(SimulatedTerrain terrain) throws IOException {
        Path outPath = Paths.get("terrain_layout.csv");
        try (BufferedWriter writer = Files.newBufferedWriter(outPath,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writer.write("i,j,is_mine");
            writer.newLine();
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    writer.write(i + "," + j + "," + terrain.isMineCell(i, j));
                    writer.newLine();
                }
            }
        }
        System.out.println("Розкладку території збережено у " + outPath.toAbsolutePath());
    }

    /** Зберігає журнал обходу однієї конфігурації у CSV для Python-скрипта візуалізації. */
    private static void writeVisitLog(int droneCount, Queue<DroneAgent.VisitRecord> log) throws IOException {
        Path outPath = Paths.get("visit_log_" + droneCount + "drones.csv");
        try (BufferedWriter writer = Files.newBufferedWriter(outPath,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writer.write("drone_id,i,j,z,elapsed_nanos");
            writer.newLine();
            for (DroneAgent.VisitRecord r : log) {
                writer.write(r.droneId() + "," + r.i() + "," + r.j() + "," + r.z() + "," + r.elapsedNanos());
                writer.newLine();
            }
        }
        System.out.println("Журнал обходу (" + droneCount + " дронів) збережено у " + outPath.toAbsolutePath());
    }

    /**
     * Зберігає фінальну карту risk/uncertainty для кожної клітинки разом з
     * ground truth (is_mine) - основа для кількісної оцінки якості детекції
     * (precision/recall), а не лише візуальної перевірки по GIF.
     */
    private static void writeFinalRiskGrid(int droneCount, RiskGrid grid, SimulatedTerrain terrain) throws IOException {
        Path outPath = Paths.get("final_risk_" + droneCount + "drones.csv");
        try (BufferedWriter writer = Files.newBufferedWriter(outPath,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writer.write("i,j,risk,uncertainty,observation_count,is_mine");
            writer.newLine();
            for (int i = 0; i < ROWS; i++) {
                for (int j = 0; j < COLS; j++) {
                    writer.write(i + "," + j + ","
                            + grid.getRisk(i, j) + "," + grid.getUncertainty(i, j) + ","
                            + grid.getObservationCount(i, j) + "," + terrain.isMineCell(i, j));
                    writer.newLine();
                }
            }
        }
        System.out.println("Фінальну карту risk (" + droneCount + " дронів) збережено у " + outPath.toAbsolutePath());
    }

    /** Кількість клітинок, що справді належать до кластера-міни (ground truth) - для журналу експериментів. */
    private static int countMineCells(SimulatedTerrain terrain) {
        int count = 0;
        for (int i = 0; i < ROWS; i++) {
            for (int j = 0; j < COLS; j++) {
                if (terrain.isMineCell(i, j)) count++;
            }
        }
        return count;
    }

    private static void printComparisonTable(List<RunResult> results) {
        double baselineMillis = results.get(0).elapsedMillis;

        System.out.println();
        System.out.println("=== Підсумкове порівняння ===");
        System.out.printf("%-8s %-15s %-12s %-10s %-12s%n",
                "Дронів", "Час (мс)", "Прискорення", "Кроків", "Детекцій");
        for (RunResult r : results) {
            double speedup = baselineMillis / r.elapsedMillis;
            System.out.printf("%-8d %-15.1f %-12.2fx %-10d %-12d%n",
                    r.droneCount, r.elapsedMillis, speedup, r.totalSteps, r.totalDetections);
        }
    }

    private static final class RunResult {
        final int droneCount;
        final double elapsedMillis;
        final int totalSteps;
        final int totalDetections;

        RunResult(int droneCount, double elapsedMillis, int totalSteps, int totalDetections) {
            this.droneCount = droneCount;
            this.elapsedMillis = elapsedMillis;
            this.totalSteps = totalSteps;
            this.totalDetections = totalDetections;
        }
    }
}