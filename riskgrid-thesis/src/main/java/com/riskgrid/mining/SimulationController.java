package com.riskgrid.mining;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Керує одним "запуском" симуляції в живому GUI: створює територію та
 * дронів, запускає їх у фонових потоках, дає доступ до стану для панелей
 * відображення (SimulationPanel, MetricsPanel), і дозволяє достроково
 * зупинити (на відміну від пакетного режиму в Main, де симуляція завжди
 * триває до повного покриття).
 */
public final class SimulationController {

    private static final double CELL_SIZE = 1.0;
    private static final double ORIGIN_X = 0.0;
    private static final double ORIGIN_Y = 0.0;
    private static final double SEARCH_RADIUS = 5.0;
    private static final double LAMBDA = 0.1;
    private static final double SENSOR_WEIGHT = 0.8;

    private final MineDetectorOnnx detector; // спільний, дорогий у створенні - живе довше за один запуск

    private RiskGrid riskGrid;
    private RoutePlanner routePlanner;
    private DroneCoordinator coordinator;
    private CoverageTracker coverage;
    private SimulatedTerrain terrain;
    private List<DroneAgent> agents;
    private ExecutorService executor;
    private AtomicBoolean stopFlag;
    private long startTimeNanos;
    private int rows, cols, droneCount, clusterCount;
    private double clusterRadiusCells, threshold = 0.5;
    private long seedUsed;
    private volatile boolean running = false;
    private volatile boolean recorded = false; // щоб не записати результат у журнал двічі

    public SimulationController(MineDetectorOnnx detector) {
        this.detector = detector;
    }

    /**
     * Запускає нову симуляцію. Викликати лише коли !isRunning().
     * clusterSeed < 0 означає "випадкова територія щоразу" (System.nanoTime());
     * невід'ємне значення дає відтворювану територію (корисно для порівнянь).
     */
    public synchronized void start(int rows, int cols, int droneCount,
                                    int clusterCount, double clusterRadiusCells,
                                    long clusterSeed,
                                    java.nio.file.Path mineImagesDir,
                                    java.nio.file.Path backgroundImagesDir) throws IOException {
        if (running) {
            throw new IllegalStateException("Симуляція вже запущена - спершу викликай stop()");
        }

        this.rows = rows;
        this.cols = cols;
        this.droneCount = droneCount;
        this.clusterCount = clusterCount;
        this.clusterRadiusCells = clusterRadiusCells;
        long seed = clusterSeed >= 0 ? clusterSeed : System.nanoTime();
        this.seedUsed = seed;
        this.recorded = false;

        this.terrain = new SimulatedTerrain(rows, cols, mineImagesDir, backgroundImagesDir,
                clusterCount, clusterRadiusCells, seed);
        this.riskGrid = new RiskGrid(rows, cols, CELL_SIZE, ORIGIN_X, ORIGIN_Y);
        this.routePlanner = new RoutePlanner(riskGrid, CELL_SIZE, ORIGIN_X, ORIGIN_Y);
        this.coordinator = new DroneCoordinator();
        this.coverage = new CoverageTracker(rows, cols);
        this.stopFlag = new AtomicBoolean(false);

        Random startRnd = new Random(seed); // ті самі стартові позиції для тієї самої seed
        this.agents = new ArrayList<>();
        this.startTimeNanos = System.nanoTime();
        for (int d = 0; d < droneCount; d++) {
            int startI = startRnd.nextInt(rows);
            int startJ = startRnd.nextInt(cols);
            RoutePlanner.Point startPoint = routePlanner.cellCenter(startI, startJ);

            agents.add(new DroneAgent(
                    d, riskGrid, routePlanner, coordinator, coverage, terrain, detector,
                    startPoint, SEARCH_RADIUS, LAMBDA, SENSOR_WEIGHT,
                    null, startTimeNanos, stopFlag)); // null: журнал у CSV в живому режимі не пишемо
        }

        this.executor = Executors.newFixedThreadPool(droneCount);
        this.running = true;
        for (DroneAgent agent : agents) {
            executor.submit(agent);
        }
        executor.shutdown(); // нові завдання більше не приймаються; вже запущені - працюють до завершення
    }

    /** Достроково зупиняє всіх дронів (наприклад, за натисканням кнопки "Зупинити"). */
    public synchronized void stop() {
        if (!running) return;
        stopFlag.set(true);
    }

    public boolean isRunning() {
        if (running && executor != null && executor.isTerminated()) {
            running = false; // симуляція сама завершилась (повне покриття або зупинку підхопили всі потоки)
        }
        return running;
    }

    public boolean isComplete() {
        return coverage != null && coverage.isComplete();
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    /**
     * Дописує результат поточного запуску в experiment_history.csv.
     * Безпечно викликати кілька разів - запис відбудеться лише один раз
     * на run (керується прапорцем recorded), навіть якщо GUI викличе це
     * і по завершенню, і після натискання "Зупинити".
     */
    public synchronized void recordResultIfNeeded() {
        if (recorded || riskGrid == null) return;
        recorded = true;

        int totalSteps = agents.stream().mapToInt(DroneAgent::getStepsCompleted).sum();
        int totalDetections = agents.stream().mapToInt(DroneAgent::getDetectionsAboveThreshold).sum();
        int uniqueCellsCovered = rows * cols - coverage.remainingCount();
        int totalMines = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (terrain.isMineCell(i, j)) totalMines++;
            }
        }
        double elapsedMillis = (System.nanoTime() - startTimeNanos) / 1_000_000.0;

        DetectionMetrics.Result metrics = DetectionMetrics.compute(
                riskGrid, terrain, coverage, rows, cols, threshold);

        ExperimentRecorder.record("gui", rows, cols, droneCount, seedUsed, threshold,
                clusterCount, clusterRadiusCells, elapsedMillis, totalSteps,
                uniqueCellsCovered, totalMines, totalDetections, metrics);
    }

    public RiskGrid getRiskGrid() { return riskGrid; }
    public SimulatedTerrain getTerrain() { return terrain; }
    public CoverageTracker getCoverage() { return coverage; }
    public List<DroneAgent> getAgents() { return agents; }
    public int getRows() { return rows; }
    public int getCols() { return cols; }
}