package com.riskgrid.mining;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.Instant;

/**
 * Дописує результат кожного запуску симуляції (пакетного через Main чи
 * живого через DemoApp) у спільний CSV-журнал experiment_history.csv.
 * Це дає накопичену історію для подальшого аналізу (прискорення,
 * порівняння конфігурацій, графіки) окремим Python-скриптом, замість
 * одноразових результатів, які губляться після кожного запуску.
 *
 * Потокобезпечний (synchronized) - хоч записи й так відбуваються
 * послідовно (по одному на завершений run), а не паралельно всередині run.
 */
public final class ExperimentRecorder {

    private static final Path HISTORY_PATH = Paths.get("experiment_history.csv");
    private static final String HEADER = "run_id,timestamp,source,rows,cols,drone_count,seed,threshold," +
            "cluster_count,cluster_radius,elapsed_ms,total_steps,unique_cells_covered,total_mines," +
            "detections_raw,accuracy,precision,recall,f1,correlation,mae";

    private ExperimentRecorder() {}

    public static synchronized void record(String source, int rows, int cols, int droneCount,
                                            long seed, double threshold,
                                            int clusterCount, double clusterRadius,
                                            double elapsedMillis, int totalSteps, int uniqueCellsCovered,
                                            int totalMines, int detectionsRaw,
                                            DetectionMetrics.Result metrics) {
        try {
            boolean exists = Files.exists(HISTORY_PATH);
            try (BufferedWriter writer = Files.newBufferedWriter(HISTORY_PATH,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
                if (!exists) {
                    writer.write(HEADER);
                    writer.newLine();
                }
                String runId = Long.toString(System.nanoTime());
                writer.write(String.join(",",
                        runId,
                        Instant.now().toString(),
                        source,
                        Integer.toString(rows),
                        Integer.toString(cols),
                        Integer.toString(droneCount),
                        Long.toString(seed),
                        Double.toString(threshold),
                        Integer.toString(clusterCount),
                        Double.toString(clusterRadius),
                        Double.toString(elapsedMillis),
                        Integer.toString(totalSteps),
                        Integer.toString(uniqueCellsCovered),
                        Integer.toString(totalMines),
                        Integer.toString(detectionsRaw),
                        Double.toString(metrics.accuracy()),
                        Double.toString(metrics.precision()),
                        Double.toString(metrics.recall()),
                        Double.toString(metrics.f1()),
                        Double.toString(metrics.correlation()),
                        Double.toString(metrics.mae())
                ));
                writer.newLine();
            }
        } catch (IOException e) {
            System.err.println("Не вдалось записати у " + HISTORY_PATH + ": " + e.getMessage());
        }
    }
}
