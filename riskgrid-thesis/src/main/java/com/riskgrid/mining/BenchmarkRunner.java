package com.riskgrid.mining;

import java.util.*;
import java.util.concurrent.*;

/**
 * Бенчмарк "чистого" обчислювального прискорення паралельного оновлення
 * карти - без шуму від черги спостережень. Масив спостережень заздалегідь
 * ділиться на K рівних шматків, кожен шматок віддається одному потоку одразу
 * (одна синхронізація на весь шматок, а не на кожен елемент), що ізолює
 * саме прискорення обчислень від накладних витрат на координацію потоків.
 */
public class BenchmarkRunner {

    private static final int[] THREAD_COUNTS = { 1, 2, 4, 8 };
    private static final int[][] MAP_SIZES = { { 50, 50 }, { 200, 200 }, { 500, 500 } };
    private static final int OBSERVATIONS_PER_RUN = 2_000_000;
    private static final int WARMUP_RUNS = 2;
    private static final int MEASURED_RUNS = 5;

    public static void main(String[] args) throws Exception {
        System.out.println("map_size,threads,time_ms,throughput_obs_per_sec,speedup");

        for (int[] size : MAP_SIZES) {
            int rows = size[0], cols = size[1];
            List<SensorPipeline.Observation> observations = generateObservations(OBSERVATIONS_PER_RUN, rows, cols, 42L);

            double baselineTimeMs = -1;

            for (int threads : THREAD_COUNTS) {
                for (int w = 0; w < WARMUP_RUNS; w++) {
                    runOnceBatch(observations, rows, cols, threads);
                }

                List<Double> times = new ArrayList<>();
                for (int r = 0; r < MEASURED_RUNS; r++) {
                    times.add(runOnceBatch(observations, rows, cols, threads));
                }
                Collections.sort(times);
                double medianMs = times.get(times.size() / 2);

                if (threads == 1) baselineTimeMs = medianMs;

                double throughput = OBSERVATIONS_PER_RUN / (medianMs / 1000.0);
                double speedup = baselineTimeMs / medianMs;

                System.out.printf(Locale.US, "%dx%d,%d,%.1f,%.0f,%.2f%n",
                        rows, cols, threads, medianMs, throughput, speedup);
            }
        }
    }

    /** Ділить спостереження на threads шматків і обробляє їх паралельно без черги. */
    private static double runOnceBatch(List<SensorPipeline.Observation> observations, int rows, int cols, int threads)
            throws InterruptedException, ExecutionException {
        RiskGrid grid = new RiskGrid(rows, cols, 1.0, 0.0, 0.0);
        ExecutorService executor = Executors.newFixedThreadPool(threads);

        int n = observations.size();
        int chunkSize = (n + threads - 1) / threads;
        List<Callable<Void>> tasks = new ArrayList<>();

        for (int t = 0; t < threads; t++) {
            int start = t * chunkSize;
            int end = Math.min(start + chunkSize, n);
            if (start >= end) continue;
            List<SensorPipeline.Observation> chunk = observations.subList(start, end);

            tasks.add(() -> {
                for (SensorPipeline.Observation obs : chunk) {
                    int[] idx = grid.toCellIndex(obs.x(), obs.y());
                    int i = idx[0], j = idx[1];
                    if (i >= 0 && i < rows && j >= 0 && j < cols) {
                        grid.update(i, j, obs.z(), obs.w(), obs.timestamp());
                    }
                }
                return null;
            });
        }

        long start = System.nanoTime();
        List<Future<Void>> futures = executor.invokeAll(tasks);
        for (Future<Void> f : futures) f.get();
        long end = System.nanoTime();

        executor.shutdown();
        return (end - start) / 1_000_000.0;
    }

    private static List<SensorPipeline.Observation> generateObservations(int count, int rows, int cols, long seed) {
        Random rnd = new Random(seed);
        List<SensorPipeline.Observation> obs = new ArrayList<>(count);
        for (int k = 0; k < count; k++) {
            double x = rnd.nextInt(rows);
            double y = rnd.nextInt(cols);
            double z = 0.01 + rnd.nextDouble() * 0.98;
            double w = 0.1 + rnd.nextDouble() * 0.9;
            obs.add(new SensorPipeline.Observation(x, y, z, w, k));
        }
        return obs;
    }
}