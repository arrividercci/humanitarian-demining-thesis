package com.riskgrid.mining;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.junit.jupiter.api.Test;

class SensorPipelineTest {

    private List<SensorPipeline.Observation> generateObservations(int count, long seed) {
        Random rnd = new Random(seed);
        List<SensorPipeline.Observation> obs = new ArrayList<>();
        for (int k = 0; k < count; k++) {
            double x = rnd.nextInt(10);
            double y = rnd.nextInt(10);
            double z = 0.01 + rnd.nextDouble() * 0.98;
            double w = 0.1 + rnd.nextDouble() * 0.9;
            obs.add(new SensorPipeline.Observation(x, y, z, w, k));
        }
        return obs;
    }

    private RiskGrid runThroughPipeline(List<SensorPipeline.Observation> observations, int numConsumers)
            throws InterruptedException {
        RiskGrid grid = new RiskGrid(10, 10, 1.0, 0.0, 0.0);
        SensorPipeline pipeline = new SensorPipeline(grid, 10_000, numConsumers);
        pipeline.startConsumers();
        for (SensorPipeline.Observation obs : observations) {
            pipeline.submitObservation(obs.x(), obs.y(), obs.z(), obs.w(), obs.timestamp());
        }
        pipeline.shutdown();
        return grid;
    }

    @Test
    void resultIsIdenticalRegardlessOfConsumerCount() throws InterruptedException {
        List<SensorPipeline.Observation> observations = generateObservations(2000, 42L);

        RiskGrid with1 = runThroughPipeline(observations, 1);
        RiskGrid with2 = runThroughPipeline(observations, 2);
        RiskGrid with4 = runThroughPipeline(observations, 4);
        RiskGrid with8 = runThroughPipeline(observations, 8);

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                double base = with1.getRisk(i, j);
                assertEquals(base, with2.getRisk(i, j), 1e-6,
                        "risk відрізняється при 2 споживачах, клітинка (" + i + "," + j + ")");
                assertEquals(base, with4.getRisk(i, j), 1e-6,
                        "risk відрізняється при 4 споживачах, клітинка (" + i + "," + j + ")");
                assertEquals(base, with8.getRisk(i, j), 1e-6,
                        "risk відрізняється при 8 споживачах, клітинка (" + i + "," + j + ")");

                assertEquals(with1.getObservationCount(i, j), with8.getObservationCount(i, j),
                        "кількість спостережень відрізняється - можливо, втрачені оновлення");
            }
        }
    }

    @Test
    void noLostUpdatesUnderHeavyConcurrentLoadOnSameCell() throws InterruptedException {
        RiskGrid grid = new RiskGrid(5, 5, 1.0, 0.0, 0.0);
        SensorPipeline pipeline = new SensorPipeline(grid, 50_000, 8);
        pipeline.startConsumers();

        int producers = 6;
        int perProducer = 2000;
        ExecutorService producerPool = Executors.newFixedThreadPool(producers);
        CountDownLatch latch = new CountDownLatch(producers);

        for (int p = 0; p < producers; p++) {
            producerPool.submit(() -> {
                try {
                    Random rnd = new Random();
                    for (int k = 0; k < perProducer; k++) {
                        double z = 0.01 + rnd.nextDouble() * 0.98;
                        pipeline.submitObservation(2.5, 2.5, z, 0.5, k);
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    latch.countDown();
                }
            });
        }

        latch.await();
        producerPool.shutdown();
        pipeline.shutdown();

        int[] idx = grid.toCellIndex(2.5, 2.5);
        long expected = (long) producers * perProducer;
        assertEquals(expected, grid.getObservationCount(idx[0], idx[1]),
                "кількість спостережень має точно збігатись - жодне не повинно загубитись");
    }
}
