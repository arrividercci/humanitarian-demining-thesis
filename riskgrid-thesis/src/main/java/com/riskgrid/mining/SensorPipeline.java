package com.riskgrid.mining;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Асинхронний конвеєр "сенсори -> черга -> оновлення карти" (рівні 1 і 2
 * з архітектурної схеми). Кілька потоків-виробників (сенсорів) незалежно
 * кладуть спостереження у спільну чергу; кілька потоків-споживачів
 * розбирають чергу й застосовують оновлення до RiskGrid паралельно.
 * Виробники й споживачі ніколи не блокують один одного напряму - черга
 * згладжує різницю в темпі роботи.
 */
public class SensorPipeline {

    /** Одне спостереження: фізична координата + оцінка сенсора + довіра. */
    public record Observation(double x, double y, double z, double w, long timestamp) {}

    /** Сигнальне значення "більше спостережень не буде" - завершує споживачів. */
    private static final Observation POISON_PILL = new Observation(0, 0, 0.5, 1.0, -1);

    private final RiskGrid grid;
    private final BlockingQueue<Observation> queue;
    private final ExecutorService consumerPool;
    private final int numConsumers;

    public SensorPipeline(RiskGrid grid, int queueCapacity, int numConsumers) {
        this.grid = grid;
        this.queue = new LinkedBlockingQueue<>(queueCapacity);
        this.numConsumers = numConsumers;
        this.consumerPool = Executors.newFixedThreadPool(numConsumers);
    }

    /** Запускає споживачів - потоки, що безперервно розбирають чергу й оновлюють карту. */
    public void startConsumers() {
        for (int k = 0; k < numConsumers; k++) {
            consumerPool.submit(this::consumeLoop);
        }
    }

    private void consumeLoop() {
        try {
            while (true) {
                Observation obs = queue.take(); // блокується, лише якщо черга порожня
                if (obs == POISON_PILL) {
                    queue.put(POISON_PILL); // передати сигнал далі іншим споживачам
                    return;
                }
                int[] idx = grid.toCellIndex(obs.x(), obs.y());
                int i = idx[0], j = idx[1];
                if (i < 0 || i >= grid.getRows() || j < 0 || j >= grid.getCols()) {
                    continue; // спостереження поза межами карти - ігноруємо
                }
                grid.update(i, j, obs.z(), obs.w(), obs.timestamp());
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * Викликається сенсором (виробником) для надсилання нового спостереження.
     * Не блокує надовго - лише якщо черга тимчасово переповнена (backpressure).
     */
    public void submitObservation(double x, double y, double z, double w, long timestamp)
            throws InterruptedException {
        queue.put(new Observation(x, y, z, w, timestamp));
    }

    /** Коректне завершення: чекає, поки черга спорожніє, і зупиняє споживачів. */
    public void shutdown() throws InterruptedException {
        queue.put(POISON_PILL);
        consumerPool.shutdown();
        consumerPool.awaitTermination(30, TimeUnit.SECONDS);
    }

    public int queueSize() {
        return queue.size();
    }
}
