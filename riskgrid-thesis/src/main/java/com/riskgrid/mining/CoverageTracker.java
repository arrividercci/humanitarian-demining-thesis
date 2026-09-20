package com.riskgrid.mining;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Відстежує, які клітинки території вже відвідані (спостереження подане
 * в RiskGrid), незалежно від поточного рівня невизначеності клітинки.
 *
 * Використовується як глобальна умова зупинки для DroneAgent: симуляція
 * триває, поки є хоч одна непокрита клітинка (isComplete() == false).
 *
 * Потокобезпечний: markVisited() атомарний через ConcurrentHashMap.putIfAbsent,
 * так само як claim-логіка в DroneCoordinator.
 */
public final class CoverageTracker {

    private final int rows;
    private final int cols;
    private final ConcurrentHashMap<Long, Boolean> visited = new ConcurrentHashMap<>();
    private final AtomicInteger remaining;
    private final AtomicInteger scanCursor = new AtomicInteger(0);

    public CoverageTracker(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.remaining = new AtomicInteger(rows * cols);
    }

    private static long key(int i, int j) {
        return ((long) i << 32) | (j & 0xffffffffL);
    }

    /** Позначає клітинку відвіданою. Повертає true, якщо це перше відвідування. */
    public boolean markVisited(int i, int j) {
        Boolean prev = visited.putIfAbsent(key(i, j), Boolean.TRUE);
        if (prev == null) {
            remaining.decrementAndGet();
            return true;
        }
        return false;
    }

    public boolean isVisited(int i, int j) {
        return visited.containsKey(key(i, j));
    }

    public boolean isComplete() {
        return remaining.get() <= 0;
    }

    public int remainingCount() {
        return Math.max(remaining.get(), 0);
    }

    /**
     * Round-robin пошук будь-якої непокритої клітинки — використовується,
     * коли RoutePlanner не знаходить кандидата в радіусі R навколо поточної
     * позиції дрона (локальна область вичерпана, але десь далі ще є
     * непокрита територія). Дрон "перелітає" туди й продовжує звідти.
     *
     * O(rows*cols) у гіршому випадку; прийнятно для розмірів сіток у цій
     * роботі (до 500x500), не є вузьким місцем порівняно з інференсом моделі.
     */
    public int[] findAnyUncovered() {
        int total = rows * cols;
        for (int attempt = 0; attempt < total; attempt++) {
            int idx = Math.floorMod(scanCursor.getAndIncrement(), total);
            int i = idx / cols;
            int j = idx % cols;
            if (!isVisited(i, j)) {
                return new int[]{i, j};
            }
        }
        return null; // все покрито
    }
}
