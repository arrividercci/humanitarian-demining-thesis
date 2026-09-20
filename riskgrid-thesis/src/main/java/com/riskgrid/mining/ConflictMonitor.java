package com.riskgrid.mining;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Лічильник одночасних дублів: рахує випадки, коли КІЛЬКА дронів одночасно
 * "зобов'язані" обробити ту саму клітинку (від моменту вибору цілі до кінця
 * обробки). Не залежить від DroneCoordinator, тож працює однаково і з
 * увімкненим, і з вимкненим захопленням клітинок.
 *
 * З увімкненим захопленням очікуваний результат - 0 (емпірична перевірка
 * інваріанта "жодні два дрони не обирають одну клітинку одночасно");
 * з вимкненим - кількість конфліктів рішень, яких метод запобігає.
 *
 * Якщо k дронів одночасно обрали одну клітинку, це дає k-1 конфліктів
 * (кожен "зайвий" дрон - одне марне обстеження).
 *
 * Пізніші повторні візити клітинки (коли перший дрон уже завершив) сюди
 * НЕ потрапляють - їх рахує DroneAgent.getRevisits().
 */
public final class ConflictMonitor {

    private final ConcurrentHashMap<Long, Integer> active = new ConcurrentHashMap<>();
    private final LongAdder conflicts = new LongAdder();

    public void enter(int i, int j) {
        int now = active.merge(DroneCoordinator.cellKey(i, j), 1, Integer::sum);
        if (now > 1) {
            conflicts.increment();
        }
    }

    public void exit(int i, int j) {
        active.compute(DroneCoordinator.cellKey(i, j), (k, v) -> (v == null || v <= 1) ? null : v - 1);
    }

    public long conflictCount() {
        return conflicts.sum();
    }
}
