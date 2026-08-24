package com.riskgrid.mining;

import java.util.Set;
import java.util.Collections;

/**
 * Планувальник маршруту (формалізація, розділ 3). Покроковий жадібний вибір
 * наступної точки обстеження за корисністю (utility-based greedy) в
 * обмеженій околиці поточної позиції - формула 3.3. Складність одного кроку
 * O(R^2 / cellSize^2), не залежить від розміру всієї карти (розділ 3.5).
 *
 * Клас потоко-безпечний: не містить мутабельного стану екземпляра, тож
 * один спільний RoutePlanner можна безпечно використовувати з кількох
 * потоків-дронів одночасно (кожен виклик selectNextTarget незалежний).
 */
public class RoutePlanner {

    /** Точка на місцевості (фізичні координати). */
    public record Point(double x, double y) {}

    /** Результат вибору цілі: координати клітинки + діагностична інформація. */
    public record TargetResult(int i, int j, int candidatesEvaluated) {}

    private final RiskGrid grid;
    private final double cellSize;
    private final double originX;
    private final double originY;

    public RoutePlanner(RiskGrid grid, double cellSize, double originX, double originY) {
        this.grid = grid;
        this.cellSize = cellSize;
        this.originX = originX;
        this.originY = originY;
    }

    /** Фізична координата центру клітинки (i, j) - обернена до формули 2.2. */
    private Point cellCenter(int i, int j) {
        double x = originX + (i + 0.5) * cellSize;
        double y = originY + (j + 0.5) * cellSize;
        return new Point(x, y);
    }

    private double dist(Point a, Point b) {
        double dx = a.x() - b.x();
        double dy = a.y() - b.y();
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Формула 3.3: корисність обстеження клітинки (i,j) з позиції p.
     * H(risk) - ентропія Бернуллі (максимальна при risk=0.5, нуль при risk->0 або 1) -
     * інформаційний виграш, λ*dist - штраф за відстань.
     */
    private double utility(int i, int j, Point p, double lambda) {
        double risk = grid.getRisk(i, j);
        double unc = grid.getUncertainty(i, j);
        double informationGain = binaryEntropy(risk) * unc;
        double travelCost = dist(cellCenter(i, j), p);
        return informationGain - lambda * travelCost;
    }

    /** Ентропія Бернуллі: -p*ln(p) - (1-p)*ln(1-p), з коректним граничним значенням 0 при p=0 або p=1. */
    private static double binaryEntropy(double p) {
        double h = 0.0;
        if (p > 0.0 && p < 1.0) {
            h = -p * Math.log(p) - (1 - p) * Math.log(1 - p);
        }
        return h;
    }

    /**
     * Формула 3.2 і 3.4: обирає наступну ціль серед клітинок у радіусі R
     * навколо поточної позиції p. Якщо кандидатів немає - подвоює радіус
     * (рекурсивно, з обмеженням, щоб не сканувати всю карту нескінченно).
     *
     * @return координати клітинки (i, j) з максимальною корисністю разом
     *         з кількістю оцінених кандидатів
     */
    public TargetResult selectNextTarget(Point p, double radius, double lambda) {
        return selectNextTarget(p, radius, lambda, Collections.emptySet(), 6);
    }

    /**
     * Те саме, що selectNextTarget(p, radius, lambda), але виключає клітинки
     * з переданого набору excludedCells - потрібно для координації кількох
     * дронів, щоб вони не обирали ту саму клітинку одночасно (розділ 3.6).
     *
     * excludedCells - типово результат DroneCoordinator.getClaimedCells();
     * очікується у вигляді пакованих ключів (див. DroneCoordinator.cellKey).
     */
    public TargetResult selectNextTarget(Point p, double radius, double lambda, Set<Long> excludedCells) {
        return selectNextTarget(p, radius, lambda, excludedCells, 6);
    }

    private TargetResult selectNextTarget(Point p, double radius, double lambda,
                                           Set<Long> excludedCells, int maxDoublings) {
        int[] pCellIdx = grid.toCellIndex(p.x(), p.y());
        int cellRadius = (int) Math.ceil(radius / cellSize);

        int bestI = -1, bestJ = -1;
        double bestUtility = Double.NEGATIVE_INFINITY;
        int candidatesEvaluated = 0;

        int iMin = Math.max(0, pCellIdx[0] - cellRadius);
        int iMax = Math.min(grid.getRows() - 1, pCellIdx[0] + cellRadius);
        int jMin = Math.max(0, pCellIdx[1] - cellRadius);
        int jMax = Math.min(grid.getCols() - 1, pCellIdx[1] + cellRadius);

        for (int i = iMin; i <= iMax; i++) {
            for (int j = jMin; j <= jMax; j++) {
                if (excludedCells.contains(DroneCoordinator.cellKey(i, j))) {
                    continue; // клітинка вже застовплена іншим дроном
                }

                Point center = cellCenter(i, j);
                if (dist(center, p) > radius) continue;

                candidatesEvaluated++;
                double u = utility(i, j, p, lambda);
                if (u > bestUtility) {
                    bestUtility = u;
                    bestI = i;
                    bestJ = j;
                }
            }
        }

        if (bestI == -1) {
            if (maxDoublings <= 0) {
                throw new IllegalStateException("Не знайдено жодного кандидата навіть після розширення радіусу");
            }
            return selectNextTarget(p, radius * 2, lambda, excludedCells, maxDoublings - 1);
        }

        return new TargetResult(bestI, bestJ, candidatesEvaluated);
    }
}