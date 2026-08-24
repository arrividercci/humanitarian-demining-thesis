package com.riskgrid.mining;

import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Карта підозрілих зон у log-odds представленні (формалізація, п. 2.1-2.4, 2.7).
 * Кожна клітинка зберігає два адитивні накопичувачі (l, W) замість прямих
 * risk/uncertainty. Оновлення - лок-фрі додавання (DoubleAdder), тому будь-яка
 * кількість потоків може одночасно оновлювати ту саму клітинку без блокувань
 * і без ризику livelock. risk/uncertainty обчислюються похідними формулами
 * лише тоді, коли потрібні для читання (планувальником, виводом тощо).
 */
public class RiskGrid {

    /** Клітинка карти: два лок-фрі накопичувачі + лічильник спостережень. */
    private static final class Cell {
        final DoubleAdder l = new DoubleAdder();   // накопичений log-odds
        final DoubleAdder W = new DoubleAdder();   // накопичена вага спостережень
        final LongAdder n = new LongAdder();       // кількість спостережень
        final AtomicLong lastUpdated = new AtomicLong(0L);
    }

    private final int rows;
    private final int cols;
    private final double cellSize;
    private final double originX;
    private final double originY;
    private final Cell[][] cells;

    public RiskGrid(int rows, int cols, double cellSize, double originX, double originY) {
        this.rows = rows;
        this.cols = cols;
        this.cellSize = cellSize;
        this.originX = originX;
        this.originY = originY;
        this.cells = new Cell[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                cells[i][j] = new Cell();
            }
        }
    }

    /** Формула 2.2: переведення фізичних координат у індекс клітинки. */
    public int[] toCellIndex(double x, double y) {
        int i = (int) Math.floor((x - originX) / cellSize);
        int j = (int) Math.floor((y - originY) / cellSize);
        return new int[] { i, j };
    }

    private boolean inBounds(int i, int j) {
        return i >= 0 && i < rows && j >= 0 && j < cols;
    }

    /**
     * Потокобезпечне оновлення клітинки (i, j) спостереженням (z, w).
     * Формула 2.4: Δl = w * ln(z / (1-z)); просте додавання до l і W.
     * Може викликатись одночасно будь-якою кількістю потоків для будь-якої
     * клітинки, включно з тією самою - конфлікт розв'язується формулою
     * злиття 2.7 (додавання комутативне й асоціативне).
     */
    public void update(int i, int j, double z, double w, long timestamp) {
        if (!inBounds(i, j)) {
            throw new IllegalArgumentException("Cell (" + i + ", " + j + ") is out of grid bounds");
        }
        if (z <= 0.0 || z >= 1.0) {
            throw new IllegalArgumentException("z must be in (0,1), got " + z);
        }
        if (w <= 0.0 || w > 1.0) {
            throw new IllegalArgumentException("w must be in (0,1], got " + w);
        }

        double deltaL = w * Math.log(z / (1.0 - z));
        Cell cell = cells[i][j];
        cell.l.add(deltaL);
        cell.W.add(w);
        cell.n.increment();
        cell.lastUpdated.accumulateAndGet(timestamp, Math::max); // монотонна мітка часу
    }

    /** Похідна величина: risk = сигмоїда від накопиченого l. */
    public double getRisk(int i, int j) {
        double l = cells[i][j].l.sum();
        return 1.0 / (1.0 + Math.exp(-l));
    }

    /** Похідна величина: uncertainty спадає з накопиченою вагою W. */
    public double getUncertainty(int i, int j) {
        double W = cells[i][j].W.sum();
        return 1.0 / (1.0 + W);
    }

    public long getObservationCount(int i, int j) {
        return cells[i][j].n.sum();
    }

    public long getLastUpdated(int i, int j) {
        return cells[i][j].lastUpdated.get();
    }

    /** Формула 2.5: чи є клітинка кандидатом на повторне обстеження. */
    public boolean needsRevisit(int i, int j, double thetaU, double thetaR, double thetaU2) {
        double risk = getRisk(i, j);
        double unc = getUncertainty(i, j);
        return unc > thetaU || (risk > thetaR && unc > thetaU2);
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }
}
