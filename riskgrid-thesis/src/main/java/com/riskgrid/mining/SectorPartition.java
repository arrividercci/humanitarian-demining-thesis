package com.riskgrid.mining;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Базовий метод для порівняння: СТАТИЧНЕ розбиття карти на сектори.
 *
 * Карта заздалегідь ділиться на n прямокутних секторів (сітка a x b, a * b = n,
 * a <= b, найближча до квадрата), і кожен дрон обстежує лише свій сектор тим
 * самим планувальником. Сектори не перетинаються, тож конфліктів між дронами
 * немає й захоплення клітинок не потрібне. Дрон, що покрив свій сектор,
 * завершує роботу (простоює до завершення інших).
 *
 * Це ізолює саме внесок динамічної координації: сенсор, карта ризику й
 * планувальник в обох підходах однакові.
 */
public final class SectorPartition {

    private SectorPartition() {}

    /**
     * Ділить сітку rows x cols на n прямокутних секторів майже однакового розміру
     * (розміри відрізняються не більше ніж на одну клітинку по кожній осі).
     * Довша сторона карти ділиться на більшу кількість частин.
     */
    public static List<RoutePlanner.Region> partition(int rows, int cols, int n) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1");
        }
        int a = 1; // найбільший дільник n, що не перевищує sqrt(n)
        for (int k = 1; (long) k * k <= n; k++) {
            if (n % k == 0) {
                a = k;
            }
        }
        int b = n / a; // a <= b
        int rowParts = rows >= cols ? b : a;
        int colParts = rows >= cols ? a : b;
        if (rowParts > rows || colParts > cols) {
            throw new IllegalArgumentException("Карта " + rows + "x" + cols + " замала для " + n + " секторів");
        }

        List<RoutePlanner.Region> sectors = new ArrayList<>();
        for (int r = 0; r < rowParts; r++) {
            int i0 = (int) ((long) r * rows / rowParts);
            int i1 = (int) ((long) (r + 1) * rows / rowParts) - 1;
            for (int c = 0; c < colParts; c++) {
                int j0 = (int) ((long) c * cols / colParts);
                int j1 = (int) ((long) (c + 1) * cols / colParts) - 1;
                sectors.add(new RoutePlanner.Region(i0, i1, j0, j1));
            }
        }
        return sectors;
    }

    private record Pair(int drone, int sector, double distance) {}

    /**
     * Жадібне призначення: пари (дрон, сектор) сортуються за відстанню від стартової
     * клітинки дрона до центру сектора, і кожна вільна пара призначається по черзі.
     * Так жоден дрон не отримує невигідно далекий сектор без потреби.
     *
     * @return sectorOfDrone[d] - індекс сектора для дрона d
     */
    public static int[] assign(List<RoutePlanner.Region> sectors, int[][] startCells, int droneCount) {
        List<Pair> pairs = new ArrayList<>();
        for (int d = 0; d < droneCount; d++) {
            for (int s = 0; s < sectors.size(); s++) {
                RoutePlanner.Region r = sectors.get(s);
                double centerI = (r.iMin() + r.iMax()) / 2.0;
                double centerJ = (r.jMin() + r.jMax()) / 2.0;
                pairs.add(new Pair(d, s, Math.hypot(startCells[d][0] - centerI, startCells[d][1] - centerJ)));
            }
        }
        pairs.sort(Comparator.comparingDouble(Pair::distance));

        int[] sectorOfDrone = new int[droneCount];
        Arrays.fill(sectorOfDrone, -1);
        boolean[] used = new boolean[sectors.size()];
        for (Pair p : pairs) {
            if (sectorOfDrone[p.drone()] < 0 && !used[p.sector()]) {
                sectorOfDrone[p.drone()] = p.sector();
                used[p.sector()] = true;
            }
        }
        return sectorOfDrone;
    }
}
