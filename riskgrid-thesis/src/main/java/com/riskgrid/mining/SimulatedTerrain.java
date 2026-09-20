package com.riskgrid.mining;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Реалізація TerrainSource: генерує тестову територію з кластерним
 * розміщенням "мін". Випадкові центри кластерів обираються детерміновано
 * за seed; клітинки в межах clusterRadiusCells від будь-якого центру
 * отримують реальне зображення з міною (test-спліт MineGuard-Thermal),
 * решта клітинок - реальне фонове зображення (без анотацій).
 *
 * Призначення зображень клітинкам фіксується один раз у конструкторі
 * (детерміновано за seed) - це гарантує однакову територію при порівнянні
 * 1 дрон vs N дронів чи різних конфігурацій паралелізму (інакше різниця
 * в результатах могла б пояснюватись різною територією, а не паралелізмом).
 *
 * Усі зображення завантажуються в пам'ять один раз у конструкторі (їх
 * мало - десятки файлів), тож getImageAt() після побудови об'єкта лише
 * читає з незмінної Map - безпечно для конкурентних викликів з кількох
 * потоків DroneAgent без додаткової синхронізації.
 */
public final class SimulatedTerrain implements TerrainSource {

    private final int rows;
    private final int cols;
    private final Path[][] assignedImage;
    private final boolean[][] mineCell;
    private final Map<Path, BufferedImage> loadedImages;

    public SimulatedTerrain(int rows, int cols,
                             Path mineImagesDir, Path backgroundImagesDir,
                             int clusterCount, double clusterRadiusCells,
                             long seed) throws IOException {
        this.rows = rows;
        this.cols = cols;
        this.assignedImage = new Path[rows][cols];
        this.mineCell = new boolean[rows][cols];

        List<Path> mineImages = listImages(mineImagesDir);
        List<Path> backgroundImages = listImages(backgroundImagesDir);
        if (mineImages.isEmpty()) {
            throw new IllegalArgumentException("Не знайдено жодного зображення з мінами в " + mineImagesDir);
        }
        if (backgroundImages.isEmpty()) {
            throw new IllegalArgumentException("Не знайдено жодного фонового зображення в " + backgroundImagesDir);
        }

        Random rnd = new Random(seed);

        // 1. Випадкові центри кластерів-мін (детерміновано за seed)
        int[][] centers = new int[clusterCount][2];
        for (int c = 0; c < clusterCount; c++) {
            centers[c][0] = rnd.nextInt(rows);
            centers[c][1] = rnd.nextInt(cols);
        }

        // 2. Кожній клітинці призначаємо конкретний файл зображення
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                boolean isMineCell = false;
                for (int[] c : centers) {
                    if (Math.hypot(i - c[0], j - c[1]) <= clusterRadiusCells) {
                        isMineCell = true;
                        break;
                    }
                }
                mineCell[i][j] = isMineCell;
                List<Path> pool = isMineCell ? mineImages : backgroundImages;
                assignedImage[i][j] = pool.get(rnd.nextInt(pool.size()));
            }
        }

        // 3. Заздалегідь завантажуємо всі використані файли в пам'ять
        Map<Path, BufferedImage> cache = new HashMap<>();
        for (Path[] row : assignedImage) {
            for (Path p : row) {
                cache.computeIfAbsent(p, this::loadImage);
            }
        }
        this.loadedImages = Collections.unmodifiableMap(cache);
    }

    private BufferedImage loadImage(Path p) {
        try {
            BufferedImage img = ImageIO.read(p.toFile());
            if (img == null) {
                throw new IOException("ImageIO не розпізнав формат файлу");
            }
            return img;
        } catch (IOException e) {
            throw new RuntimeException("Не вдалось завантажити зображення: " + p, e);
        }
    }

    private static List<Path> listImages(Path dir) throws IOException {
        try (Stream<Path> stream = Files.list(dir)) {
            return stream.filter(p -> {
                        String name = p.toString().toLowerCase();
                        return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
                    })
                    .sorted() // детермінований порядок файлів перед вибіркою через Random(seed)
                    .collect(Collectors.toList());
        }
    }

    @Override
    public BufferedImage getImageAt(int i, int j) {
        return loadedImages.get(assignedImage[i][j]);
    }

    /** Чи належить клітинка (i,j) до якогось кластера-міни (ground truth для візуалізації). */
    public boolean isMineCell(int i, int j) {
        return mineCell[i][j];
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }
}