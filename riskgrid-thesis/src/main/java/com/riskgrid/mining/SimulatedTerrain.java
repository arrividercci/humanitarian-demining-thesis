package com.riskgrid.mining;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.awt.image.RescaleOp;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
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
 * СТОХАСТИЧНА АУГМЕНТАЦІЯ ПРИ ПОВТОРНИХ ВІЗИТАХ (розділ 3.3 статті).
 * Базове зображення для клітинки одне й те саме (як і раніше), але
 * getImageAt() повертає НЕ сам закешований кадр, а його випадково
 * трансформовану копію: невеликий поворот, зсув рамки кадрування й зміна
 * яскравості - імітація іншого ракурсу/моменту прольоту. Це відповідає
 * зауваженню, що повторна обробка того самого закешованого кадру не дає
 * нової інформації, а формули (4) і (6) трактують спостереження як
 * незалежні: тепер кожен візит дає інший (хоч і статистично залежний,
 * бо сцена та сама) вхід детектора, а не буквально той самий піксельний
 * масив. Трансформація детермінована за (seed клітинки, номер візиту),
 * тож увесь експеримент лишається відтворюваним для того самого seed.
 *
 * Параметри аугментації можна вимкнути (AugmentationConfig.disabled()) -
 * тоді поведінка збігається з попередньою версією (один і той самий кадр
 * на кожен візит); це лишено для регресійних перевірок і чутливості.
 */
public final class SimulatedTerrain implements TerrainSource {

    /**
     * Параметри випадкової трансформації кадру на один візит.
     *
     * @param maxRotationDeg  максимальний кут повороту, градуси (± рівномірно)
     * @param maxCropFraction максимальний зсув рамки кадрування як частка розміру кадру (± рівномірно)
     * @param brightnessJitter максимальна відносна зміна яскравості (± рівномірно, напр. 0.12 = ±12%)
     * @param enabled         false - аугментація вимкнена (стара поведінка: один і той самий кадр)
     */
    public record AugmentationConfig(double maxRotationDeg, double maxCropFraction,
                                     double brightnessJitter, boolean enabled) {
        /** Типові значення: помірний поворот, невеликий зсув кадрування, легка зміна яскравості. */
        public static AugmentationConfig defaults() {
            return new AugmentationConfig(8.0, 0.08, 0.12, true);
        }

        /** Вимкнена аугментація - відтворює поведінку до цієї правки (один кадр на клітинку). */
        public static AugmentationConfig disabled() {
            return new AugmentationConfig(0.0, 0.0, 0.0, false);
        }
    }

    private final int rows;
    private final int cols;
    private final Path[][] assignedImage;
    private final boolean[][] mineCell;
    private final Map<Path, BufferedImage> loadedImages;
    private final AugmentationConfig aug;
    private final long augmentationSeed;

    // Лічильник візитів на клітинку - визначає, який саме (детермінований) варіант
    // трансформації повертається цього разу. Потокобезпечний: інкремент атомарний.
    private final ConcurrentHashMap<Long, AtomicInteger> visitCounts = new ConcurrentHashMap<>();

    /** Аугментація увімкнена з типовими параметрами (AugmentationConfig.defaults()). */
    public SimulatedTerrain(int rows, int cols,
                             Path mineImagesDir, Path backgroundImagesDir,
                             int clusterCount, double clusterRadiusCells,
                             long seed) throws IOException {
        this(rows, cols, mineImagesDir, backgroundImagesDir, clusterCount, clusterRadiusCells,
                seed, AugmentationConfig.defaults());
    }

    public SimulatedTerrain(int rows, int cols,
                             Path mineImagesDir, Path backgroundImagesDir,
                             int clusterCount, double clusterRadiusCells,
                             long seed, AugmentationConfig augmentation) throws IOException {
        this.rows = rows;
        this.cols = cols;
        this.assignedImage = new Path[rows][cols];
        this.mineCell = new boolean[rows][cols];
        this.aug = augmentation;
        this.augmentationSeed = seed;

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

        // 3. Заздалегідь завантажуємо всі використані файли в пам'ять (незмінні базові кадри)
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

    /**
     * Повертає кадр для клітинки (i, j). Якщо аугментація увімкнена, кожен виклик
     * повертає нову, випадково трансформовану копію базового кадру (детерміновано
     * за номером цього візиту), а не той самий об'єкт - див. коментар класу.
     */
    @Override
    public BufferedImage getImageAt(int i, int j) {
        BufferedImage base = loadedImages.get(assignedImage[i][j]);
        if (base == null || !aug.enabled()) {
            return base;
        }
        long key = DroneCoordinator.cellKey(i, j);
        int visit = visitCounts.computeIfAbsent(key, k -> new AtomicInteger(0)).getAndIncrement();
        Random rnd = new Random(mixSeeds(augmentationSeed, key, visit));
        return augment(base, rnd);
    }

    /**
     * Випадковий поворот + невеликий зсув кадрування (letterbox сірим, як у детекторі) +
     * зміна яскравості. Створює НОВЕ зображення - базовий кеш не змінюється, тож безпечно
     * для конкурентних викликів з різних потоків.
     */
    private BufferedImage augment(BufferedImage src, Random rnd) {
        int w = src.getWidth(), h = src.getHeight();

        double maxCropW = w * aug.maxCropFraction();
        double maxCropH = h * aug.maxCropFraction();
        double dx = maxCropW > 0 ? (rnd.nextDouble() * 2 - 1) * maxCropW : 0.0;
        double dy = maxCropH > 0 ? (rnd.nextDouble() * 2 - 1) * maxCropH : 0.0;
        double angleDeg = (rnd.nextDouble() * 2 - 1) * aug.maxRotationDeg();
        double brightness = 1.0 + (rnd.nextDouble() * 2 - 1) * aug.brightnessJitter();

        BufferedImage out = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = out.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.setColor(new Color(114, 114, 114)); // той самий letterbox-колір, що й у детекторі
        g.fillRect(0, 0, w, h);

        AffineTransform tx = new AffineTransform();
        tx.translate(w / 2.0 + dx, h / 2.0 + dy);
        tx.rotate(Math.toRadians(angleDeg));
        tx.translate(-w / 2.0, -h / 2.0);
        g.drawImage(src, tx, null);
        g.dispose();

        float scale = (float) Math.max(0.0, brightness);
        new RescaleOp(scale, 0f, null).filter(out, out); // RescaleOp сам клемпить у [0, 255]
        return out;
    }

    /** Детермінований мікс (seed території, ключ клітинки, номер візиту) у 64-бітний seed для Random. */
    private static long mixSeeds(long base, long key, int visit) {
        long h = base ^ (key * 0x9E3779B97F4A7C15L) ^ ((long) visit * 0xBF58476D1CE4E5B9L);
        h ^= (h >>> 33);
        h *= 0xff51afd7ed558ccdL;
        h ^= (h >>> 33);
        h *= 0xc4ceb9fe1a85ec53L;
        h ^= (h >>> 33);
        return h;
    }

    /** Чи належить клітинка (i,j) до якогось кластера-міни (ground truth для візуалізації). */
    public boolean isMineCell(int i, int j) {
        return mineCell[i][j];
    }

    public int getRows() { return rows; }
    public int getCols() { return cols; }
}