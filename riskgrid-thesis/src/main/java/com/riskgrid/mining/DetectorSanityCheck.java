package com.riskgrid.mining;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Одноразова перевірка: чи дає Java ONNX-детектор (MineDetectorOnnx) ті самі
 * значення z, що й оригінальна Python-модель (mine_detector.py) на тих самих
 * зображеннях. Запусти цей клас, потім запусти mine_detector.py на тих самих
 * файлах (шляхи будуть надруковані) і звір числа вручну.
 *
 * Використання:
 *   mvn exec:java "-Dexec.mainClass=com.riskgrid.mining.DetectorSanityCheck"
 */
public final class DetectorSanityCheck {

    private static final String ONNX_MODEL_PATH = "model/best.onnx";
    private static final Path MINE_IMAGES_DIR = java.nio.file.Paths.get("terrain-images", "mine");
    private static final Path BACKGROUND_IMAGES_DIR = java.nio.file.Paths.get("terrain-images", "background");
    private static final int SAMPLE_COUNT = 20;

    public static void main(String[] args) throws Exception {
        System.setOut(new java.io.PrintStream(System.out, true, java.nio.charset.StandardCharsets.UTF_8));

        try (MineDetectorOnnx detector = new MineDetectorOnnx(ONNX_MODEL_PATH)) {
            System.out.println("=== Зображення з мінами ===");
            checkFolder(detector, MINE_IMAGES_DIR);

            System.out.println();
            System.out.println("=== Фонові зображення ===");
            checkFolder(detector, BACKGROUND_IMAGES_DIR);
        }

        System.out.println();
        System.out.println("Тепер запусти для ТИХ САМИХ файлів (шляхи вище):");
        System.out.println("  python mine_detector.py weights/best.pt <шлях_до_файлу>");
        System.out.println("і звір Suspicion score z з тим, що надруковано вище.");
    }

    private static void checkFolder(MineDetectorOnnx detector, Path dir) throws Exception {
        List<Path> images = listImages(dir).stream().limit(SAMPLE_COUNT).collect(Collectors.toList());
        if (images.isEmpty()) {
            System.out.println("  (папка порожня або не знайдена: " + dir + ")");
            return;
        }
        for (Path img : images) {
            float z = detector.getSuspicionScore(img.toFile());
            System.out.printf("  %s  ->  z (Java) = %.4f%n", img, z);
        }
    }

    private static List<Path> listImages(Path dir) throws Exception {
        if (!Files.isDirectory(dir)) {
            return List.of();
        }
        try (Stream<Path> stream = Files.list(dir)) {
            return stream.filter(p -> {
                        String name = p.toString().toLowerCase();
                        return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
                    })
                    .sorted()
                    .collect(Collectors.toList());
        }
    }
}
