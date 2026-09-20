package com.riskgrid.mining;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Map;

/**
 * Локальний (in-process) детектор мін на основі ONNX-експорту навченої
 * YOLOv8n-моделі.
 *
 * Заміняє попередній HTTP-клієнт, який робив мережевий запит до окремого
 * Python/FastAPI сервісу. Тепер інференс виконується прямо в JVM - це
 * коректніше відображає автономність дрона (немає залежності від мережі
 * для базової функції детекції).
 *
 * Один екземпляр можна безпечно ділити між кількома потоками:
 * OrtSession.run() є потокобезпечним для конкурентних викликів на одній
 * сесії, так само як RiskGrid ділиться між дронами через DoubleAdder/LongAdder.
 */
public final class MineDetectorOnnx implements AutoCloseable {

    private static final int INPUT_SIZE = 640;

    /** Збігається з self.conf_threshold у mine_detector.py, щоб suspicion
     *  score z був однаковим між Python- та Java-реалізацією. */
    private static final float CONF_THRESHOLD = 0.25f;

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    /** Один потік інференсу на виклик - типове значення для експериментів з кількома дронами. */
    public MineDetectorOnnx(String onnxModelPath) throws OrtException {
        this(onnxModelPath, 1);
    }

    /**
     * @param intraOpThreads кількість потоків ONNX Runtime на один виклик run().
     *        Для експериментів з N дронами ОБОВ'ЯЗКОВО 1: інакше потоки ONNX
     *        конкурують з потоками дронів за ядра CPU і спотворюють прискорення.
     */
    public MineDetectorOnnx(String onnxModelPath, int intraOpThreads) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(intraOpThreads);
        opts.setInterOpNumThreads(1);
        this.session = env.createSession(onnxModelPath, opts);
        this.inputName = session.getInputNames().iterator().next();
    }

    /** Повертає suspicion score z ∈ [0,1] для зображення на диску. */
    public float getSuspicionScore(File imageFile) throws IOException, OrtException {
        BufferedImage img = ImageIO.read(imageFile);
        if (img == null) {
            throw new IOException("Не вдалося прочитати зображення: " + imageFile);
        }
        return getSuspicionScore(img);
    }

    /** Повертає suspicion score z ∈ [0,1] для вже завантаженого зображення. */
    public float getSuspicionScore(BufferedImage img) throws OrtException {
        float[] inputData = preprocess(img);
        long[] shape = {1, 3, INPUT_SIZE, INPUT_SIZE};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape)) {
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);
            try (OrtSession.Result result = session.run(inputs)) {
                float[][][] output = (float[][][]) result.get(0).getValue();
                return extractMaxConfidence(output);
            }
        }
    }

    /**
     * Letterbox-масштабування до 640x640 (збереження пропорцій, доповнення
     * сірим кольором) + нормалізація [0,1] + перетворення HWC -> CHW.
     * Відтворює стандартний препроцесинг Ultralytics YOLO.
     */
    private float[] preprocess(BufferedImage img) {
        int origW = img.getWidth();
        int origH = img.getHeight();
        double scale = Math.min((double) INPUT_SIZE / origW, (double) INPUT_SIZE / origH);
        int newW = (int) Math.round(origW * scale);
        int newH = (int) Math.round(origH * scale);
        int padX = (INPUT_SIZE - newW) / 2;
        int padY = (INPUT_SIZE - newH) / 2;

        BufferedImage resized = new BufferedImage(INPUT_SIZE, INPUT_SIZE, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = resized.createGraphics();
        g.setColor(new Color(114, 114, 114)); // стандартний letterbox-колір Ultralytics
        g.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
        // Білінійна інтерполяція (як cv2.INTER_LINEAR в Ultralytics); без цієї
        // підказки Java2D використовує nearest-neighbour, і z розходиться з Python.
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, padX, padY, newW, newH, null);
        g.dispose();

        float[] chw = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int channelSize = INPUT_SIZE * INPUT_SIZE;
        int[] pixels = resized.getRGB(0, 0, INPUT_SIZE, INPUT_SIZE, null, 0, INPUT_SIZE);
        for (int idx = 0; idx < channelSize; idx++) {
            int rgb = pixels[idx];
            chw[idx] = ((rgb >> 16) & 0xFF) / 255f;                 // канал R
            chw[channelSize + idx] = ((rgb >> 8) & 0xFF) / 255f;    // канал G
            chw[2 * channelSize + idx] = (rgb & 0xFF) / 255f;       // канал B
        }
        return chw;
    }

    /**
     * Вихід YOLOv8-ONNX для однокласової моделі ("міна") має форму
     * [1, 5, numAnchors]: рядки 0-3 - координати bbox, рядок 4 - class-score.
     *
     * Беремо максимум по class-score серед усіх якорів понад поріг - це і є
     * z. Повний NMS не потрібен: найвпевненіша детекція завжди виживає після
     * NMS, тож максимум "сирих" оцінок дає той самий z, що й Python-версія з
     * повноцінним постпроцесингом Ultralytics.
     */
    private float extractMaxConfidence(float[][][] output) {
        float[][] preds = output[0]; // [numAttrs][numAnchors]
        int classRow = 4;
        int numAnchors = preds[classRow].length;

        float maxScore = 0f;
        for (int a = 0; a < numAnchors; a++) {
            float score = preds[classRow][a];
            if (score > maxScore) {
                maxScore = score;
            }
        }
        return maxScore >= CONF_THRESHOLD ? maxScore : 0f;
    }

    @Override
    public void close() throws OrtException {
        session.close();
        // env не закриваємо тут - можливо, використовується іншими сесіями/потоками
    }
}