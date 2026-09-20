package com.riskgrid.mining;

import javax.swing.*;
import java.awt.*;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Десктоп-демонстрація методу: форма вводу (розмір карти, кількість
 * дронів, поріг), жива карта обходу території, живі метрики якості
 * детекції. Використовує той самий реальний код (RiskGrid, RoutePlanner,
 * DroneAgent, MineDetectorOnnx), що й пакетний Main - без окремої
 * реалізації математики поза Java.
 */
public final class DemoApp extends JFrame {

    // ---- Шляхи - підправ під свою структуру папок ----
    private static final Path MINE_IMAGES_DIR = Paths.get("terrain-images", "mine");
    private static final Path BACKGROUND_IMAGES_DIR = Paths.get("terrain-images", "background");
    private static final String ONNX_MODEL_PATH = "model/best.onnx";
    private static final int CLUSTER_COUNT = 3;
    private static final double CLUSTER_RADIUS_CELLS = 4.0;
    // ----------------------------------------------------

    private final SimulationController controller;
    private final SimulationPanel mapPanel = new SimulationPanel();
    private final MetricsPanel metricsPanel = new MetricsPanel();

    private final JSpinner rowsSpinner = new JSpinner(new SpinnerNumberModel(30, 5, 200, 5));
    private final JSpinner colsSpinner = new JSpinner(new SpinnerNumberModel(30, 5, 200, 5));
    private final JSpinner dronesSpinner = new JSpinner(new SpinnerNumberModel(4, 1, 32, 1));
    private final JSpinner thresholdSpinner = new JSpinner(new SpinnerNumberModel(0.422, 0.05, 0.95, 0.05));
    private final JCheckBox randomEachRun = new JCheckBox("Випадкова територія щоразу", true);
    private final JButton startButton = new JButton("Почати обстеження");
    private final JButton stopButton = new JButton("Зупинити");
    private final JLabel statusLabel = new JLabel("Готово до запуску.");

    private Timer refreshTimer;

    public DemoApp(MineDetectorOnnx detector) {
        super("Паралельний метод обстеження території - демонстрація");
        this.controller = new SimulationController(detector);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout(8, 8));

        add(buildControlPanel(), BorderLayout.NORTH);
        add(mapPanel, BorderLayout.CENTER);
        add(metricsPanel, BorderLayout.EAST);
        add(statusLabel, BorderLayout.SOUTH);

        mapPanel.setController(controller);
        metricsPanel.setController(controller);
        stopButton.setEnabled(false);

        pack();
        setSize(1000, 700);
        setLocationRelativeTo(null);
    }

    private JPanel buildControlPanel() {
        JPanel panel = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 8));
        panel.add(new JLabel("Рядків:"));
        panel.add(rowsSpinner);
        panel.add(new JLabel("Стовпців:"));
        panel.add(colsSpinner);
        panel.add(new JLabel("Дронів:"));
        panel.add(dronesSpinner);
        panel.add(new JLabel("Поріг risk:"));
        panel.add(thresholdSpinner);
        panel.add(randomEachRun);
        panel.add(startButton);
        panel.add(stopButton);

        startButton.addActionListener(e -> onStart());
        stopButton.addActionListener(e -> onStop());
        thresholdSpinner.addChangeListener(e ->
                metricsPanel.setThreshold((Double) thresholdSpinner.getValue()));

        return panel;
    }

    private void onStart() {
        int rows = (Integer) rowsSpinner.getValue();
        int cols = (Integer) colsSpinner.getValue();
        int drones = (Integer) dronesSpinner.getValue();
        double threshold = (Double) thresholdSpinner.getValue();
        long seed = randomEachRun.isSelected() ? -1L : 42L;

        metricsPanel.setThreshold(threshold);
        controller.setThreshold(threshold);
        statusLabel.setText("Генерація території та запуск дронів...");
        startButton.setEnabled(false);

        // Створення SimulatedTerrain (читання/декодування зображень) робимо
        // поза EDT, щоб не заморожувати інтерфейс на час завантаження файлів.
        new SwingWorker<Void, Void>() {
            @Override
            protected Void doInBackground() throws Exception {
                controller.start(rows, cols, drones, CLUSTER_COUNT, CLUSTER_RADIUS_CELLS, seed,
                        MINE_IMAGES_DIR, BACKGROUND_IMAGES_DIR);
                return null;
            }

            @Override
            protected void done() {
                try {
                    get(); // прокидає виняток з doInBackground(), якщо був
                    stopButton.setEnabled(true);
                    statusLabel.setText("Обстеження триває...");
                    startRefreshTimer();
                } catch (Exception ex) {
                    statusLabel.setText("Помилка: " + ex.getCause());
                    startButton.setEnabled(true);
                }
            }
        }.execute();
    }

    private void onStop() {
        controller.stop();
        stopButton.setEnabled(false);
    }

    private void startRefreshTimer() {
        if (refreshTimer != null) {
            refreshTimer.stop();
        }
        refreshTimer = new Timer(150, e -> {
            mapPanel.repaint();
            metricsPanel.refresh();

            if (!controller.isRunning()) {
                refreshTimer.stop();
                controller.recordResultIfNeeded();
                startButton.setEnabled(true);
                stopButton.setEnabled(false);
                statusLabel.setText(controller.isComplete()
                        ? "Готово - вся територія покрита. Результат збережено в experiment_history.csv"
                        : "Зупинено користувачем. Результат збережено в experiment_history.csv");
            }
        });
        refreshTimer.start();
    }

    public static void main(String[] args) throws Exception {
        MineDetectorOnnx detector = new MineDetectorOnnx(ONNX_MODEL_PATH);
        SwingUtilities.invokeLater(() -> new DemoApp(detector).setVisible(true));
    }
}