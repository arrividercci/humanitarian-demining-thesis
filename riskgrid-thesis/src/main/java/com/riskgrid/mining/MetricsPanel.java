package com.riskgrid.mining;

import javax.swing.*;
import java.awt.*;

/**
 * Показує метрики якості детекції, пораховані лише по вже відвіданих
 * клітинках (Accuracy/Precision/Recall/F1/кореляція/MAE) - оновлюються
 * по мірі того, як дрони покривають територію.
 */
public final class MetricsPanel extends JPanel {

    private SimulationController controller;
    private double threshold = 0.5;

    private final JLabel coverageLabel = new JLabel();
    private final JLabel accuracyLabel = new JLabel();
    private final JLabel precisionLabel = new JLabel();
    private final JLabel recallLabel = new JLabel();
    private final JLabel f1Label = new JLabel();
    private final JLabel corrLabel = new JLabel();
    private final JLabel maeLabel = new JLabel();

    public MetricsPanel() {
        setLayout(new GridLayout(0, 1, 4, 4));
        setBorder(BorderFactory.createTitledBorder("Метрики (по відвіданих клітинках)"));
        Font mono = new Font(Font.MONOSPACED, Font.PLAIN, 13);
        for (JLabel l : new JLabel[]{coverageLabel, accuracyLabel, precisionLabel,
                recallLabel, f1Label, corrLabel, maeLabel}) {
            l.setFont(mono);
            add(l);
        }
        resetLabels();
    }

    public void setController(SimulationController controller) {
        this.controller = controller;
        resetLabels();
    }

    public void setThreshold(double threshold) {
        this.threshold = threshold;
    }

    private void resetLabels() {
        coverageLabel.setText("Покриття:    0 / 0");
        accuracyLabel.setText("Accuracy:    -");
        precisionLabel.setText("Precision:   -");
        recallLabel.setText("Recall:      -");
        f1Label.setText("F1:          -");
        corrLabel.setText("Кореляція:   -");
        maeLabel.setText("MAE:         -");
    }

    /** Викликається періодично з Timer у EDT для перерахунку й оновлення тексту. */
    public void refresh() {
        if (controller == null || controller.getRiskGrid() == null) {
            resetLabels();
            return;
        }

        DetectionMetrics.Result m = DetectionMetrics.compute(
                controller.getRiskGrid(), controller.getTerrain(), controller.getCoverage(),
                controller.getRows(), controller.getCols(), threshold);

        int total = controller.getRows() * controller.getCols();
        coverageLabel.setText(String.format("Покриття:    %d / %d", m.n(), total));

        if (m.n() == 0) {
            accuracyLabel.setText("Accuracy:    -");
            precisionLabel.setText("Precision:   -");
            recallLabel.setText("Recall:      -");
            f1Label.setText("F1:          -");
            corrLabel.setText("Кореляція:   -");
            maeLabel.setText("MAE:         -");
            return;
        }

        accuracyLabel.setText(String.format("Accuracy:    %.3f", m.accuracy()));
        precisionLabel.setText(String.format("Precision:   %.3f", m.precision()));
        recallLabel.setText(String.format("Recall:      %.3f", m.recall()));
        f1Label.setText(String.format("F1:          %.3f", m.f1()));
        corrLabel.setText(String.format("Кореляція:   %+.3f", m.correlation()));
        maeLabel.setText(String.format("MAE:         %.3f", m.mae()));
    }
}