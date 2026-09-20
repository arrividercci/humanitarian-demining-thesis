package com.riskgrid.mining;

import javax.swing.*;
import java.awt.*;

/**
 * Малює живу карту території: колір заливки клітинки - поточний risk
 * (від жовтого до темно-червоного), тонка червона рамка - ground truth
 * (клітинка справді належить до кластера-міни), кольорові кружечки -
 * поточні позиції дронів.
 */
public final class SimulationPanel extends JPanel {

    private static final Color[] DRONE_COLORS = {
            new Color(0x1f77b4), new Color(0x2ca02c), new Color(0x9467bd), new Color(0xff7f0e),
            new Color(0x17becf), new Color(0xe377c2), new Color(0x8c564b), new Color(0xbcbd22)
    };

    private SimulationController controller;

    public SimulationPanel() {
        setBackground(Color.WHITE);
        setPreferredSize(new Dimension(600, 600));
    }

    public void setController(SimulationController controller) {
        this.controller = controller;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (controller == null || controller.getRiskGrid() == null) {
            return;
        }

        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int rows = controller.getRows();
        int cols = controller.getCols();
        RiskGrid grid = controller.getRiskGrid();
        SimulatedTerrain terrain = controller.getTerrain();
        CoverageTracker coverage = controller.getCoverage();

        int panelSize = Math.min(getWidth(), getHeight());
        double cellPx = (double) panelSize / Math.max(rows, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int x = (int) Math.round(j * cellPx);
                int y = (int) Math.round(i * cellPx);
                int w = (int) Math.round(cellPx) + 1;
                int h = (int) Math.round(cellPx) + 1;

                boolean visited = coverage.isVisited(i, j);
                Color fill = visited ? riskColor(grid.getRisk(i, j)) : new Color(0xf5f5f5);
                g2.setColor(fill);
                g2.fillRect(x, y, w, h);

                boolean isMine = terrain.isMineCell(i, j);
                g2.setColor(isMine ? new Color(0xc0392b) : new Color(0xdddddd));
                g2.setStroke(new BasicStroke(isMine ? 1.6f : 0.4f));
                g2.drawRect(x, y, w, h);
            }
        }

        for (DroneAgent agent : controller.getAgents()) {
            RoutePlanner.Point pos = agent.getCurrentPosition();
            if (pos == null) continue;
            int[] cellIdx = grid.toCellIndex(pos.x(), pos.y());
            int cx = (int) Math.round((cellIdx[1] + 0.5) * cellPx);
            int cy = (int) Math.round((cellIdx[0] + 0.5) * cellPx);
            int r = (int) Math.max(4, cellPx * 0.35);

            g2.setColor(DRONE_COLORS[agent.getId() % DRONE_COLORS.length]);
            g2.fillOval(cx - r, cy - r, 2 * r, 2 * r);
            g2.setColor(Color.BLACK);
            g2.setStroke(new BasicStroke(1.2f));
            g2.drawOval(cx - r, cy - r, 2 * r, 2 * r);
        }
    }

    /** Жовтий (z~0) -> темно-червоний (z~1), той самий градієнт, що й у Python-візуалізації. */
    private static Color riskColor(double risk) {
        risk = Math.max(0, Math.min(1, risk));
        int r1 = 0xff, g1 = 0xf7, b1 = 0xbc; // світло-жовтий
        int r2 = 0x7f, g2c = 0x00, b2 = 0x00; // темно-червоний
        int r = (int) (r1 + (r2 - r1) * risk);
        int gCh = (int) (g1 + (g2c - g1) * risk);
        int b = (int) (b1 + (b2 - b1) * risk);
        return new Color(r, gCh, b);
    }
}
