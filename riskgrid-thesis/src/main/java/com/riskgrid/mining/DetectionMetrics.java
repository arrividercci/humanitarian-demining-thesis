package com.riskgrid.mining;

/**
 * Спільний розрахунок метрик якості детекції (Accuracy/Precision/Recall/F1/
 * кореляція/MAE) проти ground truth - використовується і пакетним Main
 * (наприкінці, коли покриття повне), і живим DemoApp/MetricsPanel (у будь-
 * який момент, по вже відвіданих клітинках).
 */
public final class DetectionMetrics {

    public record Result(int n, double accuracy, double precision, double recall,
                          double f1, double correlation, double mae) {}

    private DetectionMetrics() {}

    public static Result compute(RiskGrid grid, SimulatedTerrain terrain, CoverageTracker coverage,
                                  int rows, int cols, double threshold) {
        int n = 0, tp = 0, fp = 0, fn = 0, tn = 0;
        double sumTruth = 0, sumRisk = 0, sumTruthSq = 0, sumRiskSq = 0, sumProduct = 0;
        double sumAbsError = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (!coverage.isVisited(i, j)) continue;
                n++;
                double risk = grid.getRisk(i, j);
                boolean truthMine = terrain.isMineCell(i, j);
                double truth = truthMine ? 1.0 : 0.0;
                boolean predMine = risk >= threshold;

                if (predMine && truthMine) tp++;
                else if (predMine) fp++;
                else if (truthMine) fn++;
                else tn++;

                sumTruth += truth;
                sumRisk += risk;
                sumTruthSq += truth * truth;
                sumRiskSq += risk * risk;
                sumProduct += truth * risk;
                sumAbsError += Math.abs(risk - truth);
            }
        }

        if (n == 0) {
            return new Result(0, 0, 0, 0, 0, 0, 0);
        }

        double accuracy = (tp + tn) / (double) n;
        double precision = (tp + fp) > 0 ? tp / (double) (tp + fp) : 0.0;
        double recall = (tp + fn) > 0 ? tp / (double) (tp + fn) : 0.0;
        double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
        double mae = sumAbsError / n;

        double meanTruth = sumTruth / n;
        double meanRisk = sumRisk / n;
        double covariance = sumProduct / n - meanTruth * meanRisk;
        double stdTruth = Math.sqrt(Math.max(0, sumTruthSq / n - meanTruth * meanTruth));
        double stdRisk = Math.sqrt(Math.max(0, sumRiskSq / n - meanRisk * meanRisk));
        double corr = (stdTruth > 1e-9 && stdRisk > 1e-9) ? covariance / (stdTruth * stdRisk) : 0.0;

        return new Result(n, accuracy, precision, recall, f1, corr, mae);
    }
}
