"""
evaluate_detection.py

Кількісна оцінка якості детекції: порівнює фінальне значення risk у
RiskGrid (final_risk_<N>drones.csv) з ground truth розміщенням мін
(is_mine з того самого файлу, походить із SimulatedTerrain).

Виводить:
  - матрицю помилок (TP/FP/FN/TN) при заданому порозі
  - precision, recall, F1
  - PR-криву (precision/recall для діапазону порогів) - показує, що метод
    стабільно розділяє класи не лише при одному "вдало підібраному" порозі

Використання:
    pip install pandas matplotlib scikit-learn --break-system-packages
    python evaluate_detection.py --drones 1
    python evaluate_detection.py --drones 1 --threshold 0.6
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drones", type=int, required=True,
                         help="Яку конфігурацію оцінювати (визначає final_risk_<N>drones.csv)")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Поріг risk, вище якого клітинка вважається 'підозрілою'")
    args = parser.parse_args()

    path = f"final_risk_{args.drones}drones.csv"
    df = pd.read_csv(path)

    y_true = df["is_mine"].astype(int)
    y_score = df["risk"]
    y_pred = (y_score >= args.threshold).astype(int)

    print(f"=== Оцінка детекції для {args.drones} дрон(ів), поріг risk >= {args.threshold} ===")
    print(f"Всього клітинок: {len(df)}, з них справжніх мін: {y_true.sum()}")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print()
    print("Матриця помилок:")
    print(f"                Прогноз: міна   Прогноз: чисто")
    print(f"Насправді міна       {tp:6d}          {fn:6d}")
    print(f"Насправді чисто      {fp:6d}          {tn:6d}")

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print()
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")

    # PR-крива для повного діапазону порогів
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)

    # F-beta з beta=2: надає recall удвічі більшу вагу, ніж precision -
    # обґрунтований вибір для задачі, де ціна пропущеної міни (False Negative)
    # набагато вища за ціну хибної тривоги (False Positive). Формула:
    # F_beta = (1+beta^2) * P*R / (beta^2*P + R)
    beta = 2.0
    denom = beta**2 * precisions + recalls
    f_beta_scores = np.divide((1 + beta**2) * precisions * recalls, denom,
                               out=np.zeros_like(denom), where=denom != 0)
    best_idx = f_beta_scores.argmax()
    # precision_recall_curve повертає on 1 елемент менше thresholds, ніж precisions/recalls
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    print()
    print(f"=== Оптимальний поріг за F{beta:.0f}-мірою (recall важить у {beta:.0f}x більше за precision) ===")
    print(f"Рекомендований поріг: {best_threshold:.3f}")
    print(f"  Precision при цьому порозі: {precisions[best_idx]:.3f}")
    print(f"  Recall при цьому порозі:    {recalls[best_idx]:.3f}")
    print(f"  F{beta:.0f}: {f_beta_scores[best_idx]:.3f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recalls, precisions, marker=".", markersize=2)
    ax.scatter([recall], [precision], color="red", zorder=5,
               label=f"поточний поріг = {args.threshold}")
    ax.scatter([recalls[best_idx]], [precisions[best_idx]], color="green", zorder=5,
               label=f"оптимум за F{beta:.0f} = {best_threshold:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall крива ({args.drones} дрон(ів))")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    out_path = f"pr_curve_{args.drones}drones.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPR-крива збережена у {out_path}")


if __name__ == "__main__":
    main()