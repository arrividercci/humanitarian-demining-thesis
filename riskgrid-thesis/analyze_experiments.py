"""
analyze_experiments.py

Аналізує накопичений experiment_history.csv (записується автоматично і
пакетним Main, і живим DemoApp - кожен завершений запуск додає один рядок).

Рахує:
  - прискорення відносно 1-дронової конфігурації в межах ОДНАКОВОЇ
    конфігурації території (той самий rows/cols/seed/threshold/cluster*)
  - таблицю всіх запусків
  - графіки: час vs кількість дронів, прискорення vs кількість дронів,
    Accuracy/Precision/Recall/F1 vs кількість дронів, кореляція vs MAE

Використання:
    pip install pandas matplotlib --break-system-packages
    python analyze_experiments.py
    python analyze_experiments.py --history experiment_history.csv --out-dir plots
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt


GROUP_KEYS = ["rows", "cols", "seed", "threshold", "cluster_count", "cluster_radius"]


def compute_speedup(df):
    """Додає колонку speedup: elapsed_ms(1 дрон) / elapsed_ms(N дронів) в межах
    однакової конфігурації території (одна й та сама група GROUP_KEYS)."""
    results = []
    for _, group in df.groupby(GROUP_KEYS):
        baseline_rows = group[group["drone_count"] == 1]
        if baseline_rows.empty:
            group = group.copy()
            group["speedup"] = float("nan")
        else:
            baseline_ms = baseline_rows["elapsed_ms"].mean()  # якщо кілька 1-дронових запусків - середнє
            group = group.copy()
            group["speedup"] = baseline_ms / group["elapsed_ms"]
        results.append(group)
    return pd.concat(results).sort_values(["rows", "cols", "seed", "drone_count"])


def print_summary_table(df):
    cols_to_show = ["timestamp", "source", "rows", "cols", "drone_count", "seed",
                     "elapsed_ms", "speedup", "accuracy", "precision", "recall", "f1",
                     "correlation", "mae"]
    print(df[cols_to_show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def plot_time_and_speedup(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for (rows, cols, seed), group in df.groupby(["rows", "cols", "seed"]):
        group = group.sort_values("drone_count")
        label = f"{rows}x{cols}, seed={seed}"
        axes[0].plot(group["drone_count"], group["elapsed_ms"], marker="o", label=label)
        axes[1].plot(group["drone_count"], group["speedup"], marker="o", label=label)

    axes[0].set_xlabel("Кількість дронів")
    axes[0].set_ylabel("Час покриття (мс)")
    axes[0].set_title("Час покриття vs кількість дронів")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Кількість дронів")
    axes[1].set_ylabel("Прискорення (x)")
    axes[1].set_title("Прискорення vs кількість дронів")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "time_and_speedup.png")
    fig.savefig(out_path, dpi=150)
    print(f"Збережено: {out_path}")


def plot_quality_metrics(df, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    for (rows, cols, seed), group in df.groupby(["rows", "cols", "seed"]):
        group = group.sort_values("drone_count")
        label = f"{rows}x{cols}, seed={seed}"
        ax.plot(group["drone_count"], group["accuracy"], marker="o", label=f"Accuracy ({label})")
        ax.plot(group["drone_count"], group["recall"], marker="s", linestyle="--",
                label=f"Recall ({label})")

    ax.set_xlabel("Кількість дронів")
    ax.set_ylabel("Значення метрики")
    ax.set_title("Якість детекції vs кількість дронів")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "quality_metrics.png")
    fig.savefig(out_path, dpi=150)
    print(f"Збережено: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="experiment_history.csv")
    parser.add_argument("--out-dir", default=".")
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"Файл {args.history} не знайдено. Запусти кілька симуляцій (Main або DemoApp) спершу.")
        return

    df = pd.read_csv(args.history)
    print(f"Завантажено {len(df)} записів експериментів з {args.history}\n")

    df = compute_speedup(df)
    print_summary_table(df)

    os.makedirs(args.out_dir, exist_ok=True)
    plot_time_and_speedup(df, args.out_dir)
    plot_quality_metrics(df, args.out_dir)


if __name__ == "__main__":
    main()
