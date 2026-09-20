"""
analyze_batch.py - аналіз результатів ExperimentBatch (batch_results*.csv).

Що робить:
  1. Частка завершених запусків (complete=1) для кожної конфігурації.
  2. Масштабування: медіана часу, прискорення (парне по seed відносно 1 дрона), ефективність.
  3. Абляція: захоплення клітинок увімк. vs вимк. (парне по seed) - дублі, кроки,
     повтори, співвідношення часу, тест Вілкоксона.
  4. Затримка рішення: середня, p95, p99 (мс).
  5. Коректність: F1/precision/recall та відстань карти ризику від еталона (1 дрон),
     парна різниця F1 відносно 1 дрона + тест Вілкоксона.
  6. Графіки (PNG) і таблиці (CSV) у --out-dir.

Часові порівняння використовують лише запуски з complete=1 (незавершені -
обірвані бюджетом кроків - цензуровані, їхня частка звітується окремо).

Використання:
    pip install pandas matplotlib scipy
    python analyze_batch.py --input batch_mg0.csv batch_mg001.csv --out-dir analysis
    python analyze_batch.py --input batch_mg0.csv --lang en

Кожен --input вважається окремим "variant" (ім'я файлу без .csv), тож
можна порівнювати, наприклад, планувальник без порогу (min_gain=0) і з порогом.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import wilcoxon
except Exception:  # scipy необов'язковий: без нього p-значення будуть NaN
    wilcoxon = None

TEXT = {
    "uk": dict(
        drones="Кількість дронів", speedup="Прискорення (×)", ideal="Ідеальне",
        delay="політ {d:g} мс/кл.", dup="Одночасні дублі (медіана)",
        ratio="Час без захоплення / із захопленням", off="без захоплення", on="із захопленням",
        lat="Затримка рішення, мс", p95="p95", p99="p99",
        f1="F1 (медіана, IQR)", mae="Сер. абс. різниця карти від 1 дрона",
        t_speed="Масштабування", t_abl="Абляція: захоплення клітинок",
        t_lat="Затримка рішення", t_q="Якість і коректність",
    ),
    "en": dict(
        drones="Number of drones", speedup="Speedup (x)", ideal="Ideal",
        delay="flight {d:g} ms/cell", dup="Concurrent duplicates (median)",
        ratio="Time without claiming / with claiming", off="no claiming", on="claiming",
        lat="Decision latency, ms", p95="p95", p99="p99",
        f1="F1 (median, IQR)", mae="Mean abs. map difference vs 1 drone",
        t_speed="Scaling", t_abl="Ablation: cell claiming",
        t_lat="Decision latency", t_q="Quality and correctness",
    ),
}

GROUP = ["variant", "travel_delay_ms"]


# ------------------------------------------------------------------ helpers

def q25(s):
    return s.quantile(0.25)


def q75(s):
    return s.quantile(0.75)


def paired_p(a, b):
    """Двосторонній тест Вілкоксона для парних вибірок; NaN, якщо неможливо."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if wilcoxon is None or len(a) < 6 or np.allclose(a, b):
        return float("nan")
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def show(name, table, out_dir):
    print(f"\n=== {name} ===")
    if table.empty:
        print("(порожньо)")
        return
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    table.to_csv(os.path.join(out_dir, f"table_{name}.csv"), index=False)


def load(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["variant"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    for col, default in (("stale_revisits", 0), ("min_gain", 0.0)):
        if col not in df.columns:
            df[col] = default
    return df


# ------------------------------------------------------------------ tables

def completion_table(df):
    t = (df.groupby(GROUP + ["drones", "claiming"])
           .agg(runs=("seed", "count"), completed=("complete", "sum"))
           .reset_index())
    t["completion_rate"] = t["completed"] / t["runs"]
    return t


def scaling_table(df):
    ok = df[(df.claiming == 1) & (df.complete == 1)]
    base = {(r.variant, r.travel_delay_ms, r.seed): r.elapsed_ms
            for r in ok[ok.drones == 1].itertuples()}
    rows = []
    for (variant, delay, n), sub in ok.groupby(GROUP + ["drones"]):
        sp = []
        for r in sub.itertuples():
            b = base.get((r.variant, r.travel_delay_ms, r.seed))
            if b is not None:
                sp.append(b / r.elapsed_ms)
        sp = pd.Series(sp, dtype=float)
        rows.append(dict(
            variant=variant, travel_delay_ms=delay, drones=n, n_seeds=len(sub),
            elapsed_ms_median=sub.elapsed_ms.median(),
            elapsed_ms_q25=q25(sub.elapsed_ms), elapsed_ms_q75=q75(sub.elapsed_ms),
            speedup_median=sp.median() if len(sp) else np.nan,
            speedup_q25=q25(sp) if len(sp) else np.nan,
            speedup_q75=q75(sp) if len(sp) else np.nan,
            efficiency=(sp.median() / n) if len(sp) else np.nan,
            steps_median=sub.steps.median(),
            revisits_median=sub.revisits.median(),
        ))
    return pd.DataFrame(rows)


def ablation_table(df):
    sub = df[df.drones >= 2]
    key = GROUP + ["drones", "seed"]
    cols = ["elapsed_ms", "complete", "steps", "revisits", "stale_revisits",
            "concurrent_duplicates", "claim_failures"]
    on = sub[sub.claiming == 1][key + cols]
    off = sub[sub.claiming == 0][key + cols]
    m = on.merge(off, on=key, suffixes=("_on", "_off"))
    rows = []
    for (variant, delay, n), g in m.groupby(GROUP + ["drones"]):
        both = g[(g.complete_on == 1) & (g.complete_off == 1)]
        ratio = (both.elapsed_ms_off / both.elapsed_ms_on) if len(both) else pd.Series(dtype=float)
        rows.append(dict(
            variant=variant, travel_delay_ms=delay, drones=n, n_pairs=len(g),
            completion_on=g.complete_on.mean(), completion_off=g.complete_off.mean(),
            dup_on_median=g.concurrent_duplicates_on.median(),
            dup_off_median=g.concurrent_duplicates_off.median(),
            dup_p_wilcoxon=paired_p(g.concurrent_duplicates_on, g.concurrent_duplicates_off),
            steps_on_median=g.steps_on.median(), steps_off_median=g.steps_off.median(),
            steps_p_wilcoxon=paired_p(g.steps_on, g.steps_off),
            revisits_on_median=g.revisits_on.median(), revisits_off_median=g.revisits_off.median(),
            stale_on_median=g.stale_revisits_on.median(), stale_off_median=g.stale_revisits_off.median(),
            claim_failures_on_median=g.claim_failures_on.median(),
            time_ratio_off_over_on_median=ratio.median() if len(ratio) else np.nan,
            n_pairs_both_complete=len(both),
        ))
    return pd.DataFrame(rows)


def latency_table(df):
    ok = df[(df.claiming == 1) & (df.complete == 1)]
    t = (ok.groupby(GROUP + ["drones"])
           .agg(n_seeds=("seed", "count"),
                lat_mean_ms=("lat_mean_ms", "median"),
                lat_p50_ms=("lat_p50_ms", "median"),
                lat_p95_ms=("lat_p95_ms", "median"),
                lat_p99_ms=("lat_p99_ms", "median"),
                lat_max_ms=("lat_max_ms", "max"),
                claim_failures=("claim_failures", "median"))
           .reset_index())
    return t


def quality_table(df):
    ok = df[(df.claiming == 1) & (df.complete == 1)]
    base = {(r.variant, r.travel_delay_ms, r.seed): r.f1 for r in ok[ok.drones == 1].itertuples()}
    rows = []
    for (variant, delay, n), sub in ok.groupby(GROUP + ["drones"]):
        cur, ref = [], []
        for r in sub.itertuples():
            b = base.get((r.variant, r.travel_delay_ms, r.seed))
            if b is not None:
                cur.append(r.f1)
                ref.append(b)
        diff = np.array(cur, dtype=float) - np.array(ref, dtype=float)
        rows.append(dict(
            variant=variant, travel_delay_ms=delay, drones=n, n_seeds=len(sub),
            f1_median=sub.f1.median(), f1_q25=q25(sub.f1), f1_q75=q75(sub.f1),
            precision_median=sub.precision.median(), recall_median=sub.recall.median(),
            accuracy_median=sub.accuracy.median(),
            f1_diff_vs_1drone_median=float(np.median(diff)) if len(diff) else np.nan,
            f1_diff_p_wilcoxon=paired_p(cur, ref) if n != 1 else np.nan,
            risk_mae_vs_seq_median=sub.risk_mae_vs_seq.median(),
            risk_mae_vs_seq_q75=q75(sub.risk_mae_vs_seq),
        ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def _xticks(ax, drones, T):
    ax.set_xticks(sorted(set(drones)))
    ax.set_xlabel(T["drones"])
    ax.grid(alpha=0.3)


def plot_scaling(t, T, out_dir):
    for variant, tv in t.groupby("variant"):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for delay, td in tv.groupby("travel_delay_ms"):
            td = td.sort_values("drones")
            ax.plot(td.drones, td.speedup_median, marker="o", label=T["delay"].format(d=delay))
            ax.fill_between(td.drones, td.speedup_q25, td.speedup_q75, alpha=0.2)
        ns = sorted(tv.drones.unique())
        ax.plot(ns, ns, "k--", linewidth=0.8, label=T["ideal"])
        ax.set_ylabel(T["speedup"])
        ax.set_title(f"{T['t_speed']} ({variant})")
        _xticks(ax, tv.drones, T)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig_speedup_{variant}.png"), dpi=150)
        plt.close(fig)


def plot_ablation(t, T, out_dir):
    if t.empty:
        return
    for variant, tv in t.groupby("variant"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for delay, td in tv.groupby("travel_delay_ms"):
            td = td.sort_values("drones")
            lab = T["delay"].format(d=delay)
            line, = axes[0].plot(td.drones, td.dup_off_median, marker="o",
                                 label=f"{T['off']}, {lab}")
            axes[0].plot(td.drones, td.dup_on_median, marker="s", linestyle="--",
                         color=line.get_color(), label=f"{T['on']}, {lab}")
            axes[1].plot(td.drones, td.time_ratio_off_over_on_median, marker="o", label=lab)
        axes[0].set_ylabel(T["dup"])
        axes[0].set_title(T["t_abl"] + f" ({variant})")
        axes[1].set_ylabel(T["ratio"])
        axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        for ax in axes:
            _xticks(ax, tv.drones, T)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig_ablation_{variant}.png"), dpi=150)
        plt.close(fig)


def plot_latency(t, T, out_dir):
    if t.empty:
        return
    for variant, tv in t.groupby("variant"):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for delay, td in tv.groupby("travel_delay_ms"):
            td = td.sort_values("drones")
            lab = T["delay"].format(d=delay)
            line, = ax.plot(td.drones, td.lat_p95_ms, marker="o", label=f"{T['p95']}, {lab}")
            ax.plot(td.drones, td.lat_p99_ms, marker="s", linestyle="--",
                    color=line.get_color(), label=f"{T['p99']}, {lab}")
        ax.set_ylabel(T["lat"])
        ax.set_yscale("log")
        ax.set_title(f"{T['t_lat']} ({variant})")
        _xticks(ax, tv.drones, T)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig_latency_{variant}.png"), dpi=150)
        plt.close(fig)


def plot_quality(t, T, out_dir):
    if t.empty:
        return
    for variant, tv in t.groupby("variant"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for delay, td in tv.groupby("travel_delay_ms"):
            td = td.sort_values("drones")
            lab = T["delay"].format(d=delay)
            axes[0].plot(td.drones, td.f1_median, marker="o", label=lab)
            axes[0].fill_between(td.drones, td.f1_q25, td.f1_q75, alpha=0.2)
            axes[1].plot(td.drones, td.risk_mae_vs_seq_median, marker="o", label=lab)
        axes[0].set_ylabel(T["f1"])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title(f"{T['t_q']} ({variant})")
        axes[1].set_ylabel(T["mae"])
        for ax in axes:
            _xticks(ax, tv.drones, T)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig_quality_{variant}.png"), dpi=150)
        plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=["batch_results.csv"])
    ap.add_argument("--out-dir", default="analysis")
    ap.add_argument("--lang", choices=["uk", "en"], default="uk")
    args = ap.parse_args()

    T = TEXT[args.lang]
    os.makedirs(args.out_dir, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    df = load(args.input)
    print(f"Завантажено {len(df)} запусків з {len(args.input)} файл(ів); "
          f"seed: {df.seed.nunique()}, варіанти: {sorted(df.variant.unique())}")

    comp = completion_table(df)
    scal = scaling_table(df)
    abl = ablation_table(df)
    lat = latency_table(df)
    qual = quality_table(df)

    show("completion", comp, args.out_dir)
    show("scaling", scal, args.out_dir)
    show("ablation", abl, args.out_dir)
    show("latency", lat, args.out_dir)
    show("quality", qual, args.out_dir)

    plot_scaling(scal, T, args.out_dir)
    plot_ablation(abl, T, args.out_dir)
    plot_latency(lat, T, args.out_dir)
    plot_quality(qual, T, args.out_dir)
    print(f"\nТаблиці й графіки збережено в {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
