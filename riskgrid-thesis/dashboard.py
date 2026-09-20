"""
dashboard.py

Єдина комбінована візуалізація: зліва - анімована карта обходу території
(ground truth мін + детекції дронів у реальному часі), справа - метрики
якості детекції, які оновлюються по мірі того, як дрони покривають
територію (рахуються лише по вже відвіданих клітинках).

Метрики:
  - Accuracy, Precision, Recall, F1 (при заданому порозі risk)
  - Кореляція (Пірсона) між прогнозованим risk і реальною міткою is_mine -
    показує, наскільки узгоджено оцінка системи рухається разом з реальністю
  - MAE (середня абсолютна похибка) між risk (0..1) і is_mine (0/1)

Risk клітинки перераховується кумулятивно з журналу відвідувань за тією ж
формулою, що й RiskGrid.update() (адитивний log-odds, формула 2.4):
    Δl = w * ln(z / (1-z)),  risk = sigmoid(sum(Δl))
Це гарантує, що анімація точно відображає ту саму математику, що й
формальна модель, а не якесь спрощення.

Використання:
    pip install pandas matplotlib --break-system-packages
    python dashboard.py --drones 4
    python dashboard.py --drones 1 --threshold 0.3 --out dashboard_1drone.gif
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

DRONE_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                "#17becf", "#e377c2", "#8c564b", "#bcbd22"]
Z_CMAP = LinearSegmentedColormap.from_list("suspicion", ["#fff7bc", "#d7301f", "#7f0000"])
SENSOR_WEIGHT = 0.8   # звір з SENSOR_WEIGHT у Main.java
EPS = 1e-6


def clamp_z(z):
    return np.clip(z, EPS, 1.0 - EPS)


def load_data(terrain_path, visit_log_path):
    terrain = pd.read_csv(terrain_path)
    visits = pd.read_csv(visit_log_path).sort_values("elapsed_nanos").reset_index(drop=True)
    return terrain, visits


def compute_metrics(logodds_sum, visited_mask, is_mine_grid, threshold):
    """Метрики рахуються лише по клітинках, відвіданих станом на поточний кадр."""
    idx = np.where(visited_mask)
    if len(idx[0]) == 0:
        return dict(accuracy=0, precision=0, recall=0, f1=0, corr=0, mae=0, n=0)

    risk = 1.0 / (1.0 + np.exp(-logodds_sum[idx]))
    truth = is_mine_grid[idx].astype(float)
    pred = (risk >= threshold).astype(float)

    tp = np.sum((pred == 1) & (truth == 1))
    fp = np.sum((pred == 1) & (truth == 0))
    fn = np.sum((pred == 0) & (truth == 1))
    tn = np.sum((pred == 0) & (truth == 0))

    accuracy = (tp + tn) / len(idx[0])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    if np.std(truth) > 0 and np.std(risk) > 0:
        corr = np.corrcoef(truth, risk)[0, 1]
    else:
        corr = 0.0  # кореляція не визначена, якщо всі відвідані клітинки одного класу

    mae = np.mean(np.abs(risk - truth))

    return dict(accuracy=accuracy, precision=precision, recall=recall,
                f1=f1, corr=corr, mae=mae, n=len(idx[0]))


def build_dashboard(rows, cols, terrain, visits, threshold, steps_per_frame, out_path, fps):
    is_mine_grid = np.zeros((rows, cols), dtype=bool)
    for _, row in terrain.iterrows():
        is_mine_grid[int(row["i"]), int(row["j"])] = bool(row["is_mine"])

    logodds_sum = np.zeros((rows, cols))
    visited_mask = np.zeros((rows, cols), dtype=bool)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax_map = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])

    ax_map.set_xlim(0, cols)
    ax_map.set_ylim(0, rows)
    ax_map.set_aspect("equal")
    ax_map.invert_yaxis()

    cell_patches = [[None] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            edge_color = "#c0392b" if is_mine_grid[i, j] else "#dddddd"
            edge_width = 1.6 if is_mine_grid[i, j] else 0.4
            rect = patches.Rectangle((j, i), 1, 1, facecolor="#f5f5f5",
                                      edgecolor=edge_color, linewidth=edge_width)
            ax_map.add_patch(rect)
            cell_patches[i][j] = rect

    drone_markers = {}
    bar_labels = ["Accuracy", "Precision", "Recall", "F1"]
    bars = ax_metrics.bar(bar_labels, [0, 0, 0, 0], color="#4c72b0")
    ax_metrics.set_ylim(0, 1.05)
    ax_metrics.set_title("Метрики (по відвіданих клітинках)")
    text_box = ax_metrics.text(0.02, -0.18, "", transform=ax_metrics.transAxes,
                                fontsize=11, va="top", family="monospace")

    total_frames = (len(visits) + steps_per_frame - 1) // steps_per_frame
    start = [0]  # мутабельний індекс останнього обробленого рядка visits

    def update(frame_idx):
        end = min((frame_idx + 1) * steps_per_frame, len(visits))
        chunk = visits.iloc[start[0]:end]
        start[0] = end

        for _, row in chunk.iterrows():
            i, j, z = int(row["i"]), int(row["j"]), float(row["z"])
            zc = clamp_z(z)
            delta_l = SENSOR_WEIGHT * np.log(zc / (1.0 - zc))
            logodds_sum[i, j] += delta_l
            visited_mask[i, j] = True
            risk = 1.0 / (1.0 + np.exp(-logodds_sum[i, j]))
            cell_patches[i][j].set_facecolor(Z_CMAP(risk))

        latest_per_drone = chunk.groupby("drone_id").tail(1) if len(chunk) > 0 else chunk
        for artist in drone_markers.values():
            artist.remove()
        drone_markers.clear()
        for _, row in latest_per_drone.iterrows():
            drone_id, i, j = int(row["drone_id"]), int(row["i"]), int(row["j"])
            color = DRONE_COLORS[drone_id % len(DRONE_COLORS)]
            marker, = ax_map.plot(j + 0.5, i + 0.5, "o", color=color,
                                   markersize=10, markeredgecolor="black", zorder=5)
            drone_markers[drone_id] = marker

        ax_map.set_title(f"Обхід території ({end} з {len(visits)} кроків)")

        m = compute_metrics(logodds_sum, visited_mask, is_mine_grid, threshold)
        for bar, key in zip(bars, ["accuracy", "precision", "recall", "f1"]):
            bar.set_height(m[key])
        text_box.set_text(
            f"Поріг risk:        {threshold:.2f}\n"
            f"Відвідано клітинок: {m['n']}\n"
            f"Кореляція (Пірсон): {m['corr']:+.3f}\n"
            f"MAE:                {m['mae']:.3f}"
        )

        return list(cell_patches_flat()) + list(drone_markers.values()) + list(bars)

    def cell_patches_flat():
        for row in cell_patches:
            for c in row:
                yield c

    anim = FuncAnimation(fig, update, frames=total_frames, blit=False, repeat=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    print(f"Готово: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drones", type=int, required=True)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--cols", type=int, default=30)
    parser.add_argument("--terrain", default="terrain_layout.csv")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Поріг risk для класифікації 'підозріло' (спробуй значення з evaluate_detection.py)")
    parser.add_argument("--steps-per-frame", type=int, default=5)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    visit_log_path = f"visit_log_{args.drones}drones.csv"
    out_path = args.out or f"dashboard_{args.drones}drones.gif"

    terrain, visits = load_data(args.terrain, visit_log_path)
    print(f"Завантажено {len(visits)} кроків з {visit_log_path}")

    build_dashboard(args.rows, args.cols, terrain, visits,
                     args.threshold, args.steps_per_frame, out_path, args.fps)


if __name__ == "__main__":
    main()
