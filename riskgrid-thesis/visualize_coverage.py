"""
visualize_coverage.py

Будує анімовану карту обходу території на основі CSV-логів, які записує
Main.java: terrain_layout.csv (яка клітинка - міна) і
visit_log_<N>drones.csv (хронологічний журнал відвідувань кожного дрона).

Кольорова схема:
  - Світло-сірий фон  - ще не відвідана клітинка
  - Червона рамка     - клітинка належить до кластера-міни (ground truth)
  - Заливка кольором  - suspicion score z детектора в момент відвідування
                        (від жовтого при z~0 до темно-червоного при z~1)
  - Кожен дрон має свій колір маркера поточної позиції

Використання:
    pip install matplotlib pandas --break-system-packages
    python visualize_coverage.py --drones 4
    python visualize_coverage.py --drones 4 --rows 30 --cols 30 --out coverage_4drones.gif
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

DRONE_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                "#17becf", "#e377c2", "#8c564b", "#bcbd22"]

Z_CMAP = LinearSegmentedColormap.from_list("suspicion", ["#fff7bc", "#d7301f", "#7f0000"])


def load_data(rows, cols, visit_log_path, terrain_path):
    terrain = pd.read_csv(terrain_path)
    mine_mask = [[False] * cols for _ in range(rows)]
    for _, row in terrain.iterrows():
        mine_mask[int(row["i"])][int(row["j"])] = bool(row["is_mine"])

    visits = pd.read_csv(visit_log_path)
    visits = visits.sort_values("elapsed_nanos").reset_index(drop=True)
    return mine_mask, visits


def build_animation(rows, cols, mine_mask, visits, steps_per_frame, out_path, fps):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title("Карта обходу території (0 кроків)")

    # Малюємо статичну сітку: фон + рамки для клітинок-мін (ground truth)
    cell_patches = [[None] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            edge_color = "#c0392b" if mine_mask[i][j] else "#dddddd"
            edge_width = 1.6 if mine_mask[i][j] else 0.4
            rect = patches.Rectangle((j, i), 1, 1, facecolor="#f5f5f5",
                                      edgecolor=edge_color, linewidth=edge_width)
            ax.add_patch(rect)
            cell_patches[i][j] = rect

    drone_markers = {}  # drone_id -> matplotlib artist (поточна позиція)

    total_frames = (len(visits) + steps_per_frame - 1) // steps_per_frame

    def update(frame_idx):
        end = min((frame_idx + 1) * steps_per_frame, len(visits))
        chunk = visits.iloc[:end]

        # Перефарбовуємо всі клітинки, відвідані станом на цей кадр
        last_seen = chunk.drop_duplicates(subset=["i", "j"], keep="last")
        for _, row in last_seen.iterrows():
            i, j, z = int(row["i"]), int(row["j"]), float(row["z"])
            cell_patches[i][j].set_facecolor(Z_CMAP(z))

        # Оновлюємо маркери поточної позиції кожного дрона (останній крок кожного)
        latest_per_drone = chunk.groupby("drone_id").tail(1)
        for artist in drone_markers.values():
            artist.remove()
        drone_markers.clear()
        for _, row in latest_per_drone.iterrows():
            drone_id, i, j = int(row["drone_id"]), int(row["i"]), int(row["j"])
            color = DRONE_COLORS[drone_id % len(DRONE_COLORS)]
            marker, = ax.plot(j + 0.5, i + 0.5, "o", color=color,
                               markersize=10, markeredgecolor="black", zorder=5)
            drone_markers[drone_id] = marker

        ax.set_title(f"Карта обходу території ({end} з {len(visits)} кроків)")
        return list(cell_patches_flat()) + list(drone_markers.values())

    def cell_patches_flat():
        for row in cell_patches:
            for c in row:
                yield c

    anim = FuncAnimation(fig, update, frames=total_frames, blit=False, repeat=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    print(f"Готово: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drones", type=int, required=True,
                         help="Кількість дронів (визначає, який visit_log_<N>drones.csv читати)")
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--cols", type=int, default=30)
    parser.add_argument("--terrain", default="terrain_layout.csv")
    parser.add_argument("--steps-per-frame", type=int, default=5,
                         help="Скільки кроків додавати за один кадр анімації (більше = коротша GIF)")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    visit_log_path = f"visit_log_{args.drones}drones.csv"
    out_path = args.out or f"coverage_{args.drones}drones.gif"

    mine_mask, visits = load_data(args.rows, args.cols, visit_log_path, args.terrain)
    print(f"Завантажено {len(visits)} кроків з {visit_log_path}")

    build_animation(args.rows, args.cols, mine_mask, visits,
                     args.steps_per_frame, out_path, args.fps)


if __name__ == "__main__":
    main()
