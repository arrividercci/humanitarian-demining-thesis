"""
download_terrain_subset.py

Завантажує датасет MineGuard-Thermal з Roboflow (той самий, що
використовувався для тренування YOLOv8n) і відбирає невелику підмножину
зображень для SimulatedTerrain: частину з мінами (для клітинок-цілей),
частину без анотацій, якщо такі є в датасеті (для фонових клітинок).

ПЕРЕД ЗАПУСКОМ: підстав свій api_key, workspace, project, version -
ці значення були в Colab-ноутбуці, який використовувався для тренування
(виклик Roboflow(...).workspace(...).project(...).version(...).download(...)).

Використання:
    pip install roboflow --break-system-packages
    python download_terrain_subset.py
"""
import shutil
import random
from pathlib import Path
from roboflow import Roboflow

# ---- ПІДСТАВ СВОЇ ЗНАЧЕННЯ ----
API_KEY = "IiRfiaFsPVGZXimnjrer"
WORKSPACE = "dronecv"          # орієнтовно, звір з тим, що було в Colab
PROJECT = "mineguard-thermal"  # орієнтовно, звір з тим, що було в Colab
VERSION = 15                    # орієнтовно, звір з тим, що було в Colab
# --------------------------------

N_MINE_IMAGES = 30
N_BACKGROUND_IMAGES = 30
OUTPUT_DIR = Path("terrain-images")


def has_annotations(label_path: Path) -> bool:
    """YOLO-формат: порожній .txt (або 0 байт) означає відсутність об'єктів на кадрі."""
    if not label_path.exists():
        return False
    return label_path.stat().st_size > 0


def main():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")

    dataset_dir = Path(dataset.location)
    test_images_dir = dataset_dir / "test" / "images"
    test_labels_dir = dataset_dir / "test" / "labels"

    if not test_images_dir.exists():
        print(f"УВАГА: не знайдено {test_images_dir} - перевір структуру завантаженого датасету вручну.")
        return

    mine_images = []
    background_images = []

    for img_path in test_images_dir.iterdir():
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        label_path = test_labels_dir / (img_path.stem + ".txt")
        if has_annotations(label_path):
            mine_images.append(img_path)
        else:
            background_images.append(img_path)

    print(f"Знайдено {len(mine_images)} зображень з мінами, {len(background_images)} фонових (без анотацій).")

    if not background_images:
        print("УВАГА: фонових зображень у датасеті немає - усі кадри містять хоча б одну міну.")
        print("Доведеться використати варіант B (синтетичні placeholder) для порожніх клітинок,")
        print("або окремо пошукати негативний клас, якщо він є в іншій версії/спліті проєкту.")

    random.seed(42)  # відтворюваність відбору
    mine_sample = random.sample(mine_images, min(N_MINE_IMAGES, len(mine_images)))
    background_sample = random.sample(background_images, min(N_BACKGROUND_IMAGES, len(background_images)))

    mine_out = OUTPUT_DIR / "mine"
    background_out = OUTPUT_DIR / "background"
    mine_out.mkdir(parents=True, exist_ok=True)
    background_out.mkdir(parents=True, exist_ok=True)

    for img in mine_sample:
        shutil.copy(img, mine_out / img.name)
    for img in background_sample:
        shutil.copy(img, background_out / img.name)

    print(f"Скопійовано {len(mine_sample)} зображень з мінами у {mine_out}")
    print(f"Скопійовано {len(background_sample)} фонових зображень у {background_out}")


if __name__ == "__main__":
    main()