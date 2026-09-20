"""
mine_detector.py

Інференс-модуль для отримання suspicion score z з одного кадру.
z ∈ [0, 1] — це confidence найвпевненішого детектованого об'єкта класу "міна"
на кадрі. Якщо об'єктів не знайдено — z = 0.0.

Використання як CLI (для швидкого тесту):
    python mine_detector.py path/to/best.pt path/to/image.jpg

Використання як модуль (для інтеграції з SensorPipeline):
    from mine_detector import MineDetector

    detector = MineDetector("path/to/best.pt")
    z = detector.get_suspicion_score("path/to/image.jpg")
    # або з numpy-масивом (наприклад кадр з відео/дрона):
    z = detector.get_suspicion_score(frame_array)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union

import numpy as np
from ultralytics import YOLO


class MineDetector:
    """Обгортка над донавченою YOLO-моделлю для отримання suspicion score."""

    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        """
        weights_path   -- шлях до best.pt (донавченої моделі)
        conf_threshold -- мінімальний confidence, нижче якого детекція
                           ігнорується (стандартне значення YOLO — 0.25)
        """
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def get_suspicion_score(self, image: Union[str, Path, np.ndarray]) -> float:
        """
        Повертає z ∈ [0, 1] для одного кадру.

        z визначається як максимальний confidence серед усіх детекцій
        класу "міна" на кадрі (найвпевненіша знахідка домінує — це
        консервативний підхід: якщо хоч один детектор впевнений,
        клітинка вважається підозрілою).

        Якщо об'єктів не знайдено взагалі -- повертає 0.0.
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            verbose=False,
            device="cpu",  # ROCm/HIP на AMD GPU нестабільний (задокументована проблема) - примусово CPU
        )

        # results — список (по одному елементу на кожен вхідний кадр);
        # тут завжди рівно один кадр на виклик
        boxes = results[0].boxes

        if boxes is None or len(boxes) == 0:
            return 0.0

        # boxes.conf -- тензор confidence-значень усіх знайдених об'єктів
        max_conf = float(boxes.conf.max())
        return max_conf

    def get_all_detections(self, image: Union[str, Path, np.ndarray]) -> list[dict]:
        """
        Допоміжний метод для налагодження/візуалізації: повертає всі
        детекції на кадрі (не тільки максимум), кожна як словник
        з bbox-координатами та confidence.
        """
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            verbose=False,
            device="cpu",  # ROCm/HIP на AMD GPU нестабільний (задокументована проблема) - примусово CPU
        )
        boxes = results[0].boxes
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append({
                "bbox": xyxy,  # [x1, y1, x2, y2]
                "confidence": conf,
            })
        return detections


def main():
    if len(sys.argv) != 3:
        print("Використання: python mine_detector.py <шлях_до_best.pt> <шлях_до_зображення>")
        sys.exit(1)

    weights_path = sys.argv[1]
    image_path = sys.argv[2]

    detector = MineDetector(weights_path)
    z = detector.get_suspicion_score(image_path)

    print(f"Зображення: {image_path}")
    print(f"Suspicion score z = {z:.4f}")

    detections = detector.get_all_detections(image_path)
    print(f"Всього детекцій: {len(detections)}")
    for i, det in enumerate(detections, 1):
        print(f"  [{i}] confidence={det['confidence']:.4f}  bbox={det['bbox']}")


if __name__ == "__main__":
    main()