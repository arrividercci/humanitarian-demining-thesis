"""
export_onnx.py

Експортує донавчену YOLOv8n-модель (weights/best.pt) у формат ONNX
для локального (in-process) інференсу в Java через ONNX Runtime -
замість HTTP-виклику до окремого Python-сервісу (server.py).

Використання:
    python export_onnx.py [шлях/до/best.pt]

Якщо шлях не вказано, береться weights/best.pt поруч із цим файлом.
"""
import sys
from pathlib import Path

from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "best.pt"


def main():
    if len(sys.argv) > 2:
        print("Використання: python export_onnx.py [шлях_до_best.pt]")
        sys.exit(1)

    weights_path = Path(sys.argv[1]) if len(sys.argv) == 2 else DEFAULT_WEIGHTS
    if not weights_path.exists():
        raise FileNotFoundError(f"Файл ваг не знайдено: {weights_path}")

    model = YOLO(str(weights_path))

    # opset=12 сумісний з більшістю версій onnxruntime-java;
    # simplify=True спрощує граф (потребує пакет onnxsim)
    exported_path = model.export(format="onnx", imgsz=640, opset=12, simplify=True)
    print(f"Готово: {exported_path}")
    print("Скопіюй цей .onnx файл у ресурси Java-проєкту "
          "(наприклад riskgrid-thesis/src/main/resources/model/).")


if __name__ == "__main__":
    main()
