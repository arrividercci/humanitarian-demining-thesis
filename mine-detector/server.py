"""
server.py

HTTP-сервіс над MineDetector. Приймає кадр (файл зображення) через
POST /predict і повертає suspicion score z у JSON.

Модель завантажується один раз при старті сервісу (не на кожен запит) --
саме заради цього і вибраний варіант "HTTP-сервіс" замість підпроцесу.

Запуск:
    pip install fastapi uvicorn python-multipart ultralytics
    python server.py

    (або: uvicorn server:app --host 0.0.0.0 --port 8000)

Приклад тестового запиту (curl):
    curl -X POST "http://127.0.0.1:8000/predict" \
         -F "file=@weights_test_image.jpg"

Приклад запиту з Java (SensorPipeline) -- звичайний HTTP POST
multipart/form-data, парсинг JSON-відповіді {"z": 0.9182}.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from mine_detector import MineDetector

WEIGHTS_PATH = Path(__file__).parent / "weights" / "best.pt"
CONF_THRESHOLD = 0.25

app = FastAPI(title="Mine Detector Service", version="1.0")

# Завантажуємо модель один раз при старті процесу
detector: MineDetector | None = None


class PredictResponse(BaseModel):
    z: float
    num_detections: int


@app.on_event("startup")
def load_model():
    global detector
    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"Файл ваг не знайдено: {WEIGHTS_PATH}. "
            "Поклади best.pt у папку weights/ поруч із server.py"
        )
    detector = MineDetector(str(WEIGHTS_PATH), conf_threshold=CONF_THRESHOLD)
    print(f"Модель завантажена з {WEIGHTS_PATH}")


@app.get("/health")
def health():
    """Простий healthcheck -- зручно для перевірки, що сервіс живий."""
    return {"status": "ok", "model_loaded": detector is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Приймає одне зображення (кадр) і повертає:
      - z: suspicion score (максимальний confidence детекції "міна")
      - num_detections: скільки об'єктів знайдено на кадрі
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Модель ще не завантажена")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_array = np.array(image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не вдалось прочитати зображення: {exc}")

    z = detector.get_suspicion_score(image_array)
    detections = detector.get_all_detections(image_array)

    return PredictResponse(z=z, num_detections=len(detections))


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)