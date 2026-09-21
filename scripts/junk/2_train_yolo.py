from ultralytics import YOLO
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = str(BASE_DIR / "dataset_det_final" / "data.yaml")
MODEL_NAME = "yolov8n.pt"  # O "yolo11n.pt" si prefieres el último
EPOCHS = 100
IMG_SIZE = 640
PROJECT_DIR = str(BASE_DIR / "training_runs")
RUN_NAME = "telerx_yolov8n"


def train_model():
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"No existe dataset: {DATA_YAML}")
    
    print(f"Cargando modelo base {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    print(f"Iniciando entrenamiento en {DATA_YAML}...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True
    )
    
    print("Entrenamiento completado.")
    print(f"Mejor modelo guardado en: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    os.makedirs(PROJECT_DIR, exist_ok=True)
    train_model()
