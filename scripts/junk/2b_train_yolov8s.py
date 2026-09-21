"""
Entrenamiento de YOLOv26s (Small) con mayor resolución y batch optimizado
- Modelo: yolo26s.pt (no nano)
- Resolución: 768 (superior a 640)
- Batch: 12
- Dataset: dataset/ (359 imágenes en splits 70/20/10)
"""

from ultralytics import YOLO
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = str(BASE_DIR / "dataset" / "data.yaml")
MODEL_PATH = str(BASE_DIR / "yolo26s.pt")  # Small: más parámetros que nano, mejor rendimiento
EPOCHS = 100
IMG_SIZE = 768  # Mayor que 640 para mejor detección
BATCH_SIZE = 12  # Optimizado para GPU disponible
PROJECT_DIR = str(BASE_DIR / "training_runs")
RUN_NAME = "telerx_yolo26s_768_b12"


def train_model():
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"No existe dataset: {DATA_YAML}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"No existe modelo: {MODEL_PATH}")
    
    print(f"📦 Cargando modelo base {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    print(f"\n🚀 Iniciando entrenamiento YOLOv26s")
    print(f"   Dataset: {DATA_YAML}")
    print(f"   Resolución: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   Batch: {BATCH_SIZE}")
    print(f"   Épocas: {EPOCHS}")
    print(f"   Proyecto: {RUN_NAME}\n")
    
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=False,  # No sobrescribir si existe
        patience=100,    # Early stopping
        save_period=-1,  # Solo guardar best.pt
        verbose=True,
    )
    
    print("\n✅ Entrenamiento completado.")
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    print(f"📁 Mejor modelo: {best_model}")
    print(f"📊 Ver resultados en: {results.save_dir}/results.csv")


if __name__ == "__main__":
    os.makedirs(PROJECT_DIR, exist_ok=True)
    train_model()
