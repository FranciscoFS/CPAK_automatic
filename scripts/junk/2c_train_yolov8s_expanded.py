"""
Entrenamiento de YOLOv26s (Small) con dataset expandido
- Modelo: yolo26s.pt
- Resolución: 768
- Batch: 12
- Dataset: dataset_expanded/ (combina manual + auto-etiquetado)

Ejecutar primero: scripts/2a_prepare_expanded_dataset.py
"""

from ultralytics import YOLO
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = str(BASE_DIR / "dataset_expanded" / "data.yaml")
MODEL_PATH = str(BASE_DIR / "yolo26s.pt")
EPOCHS = 100
IMG_SIZE = 768
BATCH_SIZE = 12
PROJECT_DIR = str(BASE_DIR / "training_runs")
RUN_NAME = "telerx_yolo26s_768_b12_expanded"


def train_model():
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"Dataset expandido no encontrado: {DATA_YAML}\n"
            f"Ejecutar primero: python scripts/2a_prepare_expanded_dataset.py"
        )
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"No existe modelo: {MODEL_PATH}")
    
    print(f"📦 Cargando modelo base {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    print(f"\n🚀 Iniciando entrenamiento YOLOv26s (DATASET EXPANDIDO)")
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
        exist_ok=False,
        patience=100,
        save_period=-1,
        verbose=True,
    )
    
    print("\n✅ Entrenamiento completado.")
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    print(f"📁 Mejor modelo: {best_model}")
    print(f"📊 Ver resultados en: {results.save_dir}/results.csv")


if __name__ == "__main__":
    os.makedirs(PROJECT_DIR, exist_ok=True)
    train_model()
