from ultralytics import YOLO
import os
from pathlib import Path

# Configuración
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = str(BASE_DIR / "dataset_pose_final" / "data.yaml")
# Entrenamiento desde cero con modelo small de YOLO26 Pose.
MODEL_NAME = str(BASE_DIR / "yolo26s-pose.pt")
EPOCHS = 100
IMG_SIZE = 800 
BATCH_SIZE = 8 

def train_pose_model():
    print(f"Entrenando modelo de Pose desde cero: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    print(f"Iniciando entrenamiento con datos en: {DATA_YAML}")
    
    # Entrenar con configuración explícita de aumentaciones y parámetros de Pose
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,
        project=str(BASE_DIR / "training_runs"),
        name="telerx_pose_s_rebuild1",
        exist_ok=True,
        # Aumentaciones explicitas
        fliplr=0.5,        # Habilita el uso del flip_idx que definimos
        mosaic=1.0,        
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
    
    print("Entrenamiento completado.")
    print(f"Mejor modelo guardado en: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    os.makedirs(BASE_DIR / "training_runs", exist_ok=True)
    train_pose_model()
