from ultralytics import YOLO
from pathlib import Path
import os
import random

# Ruta al modelo que acabas de entrenar (el 'last.pt' que se guardó al cortar)
MODEL_PATH = "D:/Proyectos/TeleRx/training_runs/telerx_pose_from_scratch/weights/last.pt"
VAL_DIR = Path("D:/Proyectos/TeleRx/dataset_pose_final/val/images")
OUTPUT_DIR = Path("D:/Proyectos/TeleRx/visualizations/final_v2_test")

def run_test_inference():
    if not os.path.exists(MODEL_PATH):
        print(f"No se encontró el modelo en {MODEL_PATH}")
        return
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_images = [f for f in os.listdir(VAL_DIR) if f.endswith(('.jpg', '.png'))]
    selected = random.sample(all_images, min(len(all_images), 5))
    print(f"Probando el modelo v2 en: {selected}")
    for img_name in selected:
        model.predict(
            source=VAL_DIR / img_name,
            save=True,
            project=str(OUTPUT_DIR),
            name=".",
            exist_ok=True,
            conf=0.25
        )
        print(f"Predicción guardada para: {img_name}")

if __name__ == "__main__":
    run_test_inference()
