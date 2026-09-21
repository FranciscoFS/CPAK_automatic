from ultralytics import YOLO
from pathlib import Path
import os

# Configuración
MODEL_PATH = "D:/Proyectos/TeleRx/training_runs/telerx_pose_v1/weights/best.pt"
VAL_DIR = Path("D:/Proyectos/TeleRx/dataset_pose/val/images")

def force_inference():
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado.")
        return
    model = YOLO(MODEL_PATH)
    images = [f for f in os.listdir(VAL_DIR) if f.lower().endswith(('.jpg', '.png'))]
    print(f"Forzando inferencia en 5 imágenes del set de validación (Confianza baja: 0.1)...")
    import random
    selected = random.sample(images, min(len(images), 5))
    for img_name in selected:
        print(f"Analizando: {img_name}")
        results = model.predict(VAL_DIR / img_name, conf=0.1, save=True, show=False)
    print("\nInferencia terminada. Revisa la carpeta 'runs/detect/predict' para ver si detecta Caderas/Rodillas.")

if __name__ == "__main__":
    force_inference()
