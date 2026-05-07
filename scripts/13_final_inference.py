from ultralytics import YOLO
import os
from pathlib import Path

# Cargar el modelo final entrenado
model = YOLO("D:/Proyectos/TeleRx/training_runs/telerx_pose_v2/weights/best.pt")
val_img_path = Path("D:/Proyectos/TeleRx/dataset_pose/val/images")
output_dir = Path("D:/Proyectos/TeleRx/visualizations/final_test")

# Buscamos una de cada una
files = os.listdir(val_img_path)
selected = []
for target in ["Cadera", "Rodilla", "Tobillo"]:
    for f in files:
        if target in f:
            selected.append(f)
            break

print(f"Iniciando inferencia de validación final en: {selected}")

for f in selected:
    # Inferencia
    results = model.predict(val_img_path / f, save=True, project=str(output_dir), name=".", exist_ok=True)
    print(f"Resultado guardado para: {f}")
