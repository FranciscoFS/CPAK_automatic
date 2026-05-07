from ultralytics import YOLO
import os
from pathlib import Path

# Configuración
MODEL_PATH = "D:/Proyectos/TeleRx/training_runs/telerx_pose_v2/weights/best.pt"
VAL_DIR = Path("D:/Proyectos/TeleRx/dataset_pose/val/images")
OUTPUT_DIR = Path("D:/Proyectos/TeleRx/visualizations/all_knees_debug")

def visualize_all_knees():
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado.")
        return
        
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Filtrar solo rodillas
    images = [f for f in os.listdir(VAL_DIR) if "Rodilla" in f]
    
    print(f"Generando visualizaciones para todas las {len(images)} rodillas en VAL...")
    
    for img_name in images:
        # Inferencia con save=True guarda la imagen con el esqueleto dibujado
        model.predict(
            source=VAL_DIR / img_name,
            save=True,
            project=str(OUTPUT_DIR),
            name=".",
            exist_ok=True,
            conf=0.2 # Umbral bajo para ver si hay detecciones débiles
        )
        
    print(f"\nVisualizaciones terminadas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_all_knees()
