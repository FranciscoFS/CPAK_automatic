import os
from pathlib import Path
from ultralytics import YOLO

# Configuración
MODEL_PATH = "D:/Proyectos/TeleRx/training_runs/telerx_pose_v1/weights/best.pt"
VAL_DIR = Path("D:/Proyectos/TeleRx/dataset_pose/val/images")
OUTPUT_DIR = Path("D:/Proyectos/TeleRx/visualizations/knee_pose")

def save_knee_visuals():
    if not os.path.exists(MODEL_PATH):
        print("Modelo no encontrado.")
        return
        
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Filtrar solo rodillas
    images = [f for f in os.listdir(VAL_DIR) if "Rodilla" in f]
    
    print(f"Generando visualizaciones de Pose para {len(images)} rodillas...")
    
    for img_name in images:
        # Inferencia con save=True guarda la imagen con el esqueleto dibujado automáticamente
        model.predict(
            source=VAL_DIR / img_name,
            save=True,
            project=str(OUTPUT_DIR),
            name=".", # Para no crear subcarpetas extra
            exist_ok=True,
            conf=0.3
        )
        
    print(f"\nImágenes de Pose de Rodilla guardadas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    save_knee_visuals()
