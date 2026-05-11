import os
from pathlib import Path
from ultralytics import YOLO

# Configuración
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(BASE_DIR / "training_runs" / "telerx_pose_s_rebuild1_ft_full_b16_flipfix" / "weights" / "best.pt")
VAL_DIR = BASE_DIR / "dataset_pose_final" / "val" / "images"
OUTPUT_DIR = BASE_DIR / "visualizations" / "knee_pose"

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
