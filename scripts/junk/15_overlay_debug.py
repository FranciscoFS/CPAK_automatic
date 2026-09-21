import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# Configuración
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = str(BASE_DIR / "training_runs" / "telerx_pose_s_rebuild1_ft_full_b16_flipfix" / "weights" / "best.pt")
VAL_DIR = BASE_DIR / "dataset_pose_final" / "val" / "images"
LBL_DIR = BASE_DIR / "dataset_pose_final" / "val" / "labels"
OUTPUT_DIR = BASE_DIR / "visualizations" / "overlay_debug_v2"

# Definición de qué puntos dibujar según la clase (Filtro Clínico)
JOINT_VISIBILITY = {
    "Cadera": [0],
    "Rodilla": [1, 3, 4, 5, 6, 7],
    "Tobillo": [2]
}

def draw_point(img, x, y, color, label, base_size):
    # Grosor y tamaño dinámico
    radius = max(3, int(base_size / 150))
    font_scale = base_size / 1100.0
    thickness = max(1, int(base_size / 500))
    
    cv2.circle(img, (int(x), int(y)), radius, color, -1)
    cv2.putText(img, label, (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def overlay_debug():
    if not os.path.exists(MODEL_PATH): 
        print("Modelo no encontrado.")
        return
        
    model = YOLO(MODEL_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(VAL_DIR) if f.endswith(('.jpg', '.png'))]
    
    print(f"Generando overlays (Clínicos) para {len(files)} imágenes...")
    
    for img_name in files:
        # Identificar clase por nombre
        joint_type = next((k for k in JOINT_VISIBILITY if k in img_name), None)
        if not joint_type: continue
        allowed_points = JOINT_VISIBILITY[joint_type]
        
        img_path = VAL_DIR / img_name
        lbl_path = LBL_DIR / f"{Path(img_name).stem}.txt"
        
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        h, w, _ = img.shape
        base_size = max(h, w)
        
        # 1. Dibujar Manual (Ground Truth) en AZUL
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                line = f.readline().split()
                if len(line) >= 29:
                    kpts = line[5:]
                    for i in allowed_points:
                        px = float(kpts[i*3]) * w
                        py = float(kpts[i*3+1]) * h
                        vis = int(kpts[i*3+2])
                        if vis > 0:
                            draw_point(img, px, py, (255, 0, 0), f"GT{i}", base_size)

        # 2. Dibujar Predicho en VERDE
        res = model.predict(img, conf=0.5, verbose=False)[0]
        if res.keypoints is not None and res.keypoints.xy is not None and len(res.keypoints.xy) > 0:
            kpts = res.keypoints.xy[0].cpu().numpy()
            for i in allowed_points: # Solo dibujamos los permitidos para esta clase
                px, py = kpts[i]
                if px > 0 and py > 0:
                    draw_point(img, px, py, (0, 255, 0), f"P{i}", base_size)

        cv2.imwrite(str(OUTPUT_DIR / img_name), img)
    
    print(f"Overlays corregidos guardados en {OUTPUT_DIR}")

if __name__ == "__main__":
    overlay_debug()
