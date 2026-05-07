import cv2
import os
from pathlib import Path

# Configuración
CROP_DIR = Path("D:/Proyectos/TeleRx/crops_for_keypoints")
OUTPUT_DIR = Path("D:/Proyectos/TeleRx/visualizations/skeleton_debug")
OUTPUT_DIR.mkdir(exist_ok=True)

# Esqueleto completo (8 puntos)
POINTS_NAMES = {
    0: "Cabeza Femoral", 1: "Espina Tibial", 2: "Domo Talo",
    3: "Cóndilo Medial", 4: "Cóndilo Lateral",
    5: "Plateau Medial", 6: "Plateau Lateral", 7: "Notch Femoral"
}

def draw_skeleton(img_path, lbl_path, output_path):
    img = cv2.imread(str(img_path))
    if img is None: return
    h, w, _ = img.shape
    
    if not lbl_path.exists(): return

    with open(lbl_path, 'r') as f:
        line = f.readline().split()
        if len(line) < 29: return
        
        # Saltamos id, box (5 valores)
        kpts = line[5:]
        
        for i in range(8):
            px = float(kpts[i*3]) * w
            py = float(kpts[i*3 + 1]) * h
            vis = int(kpts[i*3 + 2])
            
            if vis > 0: # Si es 1 o 2, dibujar
                color = (0, 255, 0) if vis == 2 else (0, 0, 255) # Verde si visible, Rojo si ocluido
                cv2.circle(img, (int(px), int(py)), 6, color, -1)
                cv2.putText(img, str(i), (int(px)+5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imwrite(str(output_path), img)

def debug_skeletons():
    txt_files = [f for f in os.listdir(CROP_DIR) if f.endswith(".txt")]
    print(f"Dibujando esqueletos para {len(txt_files)} archivos...")
    
    for f in txt_files:
        stem = Path(f).stem
        # Intentar buscar la imagen (png o jpg)
        img_path = None
        for ext in ['.jpg', '.png']:
            if (CROP_DIR / f"{stem}{ext}").exists():
                img_path = CROP_DIR / f"{stem}{ext}"
                break
        
        if img_path:
            draw_skeleton(img_path, CROP_DIR / f, OUTPUT_DIR / f"{stem}_skeleton.jpg")

    print(f"Imágenes de debug guardadas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    debug_skeletons()
