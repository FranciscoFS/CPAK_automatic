import cv2
import os
from pathlib import Path

# Configuración
CANONICAL_DIR = Path("D:/Proyectos/TeleRx/dataset_canonical")
OUTPUT_DIR = Path("D:/Proyectos/TeleRx/visualizations/canonical_debug")
OUTPUT_DIR.mkdir(exist_ok=True)

# Esqueleto: 0:Cabeza, 1:Espina, 2:Talo, 3:CondMed, 4:CondLat, 5:PlatMed, 6:PlatLat, 7:Notch
# Etiquetas descriptivas para identificar si Medial/Lateral se ven bien
LABEL_MAP = {
    3: "MEDIAL", 4: "LATERAL",
    5: "PMed", 6: "PLat"
}

def draw_skeleton(img_path, lbl_path, output_path):
    img = cv2.imread(str(img_path))
    if img is None: return
    h, w, _ = img.shape
    
    with open(lbl_path, 'r') as f:
        line = f.readline().split()
        if len(line) < 29: return
        kpts = line[5:]
        
        for i in range(8):
            px = float(kpts[i*3]) * w
            py = float(kpts[i*3+1]) * h
            vis = int(kpts[i*3+2])
            
            if vis > 0:
                color = (0, 255, 0) # Verde
                cv2.circle(img, (int(px), int(py)), 6, color, -1)
                txt = LABEL_MAP.get(i, str(i))
                cv2.putText(img, txt, (int(px)+5, int(py)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imwrite(str(output_path), img)

def debug_canonical():
    txt_files = [f for f in os.listdir(CANONICAL_DIR) if f.endswith(".txt")]
    print(f"Auditando {len(txt_files)} archivos normalizados...")
    
    for f in txt_files:
        stem = Path(f).stem
        img_path = CANONICAL_DIR / f"{stem}.jpg"
        if img_path.exists():
            draw_skeleton(img_path, CANONICAL_DIR / f, OUTPUT_DIR / f"{stem}_debug.jpg")

    print(f"Visualizaciones de auditoría guardadas en: {OUTPUT_DIR}")

if __name__ == "__main__":
    debug_canonical()
