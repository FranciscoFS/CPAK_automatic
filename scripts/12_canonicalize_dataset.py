import os
import cv2
from pathlib import Path

# Configuración
BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset_pose_final"
# Índices del esqueleto para intercambiar al hacer mirror (Medial <-> Lateral)
# Indices: 0:Centro, 1:Espina, 2:Talo, 3:CondMed, 4:CondLat, 5:PlatMed, 6:PlatLat, 7:Notch
FLIP_MAP = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 6, 6: 5, 7: 7}

def canonicalize():
    for split in ['train', 'val']:
        img_dir = DATASET_DIR / split / "images"
        lbl_dir = DATASET_DIR / split / "labels"
        
        for img_name in os.listdir(img_dir):
            if "Izq" in img_name:
                img_path = img_dir / img_name
                lbl_path = lbl_dir / f"{Path(img_name).stem}.txt"
                
                # 1. Voltear imagen
                img = cv2.imread(str(img_path))
                img_flipped = cv2.flip(img, 1)
                
                # 2. Procesar etiquetas
                new_points = {}
                with open(lbl_path, 'r') as f:
                    line = f.readline().split()
                    cls_id = line[0]
                    box = line[1:5]
                    kpts_data = line[5:]
                    
                    # Transformar puntos
                    # kpts_data: [x1, y1, v1, x2, y2, v2, ...]
                    for i in range(8):
                        x, y, v = float(kpts_data[i*3]), float(kpts_data[i*3+1]), int(kpts_data[i*3+2])
                        new_x = 1.0 - x
                        new_points[FLIP_MAP[i]] = (new_x, y, v)
                
                # 3. Guardar (renombrando a 'Der')
                new_stem = img_name.replace("Izq", "Der")
                cv2.imwrite(str(img_dir / new_stem), img_flipped)
                
                with open(lbl_dir / f"{Path(new_stem).stem}.txt", 'w') as f:
                    line = f"{cls_id} 0.5 0.5 0.9 0.9"
                    for i in range(8):
                        x, y, v = new_points[i]
                        line += f" {x:.6f} {y:.6f} {v}"
                    f.write(line + "\n")
                    
                # Borrar originales
                os.remove(img_path)
                os.remove(lbl_path)
                print(f"Normalizado: {img_name} -> {new_stem}")

if __name__ == "__main__":
    canonicalize()
