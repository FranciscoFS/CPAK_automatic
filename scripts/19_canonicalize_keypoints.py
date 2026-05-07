import os
import cv2
import shutil
from pathlib import Path

# Configuración
CROP_DIR = Path("D:/Proyectos/TeleRx/crops_for_keypoints")
CANONICAL_DIR = Path("D:/Proyectos/TeleRx/dataset_canonical")

# Índices para intercambiar (Medial <-> Lateral)
# 0: Cabeza, 1: Espina, 2: Talo, 3: Medial, 4: Lateral, 5: PlatMed, 6: PlatLat, 7: Notch
FLIP_MAP = {0: 0, 1: 1, 2: 2, 3: 4, 4: 3, 5: 6, 6: 5, 7: 7}

def canonicalize():
    os.makedirs(CANONICAL_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(CROP_DIR) if f.endswith(".txt")]
    print(f"Normalizando {len(files)} archivos a vista canónica...")
    
    for f in files:
        stem = Path(f).stem
        img_path = None
        for ext in ['.jpg', '.png']:
            if (CROP_DIR / f"{stem}{ext}").exists():
                img_path = CROP_DIR / f"{stem}{ext}"
                break
        
        if not img_path: continue
        
        # Determinar si es Izq o Der
        is_left = "Izq" in stem
        
        # Nombre de salida
        if is_left:
            new_name = stem.replace("Izq", "Der")
            # Verificar colisión
            counter = 1
            final_name = new_name
            while (CANONICAL_DIR / f"{final_name}.jpg").exists():
                final_name = f"{new_name}_v{counter}"
                counter += 1
        else:
            final_name = stem
            
        # PROCESAR
        img = cv2.imread(str(img_path))
        h, w, _ = img.shape
        
        if is_left:
            # Voltear
            img = cv2.flip(img, 1)
            # Leer y ajustar etiquetas
            with open(CROP_DIR / f, 'r') as file:
                line = file.readline().split()
            cls, box, kpts = line[0], line[1:5], line[5:]
            
            new_kpts = [0.0] * 24
            for i in range(8):
                x = float(kpts[i*3])
                y = float(kpts[i*3+1])
                v = int(kpts[i*3+2])
                
                # Reflejar y mapear indice
                new_x = 1.0 - x
                new_idx = FLIP_MAP[i]
                new_kpts[new_idx*3] = new_x
                new_kpts[new_idx*3+1] = y
                new_kpts[new_idx*3+2] = v
                
            with open(CANONICAL_DIR / f"{final_name}.txt", 'w') as file:
                file.write(f"{cls} {' '.join(box)} " + " ".join(map(str, new_kpts)) + "\n")
        else:
            # Copiar directo
            shutil.copy(CROP_DIR / f, CANONICAL_DIR / f"{final_name}.txt")
            
        cv2.imwrite(str(CANONICAL_DIR / f"{final_name}.jpg"), img)
        
    print(f"Dataset canonicalizado en {CANONICAL_DIR}")

if __name__ == "__main__":
    canonicalize()
