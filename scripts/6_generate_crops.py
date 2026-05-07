import os
import cv2
from pathlib import Path

# Configuración de Rutas
RAW_DATA_DIR = Path("D:/Proyectos/TeleRx/data")
AUTO_LBL_DIR = Path("D:/Proyectos/TeleRx/auto_tagged_labels") 
DATASET_ROOT = Path("D:/Proyectos/TeleRx/dataset")
SUBDIRS = ['train', 'val', 'test']
OUTPUT_DIR = Path("D:/Proyectos/TeleRx/crops_for_keypoints")

CLASS_NAMES = ['Cadera_Der', 'Cadera_Izq', 'Rodilla_Der', 'Rodilla_Izq', 'Tobillo_Der', 'Tobillo_Izq']

# Parámetros de Recorte
PADDING_PERCENT = 0.15  # 15% de margen extra alrededor de la bounding box

def find_label_file(img_stem):
    """Busca el archivo de etiqueta en el dataset manual normalizado o en las automáticas."""
    # 1. Buscar en dataset (prioridad manual/validada)
    for subdir in SUBDIRS:
        lbl_path = DATASET_ROOT / subdir / "labels" / f"{img_stem}.txt"
        if lbl_path.exists():
            return lbl_path
            
    # 2. Buscar en etiquetas automáticas
    auto_lbl_path = AUTO_LBL_DIR / f"{img_stem}.txt"
    if auto_lbl_path.exists():
        return auto_lbl_path
        
    return None

def generate_crops():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    image_extensions = ('.png', '.jpg', '.jpeg')
    all_images = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(image_extensions)]
    
    total_crops = 0
    print(f"Generando recortes para {len(all_images)} imágenes de raw data...")

    for img_name in all_images:
        img_stem = Path(img_name).stem
        lbl_path = find_label_file(img_stem)
        
        if not lbl_path:
            # print(f"Info: No se encontró etiqueta para {img_name}")
            continue
            
        img_path = RAW_DATA_DIR / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error: No se pudo leer la imagen {img_path}")
            continue
            
        h, w, _ = img.shape
        
        with open(lbl_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.split()
                if len(parts) < 5: continue
                
                cls_id = int(parts[0])
                nx, ny, nw, nh = map(float, parts[1:5])
                
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Clase_{cls_id}"
                
                # Coordenadas a píxeles
                cx, cy = nx * w, ny * h
                bw, bh = nw * w, nh * h
                
                # Aplicar Padding
                pad_w = bw * PADDING_PERCENT
                pad_h = bh * PADDING_PERCENT
                
                x1 = int(max(0, cx - (bw / 2) - pad_w))
                y1 = int(max(0, cy - (bh / 2) - pad_h))
                x2 = int(min(w, cx + (bw / 2) + pad_w))
                y2 = int(min(h, cy + (bh / 2) + pad_h))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop_img = img[y1:y2, x1:x2]
                
                # Nombre: Paciente_Clase.jpg
                crop_name = f"{img_stem}_{class_name}.jpg"
                crop_path = OUTPUT_DIR / crop_name
                
                cv2.imwrite(str(crop_path), crop_img)
                total_crops += 1

    print(f"\nFinalizado. Se generaron {total_crops} recortes en '{OUTPUT_DIR}'")

if __name__ == "__main__":
    generate_crops()
