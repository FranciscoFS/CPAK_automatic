import os
from pathlib import Path
from ultralytics import YOLO

# Configuración
MODEL_PATH = "D:/Proyectos/TeleRx/training_runs/telerx_yolov8n/weights/best.pt"
RAW_DATA_DIR = Path("D:/Proyectos/TeleRx/data")
DATASET_DIR = Path("D:/Proyectos/TeleRx/dataset")
OUTPUT_LBL_DIR = Path("D:/Proyectos/TeleRx/auto_tagged_labels")

# Umbrales
CONF_THRESHOLD = 0.20  # Muy sensible para capturar todo
SYMMETRY_MIN_CONF = 0.20  # Reflejar incluso con baja confianza

# Mapeo de simetría (Basado en tu data.yaml)
# 0: Cadera Der <-> 1: Cadera IZq
# 2: Rodilla Der <-> 3: Rodilla Izq
# 4: Tobillo DEr <-> 5: Tobillo Izq
MIRROR_MAP = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}

def get_already_tagged():
    tagged_names = set()
    for split in ['train', 'val', 'test']:
        img_path = DATASET_DIR / split / "images"
        if img_path.exists():
            for f in os.listdir(img_path):
                tagged_names.add(Path(f).stem)
    return tagged_names

def reflect_box(coords):
    # coords = [x_center, y_center, width, height]
    # Reflejar x_center respecto al centro (0.5)
    x_center, y_center, w, h = coords
    new_x = 1.0 - x_center
    return [new_x, y_center, w, h]

def auto_tag():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: No se encontró el modelo en {MODEL_PATH}")
        return

    print("Cargando modelo y aplicando lógica de simetría...")
    model = YOLO(MODEL_PATH)
    already_tagged = get_already_tagged()
    
    os.makedirs(OUTPUT_LBL_DIR, exist_ok=True)
    
    image_extensions = ('.png', '.jpg', '.jpeg')
    all_raw_images = [f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(image_extensions)]
    to_process = [img for img in all_raw_images if Path(img).stem not in already_tagged]
    
    print(f"Procesando {len(to_process)} imágenes nuevas...")
    
    for img_name in to_process:
        img_path = RAW_DATA_DIR / img_name
        results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
        
        # 1. Quedarse solo con la mejor detección por clase
        best_detections = {} # {class_id: [confidence, coords]}
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            coords = box.xywhn[0].tolist()
            
            if cls not in best_detections or conf > best_detections[cls][0]:
                best_detections[cls] = [conf, coords]
        
        # 2. Lógica de Simetría (Espejo)
        for original_cls, mirror_cls in MIRROR_MAP.items():
            # Si tenemos la original con alta confianza pero NO tenemos la espejo
            if original_cls in best_detections and mirror_cls not in best_detections:
                conf, coords = best_detections[original_cls]
                if conf >= SYMMETRY_MIN_CONF:
                    mirrored_coords = reflect_box(coords)
                    # Añadir la detección espejada (usamos la misma confianza para marcarla)
                    best_detections[mirror_cls] = [conf, mirrored_coords]
                    # print(f"  - [{img_name}] Generada clase {mirror_cls} por simetría desde {original_cls}")

        # 3. Guardar resultados y metadatos
        if best_detections:
            lbl_name = f"{img_path.stem}.txt"
            lbl_path = OUTPUT_LBL_DIR / lbl_name
            meta_path = OUTPUT_LBL_DIR / f"{img_path.stem}.meta"
            
            with open(lbl_path, 'w') as f, open(meta_path, 'w') as m:
                for cls, (conf, coords) in best_detections.items():
                    # El archivo .txt sigue siendo formato YOLO puro
                    f.write(f"{cls} {' '.join(map(str, coords))}\n")
                    
                    # El archivo .meta guarda el origen (D=Detection, S=Symmetry)
                    origin = "S" if (cls in best_detections and any(best_detections[original][1] == reflect_box(coords) for original in MIRROR_MAP if original in best_detections and MIRROR_MAP[original] == cls)) else "D"
                    # Una forma más simple de marcar el origen:
                    # Guardamos: clase, confianza, origen
                    # Pero para no complicar la lógica de arriba, simplemente marcamos si fue creada por el bucle de simetría
                    # Vamos a refinar la lógica de guardado de origen
                
                # Refinamos la lógica de guardado de origen para ser exactos
                origins = {}
                # Primero marcar todas como D
                for cls in best_detections: origins[cls] = "D"
                # Luego re-ejecutar la lógica de espejo para marcar como S
                for original_cls, mirror_cls in MIRROR_MAP.items():
                    if original_cls in best_detections and mirror_cls in best_detections:
                        # Si la espejo es el reflejo exacto de la original, es S
                        if best_detections[mirror_cls][1] == reflect_box(best_detections[original_cls][1]):
                            origins[mirror_cls] = "S"
                
                for cls, (conf, coords) in best_detections.items():
                    m.write(f"{cls} {conf:.4f} {origins[cls]}\n")
                
    print(f"\n¡Listo! Etiquetas con lógica de simetría en: {OUTPUT_LBL_DIR}")

if __name__ == "__main__":
    auto_tag()
