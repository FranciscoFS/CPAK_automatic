import os
import csv
from pathlib import Path

# Configuración
MANUAL_ROOT = Path("D:/Proyectos/TeleRx/dataset")
AUTO_DIR = Path("D:/Proyectos/TeleRx/auto_tagged_labels")
CSV_OUTPUT = Path("D:/Proyectos/TeleRx/comparison_manual_vs_auto.csv")
CLASS_NAMES = ['Cadera Der', 'Cadera IZq', 'Rodilla Der', 'Rodilla Izq', 'Tobillo DEr', 'Tobillo Izq']

def calculate_iou(boxA, boxB):
    # box = [x, y, w, h] normalizados
    def to_coords(b):
        x, y, w, h = b
        return [x - w/2, y - h/2, x + w/2, y + h/2]
    a = to_coords(boxA); b = to_coords(boxB)
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (a[2] - a[0]) * (a[3] - a[1])
    boxBArea = (b[2] - b[0]) * (b[3] - b[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def load_yolo_labels(path):
    labels = {} # {cls_id: [x, y, w, h]}
    if not path.exists(): return labels
    with open(path, 'r') as f:
        for line in f:
            parts = list(map(float, line.split()))
            if len(parts) >= 5:
                labels[int(parts[0])] = parts[1:5]
    return labels

def load_meta_info(meta_path):
    meta = {} # {cls_id: (conf, origin)}
    if not meta_path.exists(): return meta
    with open(meta_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                meta[int(parts[0])] = (float(parts[1]), parts[2])
    return meta

def compare_labels():
    # 1. Recolectar todos los archivos manuales disponibles
    manual_files = {} # {stem: path}
    for split in ['train', 'val', 'test']:
        lbl_dir = MANUAL_ROOT / split / "labels"
        if lbl_dir.exists():
            for f in os.listdir(lbl_dir):
                if f.endswith('.txt'):
                    manual_files[Path(f).stem] = lbl_dir / f

    print(f"Encontradas {len(manual_files)} etiquetas manuales para comparar.")

    results_data = []
    matches_found = 0

    for stem, manual_path in manual_files.items():
        auto_path = AUTO_DIR / f"{stem}.txt"
        meta_path = AUTO_DIR / f"{stem}.meta"
        
        if auto_path.exists():
            matches_found += 1
            manual_labels = load_yolo_labels(manual_path)
            auto_labels = load_yolo_labels(auto_path)
            meta_info = load_meta_info(meta_path)
            
            for cls_id in range(len(CLASS_NAMES)):
                if cls_id in manual_labels and cls_id in auto_labels:
                    iou = calculate_iou(manual_labels[cls_id], auto_labels[cls_id])
                    conf, origin = meta_info.get(cls_id, (0.0, "D"))
                    
                    results_data.append({
                        'file': stem,
                        'class': CLASS_NAMES[cls_id],
                        'iou': round(iou, 4),
                        'conf_saved': round(conf, 4),
                        'origin': 'Detection' if origin == 'D' else 'Symmetry'
                    })

    # Guardar CSV
    with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'class', 'iou', 'conf_saved', 'origin'])
        writer.writeheader()
        writer.writerows(results_data)

    print(f"\nBenchmark Finalizado:")
    print(f"  - Imágenes comparadas: {matches_found}")
    print(f"  - Total de articulaciones analizadas: {len(results_data)}")
    print(f"  - Reporte guardado en: {CSV_OUTPUT}")

if __name__ == "__main__":
    compare_labels()
