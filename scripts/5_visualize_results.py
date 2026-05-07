import os
import cv2
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configuración
MODEL_PATH = "D:/Proyectos/TeleRx/training_runs/telerx_yolov8n/weights/best.pt"
RAW_DATA_DIR = Path("D:/Proyectos/TeleRx") / "data"
AUTO_LBL_DIR = Path("D:/Proyectos/TeleRx") / "auto_tagged_labels"
VIS_OUTPUT_DIR = Path("D:/Proyectos/TeleRx") / "visualizations"

# Colores para las clases (BGR)
COLORS = [
    (255, 0, 0),   # Cadera Der - Azul
    (0, 255, 0),   # Cadera IZq - Verde
    (0, 0, 255),   # Rodilla Der - Rojo
    (255, 255, 0), # Rodilla Izq - Cian
    (255, 0, 255), # Tobillo DEr - Magenta
    (0, 255, 255)  # Tobillo Izq - Amarillo
]
CLASS_NAMES = ['Cadera Der', 'Cadera IZq', 'Rodilla Der', 'Rodilla Izq', 'Tobillo DEr', 'Tobillo Izq']

def draw_yolo_labels_on_image(img, lbl_path, meta_path):
    """Dibuja las etiquetas directamente sobre un objeto de imagen OpenCV."""
    h, w, _ = img.shape
    base_size = max(h, w)
    thickness = max(2, int(base_size / 500))
    font_scale = base_size / 1100.0
    text_thickness = max(1, int(thickness / 1.2))

    metadata = {}
    if meta_path.exists():
        with open(meta_path, 'r') as m:
            for line in m:
                parts = line.split()
                if len(parts) >= 3:
                    cls = int(parts[0]); conf = float(parts[1]); origin = parts[2]
                    metadata[cls] = (conf, origin)

    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.split()))
                cls = int(parts[0]); x, y, nw, nh = parts[1:]
                conf_val, origin = metadata.get(cls, (0, "D"))
                conf_str = f"{int(conf_val*100)}%" if origin == "D" else "SIMETRIA"
                x1 = int((x - nw/2) * w); y1 = int((y - nh/2) * h)
                x2 = int((x + nw/2) * w); y2 = int((y + nh/2) * h)
                color = COLORS[cls] if cls < len(COLORS) else (255, 255, 255)
                label = f"{CLASS_NAMES[cls]} {conf_str}"
                brightness = 0.114*color[0] + 0.587*color[1] + 0.299*color[2]
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
                padding = int(10 * font_scale)
                cv2.rectangle(img, (x1, y1 - th - padding*2), (x1 + tw + padding, y1), color, -1)
                cv2.putText(img, label, (x1 + padding//2, y1 - padding), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness)

def draw_yolo_labels(img_path, lbl_path, meta_path, output_path):
    img = cv2.imread(str(img_path))
    if img is None: return
    draw_yolo_labels_on_image(img, lbl_path, meta_path)
    cv2.imwrite(str(output_path), img)
def create_mosaic(num_tiles=25):
    """Crea una imagen de mosaico 5x5 con resultados anotados."""
    txt_files = [f for f in os.listdir(AUTO_LBL_DIR) if f.endswith('.txt')]
    if len(txt_files) < num_tiles:
        print(f"No hay suficientes imágenes para un mosaico de {num_tiles}. Usando todas las disponibles.")
        num_tiles = len(txt_files)
        # Ajustamos a la raíz cuadrada más cercana para que sea cuadrado o rectangular
        grid_size = int(np.sqrt(num_tiles))
        num_tiles = grid_size * grid_size
    else:
        grid_size = 5

    selected_txts = random.sample(txt_files, num_tiles)
    tiles = []
    tile_size = 600 # Un poco más pequeño por celda para que el archivo final no sea gigante

    print(f"Generando mosaico de {grid_size}x{grid_size} ({num_tiles} imágenes)...")

    for lbl_name in selected_txts:
        base_name = Path(lbl_name).stem
        meta_file = AUTO_LBL_DIR / f"{base_name}.meta"
        img_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            if os.path.exists(RAW_DATA_DIR / f"{base_name}{ext}"):
                img_file = f"{base_name}{ext}"
                break

        if img_file:
            img = cv2.imread(str(RAW_DATA_DIR / img_file))
            if img is not None:
                img_ann = img.copy()
                draw_yolo_labels_on_image(img_ann, AUTO_LBL_DIR / lbl_name, meta_file)
                img_res = cv2.resize(img_ann, (tile_size, tile_size))
                cv2.rectangle(img_res, (0,0), (tile_size, tile_size), (255,255,255), 2)
                tiles.append(img_res)

    if len(tiles) == num_tiles:
        rows = []
        for i in range(0, num_tiles, grid_size):
            row = np.hstack(tiles[i:i+grid_size])
            rows.append(row)

        mosaic = np.vstack(rows)
        output_path = VIS_OUTPUT_DIR / f"MOSAICO_resultados_{grid_size}x{grid_size}.jpg"
        cv2.imwrite(str(output_path), mosaic)
        print(f"Mosaico {grid_size}x{grid_size} guardado en: {output_path}")

def visualize_results(num_samples=10):
    os.makedirs(VIS_OUTPUT_DIR, exist_ok=True)
    txt_files = [f for f in os.listdir(AUTO_LBL_DIR) if f.endswith('.txt')]
    if not txt_files:
        print("No hay etiquetas en auto_tagged_labels/ para visualizar.")
        return
    good_cases = []; bad_cases = []
    for f in txt_files:
        with open(AUTO_LBL_DIR / f, 'r') as file:
            count = len(file.readlines())
            if count == 6: good_cases.append(f)
            else: bad_cases.append(f)
    print(f"Encontrados: {len(good_cases)} casos completos y {len(bad_cases)} casos incompletos.")
    samples_good = random.sample(good_cases, min(len(good_cases), num_samples // 2))
    samples_bad = random.sample(bad_cases, min(len(bad_cases), num_samples // 2))
    to_visualize = [("BUENO", f) for f in samples_good] + [("REVISAR", f) for f in samples_bad]
    print(f"Generando {len(to_visualize)} visualizaciones individuales...")
    for status, lbl_name in to_visualize:
        base_name = Path(lbl_name).stem
        meta_file = AUTO_LBL_DIR / f"{base_name}.meta"
        img_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            if os.path.exists(RAW_DATA_DIR / f"{base_name}{ext}"):
                img_file = f"{base_name}{ext}"; break
        if img_file:
            draw_yolo_labels(RAW_DATA_DIR / img_file, AUTO_LBL_DIR / lbl_name, meta_file, VIS_OUTPUT_DIR / f"{status}_{img_file}")

if __name__ == "__main__":
    visualize_results()
    create_mosaic()
