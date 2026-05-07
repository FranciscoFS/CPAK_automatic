import os
import cv2
import csv
from pathlib import Path
from ultralytics import YOLO

# Configuración
MODEL_PATH = "D:/Proyectos/TeleRx/training_runs/telerx_yolov8n/weights/best.pt"
VAL_IMG_DIR = Path("D:/Proyectos/TeleRx/dataset/val/images")
VAL_LBL_DIR = Path("D:/Proyectos/TeleRx/dataset/val/labels")
CSV_OUTPUT = Path("D:/Proyectos/TeleRx/benchmark_results.csv")
CLASS_NAMES = ['Cadera Der', 'Cadera IZq', 'Rodilla Der', 'Rodilla Izq', 'Tobillo DEr', 'Tobillo Izq']

def calculate_iou(boxA, boxB):
    # box = [x_center, y_center, w, h] (normalizado)
    # Convertir a [x1, y1, x2, y2]
    def to_coords(b):
        x, y, w, h = b
        return [x - w/2, y - h/2, x + w/2, y + h/2]
    
    a = to_coords(boxA)
    b = to_coords(boxB)

    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (a[2] - a[0]) * (a[3] - a[1])
    boxBArea = (b[2] - b[0]) * (b[3] - b[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def benchmark():
    if not os.path.exists(MODEL_PATH):
        print("Error: Modelo no encontrado.")
        return

    model = YOLO(MODEL_PATH)
    images = [f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results_data = []
    class_ious = {i: [] for i in range(len(CLASS_NAMES))}

    print(f"Iniciando Benchmark en {len(images)} imágenes de validación...")

    for img_name in images:
        base_name = Path(img_name).stem
        img_path = VAL_IMG_DIR / img_name
        lbl_path = VAL_LBL_DIR / f"{base_name}.txt"
        
        if not lbl_path.exists(): continue
        
        # 1. Leer etiqueta manual
        manual_boxes = {}
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = list(map(float, line.split()))
                if len(parts) >= 5:
                    manual_boxes[int(parts[0])] = parts[1:5]

        # 2. Predicción del modelo
        pred_results = model.predict(img_path, conf=0.20, verbose=False)
        auto_boxes = {}
        for box in pred_results[0].boxes:
            cls = int(box.cls[0])
            coords = box.xywhn[0].tolist()
            # Quedarse con la mejor si hay varias (aunque el modelo debería dar solo una por clase)
            if cls not in auto_boxes or box.conf[0] > auto_boxes[cls][0]:
                auto_boxes[cls] = (box.conf[0], coords)

        # 3. Comparar
        for cls_id in range(len(CLASS_NAMES)):
            if cls_id in manual_boxes and cls_id in auto_boxes:
                iou = calculate_iou(manual_boxes[cls_id], auto_boxes[cls_id][1])
                class_ious[cls_id].append(iou)
                results_data.append({
                    'img_name': img_name,
                    'class': CLASS_NAMES[cls_id],
                    'iou': round(iou, 4),
                    'conf': round(float(auto_boxes[cls_id][0]), 4)
                })

    # Guardar CSV
    with open(CSV_OUTPUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['img_name', 'class', 'iou', 'conf'])
        writer.writeheader()
        writer.writerows(results_data)

    # Resumen por consola
    print("\n--- Resultados del Benchmark (IoU Promedio) ---")
    for cls_id, ious in class_ious.items():
        avg_iou = sum(ious) / len(ious) if ious else 0
        print(f"  - {CLASS_NAMES[cls_id]}: {avg_iou:.4f} (Basado en {len(ious)} muestras)")

    print(f"\nResultados detallados guardados en: {CSV_OUTPUT}")

if __name__ == "__main__":
    benchmark()
