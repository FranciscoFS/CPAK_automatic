"""
Genera visualizaciones para el abstract del manuscrito CPAK.

Para UNA imagen de ejemplo:
  1. Imagen completa con Bounding Boxes de detección
  2. Cada BB recortado (Cadera, Rodilla, Tobillo - ambos lados)
  3. Cada BB con sus keypoints de pose superpuestos

Salida: visualizations_abstract/

Uso:
    conda run -n physis_seg python scripts/37_abstract_visuals.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
DET_MODEL_PATH = BASE_DIR / "training_runs" / "telerx_yolo26s_768_b12-4" / "weights" / "best.pt"
POSE_MODEL_PATH = BASE_DIR / "training_runs" / "telerx_pose_v2_ft" / "weights" / "best.pt"
OUTPUT_DIR = BASE_DIR / "visualizations_abstract"

# Elegir una imagen de ejemplo representativa
EXAMPLE_IMAGE = BASE_DIR / "Test_Data" / "RX_5558994-1.jpeg"

CLASS_INFO = {
    0: {"joint": "Cadera", "side": "Der"},
    1: {"joint": "Cadera", "side": "Izq"},
    2: {"joint": "Rodilla", "side": "Der"},
    3: {"joint": "Rodilla", "side": "Izq"},
    4: {"joint": "Tobillo", "side": "Der"},
    5: {"joint": "Tobillo", "side": "Izq"},
}

POSE_VISIBLE_INDICES = {
    "Cadera": [0],
    "Rodilla": [1, 3, 4, 5, 6, 7],
    "Tobillo": [2],
}

POINT_LABELS = {
    0: "Cabeza Femoral",
    1: "Centro Rodilla",
    2: "Centro Tobillo",
    3: "Condilo Medial",
    4: "Condilo Lateral",
    5: "Plateau Medial",
    6: "Plateau Lateral",
    7: "Notch Femoral",
}

SIDE_COLORS = {
    "Der": (60, 200, 60),    # Verde
    "Izq": (0, 165, 255),    # Naranja
}

POINT_COLORS = {
    0: (255, 0, 255),    # Magenta - Cabeza Femoral
    1: (0, 255, 255),    # Cyan - Centro Rodilla
    2: (0, 255, 0),      # Verde - Centro Tobillo
    3: (255, 100, 0),    # Azul - Condilo Medial
    4: (0, 100, 255),    # Naranja - Condilo Lateral
    5: (200, 200, 0),    # Turquesa - Plateau Medial
    6: (100, 0, 200),    # Violeta - Plateau Lateral
    7: (0, 200, 100),    # Verde azulado - Notch Femoral
}


def pick_best_detections(result):
    best = {}
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        if cls_id not in best or conf > best[cls_id]["confidence"]:
            best[cls_id] = {
                "class_id": cls_id,
                "confidence": conf,
                "xyxy": xyxy,
                "joint": CLASS_INFO.get(cls_id, {}).get("joint", f"Clase_{cls_id}"),
                "side": CLASS_INFO.get(cls_id, {}).get("side", "NA"),
            }
    return best


def padded_box(xyxy, image_shape, padding=0.15):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = xyxy
    bw = x2 - x1
    bh = y2 - y1
    pad_w = bw * padding
    pad_h = bh * padding
    nx1 = max(0, int(round(x1 - pad_w)))
    ny1 = max(0, int(round(y1 - pad_h)))
    nx2 = min(width, int(round(x2 + pad_w)))
    ny2 = min(height, int(round(y2 + pad_h)))
    return nx1, ny1, nx2, ny2


def infer_crop_keypoints(pose_model, crop, joint_name, pose_conf=0.20):
    result = pose_model.predict(source=crop, conf=pose_conf, verbose=False)[0]
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return {}

    pred_xy = result.keypoints.xy[0].cpu().numpy()
    points = {}
    for idx in POSE_VISIBLE_INDICES[joint_name]:
        px = float(pred_xy[idx][0])
        py = float(pred_xy[idx][1])
        if px > 1 or py > 1:
            points[idx] = (px, py)
    return points


def visualize_full_with_boxes(image, detections, output_path, output_dir):
    """Imagen completa con BB de detección y etiquetas."""
    # Guardar original simple primero
    orig_path = output_dir / "00_original_simple.jpg"
    cv2.imwrite(str(orig_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [Original] {orig_path.name}")

    canvas = image.copy()
    h, w = canvas.shape[:2]
    base = max(600, min(h, w))
    box_thickness = max(6, int(round(base / 180)))  # Más grueso
    font_scale = max(0.6, base / 1800.0)
    text_thickness = max(2, int(round(base / 800)))
    label_pad = 8

    for det in detections.values():
        side = det["side"]
        color = SIDE_COLORS.get(side, (255, 255, 255))
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        joint = det["joint"]
        conf = det["confidence"]

        # BB
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

        # Label arriba del BB
        label = f"{joint} {side}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)[0]
        tag_h = text_size[1] + (label_pad * 2)
        tag_y1 = max(0, y1 - tag_h)
        cv2.rectangle(canvas, (x1, tag_y1), (x1 + text_size[0] + label_pad * 2, y1), color, -1)
        cv2.putText(
            canvas, label,
            (x1 + label_pad, y1 - label_pad),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20), text_thickness, cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [BB Full] {output_path.name}")


def visualize_crops_only(image, detections, output_dir):
    """Cada BB recortado (crop) sin anotaciones."""
    crops_dir = output_dir / "crops_only"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for det in detections.values():
        side = det["side"]
        joint = det["joint"]
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        filename = f"{joint}_{side}.jpg"
        cv2.imwrite(str(crops_dir / filename), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [Crop] {filename}")


def visualize_crops_with_keypoints(image, detections, pose_model, output_dir, padding=0.15, pose_conf=0.20):
    """Cada BB con sus keypoints superpuestos."""
    kp_dir = output_dir / "crops_with_keypoints"
    kp_dir.mkdir(parents=True, exist_ok=True)

    for det in detections.values():
        side = det["side"]
        joint = det["joint"]
        color = SIDE_COLORS.get(side, (255, 255, 255))
        x1, y1, x2, y2 = padded_box(det["xyxy"], image.shape, padding)

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        # Inferir keypoints en el crop
        local_points = infer_crop_keypoints(pose_model, crop, joint, pose_conf)

        # Dibujar keypoints
        canvas = crop.copy()
        h_crop, w_crop = canvas.shape[:2]
        base_crop = max(200, min(h_crop, w_crop))
        # Puntos más pequeños, texto más chico para no saturar
        point_radius = max(4, int(round(base_crop / 60)))
        line_thickness = max(1, int(round(base_crop / 150)))
        font_scale = max(0.35, base_crop / 600.0)
        text_thickness = max(1, int(round(base_crop / 300)))

        # Dibujar conexiones anatómicas si es Rodilla
        if joint == "Rodilla":
            # Línea de cóndilos (3-4)
            if 3 in local_points and 4 in local_points:
                pt1 = (int(round(local_points[3][0])), int(round(local_points[3][1])))
                pt2 = (int(round(local_points[4][0])), int(round(local_points[4][1])))
                cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)
            # Línea de platillos (5-6)
            if 5 in local_points and 6 in local_points:
                pt1 = (int(round(local_points[5][0])), int(round(local_points[5][1])))
                pt2 = (int(round(local_points[6][0])), int(round(local_points[6][1])))
                cv2.line(canvas, pt1, pt2, color, line_thickness, cv2.LINE_AA)

        for idx, (px, py) in local_points.items():
            center = (int(round(px)), int(round(py)))
            pt_color = POINT_COLORS.get(idx, color)
            # Círculo relleno
            cv2.circle(canvas, center, point_radius, pt_color, -1, cv2.LINE_AA)
            # Borde blanco
            cv2.circle(canvas, center, point_radius + 2, (255, 255, 255), max(1, line_thickness // 2), cv2.LINE_AA)
            # Etiqueta solo si NO es Rodilla (para no saturar)
            if joint != "Rodilla":
                label = POINT_LABELS.get(idx, f"P{idx}")
                cv2.putText(
                    canvas, label,
                    (center[0] + point_radius + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, pt_color, text_thickness, cv2.LINE_AA,
                )

        # Recuadro del crop (más fino)
        cv2.rectangle(canvas, (0, 0), (w_crop - 1, h_crop - 1), color, max(1, line_thickness))

        # Título superior más compacto
        title = f"{joint} {side}"
        title_font = max(0.4, base_crop / 500.0)
        title_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_font, text_thickness)[0]
        cv2.rectangle(canvas, (0, 0), (title_size[0] + 10, title_size[1] + 10), color, -1)
        cv2.putText(
            canvas, title,
            (5, title_size[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, title_font, (20, 20, 20), text_thickness, cv2.LINE_AA,
        )

        filename = f"{joint}_{side}_keypoints.jpg"
        cv2.imwrite(str(kp_dir / filename), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [Crop+KP] {filename} ({len(local_points)} keypoints)")


def visualize_tiled_summary(detections, output_dir, tile_width=800):
    """Crea un mosaico con todos los crops + keypoints lado a lado para el abstract."""
    kp_dir = output_dir / "crops_with_keypoints"
    if not kp_dir.exists():
        return

    sides_order = ["Der", "Izq"]
    joints_order = ["Cadera", "Rodilla", "Tobillo"]

    # Juntar imágenes disponibles en un grid 3x2 (joints x sides)
    rows = []
    for joint in joints_order:
        row_images = []
        for side in sides_order:
            path = kp_dir / f"{joint}_{side}_keypoints.jpg"
            if path.exists():
                img = cv2.imread(str(path))
                # Redimensionar manteniendo aspect ratio
                h, w = img.shape[:2]
                scale = tile_width / w
                new_w = tile_width
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:
                img = np.ones((tile_width // 2, tile_width, 3), dtype=np.uint8) * 30
                cv2.putText(img, f"N/A", (tile_width // 3, tile_width // 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2, cv2.LINE_AA)

            # Separador vertical
            if side == "Der":
                cv2.line(img, (img.shape[1] - 1, 0), (img.shape[1] - 1, img.shape[0]), (50, 50, 50), 2)

            row_images.append(img)

        # Altura máxima de la fila
        max_h = max(im.shape[0] for im in row_images)
        row_padded = []
        for im in row_images:
            if im.shape[0] < max_h:
                pad = np.zeros((max_h - im.shape[0], im.shape[1], 3), dtype=np.uint8)
                im = np.vstack([im, pad])
            row_padded.append(im)

        row = np.hstack(row_padded)
        # Separador horizontal entre filas
        if joint != joints_order[-1]:
            sep = np.ones((4, row.shape[1], 3), dtype=np.uint8) * 50
            row = np.vstack([row, sep])
        rows.append(row)

    tiled = np.vstack(rows)
    out_path = output_dir / "tiled_crops_keypoints.jpg"
    cv2.imwrite(str(out_path), tiled, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [Tiled] {out_path.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EXAMPLE_IMAGE.exists():
        raise FileNotFoundError(f"Imagen de ejemplo no encontrada: {EXAMPLE_IMAGE}")
    if not DET_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo detección no encontrado: {DET_MODEL_PATH}")
    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo pose no encontrado: {POSE_MODEL_PATH}")

    print(f"Cargando modelo detección: {DET_MODEL_PATH.name}")
    det_model = YOLO(str(DET_MODEL_PATH))
    print(f"Cargando modelo pose: {POSE_MODEL_PATH.name}")
    pose_model = YOLO(str(POSE_MODEL_PATH))

    print(f"\nProcesando: {EXAMPLE_IMAGE.name}")
    image = cv2.imread(str(EXAMPLE_IMAGE))
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen: {EXAMPLE_IMAGE}")

    # ── 1. Detección ──────────────────────────────────────────────────
    det_result = det_model.predict(source=str(EXAMPLE_IMAGE), conf=0.20, verbose=False)[0]
    detections = pick_best_detections(det_result)

    # Organizar por articulación para detectar Der/Izq correctamente
    by_joint = {}
    for cls_id, det in detections.items():
        by_joint.setdefault(det["joint"], []).append((cls_id, det))

    # Asignar lado por posición X (el de menor X es Der en convención radiológica)
    for joint, items in by_joint.items():
        if len(items) == 2:
            sorted_items = sorted(items, key=lambda x: (x[1]["xyxy"][0] + x[1]["xyxy"][2]) / 2)
            sorted_items[0][1]["side"] = "Der"
            sorted_items[1][1]["side"] = "Izq"

    print(f"  Detecciones: {len(detections)}")
    for det in detections.values():
        print(f"    {det['joint']} {det['side']} conf={det['confidence']:.3f}")

    # ── 2. Visualizaciones ────────────────────────────────────────────
    print("\nGenerando visualizaciones...")

    # 2a. Imagen completa con BB (y original)
    visualize_full_with_boxes(image, detections, OUTPUT_DIR / "01_full_bboxes.jpg", OUTPUT_DIR)

    # 2b. Crops sin anotaciones
    visualize_crops_only(image, detections, OUTPUT_DIR)

    # 2c. Crops con keypoints
    visualize_crops_with_keypoints(image, detections, pose_model, OUTPUT_DIR,
                                   padding=0.15, pose_conf=0.20)

    # 2d. Mosaico resumen
    visualize_tiled_summary(detections, OUTPUT_DIR)

    print(f"\nTodo listo. Salida: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
