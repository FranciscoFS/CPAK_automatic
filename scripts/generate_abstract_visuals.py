"""
Genera imágenes para el abstract del artículo CPAK:

1. Radiografía completa con Bounding Boxes de detección
2. Cada crop (Cadera, Rodilla, Tobillo) por separado
3. Cada crop con sus Keypoints de pose superpuestos

Salida: visualizations_abstract/

Uso:
    conda run -n physis_seg python scripts/generate_abstract_visuals.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# ── Importar pipeline de inferencia (nombre numérico) ────────────────────
import importlib.util
spec = importlib.util.spec_from_file_location(
    "final_inference",
    str(BASE_DIR / "scripts" / "13_final_inference.py"),
)
fi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fi)

DEFAULT_DET_MODEL   = fi.DEFAULT_DET_MODEL
DEFAULT_POSE_MODEL  = fi.DEFAULT_POSE_MODEL
CLASS_INFO          = fi.CLASS_INFO
POSE_VISIBLE_INDICES = fi.POSE_VISIBLE_INDICES
POINT_LABELS        = fi.POINT_LABELS
SIDE_COLORS         = fi.SIDE_COLORS
pick_best_detections     = fi.pick_best_detections
validate_side_coherence  = fi.validate_side_coherence
reflect_missing_detections = fi.reflect_missing_detections
padded_box           = fi.padded_box
infer_crop_keypoints = fi.infer_crop_keypoints
get_overlay_style    = fi.get_overlay_style
draw_detection_boxes = fi.draw_detection_boxes

from ultralytics import YOLO

OUTPUT_DIR = BASE_DIR / "visualizations_abstract"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Elegir algunas imágenes representativas de Test_Data ─────────────────
# Seleccionamos 2-3 imágenes variadas
EXAMPLE_IMAGES = [
    BASE_DIR / "Test_Data" / "RX_5908517-4.jpeg",
    BASE_DIR / "Test_Data" / "RX_11998331-2.jpeg",
]

DET_CONF = 0.20
POSE_CONF = 0.20
PADDING = 0.15


def save_crop_with_keypoints(crop_img, keypoints_local, joint_name, side, output_path):
    """Dibuja keypoints sobre un crop y lo guarda."""
    canvas = crop_img.copy()
    h, w = canvas.shape[:2]
    style = get_overlay_style(canvas)

    color = SIDE_COLORS.get(side, (0, 255, 0))
    radius = style["point_radius"]
    thickness = style["line_thickness"]

    for kidx, (kx, ky) in keypoints_local.items():
        center = (int(round(kx)), int(round(ky)))
        # Círculo relleno del color del lado
        cv2.circle(canvas, center, radius, color, -1, cv2.LINE_AA)
        # Borde blanco
        cv2.circle(canvas, center, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)
        # Etiqueta del punto
        label = POINT_LABELS.get(kidx, f"P{kidx}")
        cv2.putText(
            canvas,
            label,
            (center[0] + style["label_pad"], center[1] - style["label_pad"]),
            cv2.FONT_HERSHEY_SIMPLEX,
            style["font_small"],
            color,
            style["text_thickness"],
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  Guardado: {output_path.name}")


def process_for_abstract(image_path: Path, det_model: YOLO, pose_model: YOLO):
    stem = image_path.stem
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ERROR: no se pudo leer {image_path}")
        return
    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"Procesando: {image_path.name}  ({w}x{h})")
    print(f"{'='*60}")

    # ── 1. Detección ────────────────────────────────────────────────────
    det_result = det_model.predict(source=str(image_path), conf=DET_CONF, verbose=False)[0]
    detections = pick_best_detections(det_result)
    detections = validate_side_coherence(detections)
    detections = reflect_missing_detections(detections, w)

    if not detections:
        print("  No se detectaron articulaciones.")
        return

    # ── 2. Imagen completa con BB ───────────────────────────────────────
    canvas_full = img.copy()
    style = get_overlay_style(canvas_full)
    draw_detection_boxes(canvas_full, detections, style)
    full_bb_path = OUTPUT_DIR / f"{stem}_full_bb.jpg"
    cv2.imwrite(str(full_bb_path), canvas_full, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [Completa+BB] {full_bb_path.name}")

    # ── 3. Por cada (joint, side) ──────────────────────────────────────
    for det in detections.values():
        joint = det["joint"]
        side = det["side"]
        color = SIDE_COLORS.get(side, (0, 255, 0))

        # Extraer crop con padding
        x1, y1, x2, y2 = padded_box(det["xyxy"], img.shape, PADDING)
        crop_img = img[y1:y2, x1:x2].copy()
        if crop_img.size == 0:
            continue

        # ── 3a. Crop solo (sin keypoints) ──────────────────────────────
        crop_only_path = OUTPUT_DIR / f"{stem}_{joint}_{side}_crop.jpg"
        cv2.imwrite(str(crop_only_path), crop_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  [Crop {joint}_{side}] {crop_only_path.name}")

        # ── 3b. Crop con keypoints ─────────────────────────────────────
        local_points = infer_crop_keypoints(pose_model, crop_img, joint, POSE_CONF)
        if local_points:
            pose_path = OUTPUT_DIR / f"{stem}_{joint}_{side}_pose.jpg"
            save_crop_with_keypoints(crop_img, local_points, joint, side, pose_path)
        else:
            print(f"  [Crop {joint}_{side}] Sin keypoints detectados")

        # ── 3c. Crop con BB original dibujado ──────────────────────────
        # Solo para referencia, dibujamos el contorno del crop en la imagen completa
        cv2.rectangle(canvas_full, (x1, y1), (x2, y2), color, style["box_thickness"])

    print(f"  Hecho.")


def main():
    print(f"Directorio de salida: {OUTPUT_DIR}")
    print(f"Modelo detección: {DEFAULT_DET_MODEL}")
    print(f"Modelo pose: {DEFAULT_POSE_MODEL}")

    # Validar modelos
    if not DEFAULT_DET_MODEL.exists():
        raise FileNotFoundError(f"No existe modelo detección: {DEFAULT_DET_MODEL}")
    if not DEFAULT_POSE_MODEL.exists():
        raise FileNotFoundError(f"No existe modelo pose: {DEFAULT_POSE_MODEL}")

    # Cargar modelos
    print("Cargando modelo de detección...")
    det_model = YOLO(str(DEFAULT_DET_MODEL))
    print("Cargando modelo de pose...")
    pose_model = YOLO(str(DEFAULT_POSE_MODEL))

    for img_path in EXAMPLE_IMAGES:
        if not img_path.exists():
            print(f"  AVISO: no existe {img_path}, se omite.")
            continue
        process_for_abstract(img_path, det_model, pose_model)

    print(f"\n✅ Todas las imágenes generadas en: {OUTPUT_DIR}")
    print("   - *_full_bb.jpg     → radiografía completa con BB")
    print("   - *_{Joint}_{Side}_crop.jpg  → crop de la articulación")
    print("   - *_{Joint}_{Side}_pose.jpg  → crop con keypoints")


if __name__ == "__main__":
    main()
