import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DET_MODEL = BASE_DIR / "training_runs" / "telerx_yolov8n" / "weights" / "best.pt"
DEFAULT_POSE_MODEL = BASE_DIR / "training_runs" / "telerx_pose_s_rebuild1_ft_full_b16_flipfix" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "visualizations" / "final_pipeline"

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
    "Der": (60, 200, 60),
    "Izq": (0, 165, 255),
}

REQUIRED_METRIC_POINTS = [0, 1, 2, 3, 4, 5, 6, 7]
AHKA_FORMULA = "MPTA_MINUS_LDFA"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline final CPAK: deteccion, pose, overlay y calculo clinico sobre una TeleRx completa."
    )
    parser.add_argument("--source", required=True, help="Ruta a una imagen o a un directorio con imagenes.")
    parser.add_argument("--det-model", default=str(DEFAULT_DET_MODEL), help="Ruta al modelo de deteccion.")
    parser.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL), help="Ruta al modelo de pose.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directorio para overlays y JSON.")
    parser.add_argument("--det-conf", type=float, default=0.20, help="Confianza minima para deteccion de zonas.")
    parser.add_argument("--pose-conf", type=float, default=0.20, help="Confianza minima para pose sobre cada crop.")
    parser.add_argument("--padding", type=float, default=0.15, help="Padding extra alrededor de cada bounding box.")
    return parser.parse_args()


def collect_images(source: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted([p for p in source.iterdir() if p.is_file() and p.suffix.lower() in exts])
    raise FileNotFoundError(f"No existe la ruta de entrada: {source}")


def validate_model_path(model_path: Path, label: str):
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo de {label}: {model_path}")


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


def padded_box(xyxy, image_shape, padding):
    """
    Expande un bounding box un porcentaje de su tamaño.
    
    El padding se aplica al bounding box de DETECCIÓN (no al modelo de pose).
    Sirve para darle más contexto anatómico al modelo de pose.
    
    Ejemplo: padding=0.15 y BB de 100x100 px → BB expandido a ~130x130 px
    
    Args:
        xyxy: coordenadas originales del BB (x1, y1, x2, y2)
        image_shape: (height, width) de la imagen
        padding: fracción decimal (0.15 = 15%)
    
    Returns:
        Coordenadas expandidas (nx1, ny1, nx2, ny2) limitadas a los bordes de la imagen
    """
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


def infer_crop_keypoints(pose_model: YOLO, crop, joint_name: str, pose_conf: float):
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


def build_side_points(image, side_detections, pose_model: YOLO, pose_conf: float, padding: float):
    """
    Procesa cada zona (Cadera, Rodilla, Tobillo) de un lado:
    
    1. Toma el BB de detección
    2. Lo expande por el porcentaje de padding (más contexto para pose)
    3. Extrae crop expandido
    4. Corre YOLO Pose en el crop expandido
    5. Mapea coordenadas locales del crop → globales de imagen
    
    El padding NO afecta cómo funciona el modelo de pose.
    Solo controla cuánto contexto ve el modelo alrededor de cada zona.
    """
    global_points = {}
    crops = {}
    for joint_name in ["Cadera", "Rodilla", "Tobillo"]:
        detection = side_detections.get(joint_name)
        if detection is None:
            continue

        x1, y1, x2, y2 = padded_box(detection["xyxy"], image.shape, padding)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        local_points = infer_crop_keypoints(pose_model, crop, joint_name, pose_conf)
        mapped_points = {}
        for idx, (px, py) in local_points.items():
            mapped_points[idx] = (x1 + px, y1 + py)
            global_points[idx] = mapped_points[idx]

        crops[joint_name] = {
            "crop_box": [x1, y1, x2, y2],
            "keypoints": {str(idx): [round(pt[0], 2), round(pt[1], 2)] for idx, pt in mapped_points.items()},
        }

    return global_points, crops


def vector(p1, p2):
    return np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)


def line_angle_deg(p1, p2, p3, p4):
    v1 = vector(p1, p2)
    v2 = vector(p3, p4)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return None

    cosine = float(np.dot(v1, v2) / (norm1 * norm2))
    cosine = max(-1.0, min(1.0, cosine))
    angle = math.degrees(math.acos(cosine))
    return min(angle, 180.0 - angle)


def line_intersection(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-8:
        return None

    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / den
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / den
    return (float(px), float(py))


def angle_at_vertex_deg(vertex, point_a, point_b):
    va = vector(vertex, point_a)
    vb = vector(vertex, point_b)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return None

    cosine = float(np.dot(va, vb) / (norm_a * norm_b))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def classify_ahka(value):
    if value < -2.0:
        return "Varo"
    if value <= 2.0:
        return "Neutro"
    return "Valgo"


def classify_jlo(value):
    if value < 177.0:
        return "Apex Distal"
    if value <= 181.0:
        return "Neutro"
    return "Apex Proximal"


# CPAK type I–IX: rows = aHKA (Varo/Neutro/Valgo), cols = JLO (Distal/Neutro/Proximal)
_CPAK_TYPE_GRID = {
    ("Varo",   "Apex Distal"):   "I",
    ("Neutro", "Apex Distal"):   "II",
    ("Valgo",  "Apex Distal"):   "III",
    ("Varo",   "Neutro"):        "IV",
    ("Neutro", "Neutro"):        "V",
    ("Valgo",  "Neutro"):        "VI",
    ("Varo",   "Apex Proximal"): "VII",
    ("Neutro", "Apex Proximal"): "VIII",
    ("Valgo",  "Apex Proximal"): "IX",
}


def classify_cpak_type(ahka_class: str, jlo_class: str) -> str:
    return _CPAK_TYPE_GRID.get((ahka_class, jlo_class), "?")


def calculate_metrics(points):
    missing = [idx for idx in REQUIRED_METRIC_POINTS if idx not in points]
    if missing:
        return {"status": "incomplete", "missing_points": missing}

    fem_intersection = line_intersection(points[0], points[7], points[3], points[4])
    tib_intersection = line_intersection(points[1], points[2], points[5], points[6])
    if fem_intersection is None or tib_intersection is None:
        return {"status": "invalid_geometry"}

    # LDFA ambiguity is common (internal vs external). We report both and select one explicitly.
    ldfa_external = angle_at_vertex_deg(fem_intersection, points[0], points[4])
    # MPTA: angle at tibial/plateau intersection using distal tibial axis ray and medial plateau ray.
    mpta = angle_at_vertex_deg(tib_intersection, points[2], points[5])
    if ldfa_external is None or mpta is None:
        return {"status": "invalid_geometry"}

    ldfa_internal = 180.0 - ldfa_external
    ldfa = ldfa_external

    ahka = mpta - ldfa
    jlo = mpta + ldfa
    return {
        "status": "ok",
        "LDFA": round(ldfa, 3),
        "LDFA_external": round(ldfa_external, 3),
        "LDFA_internal": round(ldfa_internal, 3),
        "LDFA_mode": "EXTERNAL_FIXED",
        "MPTA": round(mpta, 3),
        "aHKA": round(ahka, 3),
        "aHKA_formula": AHKA_FORMULA,
        "JLO": round(jlo, 3),
        "aHKA_class": classify_ahka(ahka),
        "JLO_class": classify_jlo(jlo),
        "CPAK": f"{classify_ahka(ahka)} | {classify_jlo(jlo)}",
        "CPAK_type": classify_cpak_type(classify_ahka(ahka), classify_jlo(jlo)),
    }


def get_overlay_style(canvas):
    h, w = canvas.shape[:2]
    base = max(600, min(h, w))
    return {
        "box_thickness": max(2, int(round(base / 650))),
        "line_thickness": max(1, int(round(base / 900))),
        "point_radius": max(4, int(round(base / 180))),
        "font_main": max(0.45, base / 2200.0),
        "font_small": max(0.4, base / 2600.0),
        "text_thickness": max(1, int(round(base / 1200))),
        "margin": max(10, int(round(base / 120))),
        "label_pad": max(5, int(round(base / 220))),
    }


def clip_text_to_width(text: str, max_width: int, font_scale: float, thickness: int):
    if max_width <= 0:
        return ""
    current = text
    while current:
        text_width = cv2.getTextSize(current, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
        if text_width <= max_width:
            return current
        if len(current) <= 4:
            return "..."
        current = current[:-4] + "..."
    return ""


def project_line_to_image(p1, p2, width: int, height: int):
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1

    candidates = []

    if abs(dx) > 1e-8:
        t_left = (0.0 - x1) / dx
        y_left = y1 + t_left * dy
        if 0.0 <= y_left <= (height - 1):
            candidates.append((0.0, y_left))

        t_right = ((width - 1) - x1) / dx
        y_right = y1 + t_right * dy
        if 0.0 <= y_right <= (height - 1):
            candidates.append((float(width - 1), y_right))

    if abs(dy) > 1e-8:
        t_top = (0.0 - y1) / dy
        x_top = x1 + t_top * dx
        if 0.0 <= x_top <= (width - 1):
            candidates.append((x_top, 0.0))

        t_bottom = ((height - 1) - y1) / dy
        x_bottom = x1 + t_bottom * dx
        if 0.0 <= x_bottom <= (width - 1):
            candidates.append((x_bottom, float(height - 1)))

    unique = []
    for pt in candidates:
        if not any(abs(pt[0] - q[0]) < 1e-6 and abs(pt[1] - q[1]) < 1e-6 for q in unique):
            unique.append(pt)

    if len(unique) < 2:
        return None

    best_pair = None
    best_dist = -1.0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            dist = (unique[i][0] - unique[j][0]) ** 2 + (unique[i][1] - unique[j][1]) ** 2
            if dist > best_dist:
                best_dist = dist
                best_pair = (unique[i], unique[j])

    if best_pair is None:
        return None

    p_start = (int(round(best_pair[0][0])), int(round(best_pair[0][1])))
    p_end = (int(round(best_pair[1][0])), int(round(best_pair[1][1])))
    return p_start, p_end


def draw_projected_line(canvas, p1, p2, color, thickness):
    projected = project_line_to_image(p1, p2, canvas.shape[1], canvas.shape[0])
    if projected is None:
        return
    cv2.line(canvas, projected[0], projected[1], color, thickness, cv2.LINE_AA)


def draw_extended_line(canvas, p1, p2, color, thickness, extend: float = 0.20):
    """Dibuja una línea entre p1 y p2 extendida `extend` fracción más allá de cada extremo."""
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dx = x2 - x1
    dy = y2 - y1
    ex1 = (int(round(x1 - dx * extend)), int(round(y1 - dy * extend)))
    ex2 = (int(round(x2 + dx * extend)), int(round(y2 + dy * extend)))
    h, w = canvas.shape[:2]
    ex1 = (max(0, min(w - 1, ex1[0])), max(0, min(h - 1, ex1[1])))
    ex2 = (max(0, min(w - 1, ex2[0])), max(0, min(h - 1, ex2[1])))
    cv2.line(canvas, ex1, ex2, color, thickness, cv2.LINE_AA)


def draw_detection_boxes(canvas, detections, style):
    for det in detections.values():
        side = det["side"]
        color = SIDE_COLORS.get(side, (255, 255, 255))
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        label = f"{det['joint']}_{side} {det['confidence']:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, style["box_thickness"])
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, style["font_small"], style["text_thickness"])[0]
        pad = style["label_pad"]
        tag_h = text_size[1] + (pad * 2)
        tag_w = text_size[0] + (pad * 2)
        tag_y1 = max(0, y1 - tag_h)
        tag_y2 = y1
        tag_x2 = min(canvas.shape[1], x1 + tag_w)
        cv2.rectangle(canvas, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + pad, tag_y2 - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            style["font_small"],
            (20, 20, 20),
            style["text_thickness"],
            cv2.LINE_AA,
        )


def draw_side_overlay(canvas, side: str, points, metrics, style, draw_axes: bool = True, draw_points: bool = True):
    color = SIDE_COLORS.get(side, (255, 255, 255))
    point_radius = style["point_radius"]

    if draw_axes:
        # Ejes largos: proyección completa hasta el borde de la imagen
        for start, end in [(0, 7), (1, 2)]:
            if start in points and end in points:
                draw_projected_line(canvas, points[start], points[end], color, style["line_thickness"])
        # Cóndilos y platillos: solo 20% más allá de cada punto
        for start, end in [(3, 4), (5, 6)]:
            if start in points and end in points:
                draw_extended_line(canvas, points[start], points[end], color, style["line_thickness"], extend=0.20)

    if draw_points:
        for idx, (px, py) in points.items():
            center = (int(round(px)), int(round(py)))
            cv2.circle(canvas, center, point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, center, point_radius + style["line_thickness"], (255, 255, 255), style["line_thickness"], cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"P{idx}",
                (center[0] + style["label_pad"], center[1] - style["label_pad"]),
                cv2.FONT_HERSHEY_SIMPLEX,
                style["font_small"],
                color,
                style["text_thickness"],
                cv2.LINE_AA,
            )


def render_overlay_from_payload(image_path: Path, payload: dict, draw_boxes: bool = True, draw_axes: bool = True):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen para render: {image_path}")

    canvas = image.copy()
    style = get_overlay_style(canvas)

    detections = {}
    for cls_id_str, det in payload.get("detections", {}).items():
        detections[int(cls_id_str)] = {
            "joint": det["joint"],
            "side": det["side"],
            "confidence": float(det["confidence"]),
            "xyxy": [float(v) for v in det["xyxy"]],
        }

    if draw_boxes:
        draw_detection_boxes(canvas, detections, style)

    for side_name in ["Der", "Izq"]:
        side_info = payload.get("sides", {}).get(side_name, {})
        points = {
            int(idx): (float(vals[0]), float(vals[1]))
            for idx, vals in side_info.get("points", {}).items()
        }
        metrics = side_info.get("metrics", {})
        draw_side_overlay(canvas, side_name, points, metrics, style, draw_axes=draw_axes, draw_points=True)

    return canvas


def process_image(
    det_model: YOLO,
    pose_model: YOLO,
    image_path: Path,
    output_dir: Path,
    det_conf: float,
    pose_conf: float,
    padding: float,
    draw_boxes: bool = True,
    draw_axes: bool = True,
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    det_result = det_model.predict(source=str(image_path), conf=det_conf, verbose=False)[0]
    detections = pick_best_detections(det_result)

    sides = {"Der": {}, "Izq": {}}
    for det in detections.values():
        if det["side"] in sides:
            sides[det["side"]][det["joint"]] = det

    result_payload = {"image": str(image_path), "detections": {}, "sides": {}}
    for cls_id, det in detections.items():
        result_payload["detections"][str(cls_id)] = {
            "joint": det["joint"],
            "side": det["side"],
            "confidence": round(det["confidence"], 4),
            "xyxy": [round(v, 2) for v in det["xyxy"]],
        }

    for side_name, side_detections in sides.items():
        points, crops = build_side_points(image, side_detections, pose_model, pose_conf, padding)
        metrics = calculate_metrics(points)
        result_payload["sides"][side_name] = {
            "detections_found": sorted(side_detections.keys()),
            "points": {str(idx): [round(pt[0], 2), round(pt[1], 2)] for idx, pt in points.items()},
            "crops": crops,
            "metrics": metrics,
        }

    canvas = render_overlay_from_payload(image_path, result_payload, draw_boxes=draw_boxes, draw_axes=draw_axes)

    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / f"{image_path.stem}_final_overlay.jpg"
    json_path = output_dir / f"{image_path.stem}_final_metrics.json"
    cv2.imwrite(str(overlay_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    json_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return overlay_path, json_path, result_payload


def main():
    args = parse_args()

    source = Path(args.source)
    det_model_path = Path(args.det_model)
    pose_model_path = Path(args.pose_model)
    output_dir = Path(args.output_dir)

    validate_model_path(det_model_path, "deteccion")
    validate_model_path(pose_model_path, "pose")

    images = collect_images(source)
    if not images:
        raise RuntimeError(f"No se encontraron imagenes para procesar en: {source}")

    det_model = YOLO(str(det_model_path))
    pose_model = YOLO(str(pose_model_path))

    print(f"Modelo deteccion: {det_model_path}")
    print(f"Modelo pose: {pose_model_path}")
    print(f"Imagenes a procesar: {len(images)}")

    for image_path in images:
        overlay_path, json_path, payload = process_image(
            det_model=det_model,
            pose_model=pose_model,
            image_path=image_path,
            output_dir=output_dir,
            det_conf=args.det_conf,
            pose_conf=args.pose_conf,
            padding=args.padding,
        )
        print(f"\nProcesada: {image_path.name}")
        print(f"  Overlay: {overlay_path}")
        print(f"  JSON: {json_path}")
        for side_name, side_info in payload["sides"].items():
            metrics = side_info["metrics"]
            if metrics.get("status") == "ok":
                print(
                    f"  {side_name}: LDFA={metrics['LDFA']:.2f} MPTA={metrics['MPTA']:.2f} "
                    f"aHKA={metrics['aHKA']:.2f} JLO={metrics['JLO']:.2f} | {metrics['CPAK']}"
                )
            else:
                print(f"  {side_name}: sin calculo completo ({metrics})")


if __name__ == "__main__":
    main()
