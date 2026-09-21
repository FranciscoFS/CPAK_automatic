from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


CLASS_NAMES = [
    "Cadera Der",
    "Cadera Izq",
    "Rodilla Der",
    "Rodilla Izq",
    "Tobillo Der",
    "Tobillo Izq",
]

PAIR_CHECKS = [(0, 1), (2, 3), (4, 5)]


@dataclass
class FileAudit:
    stem: str
    issues: List[str] = field(default_factory=list)
    severity: int = 0
    boxes: Dict[int, Tuple[float, float, float, float]] = field(default_factory=dict)

    def add(self, message: str, weight: int = 1) -> None:
        self.issues.append(message)
        self.severity += weight


def find_image(data_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        p = data_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def parse_label_line(line: str) -> Tuple[int, float, float, float, float] | None:
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    try:
        cls_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])
    except ValueError:
        return None
    return cls_id, x, y, w, h


def parse_meta_line(line: str) -> Tuple[int, float, str] | None:
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    try:
        cls_id = int(parts[0])
        conf = float(parts[1])
        origin = parts[2]
    except ValueError:
        return None
    return cls_id, conf, origin


def audit_file(txt_path: Path, meta_path: Path, low_conf: float) -> FileAudit:
    audit = FileAudit(stem=txt_path.stem)

    lines = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        audit.add("label vacio", 4)
        return audit

    seen_classes: List[int] = []
    for idx, line in enumerate(lines, start=1):
        row = parse_label_line(line)
        if row is None:
            audit.add(f"linea {idx} invalida (esperado: class x y w h)", 3)
            continue

        cls_id, x, y, w, h = row
        if not 0 <= cls_id < len(CLASS_NAMES):
            audit.add(f"linea {idx}: clase fuera de rango ({cls_id})", 3)
            continue

        seen_classes.append(cls_id)
        audit.boxes[cls_id] = (x, y, w, h)

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            audit.add(f"linea {idx}: valores fuera de [0,1]", 3)

        x1 = x - (w / 2.0)
        y1 = y - (h / 2.0)
        x2 = x + (w / 2.0)
        y2 = y + (h / 2.0)
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            audit.add(f"linea {idx}: bbox sale de la imagen", 2)

        area = w * h
        if area < 0.002:
            audit.add(f"linea {idx}: bbox muy pequena (area={area:.4f})", 1)
        if area > 0.18:
            audit.add(f"linea {idx}: bbox muy grande (area={area:.4f})", 1)

    if len(seen_classes) != 6:
        audit.add(f"cantidad de etiquetas inesperada: {len(seen_classes)} (esperado 6)", 3)

    missing = sorted(set(range(6)) - set(seen_classes))
    if missing:
        missing_names = ", ".join(CLASS_NAMES[c] for c in missing)
        audit.add(f"clases faltantes: {missing_names}", 3)

    dup = sorted({c for c in seen_classes if seen_classes.count(c) > 1})
    if dup:
        dup_names = ", ".join(CLASS_NAMES[c] for c in dup)
        audit.add(f"clases duplicadas: {dup_names}", 2)

    for right_cls, left_cls in PAIR_CHECKS:
        if right_cls in audit.boxes and left_cls in audit.boxes:
            xr = audit.boxes[right_cls][0]
            xl = audit.boxes[left_cls][0]
            if xr >= xl:
                audit.add(
                    f"orden lateral sospechoso: {CLASS_NAMES[right_cls]} x={xr:.3f} no esta a la izquierda de {CLASS_NAMES[left_cls]} x={xl:.3f}",
                    1,
                )

    if all(k in audit.boxes for k in (0, 2, 4)):
        y0, y2, y4 = audit.boxes[0][1], audit.boxes[2][1], audit.boxes[4][1]
        if not (y0 < y2 < y4):
            audit.add("orden vertical sospechoso en lado derecho (cadera->rodilla->tobillo)", 1)

    if all(k in audit.boxes for k in (1, 3, 5)):
        y1, y3, y5 = audit.boxes[1][1], audit.boxes[3][1], audit.boxes[5][1]
        if not (y1 < y3 < y5):
            audit.add("orden vertical sospechoso en lado izquierdo (cadera->rodilla->tobillo)", 1)

    if meta_path.exists():
        meta_rows = meta_path.read_text(encoding="utf-8", errors="replace").splitlines()
        meta_classes: List[int] = []
        for idx, line in enumerate(meta_rows, start=1):
            row = parse_meta_line(line)
            if row is None:
                audit.add(f"meta linea {idx} invalida", 2)
                continue
            cls_id, conf, origin = row
            meta_classes.append(cls_id)

            if cls_id not in range(6):
                audit.add(f"meta linea {idx}: clase fuera de rango ({cls_id})", 2)
            if not (0.0 <= conf <= 1.0):
                audit.add(f"meta linea {idx}: confidence fuera de [0,1] ({conf:.3f})", 2)
            elif conf < low_conf:
                audit.add(f"meta linea {idx}: confidence baja ({conf:.3f})", 1)

            if origin not in ("D", "S"):
                audit.add(f"meta linea {idx}: origin desconocido ({origin})", 1)

        if len(meta_classes) != len(seen_classes):
            audit.add(
                f"txt/meta con distinta cantidad de etiquetas ({len(seen_classes)} vs {len(meta_classes)})",
                2,
            )

        if set(meta_classes) != set(seen_classes):
            audit.add("txt/meta con clases inconsistentes", 2)
    else:
        audit.add("archivo .meta faltante", 2)

    return audit


def draw_review_image(image_path: Path, audit: FileAudit, out_path: Path) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False

    h, w = image.shape[:2]
    color_ok = (65, 190, 65)
    color_warn = (20, 60, 220)

    for cls_id, (x, y, bw, bh) in sorted(audit.boxes.items()):
        x1 = int((x - bw / 2.0) * w)
        y1 = int((y - bh / 2.0) * h)
        x2 = int((x + bw / 2.0) * w)
        y2 = int((y + bh / 2.0) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), color_ok, 2)
        label = f"{cls_id}:{CLASS_NAMES[cls_id]}"
        cv2.putText(image, label, (max(2, x1), max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_ok, 1)

    y_text = 24
    cv2.putText(
        image,
        f"severity={audit.severity} issues={len(audit.issues)}",
        (8, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color_warn,
        2,
    )
    y_text += 24
    for msg in audit.issues[:6]:
        cv2.putText(image, msg[:95], (8, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_warn, 1)
        y_text += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    return cv2.imwrite(str(out_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verifica calidad de auto_tagged_labels y genera reporte")
    parser.add_argument("--labels-dir", type=Path, default=Path("auto_tagged_labels"), help="Carpeta con .txt y .meta")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Carpeta con imagenes fuente")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/auto_label_qc"), help="Carpeta de salida")
    parser.add_argument("--low-conf", type=float, default=0.55, help="Umbral para marcar confidence baja")
    parser.add_argument("--review-top", type=int, default=40, help="Cuantas imagenes sospechosas exportar")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    labels_dir = (base_dir / args.labels_dir).resolve()
    data_dir = (base_dir / args.data_dir).resolve()
    out_dir = (base_dir / args.out_dir).resolve()

    if not labels_dir.exists():
        raise FileNotFoundError(f"No existe labels_dir: {labels_dir}")

    txt_files = sorted(labels_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No hay .txt en {labels_dir}")

    audits: List[FileAudit] = []
    for txt in txt_files:
        audits.append(audit_file(txt, txt.with_suffix(".meta"), args.low_conf))

    total = len(audits)
    with_issues = [a for a in audits if a.issues]
    severe = [a for a in audits if a.severity >= 6]

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "auto_labels_qc_details.csv"
    json_path = out_dir / "auto_labels_qc_summary.json"
    review_dir = out_dir / "review_images"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_stem", "severity", "issue_count", "issues"])
        for a in sorted(audits, key=lambda z: z.severity, reverse=True):
            writer.writerow([a.stem, a.severity, len(a.issues), " | ".join(a.issues)])

    summary = {
        "labels_dir": str(labels_dir),
        "data_dir": str(data_dir),
        "total_files": total,
        "files_with_issues": len(with_issues),
        "files_without_issues": total - len(with_issues),
        "severe_files": len(severe),
        "low_conf_threshold": args.low_conf,
        "review_images_requested": args.review_top,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    exported = 0
    for a in sorted(with_issues, key=lambda z: z.severity, reverse=True)[: args.review_top]:
        img_path = find_image(data_dir, a.stem)
        if img_path is None:
            continue
        out_img = review_dir / f"{a.stem}_qc.jpg"
        if draw_review_image(img_path, a, out_img):
            exported += 1

    print("=== Auto Labels QC ===")
    print(f"labels: {labels_dir}")
    print(f"total_files: {total}")
    print(f"with_issues: {len(with_issues)}")
    print(f"severe_files: {len(severe)}")
    print(f"details_csv: {csv_path}")
    print(f"summary_json: {json_path}")
    print(f"review_images_exported: {exported}")


if __name__ == "__main__":
    main()
