"""
Screening de TeleRx en Test_Data_Historic.
Usa solo el modelo de deteccion YOLO para clasificar imagenes como:
  - TeleRx:       4-6 detecciones de las 6 clases esperadas
  - TeleRx_parcial: 3 detecciones (dudosa)
  - No_TeleRx:    0-2 detecciones
Genera CSV con resultados y mueve/copia las validas a una carpeta limpia.
"""

import argparse
import csv
import shutil
from pathlib import Path
from collections import Counter

from ultralytics import YOLO

# ── Constantes ──────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "Cadera/Der", 1: "Cadera/Izq", 2: "Rodilla/Der",
               3: "Rodilla/Izq", 4: "Tobillo/Der", 5: "Tobillo/Izq"}
ALL_CLASSES = set(range(6))  # IDs 0-5

EXCLUDE_DIRS = {"_formato_viejo", "_old", "_backup", "_archive"}


def find_images(source_dir: Path):
    """Recorre source_dir buscando imagenes, excluyendo subcarpetas en EXCLUDE_DIRS."""
    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images.extend(source_dir.rglob(ext))

    # Filtrar excluyendo rutas que contengan EXCLUDE_DIRS
    filtered = []
    for img in images:
        parts = set(p.name for p in img.relative_to(source_dir).parents)
        if parts & EXCLUDE_DIRS:
            continue
        filtered.append(img)
    return sorted(filtered)


def classify(detected_classes: set) -> str:
    n = len(detected_classes)
    if n >= 4:
        return "TeleRx"
    elif n == 3:
        return "TeleRx_parcial"
    else:
        return "No_TeleRx"


def main():
    parser = argparse.ArgumentParser(description="Screening de TeleRx con YOLO detection")
    parser.add_argument("--source", default="Test_Data_Historic/_pngs",
                        help="Carpeta con imagenes a evaluar")
    parser.add_argument("--det-model",
                        default="training_runs/telerx_yolo26s_768_b12-4/weights/best.pt",
                        help="Modelo de deteccion YOLO")
    parser.add_argument("--det-conf", type=float, default=0.20,
                        help="Umbral de confianza para deteccion")
    parser.add_argument("--det-batch-size", type=int, default=32,
                        help="Batch size para inference")
    parser.add_argument("--output-csv", default="reports/screening_teleRx.csv",
                        help="CSV de salida con resultados")
    parser.add_argument("--copy-valid-dir", default=None,
                        help="Si se indica, copia las TeleRx validas a esta carpeta")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        raise FileNotFoundError(f"No existe: {source_dir}")

    print(f"Cargando modelo de deteccion: {args.det_model}")
    det_model = YOLO(args.det_model)

    images = find_images(source_dir)
    print(f"Imagenes encontradas: {len(images)}")

    results = []
    total_det = Counter()  # contador global de clases detectadas

    # Procesar en batches
    for start in range(0, len(images), args.det_batch_size):
        batch = images[start:start + args.det_batch_size]
        preds = det_model.predict(
            source=[str(p) for p in batch],
            conf=args.det_conf,
            verbose=False,
        )

        for img_path, pred in zip(batch, preds):
            # Clases detectadas en esta imagen
            cls_ids = set(pred.boxes.cls.int().tolist()) if pred.boxes else set()
            # Solo IDs 0-5
            cls_ids = cls_ids & ALL_CLASSES
            category = classify(cls_ids)

            clases_str = "; ".join(sorted(CLASS_NAMES[c] for c in cls_ids))
            n_det = len(cls_ids)
            max_conf = round(pred.boxes.conf.max().item(), 3) if pred.boxes is not None and len(pred.boxes) > 0 else 0.0

            results.append({
                "image": str(img_path),
                "relative_path": str(img_path.relative_to(source_dir)),
                "n_classes": n_det,
                "max_confidence": max_conf,
                "classes_detected": clases_str,
                "classification": category,
            })

            total_det[category] += 1

        done = min(start + args.det_batch_size, len(images))
        print(f"  Procesadas: {done}/{len(images)}  "
              f"[TeleRx: {total_det.get('TeleRx', 0)}, "
              f"Parcial: {total_det.get('TeleRx_parcial', 0)}, "
              f"No: {total_det.get('No_TeleRx', 0)}]")

    # ── Guardar CSV ────────────────────────────────────────────────────────
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ── Copiar validas si se pide ──────────────────────────────────────────
    if args.copy_valid_dir:
        valid_dir = Path(args.copy_valid_dir)
        valid_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for r in results:
            if r["classification"] in ("TeleRx", "TeleRx_parcial"):
                dst = valid_dir / r["relative_path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(r["image"], dst)
                copied += 1
        print(f"  Copiadas {copied} imagenes a {valid_dir}")

    # ── Resumen ────────────────────────────────────────────────────────────
    total = len(results)
    print(f"\n{'='*60}")
    print("RESUMEN DE SCREENING")
    print(f"{'='*60}")
    print(f"Total imagenes evaluadas: {total}")
    for cat in ["TeleRx", "TeleRx_parcial", "No_TeleRx"]:
        n = total_det.get(cat, 0)
        print(f"  {cat:16}: {n:4}  ({n/total*100:.1f}%)")
    print(f"\nCSV: {out_csv.resolve()}")


if __name__ == "__main__":
    main()
