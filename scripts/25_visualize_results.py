"""
25_visualize_results.py

Genera tres tipos de visualizaciones para evaluar el modelo reentrenado:

  1. overlays/  — imágenes individuales con GT (verde) y predicción (naranja)
                  para los N peores y M mejores casos por error.
  2. reports/error_distribution.png  — histograma de mean_err_px (nuevo vs viejo modelo).
  3. reports/model_comparison.png    — barras comparando swap_count y avg_err entre modelos.

Uso:
  python scripts/25_visualize_results.py [--top-worst 10] [--top-best 6]
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constantes de keypoints
# ---------------------------------------------------------------------------
# Índices en el label YOLO (0-based):
#   0 Hip_Med, 1 Hip_Lat, 2 (no usado), 3 Condilo_Med, 4 Condilo_Lat,
#   5 Platillo_Med, 6 Platillo_Lat, 7 Tobillo
KNEE_IDX = [3, 4, 5, 6]
KPT_LABELS = {
    3: "Cond_Med",
    4: "Cond_Lat",
    5: "Plat_Med",
    6: "Plat_Lat",
}
# Colores BGR para cada keypoint (GT usa relleno, Pred usa borde grueso)
KPT_COLORS = {
    3: (0, 200, 0),    # verde
    4: (255, 80, 0),   # azul
    5: (0, 200, 200),  # amarillo-verde
    6: (200, 0, 200),  # magenta
}
GT_ALPHA_COLOR = (50, 220, 50)     # relleno GT
PRED_ALPHA_COLOR = (0, 165, 255)   # relleno Pred


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_label(label_path: Path, w: int, h: int):
    """Devuelve dict idx -> (px, py) para puntos con visibilidad > 0."""
    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return {}
    parts = txt.split()
    if len(parts) < 5 + 8 * 3:
        return {}
    kpt_values = list(map(float, parts[5:]))
    pts = {}
    for i in range(8):
        x = kpt_values[i * 3]
        y = kpt_values[i * 3 + 1]
        v = int(kpt_values[i * 3 + 2])
        if v > 0:
            pts[i] = (x * w, y * h)
    return pts


def dist_px(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def draw_overlay(image_path: Path, label_path: Path, model: YOLO, conf: float) -> np.ndarray:
    """Dibuja GT y predicción sobre la imagen, devuelve ndarray BGR."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    gt_pts = parse_label(label_path, w, h)

    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    pred_xy = None
    if result.keypoints is not None and len(result.keypoints.xy) > 0:
        pred_xy = result.keypoints.xy[0].cpu().numpy()  # shape (8, 2)

    # Ligero oscurecimiento para que los puntos resalten
    overlay = img.copy()
    cv2.addWeighted(img, 0.75, np.zeros_like(img), 0.25, 0, overlay)

    r_gt = max(6, w // 80)
    r_pred = max(6, w // 80)
    font_scale = max(0.4, w / 1200)
    thick = max(1, w // 400)

    for idx in KNEE_IDX:
        color = KPT_COLORS[idx]
        label = KPT_LABELS[idx]

        gp = gt_pts.get(idx)
        pp = None
        if pred_xy is not None:
            px, py = float(pred_xy[idx][0]), float(pred_xy[idx][1])
            if px > 0 or py > 0:
                pp = (px, py)

        # Línea de error GT -> Pred
        if gp is not None and pp is not None:
            err = dist_px(gp, pp)
            cv2.line(overlay,
                     (int(gp[0]), int(gp[1])),
                     (int(pp[0]), int(pp[1])),
                     (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
            # Texto de error en el punto medio
            mx = int((gp[0] + pp[0]) / 2)
            my = int((gp[1] + pp[1]) / 2)
            cv2.putText(overlay, f"{err:.1f}px", (mx + 4, my - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75,
                        (255, 255, 255), thick, cv2.LINE_AA)

        # GT — círculo relleno
        if gp is not None:
            cv2.circle(overlay, (int(gp[0]), int(gp[1])), r_gt, color, -1, cv2.LINE_AA)
            cv2.circle(overlay, (int(gp[0]), int(gp[1])), r_gt, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, label,
                        (int(gp[0]) + r_gt + 2, int(gp[1]) - r_gt),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA)

        # Pred — círculo con borde blanco grueso
        if pp is not None:
            cv2.circle(overlay, (int(pp[0]), int(pp[1])), r_pred, (255, 255, 255), thick + 2, cv2.LINE_AA)
            cv2.circle(overlay, (int(pp[0]), int(pp[1])), r_pred, color, thick, cv2.LINE_AA)

    # Leyenda
    leg_y = 28
    cv2.rectangle(overlay, (0, 0), (220, 60), (20, 20, 20), -1)
    cv2.circle(overlay, (14, leg_y - 8), 8, (0, 200, 0), -1)
    cv2.putText(overlay, "GT (relleno)", (26, leg_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (200, 255, 200), 1)
    cv2.circle(overlay, (14, leg_y + 16), 8, (255, 255, 255), 2)
    cv2.putText(overlay, "Pred (borde)", (26, leg_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (200, 200, 255), 1)

    return overlay


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path):
    rows = []
    with csv_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def get_ok_rows_sorted(rows):
    ok = [r for r in rows if r["status"] == "ok" and r["mean_err_px"] != ""]
    ok.sort(key=lambda r: float(r["mean_err_px"]), reverse=True)
    return ok


# ---------------------------------------------------------------------------
# Visualización 1: overlays individuales
# ---------------------------------------------------------------------------

def make_overlays(dataset_dir: Path, model: YOLO, conf: float,
                  csv_path: Path, out_dir: Path,
                  n_worst: int, n_best: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(csv_path)
    sorted_rows = get_ok_rows_sorted(rows)

    selected = []
    # Peores N
    for r in sorted_rows[:n_worst]:
        selected.append(("worst", r))
    # Mejores M (menor error)
    for r in sorted_rows[-n_best:]:
        selected.append(("best", r))

    print(f"\n[Overlays] Generando {len(selected)} imágenes en {out_dir} …")
    for tag, row in selected:
        fname = row["file"]
        img_path = dataset_dir / fname
        lbl_path = dataset_dir / (Path(fname).stem + ".txt")
        if not img_path.exists():
            print(f"  SKIP (imagen no encontrada): {fname}")
            continue

        vis = draw_overlay(img_path, lbl_path, model, conf)
        if vis is None:
            print(f"  SKIP (no se pudo leer): {fname}")
            continue

        mean_err = float(row["mean_err_px"])
        cswap = row["condyles_swap_likely"]
        pswap = row["plateaus_swap_likely"]

        # Cabecera en la imagen
        info = (f"{tag.upper()}  mean={mean_err:.1f}px  "
                f"cond_swap={cswap}  plat_swap={pswap}")
        cv2.rectangle(vis, (0, vis.shape[0] - 36), (vis.shape[1], vis.shape[0]),
                      (20, 20, 20), -1)
        cv2.putText(vis, info, (8, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        out_name = f"{tag}_{mean_err:07.2f}_{Path(fname).stem}.jpg"
        cv2.imwrite(str(out_dir / out_name), vis, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {out_name}")


# ---------------------------------------------------------------------------
# Visualización 2: histograma de errores
# ---------------------------------------------------------------------------

def make_error_histogram(csv_old: Path, csv_new: Path, out_path: Path):
    def get_errors(path):
        rows = load_csv(path)
        return [float(r["mean_err_px"]) for r in rows
                if r["status"] == "ok" and r["mean_err_px"] != ""]

    errs_old = get_errors(csv_old)
    errs_new = get_errors(csv_new)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Distribución de error medio por imagen (keypoints de rodilla)",
                 fontsize=13, fontweight="bold")

    for ax, errs, label, color in [
        (axes[0], errs_old, "Modelo viejo\n(datos corregidos)", "#e07b39"),
        (axes[1], errs_new, "Modelo NUEVO\n(reentrenado)", "#3a7ebf"),
    ]:
        bins = np.linspace(0, max(max(errs_old), max(errs_new)) * 1.05, 35)
        ax.hist(errs, bins=bins, color=color, edgecolor="white", linewidth=0.5, alpha=0.85)
        mean_v = np.mean(errs)
        median_v = np.median(errs)
        ax.axvline(mean_v, color="red", linestyle="--", linewidth=1.5,
                   label=f"Media: {mean_v:.1f} px")
        ax.axvline(median_v, color="gold", linestyle=":", linewidth=1.5,
                   label=f"Mediana: {median_v:.1f} px")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("mean_err_px (píxeles)")
        ax.set_ylabel("Cantidad de imágenes")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\n[Histograma] Guardado en {out_path}")


# ---------------------------------------------------------------------------
# Visualización 3: panel comparativo de métricas
# ---------------------------------------------------------------------------

def load_summary(path: Path):
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def make_comparison_bars(summary_old: Path, summary_new: Path, out_path: Path):
    old = load_summary(summary_old)
    new = load_summary(summary_new)

    metrics = {
        "avg_mean_err_px": "Error medio (px)",
        "condyles_swap_likely_count": "Condylos swapeados (#)",
        "plateaus_swap_likely_count": "Platillos swapeados (#)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Comparativa de modelos — rodilla keypoint drift",
                 fontsize=13, fontweight="bold")

    colors_old = "#e07b39"
    colors_new = "#3a7ebf"

    for ax, (key, title) in zip(axes, metrics.items()):
        v_old = float(old.get(key, 0))
        v_new = float(new.get(key, 0))
        bars = ax.bar(["Modelo\nviejo", "Modelo\nnuevo"],
                      [v_old, v_new],
                      color=[colors_old, colors_new],
                      edgecolor="white", width=0.45)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(v_old, v_new) * 1.3 + 1)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, [v_old, v_new]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(v_old, v_new) * 0.02,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        # Anotación de mejora porcentual
        if v_old > 0:
            pct = (v_old - v_new) / v_old * 100
            sign = "▼" if pct >= 0 else "▲"
            color_ann = "#2ca02c" if pct >= 0 else "#d62728"
            ax.text(0.5, 0.92, f"{sign} {abs(pct):.1f}%",
                    transform=ax.transAxes, ha="center", fontsize=11,
                    color=color_ann, fontweight="bold")

    patch_old = mpatches.Patch(color=colors_old, label="Modelo viejo (datos corregidos)")
    patch_new = mpatches.Patch(color=colors_new, label="Modelo nuevo (reentrenado)")
    fig.legend(handles=[patch_old, patch_new], loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=140, bbox_inches="tight")
    plt.close()
    print(f"[Comparativa] Guardada en {out_path}")


# ---------------------------------------------------------------------------
# Visualización 4: mosaico resumen (worst + best en una sola imagen)
# ---------------------------------------------------------------------------

def make_mosaic(overlay_dir: Path, out_path: Path, n_cols: int = 4):
    worst = sorted(overlay_dir.glob("worst_*.jpg"))
    best = sorted(overlay_dir.glob("best_*.jpg"))
    imgs_paths = worst + best
    if not imgs_paths:
        print("[Mosaico] Sin imágenes, salteando.")
        return

    imgs = []
    target_h = 320
    for p in imgs_paths:
        im = cv2.imread(str(p))
        if im is None:
            continue
        ratio = target_h / im.shape[0]
        im = cv2.resize(im, (int(im.shape[1] * ratio), target_h), interpolation=cv2.INTER_AREA)
        imgs.append(im)

    if not imgs:
        return

    target_w = max(im.shape[1] for im in imgs)
    padded = []
    for im in imgs:
        if im.shape[1] < target_w:
            pad = np.zeros((im.shape[0], target_w - im.shape[1], 3), dtype=np.uint8)
            im = np.hstack([im, pad])
        padded.append(im)

    n_cols = min(n_cols, len(padded))
    n_rows = math.ceil(len(padded) / n_cols)
    blank = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    rows_imgs = []
    for r in range(n_rows):
        row_imgs = padded[r * n_cols: (r + 1) * n_cols]
        while len(row_imgs) < n_cols:
            row_imgs.append(blank.copy())
        rows_imgs.append(np.hstack(row_imgs))

    mosaic = np.vstack(rows_imgs)

    # Separadores
    sep_color = (60, 60, 60)
    for col in range(1, n_cols):
        x = col * target_w
        cv2.line(mosaic, (x, 0), (x, mosaic.shape[0]), sep_color, 2)
    for row in range(1, n_rows):
        y = row * target_h
        cv2.line(mosaic, (0, y), (mosaic.shape[1], y), sep_color, 2)

    # Etiqueta de sección
    cv2.putText(mosaic, "PEORES CASOS", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 1, cv2.LINE_AA)
    if len(worst) < len(padded):
        x_best = len(worst) % n_cols * target_w if len(worst) % n_cols != 0 else 0
        y_best = (len(worst) // n_cols) * target_h
        cv2.putText(mosaic, "MEJORES CASOS", (x_best + 8, y_best + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[Mosaico] Guardado en {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    BASE = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Visualizaciones de resultados pose model")
    parser.add_argument("--model", default=str(BASE / "training_runs/telerx_pose_s_rebuild1/weights/best.pt"))
    parser.add_argument("--dataset", default=str(BASE / "dataset_canonical"))
    parser.add_argument("--csv-new", default=str(BASE / "reports/knee_keypoint_drift_retrained.csv"))
    parser.add_argument("--csv-old", default=str(BASE / "reports/knee_keypoint_drift_after_rebuild.csv"))
    parser.add_argument("--summary-new", default=str(BASE / "reports/knee_keypoint_drift_retrained_summary.txt"))
    parser.add_argument("--summary-old", default=str(BASE / "reports/knee_keypoint_drift_after_rebuild_summary.txt"))
    parser.add_argument("--out-dir", default=str(BASE / "reports/visualizations"))
    parser.add_argument("--top-worst", type=int, default=10, help="Cuántos peores casos visualizar")
    parser.add_argument("--top-best",  type=int, default=6,  help="Cuántos mejores casos visualizar")
    parser.add_argument("--conf", type=float, default=0.2)
    args = parser.parse_args()

    model_path   = Path(args.model)
    dataset_dir  = Path(args.dataset)
    csv_new      = Path(args.csv_new)
    csv_old      = Path(args.csv_old)
    summary_new  = Path(args.summary_new)
    summary_old  = Path(args.summary_old)
    out_dir      = Path(args.out_dir)
    overlay_dir  = out_dir / "overlays"

    print(f"Modelo:  {model_path}")
    print(f"Dataset: {dataset_dir}")

    model = YOLO(str(model_path))

    # 1. Overlays
    make_overlays(dataset_dir, model, args.conf,
                  csv_new, overlay_dir,
                  args.top_worst, args.top_best)

    # 2. Histograma
    make_error_histogram(csv_old, csv_new, out_dir / "error_distribution.png")

    # 3. Comparativa de barras
    make_comparison_bars(summary_old, summary_new, out_dir / "model_comparison.png")

    # 4. Mosaico
    make_mosaic(overlay_dir, out_dir / "mosaic.jpg", n_cols=4)

    print(f"\nListo. Todas las visualizaciones en: {out_dir}")


if __name__ == "__main__":
    main()
