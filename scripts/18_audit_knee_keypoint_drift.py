import argparse
import csv
import math
from pathlib import Path

import cv2
from ultralytics import YOLO

# Knee keypoint indices used across this project:
# 3: Condyle Medial, 4: Condyle Lateral, 5: Plateau Medial, 6: Plateau Lateral
KNEE_POINT_INDICES = [3, 4, 5, 6]
PAIR_DEFS = [(3, 4, "condyles"), (5, 6, "plateaus")]


def parse_label_file(label_path: Path):
    if not label_path.exists():
        return None

    txt = label_path.read_text(encoding="utf-8").strip()
    if not txt:
        return None

    parts = txt.split()
    if len(parts) < 5 + 8 * 3:
        return None

    kpt_values = list(map(float, parts[5:]))
    points = {}
    for i in range(8):
        x = kpt_values[i * 3]
        y = kpt_values[i * 3 + 1]
        v = int(kpt_values[i * 3 + 2])
        points[i] = (x, y, v)

    return points


def dist_px(px1, py1, px2, py2):
    return math.sqrt((px1 - px2) ** 2 + (py1 - py2) ** 2)


def evaluate_image(model: YOLO, image_path: Path, label_path: Path, conf: float):
    gt_points = parse_label_file(label_path)
    if gt_points is None:
        return None

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
        return {
            "file": image_path.name,
            "status": "no_prediction",
            "n_valid_gt": sum(1 for i in KNEE_POINT_INDICES if gt_points[i][2] > 0),
            "mean_err_px": "",
            "max_err_px": "",
            "condyles_swap_likely": "",
            "plateaus_swap_likely": "",
            "condyles_direct_err": "",
            "condyles_swap_err": "",
            "plateaus_direct_err": "",
            "plateaus_swap_err": "",
            "worst_point": "",
            "worst_err_px": "",
        }

    pred_xy = result.keypoints.xy[0].cpu().numpy()

    errors = {}
    for idx in KNEE_POINT_INDICES:
        gx, gy, gv = gt_points[idx]
        if gv <= 0:
            continue

        gxp = gx * w
        gyp = gy * h
        px, py = float(pred_xy[idx][0]), float(pred_xy[idx][1])
        errors[idx] = dist_px(px, py, gxp, gyp)

    if not errors:
        return None

    worst_idx = max(errors, key=lambda k: errors[k])
    worst_err = errors[worst_idx]

    pair_stats = {}
    for a, b, name in PAIR_DEFS:
        if gt_points[a][2] <= 0 or gt_points[b][2] <= 0:
            pair_stats[name] = {
                "direct": None,
                "swap": None,
                "swap_likely": None,
            }
            continue

        ga_x, ga_y, _ = gt_points[a]
        gb_x, gb_y, _ = gt_points[b]
        ga_x *= w
        ga_y *= h
        gb_x *= w
        gb_y *= h

        pa_x, pa_y = float(pred_xy[a][0]), float(pred_xy[a][1])
        pb_x, pb_y = float(pred_xy[b][0]), float(pred_xy[b][1])

        direct = dist_px(pa_x, pa_y, ga_x, ga_y) + dist_px(pb_x, pb_y, gb_x, gb_y)
        swap = dist_px(pa_x, pa_y, gb_x, gb_y) + dist_px(pb_x, pb_y, ga_x, ga_y)

        # If swapped assignment is clearly better, this pair is likely inverted.
        swap_likely = swap + 4.0 < direct

        pair_stats[name] = {
            "direct": direct,
            "swap": swap,
            "swap_likely": swap_likely,
        }

    mean_err = sum(errors.values()) / len(errors)
    max_err = max(errors.values())

    return {
        "file": image_path.name,
        "status": "ok",
        "n_valid_gt": len(errors),
        "mean_err_px": round(mean_err, 3),
        "max_err_px": round(max_err, 3),
        "condyles_swap_likely": pair_stats["condyles"]["swap_likely"],
        "plateaus_swap_likely": pair_stats["plateaus"]["swap_likely"],
        "condyles_direct_err": "" if pair_stats["condyles"]["direct"] is None else round(pair_stats["condyles"]["direct"], 3),
        "condyles_swap_err": "" if pair_stats["condyles"]["swap"] is None else round(pair_stats["condyles"]["swap"], 3),
        "plateaus_direct_err": "" if pair_stats["plateaus"]["direct"] is None else round(pair_stats["plateaus"]["direct"], 3),
        "plateaus_swap_err": "" if pair_stats["plateaus"]["swap"] is None else round(pair_stats["plateaus"]["swap"], 3),
        "worst_point": worst_idx,
        "worst_err_px": round(worst_err, 3),
    }


def collect_knee_images(dataset_dir: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    all_images = [
        p for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]
    return sorted([p for p in all_images if "Rodilla" in p.stem])


def write_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "file",
        "status",
        "n_valid_gt",
        "mean_err_px",
        "max_err_px",
        "condyles_swap_likely",
        "plateaus_swap_likely",
        "condyles_direct_err",
        "condyles_swap_err",
        "plateaus_direct_err",
        "plateaus_swap_err",
        "worst_point",
        "worst_err_px",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows, out_txt: Path):
    ok_rows = [r for r in rows if r["status"] == "ok"]
    nopred_rows = [r for r in rows if r["status"] == "no_prediction"]

    cond_swap = sum(1 for r in ok_rows if r["condyles_swap_likely"] is True)
    plat_swap = sum(1 for r in ok_rows if r["plateaus_swap_likely"] is True)

    avg_mean_err = 0.0
    if ok_rows:
        avg_mean_err = sum(float(r["mean_err_px"]) for r in ok_rows) / len(ok_rows)

    lines = [
        "Knee keypoint drift audit",
        f"total_rows={len(rows)}",
        f"ok_rows={len(ok_rows)}",
        f"no_prediction_rows={len(nopred_rows)}",
        f"avg_mean_err_px={avg_mean_err:.3f}",
        f"condyles_swap_likely_count={cond_swap}",
        f"plateaus_swap_likely_count={plat_swap}",
    ]

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict all canonical knee crops and compare knee keypoint positions against labels."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="training_runs/telerx_pose_v2/weights/best.pt",
        help="Path to pose model (.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dataset_canonical",
        help="Path to canonical dataset folder containing images and labels",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.2,
        help="Confidence threshold for prediction",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="reports/knee_keypoint_drift.csv",
        help="Output CSV report path",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="reports/knee_keypoint_drift_summary.txt",
        help="Output summary txt path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of worst samples to print in console",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    dataset_dir = Path(args.dataset)
    out_csv = Path(args.out_csv)
    out_summary = Path(args.out_summary)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    images = collect_knee_images(dataset_dir)
    if not images:
        raise RuntimeError("No knee images found in dataset folder (expected names containing 'Rodilla').")

    print(f"Loaded model: {model_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Knee images found: {len(images)}")

    model = YOLO(str(model_path))

    rows = []
    for i, img_path in enumerate(images, start=1):
        label_path = img_path.with_suffix(".txt")
        row = evaluate_image(model, img_path, label_path, args.conf)
        if row is None:
            continue
        rows.append(row)

        if i % 100 == 0:
            print(f"Processed {i}/{len(images)}")

    # Sort worst first by max error when available.
    def sort_key(r):
        if r["status"] != "ok":
            return -1.0
        return float(r["max_err_px"])

    rows_sorted = sorted(rows, key=sort_key, reverse=True)

    write_csv(rows_sorted, out_csv)
    write_summary(rows_sorted, out_summary)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved summary: {out_summary}")

    print("\nTop worst samples:")
    shown = 0
    for r in rows_sorted:
        if r["status"] != "ok":
            continue
        print(
            f"- {r['file']} | max_err_px={r['max_err_px']} | "
            f"cond_swap={r['condyles_swap_likely']} | plat_swap={r['plateaus_swap_likely']}"
        )
        shown += 1
        if shown >= args.top_n:
            break


if __name__ == "__main__":
    main()
