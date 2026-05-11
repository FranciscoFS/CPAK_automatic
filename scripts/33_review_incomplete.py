"""
Revisa casos incompletos/invalid_geometry del batch CPAK.
Genera:
- Resumen por estado y lado
- Top de motivos de incompletitud
- CSV detallado para revisión manual
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "visualizations" / "final_pipeline_data_batch"
OUT_CSV = ROOT / "reports" / "incomplete_cases_review.csv"

json_files = sorted(BATCH_DIR.glob("*_final_metrics.json"))

rows = []
status_counter = Counter()
status_by_side = {"Der": Counter(), "Izq": Counter()}
missing_joint_counter = Counter()
missing_point_counter = Counter()
combo_missing_counter = Counter()

for jf in json_files:
    payload = json.loads(jf.read_text(encoding="utf-8"))
    image_name = payload.get("image_name", jf.name.replace("_final_metrics.json", ""))
    sides = payload.get("sides", {})

    for side in ("Der", "Izq"):
        side_data = sides.get(side, {})
        metrics = side_data.get("metrics", {})
        status = metrics.get("status", "missing_side")
        status_counter[status] += 1
        status_by_side[side][status] += 1

        if status == "ok":
            continue

        detections_found = side_data.get("detections_found", []) or []
        expected_joints = ["Cadera", "Rodilla", "Tobillo"]
        missing_joints = [j for j in expected_joints if j not in detections_found]

        missing_points = [str(x) for x in (metrics.get("missing_points", []) or [])]
        error = metrics.get("error", "")

        if missing_joints:
            for j in missing_joints:
                missing_joint_counter[f"{side}:{j}"] += 1

        if missing_points:
            for p in missing_points:
                missing_point_counter[p] += 1
            combo_missing_counter[", ".join(sorted(missing_points))] += 1

        rows.append(
            {
                "image_name": image_name,
                "json_file": jf.name,
                "side": side,
                "status": status,
                "cpak_type": metrics.get("CPAK_type", ""),
                "aHKA": metrics.get("aHKA", ""),
                "JLO": metrics.get("JLO", ""),
                "detections_found": ";".join(detections_found),
                "missing_joints": ";".join(missing_joints),
                "missing_points": ";".join(missing_points),
                "error": error,
            }
        )

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image_name",
            "json_file",
            "side",
            "status",
            "cpak_type",
            "aHKA",
            "JLO",
            "detections_found",
            "missing_joints",
            "missing_points",
            "error",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("=" * 72)
print("REVISION DE CASOS INCOMPLETOS")
print("=" * 72)
print(f"JSON analizados: {len(json_files)}")
print(f"Total lados evaluados: {2 * len(json_files)}")
print(f"Lados NO-ok: {len(rows)}")

print("\nEstados globales:")
for k, v in status_counter.most_common():
    print(f"  {k}: {v}")

for side in ("Der", "Izq"):
    print(f"\nEstados en lado {side}:")
    for k, v in status_by_side[side].most_common():
        print(f"  {k}: {v}")

print("\nTop articulaciones faltantes:")
for k, v in missing_joint_counter.most_common(10):
    print(f"  {k}: {v}")

print("\nTop keypoints faltantes:")
for k, v in missing_point_counter.most_common(12):
    print(f"  {k}: {v}")

print("\nTop combinaciones de keypoints faltantes:")
for k, v in combo_missing_counter.most_common(12):
    print(f"  [{k}]: {v}")

print(f"\nCSV detallado: {OUT_CSV}")
