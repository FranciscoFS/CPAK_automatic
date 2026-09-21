"""
Resumen de inferencia batch sobre carpeta data/.
Lee todos los JSON en outputs_data_json_batch_v2/ y produce estadísticas.
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

RESULTS_DIR = Path(__file__).resolve().parents[1] / "outputs_data_json_batch_v2"

jsons = sorted(RESULTS_DIR.glob("*_final_metrics.json"))
print(f"Total archivos JSON: {len(jsons)}\n")

sides_names = ["Der", "Izq"]

# Contadores globales
total_images = len(jsons)
complete_both  = 0   # ambos lados con status ok
complete_one   = 0   # solo un lado ok
complete_none  = 0   # ningún lado ok

ldfa_vals  = {"Der": [], "Izq": []}
mpta_vals  = {"Der": [], "Izq": []}
ahka_vals  = {"Der": [], "Izq": []}
jlo_vals   = {"Der": [], "Izq": []}
cpak_type  = {"Der": Counter(), "Izq": Counter()}
cpak_class = {"Der": Counter(), "Izq": Counter()}
jlo_class  = {"Der": Counter(), "Izq": Counter()}
ahka_class = {"Der": Counter(), "Izq": Counter()}

missing_joints_counter = Counter()   # qué articulación falta más

for p in jsons:
    payload = json.loads(p.read_text(encoding="utf-8"))
    sides = payload.get("sides", {})

    ok_sides = 0
    for side in sides_names:
        m = sides.get(side, {}).get("metrics", {})
        if m.get("status") == "ok":
            ok_sides += 1
            ldfa_vals[side].append(m["LDFA"])
            mpta_vals[side].append(m["MPTA"])
            ahka_vals[side].append(m["aHKA"])
            jlo_vals[side].append(m["JLO"])
            cpak_type[side][m.get("CPAK_type", "?")] += 1
            cpak_class[side][m.get("aHKA_class", "?")] += 1
            jlo_class[side][m.get("JLO_class", "?")] += 1
            ahka_class[side][m.get("aHKA_class", "?")] += 1
        else:
            missing = m.get("missing_points", [])
            det_found = sides.get(side, {}).get("detections_found", [])
            for j in ["Cadera", "Rodilla", "Tobillo"]:
                if j not in det_found:
                    missing_joints_counter[f"{side}:{j}"] += 1

    if ok_sides == 2:
        complete_both += 1
    elif ok_sides == 1:
        complete_one += 1
    else:
        complete_none += 1


def stats(vals):
    if not vals:
        return {"n": 0}
    import statistics
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), 2),
        "median": round(statistics.median(vals), 2),
        "std": round(statistics.stdev(vals), 2) if len(vals) > 1 else 0,
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }


# ── IMPRIMIR RESUMEN ────────────────────────────────────────────────────────

print("=" * 60)
print("RESUMEN DE INFERENCIA BATCH")
print("=" * 60)
print(f"Imágenes totales         : {total_images}")
print(f"Cálculo completo (ambos) : {complete_both}  ({complete_both/total_images*100:.1f}%)")
print(f"Cálculo parcial (uno)    : {complete_one}  ({complete_one/total_images*100:.1f}%)")
print(f"Sin cálculo (ninguno)    : {complete_none}  ({complete_none/total_images*100:.1f}%)")

for side in sides_names:
    n_ok = len(ldfa_vals[side])
    print(f"\n── Lado {side}  ({n_ok}/{total_images} con cálculo ok) ──")

    for metric_name, vals in [("LDFA", ldfa_vals[side]), ("MPTA", mpta_vals[side]),
                               ("aHKA", ahka_vals[side]), ("JLO",  jlo_vals[side])]:
        s = stats(vals)
        if s["n"]:
            print(f"  {metric_name:6}: mean={s['mean']:7.2f}  median={s['median']:7.2f}  "
                  f"std={s['std']:6.2f}  [{s['min']}, {s['max']}]")

    print(f"\n  Eje (aHKA): {dict(ahka_class[side].most_common())}")
    print(f"  JLO       : {dict(jlo_class[side].most_common())}")
    print(f"\n  CPAK tipo : {dict(cpak_type[side].most_common())}")

print("\n── Articulaciones no detectadas (top 10) ──")
for k, v in missing_joints_counter.most_common(10):
    print(f"  {k}: {v}  ({v/total_images*100:.1f}%)")

print("\n" + "=" * 60)
