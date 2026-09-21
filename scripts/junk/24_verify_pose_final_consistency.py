from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = BASE_DIR / "dataset_pose_final"


def parse_label(label_path: Path):
    vals = label_path.read_text(encoding="utf-8").strip().split()
    if len(vals) < 5 + 8 * 3:
        return None
    k = list(map(float, vals[5:]))
    out = {}
    for i in range(8):
        out[i] = (k[i * 3], k[i * 3 + 1], int(k[i * 3 + 2]))
    return out


def check_split(split: str):
    lbl_dir = DATASET_DIR / split / "labels"
    files = sorted(lbl_dir.glob("*.txt"))

    total = 0
    cond_valid = 0
    plat_valid = 0
    cond_med_gt_lat = 0
    plat_med_gt_lat = 0

    for f in files:
        if "Rodilla" not in f.stem:
            continue
        data = parse_label(f)
        if data is None:
            continue

        total += 1

        if data[3][2] > 0 and data[4][2] > 0:
            cond_valid += 1
            if data[3][0] > data[4][0]:
                cond_med_gt_lat += 1

        if data[5][2] > 0 and data[6][2] > 0:
            plat_valid += 1
            if data[5][0] > data[6][0]:
                plat_med_gt_lat += 1

    print(f"split={split}")
    print(f"  knee_labels={total}")
    print(
        f"  condyles_med_gt_lat={cond_med_gt_lat}/{cond_valid}"
        f" ({(100.0 * cond_med_gt_lat / max(cond_valid,1)):.2f}%)"
    )
    print(
        f"  plateaus_med_gt_lat={plat_med_gt_lat}/{plat_valid}"
        f" ({(100.0 * plat_med_gt_lat / max(plat_valid,1)):.2f}%)"
    )


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_DIR}")

    check_split("train")
    check_split("val")


if __name__ == "__main__":
    main()
