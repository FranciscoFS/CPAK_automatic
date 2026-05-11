import argparse
from pathlib import Path

PAIR_DEFS = [(3, 4), (5, 6)]
MODES = ("direct", "swap", "mirr_direct", "mirr_swap")


def read_label_kpts(label_path: Path):
    vals = label_path.read_text(encoding="utf-8").strip().split()
    if len(vals) < 5 + 8 * 3:
        return None

    kpt = list(map(float, vals[5:]))
    out = {}
    for i in range(8):
        out[i] = (kpt[i * 3], kpt[i * 3 + 1], int(kpt[i * 3 + 2]))
    return out


def pair_cost(a, b, mode):
    total = 0.0
    n = 0

    for med, lat in PAIR_DEFS:
        axm, aym, avm = a[med]
        axl, ayl, avl = a[lat]
        bxm, bym, bvm = b[med]
        bxl, byl, bvl = b[lat]

        if avm <= 0 or avl <= 0 or bvm <= 0 or bvl <= 0:
            continue

        if mode == "direct":
            aa = [(axm, aym), (axl, ayl)]
            bb = [(bxm, bym), (bxl, byl)]
        elif mode == "swap":
            aa = [(axm, aym), (axl, ayl)]
            bb = [(bxl, byl), (bxm, bym)]
        elif mode == "mirr_direct":
            aa = [(axm, aym), (axl, ayl)]
            bb = [(1.0 - bxm, bym), (1.0 - bxl, byl)]
        elif mode == "mirr_swap":
            aa = [(axm, aym), (axl, ayl)]
            bb = [(1.0 - bxl, byl), (1.0 - bxm, bym)]
        else:
            raise ValueError(mode)

        for (x1, y1), (x2, y2) in zip(aa, bb):
            total += abs(x1 - x2) + abs(y1 - y2)
            n += 1

    return total / max(n, 1)


def collect_pairs(dataset_dir: Path):
    pairs = []
    for v1 in dataset_dir.glob("*_Rodilla_Der_v1.txt"):
        base = Path(str(v1).replace("_v1.txt", ".txt"))
        if base.exists():
            pairs.append((base, v1))
    return sorted(pairs, key=lambda p: p[0].name)


def main():
    parser = argparse.ArgumentParser(description="Compare GT labels between Rodilla_Der and Rodilla_Der_v1 pairs.")
    parser.add_argument("--dataset", type=str, default="dataset_canonical")
    parser.add_argument("--show", type=int, default=12, help="How many sample pairs to print.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    pairs = collect_pairs(dataset_dir)
    if not pairs:
        print("No Der/Der_v1 knee pairs found.")
        return

    relation_counts = {m: 0 for m in MODES}
    rows = []

    for base, v1 in pairs:
        a = read_label_kpts(base)
        b = read_label_kpts(v1)
        if a is None or b is None:
            continue

        costs = {m: pair_cost(a, b, m) for m in MODES}
        best = min(costs, key=costs.get)
        relation_counts[best] += 1

        ordered = sorted(costs.items(), key=lambda kv: kv[1])
        margin = ordered[1][1] - ordered[0][1]
        rows.append((base.name, best, margin, costs, a, b))

    print(f"total_pairs={len(rows)}")
    print(f"best_relation_counts={relation_counts}")

    print("\nSamples:")
    for row in rows[: args.show]:
        name, best, margin, costs, a, b = row
        print(f"- {name}: best={best}, margin={margin:.6f}, costs={{direct:{costs['direct']:.6f}, swap:{costs['swap']:.6f}, mirr_direct:{costs['mirr_direct']:.6f}, mirr_swap:{costs['mirr_swap']:.6f}}}")
        print(
            "  idx3/4 base=",
            (a[3], a[4]),
            "v1=",
            (b[3], b[4]),
            "| idx5/6 base=",
            (a[5], a[6]),
            "v1=",
            (b[5], b[6]),
        )


if __name__ == "__main__":
    main()
