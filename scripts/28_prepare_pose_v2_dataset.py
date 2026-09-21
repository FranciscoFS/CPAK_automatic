"""
Prepara dataset_pose_v2 combinando dataset_pose_final (943 imgs) con los
791 crops nuevos taggeados en crops_for_keypoints/.

Resultado:
    dataset_pose_v2/
        train/images/  train/labels/
        val/images/    val/labels/
        data.yaml

Uso:
    conda run -n physis_seg python scripts/28_prepare_pose_v2_dataset.py
"""
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
CROPS_DIR     = BASE_DIR / "crops_for_keypoints"
SRC_DATASET   = BASE_DIR / "dataset_pose_final"
DST_DATASET   = BASE_DIR / "dataset_pose_v2"
VAL_RATIO     = 0.10
TEST_RATIO    = 0.10
SEED          = 42

random.seed(SEED)


def copy_pair(img_src: Path, lbl_src: Path, split: str):
    for kind, src in (("images", img_src), ("labels", lbl_src)):
        dst = DST_DATASET / split / kind / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    # ── 1. Copiar dataset_pose_final tal cual ──────────────────────────────
    print("Copiando dataset_pose_final...")
    copied = 0
    for split in ("train", "val"):
        img_dir = SRC_DATASET / split / "images"
        lbl_dir = SRC_DATASET / split / "labels"
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                copy_pair(img, lbl, split)
                copied += 1
    print(f"  {copied} imágenes copiadas desde dataset_pose_final")

    # ── 2. Identificar crops nuevos ────────────────────────────────────────
    existing = set(p.stem for p in (DST_DATASET / "train" / "images").glob("*.jpg")) | \
               set(p.stem for p in (DST_DATASET / "val"   / "images").glob("*.jpg"))

    new_crops = [
        p for p in CROPS_DIR.glob("*.jpg")
        if (CROPS_DIR / (p.stem + ".txt")).exists() and p.stem not in existing
    ]
    print(f"\nCrops nuevos encontrados: {len(new_crops)}")

    # ── 3. Split aleatorio de nuevos (80/10/10) ───────────────────────────
    random.shuffle(new_crops)
    n_val   = max(1, int(len(new_crops) * VAL_RATIO))
    n_test  = max(1, int(len(new_crops) * TEST_RATIO))
    n_train = len(new_crops) - n_val - n_test
    new_val   = new_crops[:n_val]
    new_test  = new_crops[n_val:n_val + n_test]
    new_train = new_crops[n_val + n_test:]

    for img in new_train:
        lbl = CROPS_DIR / (img.stem + ".txt")
        copy_pair(img, lbl, "train")
    for img in new_val:
        lbl = CROPS_DIR / (img.stem + ".txt")
        copy_pair(img, lbl, "val")
    for img in new_test:
        lbl = CROPS_DIR / (img.stem + ".txt")
        copy_pair(img, lbl, "test")

    print(f"  Nuevos → train: {len(new_train)}  val: {len(new_val)}  test: {len(new_test)}")

    # ── 4. Resumen final ───────────────────────────────────────────────────
    total_train = len(list((DST_DATASET / "train" / "images").glob("*.jpg")))
    total_val   = len(list((DST_DATASET / "val"   / "images").glob("*.jpg")))
    total_test  = len(list((DST_DATASET / "test"  / "images").glob("*.jpg")))
    print(f"\nDataset final:")
    print(f"  train: {total_train}  val: {total_val}  test: {total_test}  total: {total_train + total_val + total_test}")

    # ── 5. data.yaml ──────────────────────────────────────────────────────
    yaml_content = f"""path: {DST_DATASET.as_posix()}
train: train/images
val: val/images
test: test/images

kpt_shape: [8, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7]

nc: 3
names:
  0: Hip
  1: Knee
  2: Ankle
"""
    (DST_DATASET / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\ndata.yaml guardado en {DST_DATASET / 'data.yaml'}")
    print("Listo.")


if __name__ == "__main__":
    main()
