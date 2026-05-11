import os
import shutil
import random
import stat
import argparse
from pathlib import Path
import yaml

# Rutas
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DIR = BASE_DIR / "crops_for_keypoints"
POSE_DATASET_DIR = BASE_DIR / "dataset_pose_final"


def _handle_remove_readonly(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _find_image_for_stem(source_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = source_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate.name
    return None


def prepare_final_dataset(source_dir: Path):
    os.makedirs(POSE_DATASET_DIR, exist_ok=True)

    # Rehacer split limpio para evitar mezclar residuos de ejecuciones anteriores.
    for split in ["train", "val"]:
        split_dir = POSE_DATASET_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir, onerror=_handle_remove_readonly)

    label_files = [
        f for f in os.listdir(source_dir) if f.lower().endswith(".txt")
    ]

    pairs = []
    missing_images = []
    for lbl in label_files:
        stem = Path(lbl).stem
        img = _find_image_for_stem(source_dir, stem)
        if img is None:
            missing_images.append(lbl)
            continue
        pairs.append((img, lbl))

    print(f"Organizando {len(pairs)} pares imagen-label desde: {source_dir}")
    if missing_images:
        print(f"Aviso: {len(missing_images)} labels no tienen imagen asociada.")

    random.seed(42)
    random.shuffle(pairs)
    
    # Split 80/20
    train_end = int(len(pairs) * 0.8)
    splits = {'train': pairs[:train_end], 'val': pairs[train_end:]}
    
    for split, data in splits.items():
        img_dest = POSE_DATASET_DIR / split / "images"
        lbl_dest = POSE_DATASET_DIR / split / "labels"
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        for img, lbl in data:
            shutil.copy(source_dir / img, img_dest / img)
            shutil.copy(source_dir / lbl, lbl_dest / lbl)
            
    # data.yaml
    yaml_data = {
        'path': str(POSE_DATASET_DIR.absolute()).replace("\\", "/"),
        'train': 'train/images',
        'val': 'val/images',
        'kpt_shape': [8, 3],
        'flip_idx': [0, 1, 2, 3, 4, 5, 6, 7],
        'nc': 3,
        'names': {0: 'Hip', 1: 'Knee', 2: 'Ankle'}
    }
    
    with open(POSE_DATASET_DIR / "data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    print(f"Dataset listo en {POSE_DATASET_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construye dataset_pose_final desde crops etiquetados"
    )
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help="Carpeta con imagenes y labels .txt (default: crops_for_keypoints)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_final_dataset(Path(args.source_dir))
