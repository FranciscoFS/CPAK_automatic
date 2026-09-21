"""
Prepara un dataset ampliado combinando:
1. Labels existentes en dataset/ (359 imágenes etiquetadas manualmente)
2. Labels automáticos en auto_tagged_labels/ (597 imágenes)

Genera: dataset_expanded/ con splits 70/20/10
"""

import os
import shutil
import argparse
from pathlib import Path
import random
import yaml
import csv


BASE_DIR = Path(__file__).resolve().parents[1]

# Fuentes
DATASET_MANUAL_TRAIN = BASE_DIR / "dataset" / "train"
DATASET_MANUAL_VAL = BASE_DIR / "dataset" / "val"
DATASET_MANUAL_TEST = BASE_DIR / "dataset" / "test"

AUTO_TAGGED_LABELS = BASE_DIR / "auto_tagged_labels"
DATA_IMAGES = BASE_DIR / "data"

# Destino
DEST_ROOT = BASE_DIR / "dataset_expanded"

# Proporciones
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1


def load_allowed_auto_stems(qc_csv_path: Path, max_severity: int):
    """Carga stems permitidos desde CSV de control de calidad."""
    if not qc_csv_path.exists():
        raise FileNotFoundError(f"No existe el CSV de QC: {qc_csv_path}")

    allowed = set()
    with open(qc_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                severity = int(row.get("severity", "999"))
            except ValueError:
                continue
            if severity <= max_severity:
                allowed.add(row["image_stem"])

    return allowed


def collect_all_samples(allowed_auto_stems=None):
    """Recolecta tuplas (image_path, label_path) de todas las fuentes."""
    samples = []
    
    # 1. Recolectar del dataset manual existente
    print("📂 Recolectando dataset manual existente...")
    for split_dir in [DATASET_MANUAL_TRAIN, DATASET_MANUAL_VAL, DATASET_MANUAL_TEST]:
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        if img_dir.exists():
            for img_file in img_dir.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    base_name = img_file.stem
                    lbl_file = lbl_dir / f"{base_name}.txt"
                    if lbl_file.exists():
                        samples.append((img_file, lbl_file))
                        print(f"  ✓ {img_file.name}")
    
    print(f"\n  Total dataset manual: {len(samples)} pares\n")
    
    # 2. Recolectar del auto_tagged_labels
    print("📂 Recolectando labels automáticos...")
    auto_count = 0
    auto_skipped = 0
    for lbl_file in AUTO_TAGGED_LABELS.glob("*.txt"):
        img_base = lbl_file.stem

        if allowed_auto_stems is not None and img_base not in allowed_auto_stems:
            auto_skipped += 1
            continue

        # Buscar imagen en data/
        img_candidates = list(DATA_IMAGES.glob(f"{img_base}.*"))
        if img_candidates:
            img_file = img_candidates[0]
            if img_file.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                samples.append((img_file, lbl_file))
                auto_count += 1
                if auto_count <= 5:
                    print(f"  ✓ {img_file.name}")
                elif auto_count == 6:
                    print(f"  ... ({len(list(AUTO_TAGGED_LABELS.glob('*.txt'))) - auto_count} más)")
    
    print(f"\n  Total labels automáticos encontrados: {auto_count}")
    if allowed_auto_stems is not None:
        print(f"  Labels automáticos descartados por QC: {auto_skipped}")
    print(f"  Total combinado: {len(samples)} pares\n")
    
    return samples


def create_expanded_dataset(samples):
    """Crea dataset_expanded/ con splits 70/20/10."""
    
    # Mezclar aleatoriamente
    random.seed(42)
    random.shuffle(samples)
    
    # Calcular cortes
    total = len(samples)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits = {
        'train': samples[:train_end],
        'val': samples[train_end:val_end],
        'test': samples[val_end:]
    }
    
    print(f"📊 Splits generados:")
    for split_name, split_samples in splits.items():
        print(f"   {split_name.upper()}: {len(split_samples)} imágenes ({len(split_samples)/total*100:.1f}%)")
    
    print(f"\n📁 Creando directorio {DEST_ROOT}...\n")
    
    # Crear directorios y copiar archivos
    for split_name, split_samples in splits.items():
        img_dest = DEST_ROOT / split_name / "images"
        lbl_dest = DEST_ROOT / split_name / "labels"
        
        img_dest.mkdir(parents=True, exist_ok=True)
        lbl_dest.mkdir(parents=True, exist_ok=True)
        
        for idx, (img_src, lbl_src) in enumerate(split_samples):
            # Copiar imagen
            img_dst = img_dest / img_src.name
            shutil.copy2(img_src, img_dst)
            
            # Copiar label
            lbl_dst = lbl_dest / lbl_src.name
            shutil.copy2(lbl_src, lbl_dst)
            
            if (idx + 1) % max(1, len(split_samples) // 5) == 0:
                print(f"  {split_name}: {idx + 1}/{len(split_samples)}")
    
    # Crear data.yaml
    data_yaml = {
        'path': str(DEST_ROOT.absolute()).replace("\\", "/"),
        'train': str((DEST_ROOT / "train/images").absolute()).replace("\\", "/"),
        'val': str((DEST_ROOT / "val/images").absolute()).replace("\\", "/"),
        'test': str((DEST_ROOT / "test/images").absolute()).replace("\\", "/"),
        'nc': 6,
        'names': ['Cadera Der', 'Cadera Izq', 'Rodilla Der', 'Rodilla Izq', 'Tobillo Der', 'Tobillo Izq']
    }
    
    yaml_path = DEST_ROOT / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✅ Dataset expandido creado en: {DEST_ROOT}")
    print(f"   data.yaml: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset expandido combinando manual + auto labels")
    parser.add_argument(
        "--qc-csv",
        type=Path,
        default=BASE_DIR / "reports" / "auto_label_qc" / "auto_labels_qc_details.csv",
        help="CSV de calidad generado por scripts/34_verify_auto_tagged_labels.py",
    )
    parser.add_argument(
        "--max-severity",
        type=int,
        default=None,
        help="Si se define, solo se incluyen auto labels con severity <= este valor",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PREPARACIÓN DE DATASET EXPANDIDO")
    print("=" * 60 + "\n")

    allowed_auto_stems = None
    if args.max_severity is not None:
        allowed_auto_stems = load_allowed_auto_stems(args.qc_csv, args.max_severity)
        print(f"Filtro QC activo: severity <= {args.max_severity}")
        print(f"Auto labels permitidos por QC: {len(allowed_auto_stems)}\n")

    samples = collect_all_samples(allowed_auto_stems=allowed_auto_stems)
    create_expanded_dataset(samples)
    
    print("\n🎯 Próximo paso: entrenar con scripts/2c_train_yolov8s_expanded.py")
