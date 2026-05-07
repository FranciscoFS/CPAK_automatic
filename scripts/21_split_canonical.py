import os
import shutil
import random
from pathlib import Path
import yaml

# Rutas
CANONICAL_DIR = Path("D:/Proyectos/TeleRx/dataset_canonical")
POSE_DATASET_DIR = Path("D:/Proyectos/TeleRx/dataset_pose_final")

def prepare_final_dataset():
    os.makedirs(POSE_DATASET_DIR, exist_ok=True)
    
    # Obtener imágenes (todas son .jpg ahora)
    files = [f for f in os.listdir(CANONICAL_DIR) if f.endswith(".jpg")]
    pairs = [(f, f.replace(".jpg", ".txt")) for f in files if os.path.exists(CANONICAL_DIR / f.replace(".jpg", ".txt"))]

    print(f"Organizando {len(pairs)} imágenes canonicalizadas...")

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
            shutil.copy(CANONICAL_DIR / img, img_dest / img)
            shutil.copy(CANONICAL_DIR / lbl, lbl_dest / lbl)
            
    # data.yaml
    yaml_data = {
        'path': str(POSE_DATASET_DIR.absolute()).replace("\\", "/"),
        'train': 'train/images',
        'val': 'val/images',
        'kpt_shape': [8, 3],
        'flip_idx': [0, 1, 2, 4, 3, 6, 5, 7],
        'nc': 3,
        'names': {0: 'Hip', 1: 'Knee', 2: 'Ankle'}
    }
    
    with open(POSE_DATASET_DIR / "data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    print(f"Dataset listo en {POSE_DATASET_DIR}")

if __name__ == "__main__":
    prepare_final_dataset()
