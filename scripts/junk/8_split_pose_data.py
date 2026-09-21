import os
import shutil
import random
from pathlib import Path
import yaml

# Rutas
CROP_DIR = Path("D:/Proyectos/TeleRx/crops_for_keypoints")
POSE_DATASET_DIR = Path("D:/Proyectos/TeleRx/dataset_pose")

def prepare_pose_dataset():
    os.makedirs(POSE_DATASET_DIR, exist_ok=True)
    
    # Obtener imágenes tagueadas
    image_files = [f for f in os.listdir(CROP_DIR) if f.lower().endswith(('.jpg', '.png'))]
    tagged_pairs = []
    
    for img_file in image_files:
        stem = Path(img_file).stem
        lbl_file = f"{stem}.txt"
        if (CROP_DIR / lbl_file).exists():
            # Revisar que el archivo no esté vacío (saltamos si fue un error)
            with open(CROP_DIR / lbl_file, 'r') as f:
                if len(f.readlines()) > 0:
                    tagged_pairs.append((img_file, lbl_file))

    print(f"Se encontraron {len(tagged_pairs)} recortes tagueados.")
    if len(tagged_pairs) == 0:
        return

    random.seed(42)
    random.shuffle(tagged_pairs)
    
    # Split 80% train / 20% val
    train_end = int(len(tagged_pairs) * 0.8)
    splits = {
        'train': tagged_pairs[:train_end],
        'val': tagged_pairs[train_end:]
    }
    
    for split_name, pairs in splits.items():
        img_dest = POSE_DATASET_DIR / split_name / "images"
        lbl_dest = POSE_DATASET_DIR / split_name / "labels"
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        for img_f, lbl_f in pairs:
            shutil.copy(CROP_DIR / img_f, img_dest / img_f)
            shutil.copy(CROP_DIR / lbl_f, lbl_dest / lbl_f)
            
    # Crear data.yaml de Pose (según ESQUEMA.md)
    yaml_data = {
        'path': str(POSE_DATASET_DIR.absolute()).replace("\\", "/"),
        'train': 'train/images',
        'val': 'val/images',
        'kpt_shape': [8, 3], # 8 Puntos (Incluye Notch), 3 dimensiones (x, y, visibilidad)
        'flip_idx': [0, 1, 2, 4, 3, 6, 5, 7], # Mapeo de simetría (Medial <-> Lateral)
        'nc': 3,
        'names': {
            0: 'Hip_Crop',
            1: 'Knee_Crop',
            2: 'Ankle_Crop'
        }
    }
    
    with open(POSE_DATASET_DIR / "data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
    print(f"Dataset organizado en {POSE_DATASET_DIR}")

if __name__ == "__main__":
    prepare_pose_dataset()
