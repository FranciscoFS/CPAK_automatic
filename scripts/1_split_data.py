import os
import shutil
import random
from pathlib import Path
import yaml

# Configuración de rutas
BASE_DIR = Path("D:/Proyectos/TeleRx")
SOURCE_IMG_DIR = BASE_DIR / "CPAK.yolo26/train/images"
SOURCE_LBL_DIR = BASE_DIR / "CPAK.yolo26/train/labels"
DEST_ROOT = BASE_DIR / "dataset"

# Proporciones
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

def split_dataset():
    # Obtener todos los archivos de imagen
    image_files = [f for f in os.listdir(SOURCE_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Asegurarnos de que el nombre base coincida con la etiqueta
    valid_pairs = []
    for img_file in image_files:
        base_name = Path(img_file).stem
        lbl_file = f"{base_name}.txt"
        if os.path.exists(SOURCE_LBL_DIR / lbl_file):
            valid_pairs.append((img_file, lbl_file))
        else:
            print(f"Warning: No se encontró etiqueta para {img_file}")

    print(f"Total de pares válidos encontrados: {len(valid_pairs)}")
    
    # Mezclar aleatoriamente
    random.seed(42)
    random.shuffle(valid_pairs)
    
    # Calcular cortes
    total = len(valid_pairs)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)
    
    splits = {
        'train': valid_pairs[:train_end],
        'val': valid_pairs[train_end:val_end],
        'test': valid_pairs[val_end:]
    }
    
    # Crear directorios y copiar archivos
    for split_name, pairs in splits.items():
        img_dest = DEST_ROOT / split_name / "images"
        lbl_dest = DEST_ROOT / split_name / "labels"
        
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lbl_dest, exist_ok=True)
        
        print(f"Copiando {len(pairs)} archivos a {split_name}...")
        for img_file, lbl_file in pairs:
            shutil.copy(SOURCE_IMG_DIR / img_file, img_dest / img_file)
            shutil.copy(SOURCE_LBL_DIR / lbl_file, lbl_dest / lbl_file)

    # Crear data.yaml
    data_yaml = {
        'train': str((DEST_ROOT / "train/images").absolute()).replace("\\", "/"),
        'val': str((DEST_ROOT / "val/images").absolute()).replace("\\", "/"),
        'test': str((DEST_ROOT / "test/images").absolute()).replace("\\", "/"),
        'nc': 6,
        'names': ['Cadera Der', 'Cadera IZq', 'Rodilla Der', 'Rodilla Izq', 'Tobillo DEr', 'Tobillo Izq']
    }
    
    with open(DEST_ROOT / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
        
    print(f"Dataset organizado en {DEST_ROOT}")
    print(f"Archivo data.yaml creado.")

if __name__ == "__main__":
    split_dataset()
