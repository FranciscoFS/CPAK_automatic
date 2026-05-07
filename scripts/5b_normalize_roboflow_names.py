import os
import re
from pathlib import Path

# Configuración
DATASET_ROOT = Path("D:/Proyectos/TeleRx/dataset")
SUBDIRS = ['train', 'val', 'test']
FOLDERS = ['images', 'labels']

# Patrón para detectar el sufijo de Roboflow
# Busca "_jpg.rf.hash" o "_png.rf.hash" antes de la extensión
ROBOFLOW_PATTERN = re.compile(r'(_jpg\.rf\.[a-fA-F0-9a-zA-Z]+|_png\.rf\.[a-fA-F0-9a-zA-Z]+)')

def normalize_names():
    total_renamed = 0
    
    for subdir in SUBDIRS:
        for folder in FOLDERS:
            target_dir = DATASET_ROOT / subdir / folder
            if not target_dir.exists():
                continue
                
            print(f"Procesando {target_dir}...")
            
            for filename in os.listdir(target_dir):
                # Buscar si el patrón existe en el nombre del archivo
                if ROBOFLOW_PATTERN.search(filename):
                    # Reemplazar el patrón con una cadena vacía
                    new_filename = ROBOFLOW_PATTERN.sub('', filename)
                    
                    old_path = target_dir / filename
                    new_path = target_dir / new_filename
                    
                    # Evitar colisiones si el archivo ya existe
                    if new_path.exists():
                        print(f"Aviso: {new_filename} ya existe. Omitiendo {filename}")
                    else:
                        os.rename(old_path, new_path)
                        total_renamed += 1
                        
    print(f"\nNormalización completada. Se renombraron {total_renamed} archivos.")

if __name__ == "__main__":
    normalize_names()
