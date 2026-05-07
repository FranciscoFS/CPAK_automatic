import os
from pathlib import Path

# Configuración
CROP_DIR = Path("D:/Proyectos/TeleRx/crops_for_keypoints")
REQUIRED_COLUMNS = 29 # 1 (class) + 4 (box) + 8*3 (keypoints) = 29

def repair_labels():
    files = [f for f in os.listdir(CROP_DIR) if f.endswith(".txt")]
    fixed_count = 0
    
    print(f"Revisando {len(files)} archivos de etiquetas...")
    
    for f in files:
        path = CROP_DIR / f
        with open(path, 'r') as file:
            line = file.read().strip()
        
        parts = line.split()
        
        # Si tiene menos de 29 columnas, necesitamos rellenar
        if len(parts) < REQUIRED_COLUMNS:
            needed = REQUIRED_COLUMNS - len(parts)
            # Cada punto invisible es " 0.0 0.0 0"
            # Como cada punto son 3 valores (x, y, v), completamos
            padding = " 0.0 0.0 0" * (needed // 3)
            
            with open(path, 'w') as file:
                file.write(line + padding + "\n")
            fixed_count += 1
            
    print(f"Reparación completada. Se corrigieron {fixed_count} archivos.")
    print("Nota: Ahora todos los archivos tienen exactamente 29 columnas.")

if __name__ == "__main__":
    repair_labels()
