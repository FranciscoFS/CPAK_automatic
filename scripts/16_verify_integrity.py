import os
from pathlib import Path

# Configuración
BASE_DIR = Path(__file__).resolve().parents[1]
ORIGINAL_DATA_DIR = BASE_DIR / "dataset"
CANONICAL_DIR = BASE_DIR / "dataset_pose_final"

def verify_integrity():
    # Buscamos un par de archivos para comparar si la "lógica de lado" se mantiene
    # Compararemos si un 'Rodilla_Der.txt' original se mapeó correctamente a 'Rodilla_Der.txt' en pose
    
    # Vamos a tomar un par de ejemplos aleatorios
    train_dir = CANONICAL_DIR / "train" / "labels"
    for f in os.listdir(train_dir)[:5]:
        print(f"\nAnalizando archivo: {f}")
        # En el dataset original (Script 1), renombramos todo a nombres sin sufijos
        # Vamos a ver qué dice el archivo nuevo
        with open(train_dir / f, 'r') as file:
            line = file.readline().split()
            # La línea YOLO pose tiene: id, x, y, w, h, p0x, p0y, p0v, p1x, p1y, p1v ...
            # Tu esqueleto: 
            # 1: Espina, 3: Medial, 4: Lateral, 5: PlatMed, 6: PlatLat, 7: Notch
            kpts = line[5:]
            # Medial (ID 3) vs Lateral (ID 4)
            # Medial es el punto 3 (índices 9,10,11 en el array kpts)
            # Lateral es el punto 4 (índices 12,13,14 en el array kpts)
            medial_x = float(kpts[9])
            lateral_x = float(kpts[12])
            
            print(f"  Medial X: {medial_x:.4f}")
            print(f"  Lateral X: {lateral_x:.4f}")
            
            if "Der" in f:
                if medial_x < lateral_x:
                    print("  [OK] Rodilla Derecha: Medial a la izquierda (menor X)")
                else:
                    print("  [ERROR] Rodilla Derecha: Medial a la derecha (mayor X) - ¡Está invertido!")
            elif "Izq" in f: # Aunque ya no deberías tener Izq, por si acaso
                if medial_x > lateral_x:
                    print("  [OK] Rodilla Izquierda: Medial a la derecha (mayor X)")
                else:
                    print("  [ERROR] Rodilla Izquierda: Medial a la izquierda (menor X) - ¡Está invertido!")

if __name__ == "__main__":
    verify_integrity()
