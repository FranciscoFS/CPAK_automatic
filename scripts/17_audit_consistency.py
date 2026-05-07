import os
from pathlib import Path

# Configuración
CROP_DIR = Path("D:/Proyectos/TeleRx/crops_for_keypoints")
POSE_DIR = Path("D:/Proyectos/TeleRx/dataset_pose")

def audit_consistency():
    # Encontrar algunos archivos de rodilla en el dataset pose
    val_labels = POSE_DIR / "val" / "labels"
    files = [f for f in os.listdir(val_labels) if "Rodilla" in f]
    
    print(f"Comparando {len(files)} rodillas en dataset_pose contra originales...")
    
    for f in files:
        # El archivo en dataset_pose ahora se llama todo 'Der' (debido a la normalización)
        # Necesitamos buscar su origen en CROP_DIR. 
        # Si el archivo original era 'Rodilla_Izq.txt', el script 12 lo voltea y renombra a 'Rodilla_Der.txt'
        # Entonces el 'stem' base original es el mismo.
        
        orig_lbl = CROP_DIR / f
        pose_lbl = val_labels / f
        
        if not orig_lbl.exists():
            continue
            
        with open(orig_lbl, 'r') as f1, open(pose_lbl, 'r') as f2:
            l1 = f1.readline().split()
            l2 = f2.readline().split()
            
            # Comparar el lado medial (idx 9,10,11) y lateral (idx 12,13,14)
            # En el original: Medial=3, Lateral=4
            m_orig = float(l1[9])
            l_orig = float(l1[12])
            
            # En el pose (ya normalizado): Medial=3, Lateral=4
            m_pose = float(l2[9])
            l_pose = float(l2[12])
            
            # Si la original era Izquierda, fue volteada. 
            # Entonces el medial original (lado derecho de la imagen) 
            # al voltearse debe quedar en el lado izquierdo (x menor)
            
            print(f"\nArchivo: {f}")
            print(f"  Orig: MedialX={m_orig:.3f}, LateralX={l_orig:.3f}")
            print(f"  Pose: MedialX={m_pose:.3f}, LateralX={l_pose:.3f}")

if __name__ == "__main__":
    audit_consistency()
