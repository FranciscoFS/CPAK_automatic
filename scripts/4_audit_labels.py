import os
from pathlib import Path
from collections import Counter

# Configuración
LBL_DIR = Path("D:/Proyectos/TeleRx/auto_tagged_labels")
CLASS_NAMES = ['Cadera Der', 'Cadera IZq', 'Rodilla Der', 'Rodilla Izq', 'Tobillo DEr', 'Tobillo Izq']

import os
import csv
from pathlib import Path
from collections import Counter

# Configuración
LBL_DIR = Path("D:/Proyectos/TeleRx/auto_tagged_labels")
CLASS_NAMES = ['Cadera Der', 'Cadera IZq', 'Rodilla Der', 'Rodilla Izq', 'Tobillo DEr', 'Tobillo Izq']
CSV_OUTPUT = Path("D:/Proyectos/TeleRx/consolidated_audit.csv")

def audit_labels():
    if not LBL_DIR.exists():
        print(f"Error: No se encuentra la carpeta {LBL_DIR}")
        return

    txt_files = [f for f in os.listdir(LBL_DIR) if f.endswith('.txt')]
    total_files = len(txt_files)
    
    if total_files == 0:
        print("No se encontraron archivos de etiquetas para auditar.")
        return

    stats_counts = Counter()
    stats_origins = Counter()
    incomplete_images = []
    csv_data = []

    print(f"--- Reporte de Auditoría Pro: {total_files} imágenes procesadas ---")
    
    for txt_file in txt_files:
        base_name = Path(txt_file).stem
        meta_file = LBL_DIR / f"{base_name}.meta"
        
        # Leer TXT
        with open(LBL_DIR / txt_file, 'r') as f:
            lines = f.readlines()
            num_labels = len(lines)
            stats_counts[num_labels] += 1
            if num_labels < 6:
                present_classes = [int(line.split()[0]) for line in lines]
                missing = [CLASS_NAMES[i] for i in range(6) if i not in present_classes]
                incomplete_images.append((txt_file, num_labels, missing))

        # Leer META y preparar datos para CSV
        if meta_file.exists():
            with open(meta_file, 'r') as m:
                for line in m:
                    parts = line.split()
                    if len(parts) >= 3:
                        cls_id = int(parts[0])
                        conf = float(parts[1])
                        origin = parts[2]
                        stats_origins[origin] += 1
                        csv_data.append({
                            'img_name': base_name,
                            'class_name': CLASS_NAMES[cls_id],
                            'confidence': conf,
                            'origin': 'Detection' if origin == 'D' else 'Symmetry'
                        })

    # Guardar CSV
    with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['img_name', 'class_name', 'confidence', 'origin'])
        writer.writeheader()
        writer.writerows(csv_data)

    # Mostrar resumen en consola
    print("\nResumen de etiquetas por imagen:")
    for count in sorted(stats_counts.keys(), reverse=True):
        print(f"  - {count} etiquetas: {stats_counts[count]} imágenes")

    print("\nResumen de origen de etiquetas:")
    total_labels = sum(stats_origins.values())
    if total_labels > 0:
        d_pct = (stats_origins['D'] / total_labels) * 100
        s_pct = (stats_origins['S'] / total_labels) * 100
        print(f"  - Detecciones Reales (D): {stats_origins['D']} ({d_pct:.1f}%)")
        print(f"  - Generadas por Simetría (S): {stats_origins['S']} ({s_pct:.1f}%)")

    print(f"\n¡CSV consolidado generado en: {CSV_OUTPUT}")
    print(f"Reporte de texto generado en: D:/Proyectos/TeleRx/audit_report.txt")

    if incomplete_images:
        print("\n--- Imágenes Incompletas (menos de 6 etiquetas) ---")
        # Mostrar las primeras 20 para no saturar
        for name, count, missing in incomplete_images[:20]:
            print(f"  - {name}: {count}/6 (Falta: {', '.join(missing)})")
        
        if len(incomplete_images) > 20:
            print(f"  ... y {len(incomplete_images) - 20} imágenes más.")

    # Guardar reporte en un TXT para el usuario
    with open("D:/Proyectos/TeleRx/audit_report.txt", "w") as r:
        r.write(f"Reporte de Auditoría TeleRx\n")
        r.write(f"Total imágenes: {total_files}\n\n")
        for name, count, missing in incomplete_images:
            r.write(f"{name}: {count}/6 - Faltan: {', '.join(missing)}\n")

    print(f"\nReporte detallado guardado en: D:/Proyectos/TeleRx/audit_report.txt")

if __name__ == "__main__":
    audit_labels()
