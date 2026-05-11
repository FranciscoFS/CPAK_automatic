import csv
from pathlib import Path

# Configuración
BASE_DIR = Path(__file__).resolve().parents[1]
BENCHMARK_FILE = BASE_DIR / "benchmark_results.csv"

def evaluate_knees_only():
    if not BENCHMARK_FILE.exists():
        print("Error: No se encontró benchmark_results.csv. Ejecuta el Script 7 primero.")
        return

    knee_ious = []
    
    with open(BENCHMARK_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['class'] == 'Rodilla Der' or row['class'] == 'Rodilla Izq':
                knee_ious.append(float(row['iou']))
    
    if not knee_ious:
        print("No se encontraron resultados de rodillas en el benchmark.")
        return
        
    avg_iou = sum(knee_ious) / len(knee_ious)
    print(f"--- Evaluación Específica: Rodillas ---")
    print(f"Muestras analizadas: {len(knee_ious)}")
    print(f"IoU Promedio (Rodillas): {avg_iou:.4f}")
    
    # Filtrar malos casos (IoU < 0.7)
    bad_knees = [iou for iou in knee_ious if iou < 0.7]
    print(f"Casos con IoU < 0.7: {len(bad_knees)}")

if __name__ == "__main__":
    evaluate_knees_only()
