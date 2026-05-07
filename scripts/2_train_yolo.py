from ultralytics import YOLO
import os

# Configuración
DATA_YAML = "D:/Proyectos/TeleRx/dataset/data.yaml"
MODEL_NAME = "yolov8n.pt"  # O "yolo11n.pt" si prefieres el último
EPOCHS = 100
IMG_SIZE = 640

def train_model():
    print(f"Cargando modelo base {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    print(f"Iniciando entrenamiento en {DATA_YAML}...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=0,  # Usa 'auto' para detectar GPU o CPU automáticamente
        project="D:/Proyectos/TeleRx/training_runs",
        name="telerx_yolov8n",
        exist_ok=True
    )
    
    print("Entrenamiento completado.")
    print(f"Mejor modelo guardado en: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    # Asegurarse de que el directorio de entrenamiento existe
    os.makedirs("D:/Proyectos/TeleRx/training_runs", exist_ok=True)
    train_model()
