from ultralytics import YOLO
import os

# Configuración
DATA_YAML = "D:/Proyectos/TeleRx/dataset_pose_final/data.yaml"
# Partimos de 0 con el modelo Nano de YOLO26
MODEL_NAME = "yolo26n-pose.pt" 
EPOCHS = 100
IMG_SIZE = 800 
BATCH_SIZE = 4 

def train_pose_model():
    print(f"Entrenando modelo de Pose desde cero: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    
    print(f"Iniciando entrenamiento con datos en: {DATA_YAML}")
    
    # Entrenar con configuración explícita de aumentaciones y parámetros de Pose
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,
        project="D:/Proyectos/TeleRx/training_runs",
        name="telerx_pose_from_scratch", 
        exist_ok=True,
        # Aumentaciones explicitas
        fliplr=0.5,        # Habilita el uso del flip_idx que definimos
        mosaic=1.0,        
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4
    )
    
    print("Entrenamiento completado.")
    print(f"Mejor modelo guardado en: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    os.makedirs("D:/Proyectos/TeleRx/training_runs", exist_ok=True)
    train_pose_model()
