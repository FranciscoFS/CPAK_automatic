from ultralytics import YOLO
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = str(BASE_DIR / "dataset_pose_final" / "data.yaml")
START_WEIGHTS = str(BASE_DIR / "training_runs" / "telerx_pose_s_rebuild1_ft_flipfix" / "weights" / "best.pt")

EPOCHS = 30
IMG_SIZE = 800
BATCH_SIZE = 16
RUN_NAME = "telerx_pose_s_rebuild1_ft_full_b16_flipfix"


def main():
    if not Path(START_WEIGHTS).exists():
        raise FileNotFoundError(f"No existe checkpoint inicial: {START_WEIGHTS}")

    print(f"Fine-tune desde: {START_WEIGHTS}")
    print(f"Dataset: {DATA_YAML}")

    model = YOLO(START_WEIGHTS)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=0,
        project=str(BASE_DIR / "training_runs"),
        name=RUN_NAME,
        exist_ok=True,
        # Mantener aumento horizontal, ahora con flip_idx correcto (identidad)
        fliplr=0.5,
        mosaic=1.0,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    print("Fine-tune completado")
    print(f"best: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    os.makedirs(BASE_DIR / "training_runs", exist_ok=True)
    main()
