"""
Fine-tune del modelo de pose sobre dataset_pose_v2 (943 + 791 = ~1734 imgs).
Parte desde el best.pt del entrenamiento anterior.

Uso:
    conda run -n physis_seg python scripts/29_finetune_pose_v2.py
"""
from pathlib import Path
from ultralytics import YOLO
import os
import torch

BASE_DIR     = Path(__file__).resolve().parents[1]
DATA_YAML    = str(BASE_DIR / "dataset_pose_v2" / "data.yaml")
START_WEIGHTS = str(BASE_DIR / "training_runs" / "telerx_pose_s_rebuild1_ft_full_b16_flipfix" / "weights" / "best.pt")

EPOCHS     = 40
IMG_SIZE   = 800
BATCH_SIZE = 16
RUN_NAME   = "telerx_pose_v2_ft"


def get_required_device() -> int:
    """Force GPU training on CUDA:0; fail fast if unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no disponible. Este script esta configurado para correr solo en GPU (device=0)."
        )
    if torch.cuda.device_count() < 1:
        raise RuntimeError(
            "No se detectaron GPUs CUDA. Este script requiere al menos una GPU (device=0)."
        )
    return 0


def main():
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Primero corre 28_prepare_pose_v2_dataset.py\nNo existe: {DATA_YAML}")
    if not Path(START_WEIGHTS).exists():
        raise FileNotFoundError(f"No existe checkpoint: {START_WEIGHTS}")

    print(f"Fine-tune desde : {START_WEIGHTS}")
    print(f"Dataset         : {DATA_YAML}")
    print(f"Épocas          : {EPOCHS}  img: {IMG_SIZE}  batch: {BATCH_SIZE}")

    device = get_required_device()
    print(f"Device          : CUDA:{device} ({torch.cuda.get_device_name(device)})")

    model = YOLO(START_WEIGHTS)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        project=str(BASE_DIR / "training_runs"),
        name=RUN_NAME,
        exist_ok=True,
        fliplr=0.5,
        mosaic=1.0,
        degrees=10.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        patience=20,
    )

    print("\nFine-tune completado.")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"Mejor modelo: {best_pt}")

    # ── Evaluación final sobre test set ───────────────────────────────────
    print("\nEvaluando sobre test set...")
    best_model = YOLO(str(best_pt))
    test_results = best_model.val(
        data=DATA_YAML,
        split="test",
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
    )
    print(f"Test  mAP50: {test_results.pose.map50:.4f}  mAP50-95: {test_results.pose.map:.4f}")


if __name__ == "__main__":
    os.makedirs(BASE_DIR / "training_runs", exist_ok=True)
    main()
