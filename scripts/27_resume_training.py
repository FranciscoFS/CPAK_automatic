"""
Resume training for telerx_pose_s_rebuild1_ft_full_b16_flipfix from last.pt.

Uses workers=0 to avoid Windows multiprocessing deadlock after a CUDA crash.
Sets CUDA_LAUNCH_BLOCKING=1 for cleaner error messages if crash recurs.
"""
import os

# Must be set before importing torch/ultralytics
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[1]
LAST_PT = BASE_DIR / "training_runs" / "telerx_pose_s_rebuild1_ft_full_b16_flipfix" / "weights" / "last.pt"


def main():
    if not LAST_PT.exists():
        raise FileNotFoundError(f"No existe last.pt: {LAST_PT}")

    print(f"Resumiendo desde: {LAST_PT}")

    model = YOLO(str(LAST_PT))

    # resume=True reuses the checkpoint's optimizer/scheduler state and epoch count.
    # workers=0 is the key fix for the Windows multiprocessing deadlock after a CUDA crash.
    # Remaining overrides are forwarded on top of the checkpoint's saved args.
    results = model.train(
        resume=True,
        workers=0,
    )

    print("Entrenamiento completado.")
    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"best.pt: {best}")


if __name__ == "__main__":
    main()
