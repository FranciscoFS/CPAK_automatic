import argparse
import importlib.util
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DET_MODEL = BASE_DIR / "training_runs" / "telerx_yolo26s_768_b12-4" / "weights" / "best.pt"
DEFAULT_POSE_MODEL = BASE_DIR / "training_runs" / "telerx_pose_v2_ft" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "visualizations" / "final_pipeline_batch"


def parse_args():
    parser = argparse.ArgumentParser(
        description="CPAK batch pipeline: detection batch + pose batch on crops + per-image clinical metrics."
    )
    parser.add_argument("--source", required=True, help="Path to image file or folder.")
    parser.add_argument("--det-model", default=str(DEFAULT_DET_MODEL), help="Detection model path.")
    parser.add_argument("--pose-model", default=str(DEFAULT_POSE_MODEL), help="Pose model path.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output dir for JSON/overlay.")
    parser.add_argument("--det-conf", type=float, default=0.20, help="Detection confidence threshold.")
    parser.add_argument("--pose-conf", type=float, default=0.20, help="Pose confidence threshold.")
    parser.add_argument("--padding", type=float, default=0.15, help="Extra box padding before pose.")
    parser.add_argument("--det-batch-size", type=int, default=16, help="Batch size for detection stage.")
    parser.add_argument("--pose-batch-size", type=int, default=64, help="Batch size for pose stage (crops).")
    parser.add_argument("--json-only", action="store_true", help="Save only JSON and skip overlays.")
    parser.add_argument("--timing", action="store_true", help="Record and report per-image processing time.")
    parser.add_argument(
        "--render-from-json",
        default=None,
        help="Carpeta con JSONs ya generados. Renderiza overlays directamente sin correr inferencia.",
    )
    return parser.parse_args()


def load_base_pipeline_module():
    module_path = Path(__file__).with_name("13_final_inference.py")
    spec = importlib.util.spec_from_file_location("final_inference_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load base pipeline module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_joint_points_from_pose_result(pose_result, joint_name: str, pose_visible_indices: dict):
    if pose_result.keypoints is None or pose_result.keypoints.xy is None or len(pose_result.keypoints.xy) == 0:
        return {}

    pred_xy = pose_result.keypoints.xy[0].cpu().numpy()
    points = {}
    for idx in pose_visible_indices[joint_name]:
        px = float(pred_xy[idx][0])
        py = float(pred_xy[idx][1])
        if px > 1 or py > 1:
            points[idx] = (px, py)
    return points


def build_image_context(image_path: Path, image, detections: dict):
    sides = {"Der": {}, "Izq": {}}
    for det in detections.values():
        if det["side"] in sides:
            sides[det["side"]][det["joint"]] = det

    payload = {"image": str(image_path), "detections": {}, "sides": {}}
    for cls_id, det in detections.items():
        payload["detections"][str(cls_id)] = {
            "joint": det["joint"],
            "side": det["side"],
            "confidence": round(det["confidence"], 4),
            "xyxy": [round(v, 2) for v in det["xyxy"]],
            "reflected": det.get("reflected", False),
        }

    return {
        "image_path": image_path,
        "image": image,
        "sides": sides,
        "payload": payload,
        "side_points": {"Der": {}, "Izq": {}},
        "side_crops": {"Der": {}, "Izq": {}},
    }


def render_from_json_dir(json_dir: Path, output_dir: Path, base):
    """Renderiza overlays directamente desde JSONs existentes, sin correr modelos."""
    jsons = sorted(json_dir.glob("*_final_metrics.json"))
    if not jsons:
        raise RuntimeError(f"No se encontraron JSONs en: {json_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for j in jsons:
        payload = json.loads(j.read_text(encoding="utf-8"))
        img_path = Path(payload.get("image", ""))
        if not img_path.exists():
            img_path = Path("data") / img_path.name
        if not img_path.exists():
            print(f"  [Saltar] imagen no encontrada: {j.name}")
            continue
        canvas = base.render_overlay_from_payload(img_path, payload)
        out_path = output_dir / f"{j.stem.replace('_final_metrics', '')}_final_overlay.jpg"
        cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        done += 1
        print(f"  Renderizado: {out_path.name}")
    print(f"\nTotal overlays generados: {done}/{len(jsons)}")


def main():
    args = parse_args()

    base = load_base_pipeline_module()

    # Modo render-only: solo genera overlays desde JSONs existentes.
    if args.render_from_json:
        json_dir = Path(args.render_from_json)
        output_dir = Path(args.output_dir)
        render_from_json_dir(json_dir, output_dir, base)
        return

    if args.det_batch_size < 1 or args.pose_batch_size < 1:
        raise ValueError("det-batch-size and pose-batch-size must be >= 1")

    source = Path(args.source)
    det_model_path = Path(args.det_model)
    pose_model_path = Path(args.pose_model)
    output_dir = Path(args.output_dir)

    base.validate_model_path(det_model_path, "deteccion")
    base.validate_model_path(pose_model_path, "pose")

    images = base.collect_images(source)
    if not images:
        raise RuntimeError(f"No images found in: {source}")

    det_model = YOLO(str(det_model_path))
    pose_model = YOLO(str(pose_model_path))

    print(f"Detection model: {det_model_path}")
    print(f"Pose model: {pose_model_path}")
    print(f"Images to process: {len(images)}")
    print(f"Detection batch size: {args.det_batch_size}")
    print(f"Pose batch size: {args.pose_batch_size}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for batch_start in range(0, len(images), args.det_batch_size):
        batch_paths = images[batch_start : batch_start + args.det_batch_size]
        batch_sources = [str(p) for p in batch_paths]

        det_results = det_model.predict(source=batch_sources, conf=args.det_conf, verbose=False, device=0)

        contexts = []
        pose_requests = []

        for image_path, det_result in zip(batch_paths, det_results):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"\nSkipped (unreadable): {image_path.name}")
                continue

            detections = base.pick_best_detections(det_result)
            detections = base.validate_side_coherence(detections)
            detections = base.reflect_missing_detections(detections, image.shape[1])

            context = build_image_context(image_path, image, detections)
            context_idx = len(contexts)
            contexts.append(context)

            for side_name in ["Der", "Izq"]:
                side_detections = context["sides"][side_name]
                for joint_name in ["Cadera", "Rodilla", "Tobillo"]:
                    detection = side_detections.get(joint_name)
                    if detection is None:
                        continue

                    x1, y1, x2, y2 = base.padded_box(detection["xyxy"], image.shape, args.padding)
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    pose_requests.append(
                        {
                            "context_idx": context_idx,
                            "side_name": side_name,
                            "joint_name": joint_name,
                            "crop_box": [x1, y1, x2, y2],
                            "crop": crop,
                        }
                    )

        for pose_start in range(0, len(pose_requests), args.pose_batch_size):
            chunk = pose_requests[pose_start : pose_start + args.pose_batch_size]
            crop_sources = [req["crop"] for req in chunk]
            pose_results = pose_model.predict(source=crop_sources, conf=args.pose_conf, verbose=False, device=0)

            for req, pose_result in zip(chunk, pose_results):
                ctx = contexts[req["context_idx"]]
                side_name = req["side_name"]
                joint_name = req["joint_name"]
                x1, y1, _, _ = req["crop_box"]

                local_points = extract_joint_points_from_pose_result(
                    pose_result, joint_name, base.POSE_VISIBLE_INDICES
                )

                mapped_points = {}
                for idx, (px, py) in local_points.items():
                    gx = x1 + px
                    gy = y1 + py
                    mapped_points[idx] = (gx, gy)
                    ctx["side_points"][side_name][idx] = (gx, gy)

                ctx["side_crops"][side_name][joint_name] = {
                    "crop_box": req["crop_box"],
                    "keypoints": {
                        str(idx): [round(pt[0], 2), round(pt[1], 2)] for idx, pt in mapped_points.items()
                    },
                }

        timing_data = [] if args.timing else None

        for ctx in contexts:
            t_start = time.perf_counter() if args.timing else None
            image_path = ctx["image_path"]
            payload = ctx["payload"]

            for side_name in ["Der", "Izq"]:
                side_detections = ctx["sides"][side_name]
                points = ctx["side_points"][side_name]
                crops = ctx["side_crops"][side_name]
                metrics = base.calculate_metrics(points)
                payload["sides"][side_name] = {
                    "detections_found": sorted(side_detections.keys()),
                    "points": {str(idx): [round(pt[0], 2), round(pt[1], 2)] for idx, pt in points.items()},
                    "crops": crops,
                    "metrics": metrics,
                }

            if args.timing and t_start is not None:
                elapsed = round(time.perf_counter() - t_start, 3)
                payload["processing_time_s"] = elapsed
                timing_data.append((image_path.name, elapsed))

            overlay_path = None
            if not args.json_only:
                canvas = base.render_overlay_from_payload(image_path, payload)
                overlay_path = output_dir / f"{image_path.stem}_final_overlay.jpg"
                cv2.imwrite(str(overlay_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])

            json_path = output_dir / f"{image_path.stem}_final_metrics.json"
            json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"\nProcesada: {image_path.name}")
            if overlay_path is not None:
                print(f"  Overlay: {overlay_path}")
            print(f"  JSON: {json_path}")
            for side_name, side_info in payload["sides"].items():
                metrics = side_info["metrics"]
                if metrics.get("status") == "ok":
                    print(
                        f"  {side_name}: LDFA={metrics['LDFA']:.2f} MPTA={metrics['MPTA']:.2f} "
                        f"aHKA={metrics['aHKA']:.2f} JLO={metrics['JLO']:.2f} | {metrics['CPAK']}"
                    )
                else:
                    print(f"  {side_name}: sin calculo completo ({metrics})")

        if args.timing and timing_data:
            import statistics
            times = [t for _, t in timing_data]
            mean_t = statistics.mean(times)
            std_t = statistics.stdev(times) if len(times) > 1 else 0.0
            total_t = sum(times)
            print(f"\n--- Timing ---")
            print(f"Total: {total_t:.2f}s | Imagenes: {len(times)}")
            print(f"Promedio: {mean_t:.3f}s | Desv. est.: {std_t:.3f}s | Mediana: {statistics.median(times):.3f}s")
            print(f"Min: {min(times):.3f}s | Max: {max(times):.3f}s")


if __name__ == "__main__":
    main()
