"""
Run license plate detection (+ optional OCR) on an image, a video file,
a folder of images, or a live webcam feed.

Usage:
    python src/detect.py --source path/to/image.jpg
    python src/detect.py --source path/to/video.mp4
    python src/detect.py --source path/to/folder/
    python src/detect.py --source 0                     # webcam
    python src/detect.py --source video.mp4 --no-ocr     # detection only, faster
    python src/detect.py --source image.jpg --weights models/best.pt --conf 0.5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.ocr import get_ocr_engine
from src.utils import (
    CsvLogger,
    crop_with_margin,
    draw_detection,
    ensure_dir,
    is_image_file,
    is_video_file,
    load_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="License plate detection + OCR")
    parser.add_argument("--source", required=True,
                         help="Image path, video path, folder path, or webcam index (e.g. 0)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--weights", default=None, help="Override weights path from config")
    parser.add_argument("--conf", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR, only draw boxes")
    parser.add_argument("--no-display", action="store_true", help="Don't open a preview window")
    parser.add_argument("--output", default=None, help="Override output directory")
    return parser.parse_args()


def resolve_weights(weights_arg: str | None, cfg: dict) -> str:
    candidate = weights_arg or cfg["detection"]["weights"]
    if Path(candidate).exists():
        return candidate
    print(
        f"[detect] '{candidate}' not found — falling back to the pretrained "
        f"'{cfg['detection']['base_model']}' (generic object weights, NOT trained "
        f"on plates). Train first with src/train.py for real plate detection."
    )
    return cfg["detection"]["base_model"]


def process_frame(frame, model, ocr, conf_thresh, iou_thresh, csv_logger, source_name, frame_idx):
    """Detect plates in a single frame, run OCR, draw boxes. Returns annotated frame."""
    results = model.predict(frame, conf=conf_thresh, iou=iou_thresh, verbose=False)[0]

    for box in results.boxes:
        xyxy = box.xyxy[0].tolist()
        det_conf = float(box.conf[0])

        plate_text, ocr_conf = "", 0.0
        if ocr is not None:
            crop = crop_with_margin(frame, xyxy)
            plate_text, ocr_conf = ocr.read(crop)

        label = plate_text if plate_text else "plate"
        draw_detection(frame, xyxy, label, det_conf)

        if csv_logger is not None and plate_text:
            csv_logger.log(source_name, frame_idx, plate_text, det_conf, ocr_conf, xyxy)

    return frame


def run_on_image(path, model, ocr, cfg, csv_logger, out_dir, no_display):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[detect] Could not read image: {path}")
        return
    annotated = process_frame(
        frame, model, ocr,
        cfg["detection"]["conf_threshold"], cfg["detection"]["iou_threshold"],
        csv_logger, str(path), 0,
    )
    out_path = Path(out_dir) / f"{Path(path).stem}_annotated.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"[detect] Saved: {out_path}")

    if not no_display:
        cv2.imshow("License Plate Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video(source, model, ocr, cfg, csv_logger, out_dir, no_display):
    is_webcam = str(source).isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else str(source))
    if not cap.isOpened():
        print(f"[detect] Could not open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not is_webcam:
        out_path = Path(out_dir) / f"{Path(str(source)).stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    t0 = time.time()
    source_name = "webcam" if is_webcam else str(source)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            annotated = process_frame(
                frame, model, ocr,
                cfg["detection"]["conf_threshold"], cfg["detection"]["iou_threshold"],
                csv_logger, source_name, frame_idx,
            )

            if writer is not None:
                writer.write(annotated)

            if not no_display:
                cv2.imshow("License Plate Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"[detect] Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx / max(elapsed, 1e-6):.1f} fps)")
    if writer is not None:
        print(f"[detect] Saved annotated video to: {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.conf is not None:
        cfg["detection"]["conf_threshold"] = args.conf

    weights = resolve_weights(args.weights, cfg)
    model = YOLO(weights)

    ocr = None if args.no_ocr else get_ocr_engine(cfg)
    if ocr is None and not args.no_ocr:
        print("[detect] OCR engine unavailable — continuing with detection only.")

    out_dir = ensure_dir(args.output or cfg["io"]["output_dir"])
    csv_logger = CsvLogger(Path(out_dir) / "results.csv") if cfg["io"]["save_csv"] else None

    source = args.source
    src_path = Path(source)

    if source.isdigit():
        run_on_video(source, model, ocr, cfg, csv_logger, out_dir, args.no_display)
    elif src_path.is_dir():
        images = sorted([p for p in src_path.iterdir() if is_image_file(p)])
        print(f"[detect] Found {len(images)} images in {src_path}")
        for img_path in images:
            run_on_image(img_path, model, ocr, cfg, csv_logger, out_dir, no_display=True)
    elif is_video_file(src_path):
        run_on_video(source, model, ocr, cfg, csv_logger, out_dir, args.no_display)
    elif is_image_file(src_path):
        run_on_image(src_path, model, ocr, cfg, csv_logger, out_dir, args.no_display)
    else:
        raise ValueError(f"Unrecognized source type: {source}")


if __name__ == "__main__":
    main()
