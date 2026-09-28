"""
Train a YOLOv8 license-plate detector.

Usage:
    python src/train.py
    python src/train.py --epochs 150 --batch 32 --model yolov8s.pt
    python src/train.py --data data/dataset.yaml --config config.yaml

The dataset must already be in YOLO format under data/images/{train,val}
and data/labels/{train,val} — see data/dataset.yaml and
scripts/download_dataset.py for how to obtain / arrange one.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 license plate detector")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", default="data/dataset.yaml", help="Path to YOLO dataset yaml")
    parser.add_argument("--model", default=None, help="Base checkpoint, e.g. yolov8n.pt / yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", default=None, help="'0' for GPU 0, or 'cpu'")
    parser.add_argument("--resume", action="store_true", help="Resume last interrupted run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    det_cfg = cfg["detection"]

    model_ckpt = args.model or det_cfg.get("base_model", "yolov8n.pt")
    model = YOLO(model_ckpt)

    results = model.train(
        data=args.data,
        epochs=args.epochs or train_cfg.get("epochs", 100),
        batch=args.batch or train_cfg.get("batch_size", 16),
        imgsz=args.imgsz or train_cfg.get("img_size", 640),
        patience=train_cfg.get("patience", 20),
        device=args.device if args.device is not None else train_cfg.get("device", 0),
        project=train_cfg.get("project", "runs/train"),
        name=train_cfg.get("name", "lp_detector"),
        resume=args.resume,
    )

    # Copy the best weights to the location detect.py / app.py expect by default.
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    target = Path(det_cfg.get("weights", "models/best.pt"))
    target.parent.mkdir(parents=True, exist_ok=True)

    if best_path.exists():
        import shutil

        shutil.copy(best_path, target)
        print(f"\nBest weights copied to: {target}")
    else:
        print(f"\nWARNING: expected weights at {best_path} but none were found.")

    print(f"Full training run saved under: {results.save_dir}")


if __name__ == "__main__":
    main()
