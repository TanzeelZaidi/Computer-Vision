"""
Helper to pull a license-plate dataset from Roboflow Universe in YOLOv8 format.

Roboflow hosts several ready-made, labeled license-plate datasets. This script
just wraps the roboflow SDK so the rest of the project has a consistent
data/images + data/labels layout to train against.

Usage:
    export ROBOFLOW_API_KEY=your_key_here      # free at https://roboflow.com
    python scripts/download_dataset.py --workspace <ws> --project <proj> --version 1

If you'd rather use your own dataset, skip this script entirely and just
arrange your own images/labels to match data/dataset.yaml, or export any
Roboflow project directly as "YOLOv8" format and drop it into data/.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a YOLOv8-format license plate dataset")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace slug")
    parser.add_argument("--project", required=True, help="Roboflow project slug")
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--dest", default="data", help="Destination directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit(
            "Set ROBOFLOW_API_KEY in your environment first. "
            "Get a free key at https://roboflow.com after creating an account."
        )

    try:
        from roboflow import Roboflow
    except ImportError as e:
        raise SystemExit(
            "roboflow package not installed. Run: pip install roboflow"
        ) from e

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset = project.version(args.version).download("yolov8")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        src_images = Path(dataset.location) / split / "images"
        src_labels = Path(dataset.location) / split / "labels"
        split_name = "val" if split == "valid" else split

        if src_images.exists():
            shutil.copytree(src_images, dest / "images" / split_name, dirs_exist_ok=True)
        if src_labels.exists():
            shutil.copytree(src_labels, dest / "labels" / split_name, dirs_exist_ok=True)

    print(f"[download_dataset] Dataset arranged under: {dest.resolve()}")
    print("[download_dataset] Update data/dataset.yaml 'path' if you used a custom --dest.")


if __name__ == "__main__":
    main()
