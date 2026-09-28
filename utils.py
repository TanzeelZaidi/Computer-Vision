"""
Shared utility functions: config loading, image drawing, plate text cleaning,
and CSV logging used across detect.py, train.py, ocr.py and app.py.
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the project YAML config into a plain dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_plate_text(raw_text: str) -> str:
    """
    Normalize OCR output into a plausible plate string:
    uppercase, strip whitespace, drop characters that never appear on plates.
    """
    if not raw_text:
        return ""
    text = raw_text.upper().strip()
    text = re.sub(r"[^A-Z0-9\-]", "", text)
    return text


def draw_detection(
    frame: np.ndarray,
    box: Iterable[float],
    label: str,
    confidence: float,
    color: tuple[int, int, int] = (0, 220, 90),
) -> np.ndarray:
    """Draw a single bounding box + label banner onto a frame (in place, returns frame)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"{label} {confidence:.2f}" if label else f"{confidence:.2f}"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    banner_y1 = max(0, y1 - th - baseline - 6)
    cv2.rectangle(frame, (x1, banner_y1), (x1 + tw + 6, y1), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 3, y1 - 5 if y1 - 5 > 0 else th),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def crop_with_margin(
    frame: np.ndarray, box: Iterable[float], margin: float = 0.08
) -> np.ndarray:
    """Crop a bounding box out of a frame with a small margin, clamped to image bounds."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * margin))
    y1 = max(0, int(y1 - bh * margin))
    x2 = min(w, int(x2 + bw * margin))
    y2 = min(h, int(y2 + bh * margin))
    return frame[y1:y2, x1:x2]


class CsvLogger:
    """Append detection + OCR results to a CSV file, creating headers on first write."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        ensure_dir(self.csv_path.parent)
        self._write_header_if_needed()

    def _write_header_if_needed(self) -> None:
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "source", "frame_index", "plate_text",
                     "det_confidence", "ocr_confidence", "x1", "y1", "x2", "y2"]
                )

    def log(
        self,
        source: str,
        frame_index: int,
        plate_text: str,
        det_confidence: float,
        ocr_confidence: float,
        box: Iterable[float],
    ) -> None:
        x1, y1, x2, y2 = box
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    source,
                    frame_index,
                    plate_text,
                    round(float(det_confidence), 4),
                    round(float(ocr_confidence), 4),
                    int(x1), int(y1), int(x2), int(y2),
                ]
            )


def is_image_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
