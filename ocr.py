"""
Plate text recognition. Wraps EasyOCR with plate-specific preprocessing
(grayscale, contrast boost, upscaling) so small / angled plate crops read
more reliably than feeding EasyOCR the raw crop.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.utils import clean_plate_text


class PlateOCR:
    def __init__(
        self,
        languages: list[str] | None = None,
        allow_list: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-",
        min_confidence: float = 0.4,
        gpu: bool = True,
    ):
        import easyocr  # imported lazily so --help / training doesn't require it

        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)
        self.allow_list = allow_list
        self.min_confidence = min_confidence

    @staticmethod
    def _preprocess(crop: np.ndarray) -> np.ndarray:
        """Upscale + denoise + boost contrast on a plate crop before OCR."""
        if crop.size == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Upscale small crops — OCR accuracy drops sharply below ~100px wide.
        h, w = gray.shape[:2]
        if w < 300:
            scale = 300 / max(w, 1)
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray

    def read(self, crop: np.ndarray) -> tuple[str, float]:
        """
        Run OCR on a cropped plate image.
        Returns (cleaned_text, confidence). Returns ("", 0.0) if nothing usable is read.
        """
        if crop is None or crop.size == 0:
            return "", 0.0

        processed = self._preprocess(crop)

        results = self.reader.readtext(
            processed,
            allowlist=self.allow_list,
            detail=1,
            paragraph=False,
        )

        if not results:
            return "", 0.0

        # Concatenate all detected text fragments left-to-right (plates can
        # sometimes be split into two OCR fragments), weight confidence by length.
        results.sort(key=lambda r: r[0][0][0])  # sort by left x-coordinate of box
        texts, confs = [], []
        for _, text, conf in results:
            if conf >= self.min_confidence:
                texts.append(text)
                confs.append(conf)

        if not texts:
            return "", 0.0

        full_text = clean_plate_text("".join(texts))
        avg_conf = sum(confs) / len(confs)
        return full_text, avg_conf


def get_ocr_engine(cfg: dict) -> Optional[PlateOCR]:
    """Factory that builds the configured OCR engine, or None for 'tesseract'/'none'."""
    ocr_cfg = cfg.get("ocr", {})
    engine = ocr_cfg.get("engine", "easyocr")

    if engine == "easyocr":
        return PlateOCR(
            languages=ocr_cfg.get("languages", ["en"]),
            allow_list=ocr_cfg.get("allow_list", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"),
            min_confidence=ocr_cfg.get("min_confidence", 0.4),
            gpu=False,
        )
    if engine == "tesseract":
        raise NotImplementedError(
            "Tesseract backend not bundled by default. Install pytesseract + "
            "the Tesseract binary, then implement TesseractOCR similarly to PlateOCR."
        )
    return None
