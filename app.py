"""
Streamlit demo — upload an image (or video) and see detected plates + OCR
text in the browser.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

from src.ocr import get_ocr_engine
from src.utils import crop_with_margin, draw_detection, load_config


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


@st.cache_resource
def load_ocr(_cfg: dict):
    return get_ocr_engine(_cfg)


def main() -> None:
    st.set_page_config(page_title="License Plate Detector", page_icon="🚘", layout="centered")
    st.title("🚘 License Plate Detection & Recognition")
    st.caption("YOLOv8 detection + EasyOCR text recognition")

    cfg = load_config("config.yaml")

    with st.sidebar:
        st.header("Settings")
        weights_path = st.text_input("Weights path", value=cfg["detection"]["weights"])
        conf = st.slider("Confidence threshold", 0.05, 0.95, cfg["detection"]["conf_threshold"], 0.05)
        run_ocr = st.checkbox("Run OCR on detected plates", value=True)

    if not Path(weights_path).exists():
        st.warning(
            f"Weights not found at `{weights_path}`. Falling back to the generic "
            f"`{cfg['detection']['base_model']}` checkpoint — train your own model "
            f"with `python src/train.py` for real plate detection."
        )
        weights_path = cfg["detection"]["base_model"]

    model = load_model(weights_path)
    ocr = load_ocr(cfg) if run_ocr else None

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tmp.write(uploaded.read())
        tmp.close()

        frame = cv2.imread(tmp.name)
        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)

        with st.spinner("Detecting plates..."):
            results = model.predict(frame, conf=conf, verbose=False)[0]

            rows = []
            annotated = frame.copy()
            for box in results.boxes:
                xyxy = box.xyxy[0].tolist()
                det_conf = float(box.conf[0])

                plate_text, ocr_conf = "", 0.0
                if ocr is not None:
                    crop = crop_with_margin(frame, xyxy)
                    plate_text, ocr_conf = ocr.read(crop)

                draw_detection(annotated, xyxy, plate_text or "plate", det_conf)
                rows.append(
                    {
                        "plate_text": plate_text or "—",
                        "detection_confidence": round(det_conf, 3),
                        "ocr_confidence": round(ocr_conf, 3),
                    }
                )

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detections", use_container_width=True)

        if rows:
            st.subheader(f"Found {len(rows)} plate(s)")
            st.dataframe(rows, use_container_width=True)
        else:
            st.info("No plates detected above the confidence threshold.")


if __name__ == "__main__":
    main()
