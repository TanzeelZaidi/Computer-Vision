# 🚘 License Plate Detection & Recognition

Real-time license plate **detection** (YOLOv8) and **text recognition** (EasyOCR) for images, videos, and live webcam feeds — with a training pipeline, CLI inference tool, and a Streamlit web demo.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/model-YOLOv8-purple)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/<your-username>/license-plate-detection/actions/workflows/ci.yml/badge.svg)

<p align="center">
  <img src="assets/demo.gif" alt="Demo" width="600"/>
  <br/>
  <em>Add your own demo.gif / demo.png to assets/ once you have output — see "Adding your own demo media" below.</em>
</p>

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training Your Own Model](#training-your-own-model)
- [Inference (CLI)](#inference-cli)
- [Web Demo (Streamlit)](#web-demo-streamlit)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Results](#results)
- [Testing & CI](#testing--ci)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [Deploying to GitHub](#deploying-to-github)
- [Citation / Acknowledgements](#citation--acknowledgements)
- [License](#license)

---

## Features

- **YOLOv8-based detection** — fast, accurate plate localization, trainable on any labeled plate dataset (custom or Roboflow).
- **EasyOCR text recognition** — reads the plate string out of each detected crop, with preprocessing (upscaling, CLAHE contrast, denoising) tuned for small/angled plates.
- **Three input modes** — single image, video file, folder of images, or a live webcam feed, all from one CLI script.
- **CSV logging** — every detected plate (text, confidence, bounding box, timestamp) is appended to `outputs/results.csv`.
- **Streamlit web app** — drag-and-drop an image in the browser and see detections + OCR results instantly.
- **Config-driven** — thresholds, model paths, and training hyperparameters live in one `config.yaml`.
- **CI-ready** — GitHub Actions workflow runs lint + unit tests on every push.

## Project Structure

```
license-plate-detection/
├── app.py                     # Streamlit web demo
├── config.yaml                # central configuration (thresholds, paths, hyperparams)
├── requirements.txt
├── LICENSE
├── README.md
├── .github/workflows/ci.yml   # lint + test on push/PR
├── data/
│   ├── dataset.yaml           # YOLO dataset definition (classes, paths)
│   └── images/ labels/        # train/val/test splits go here (YOLO format)
├── models/                    # trained weights land here (best.pt) — gitignored
├── runs/                      # YOLO training logs/checkpoints — gitignored
├── scripts/
│   └── download_dataset.py    # optional helper to pull a Roboflow dataset
├── src/
│   ├── detect.py              # main inference script (image/video/webcam)
│   ├── train.py                # YOLOv8 training script
│   ├── ocr.py                  # EasyOCR wrapper + preprocessing
│   └── utils.py                 # shared helpers (drawing, CSV logging, config)
└── tests/
    └── test_utils.py           # unit tests for src/utils.py
```

## How It Works

```
 Input (image / video / webcam)
            │
            ▼
   ┌─────────────────┐
   │   YOLOv8 model   │   src/detect.py → model.predict()
   │ (plate detector) │
   └────────┬────────┘
            │  bounding boxes + confidence
            ▼
   ┌─────────────────┐
   │  Crop + enhance  │   src/utils.crop_with_margin()
   │  each plate box  │   src/ocr.PlateOCR._preprocess()
   └────────┬────────┘
            │  cleaned plate crop
            ▼
   ┌─────────────────┐
   │     EasyOCR      │   src/ocr.py → reader.readtext()
   │ (text recognition)│
   └────────┬────────┘
            │  plate string + OCR confidence
            ▼
   Annotated frame + outputs/results.csv
```

## Installation

**Requirements:** Python 3.9+, pip, and (optionally) a CUDA-capable GPU for faster training/inference.

```bash
git clone https://github.com/<your-username>/license-plate-detection.git
cd license-plate-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** the first run of `src/ocr.py` will download EasyOCR's recognition weights (~100 MB) automatically. This requires an internet connection once, after which it's cached locally.

## Quick Start

There's no trained model bundled with this repo (weight files are gitignored — keep the repo light). You have two options:

**Option A — train your own** (recommended, see [Training](#training-your-own-model)):

```bash
python src/train.py --epochs 100
```

**Option B — run detection-only mode with a generic pretrained YOLOv8** while you don't yet have plate-specific weights. It will detect generic objects, not plates specifically — it's only there so the pipeline runs end-to-end immediately:

```bash
python src/detect.py --source assets/sample.jpg
```

Once trained, `models/best.pt` is used automatically:

```bash
python src/detect.py --source path/to/car.jpg
```

## Training Your Own Model

1. **Get a labeled dataset.** Two options:
   - Use `scripts/download_dataset.py` to pull a ready-made YOLOv8-format license-plate dataset from [Roboflow Universe](https://universe.roboflow.com/?q=license%20plate) (search "license plate detection", pick any YOLOv8-exportable project):
     ```bash
     export ROBOFLOW_API_KEY=your_key_here
     python scripts/download_dataset.py --workspace <workspace> --project <project> --version 1
     ```
   - Or arrange your own images + YOLO-format `.txt` labels under `data/images/{train,val}` and `data/labels/{train,val}` (see comments in `data/dataset.yaml` for the exact layout and label format).

2. **Train:**
   ```bash
   python src/train.py --epochs 100 --batch 16 --model yolov8n.pt
   ```
   - Swap `yolov8n.pt` → `yolov8s.pt` / `yolov8m.pt` for higher accuracy at the cost of speed.
   - All hyperparameters can instead be set once in `config.yaml` under `training:`.

3. **Result:** the best checkpoint is copied automatically to `models/best.pt`, which `detect.py` and `app.py` pick up by default. Full logs, metrics, and sample validation predictions are saved under `runs/train/lp_detector/`.

## Inference (CLI)

```bash
# Single image
python src/detect.py --source samples/car1.jpg

# Video file
python src/detect.py --source samples/traffic.mp4

# Folder of images (batch mode)
python src/detect.py --source samples/

# Live webcam
python src/detect.py --source 0

# Detection only, skip OCR (faster)
python src/detect.py --source samples/car1.jpg --no-ocr

# Custom weights / confidence threshold
python src/detect.py --source samples/car1.jpg --weights models/best.pt --conf 0.5

# Headless (no preview window, e.g. on a server)
python src/detect.py --source samples/traffic.mp4 --no-display
```

Outputs land in `outputs/`:
- Annotated image/video files
- `outputs/results.csv` — one row per detected plate: timestamp, source, frame index, plate text, detection confidence, OCR confidence, bounding box

## Web Demo (Streamlit)

```bash
streamlit run app.py
```

Opens a browser tab where you can upload an image, adjust the confidence threshold live, and see the annotated result plus a results table.

## Configuration

All tunable settings live in [`config.yaml`](config.yaml):

| Section      | Key               | Description                                      |
|--------------|-------------------|---------------------------------------------------|
| `detection`  | `weights`         | Path to trained weights (falls back to base model if missing) |
| `detection`  | `conf_threshold`  | Minimum detection confidence to keep a box        |
| `detection`  | `iou_threshold`   | NMS IoU threshold                                  |
| `training`   | `epochs`, `batch_size`, `img_size` | Training hyperparameters          |
| `training`   | `device`          | GPU index, or `"cpu"`                              |
| `ocr`        | `engine`          | `easyocr` (default) or `tesseract` (stub — implement in `src/ocr.py`) |
| `ocr`        | `min_confidence`  | Minimum OCR confidence to accept a text fragment   |
| `io`         | `output_dir`      | Where annotated outputs + CSV are saved            |

CLI flags (`--conf`, `--weights`, etc.) override the config file for a single run.

## Dataset

This project doesn't ship a dataset — plate images generally include real, identifiable vehicle/location data, so it's left to you to source one appropriate for your use case and jurisdiction. Good starting points:

- [Roboflow Universe — License Plate Recognition](https://universe.roboflow.com/?q=license%20plate) (many pre-labeled, YOLOv8-exportable options)
- [Kaggle — Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
- Your own footage, labeled with a tool like [LabelImg](https://github.com/heartexlabs/labelImg) or [Roboflow Annotate](https://roboflow.com/annotate) and exported in YOLOv8 format.

See `data/dataset.yaml` for the exact folder layout and label format expected by `src/train.py`.

## Results

_Fill this in after training on your dataset — this table is a placeholder showing what to report:_

| Metric        | Value   |
|---------------|---------|
| mAP@0.5       | —       |
| mAP@0.5:0.95  | —       |
| Precision     | —       |
| Recall        | —       |
| Inference FPS (GPU) | — |
| OCR accuracy (exact match) | — |

Training curves and validation predictions are auto-saved under `runs/train/lp_detector/` (e.g. `results.png`, `confusion_matrix.png`, `val_batch0_pred.jpg`) — pull the ones you want into this section or into `assets/`.

## Testing & CI

```bash
pytest tests/ -v
```

`.github/workflows/ci.yml` runs `flake8` linting and the test suite automatically on every push/PR to `main`.

## Roadmap

- [ ] Add a `Dockerfile` for one-command containerized deployment
- [ ] Multi-line / motorcycle plate support in the OCR post-processing
- [ ] Plate-format validation per country/region (regex rules in `config.yaml`)
- [ ] Export to ONNX/TensorRT for edge deployment
- [ ] FastAPI inference endpoint as an alternative to the Streamlit demo

## Troubleshooting

- **`FileNotFoundError: Config file not found`** — run scripts from the project root (`python src/detect.py ...`), not from inside `src/`.
- **EasyOCR is slow on CPU** — this is expected; pass `--no-ocr` for detection-only speed, or run on a GPU machine and set `gpu=True` in `src/ocr.py`.
- **No plates detected at all** — you're likely still on the generic base checkpoint. Train first with `src/train.py`, or lower `--conf`.
- **Webcam won't open (`--source 0`)** — try a different index (`1`, `2`, …) if you have multiple cameras, and confirm no other app is holding the camera.

## Deploying to GitHub

```bash
cd license-plate-detection
git init
git add .
git commit -m "Initial commit: license plate detection & recognition pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/license-plate-detection.git
git push -u origin main
```

Then, on GitHub:
1. Add a repo description + topics (`yolov8`, `computer-vision`, `ocr`, `license-plate-recognition`).
2. Replace `<your-username>` in this README's badges and clone URL.
3. (Optional) Enable GitHub Pages or a Streamlit Community Cloud deployment pointing at `app.py` for a live public demo link.
4. Drop a real `assets/demo.gif` in once you've run inference on a sample video, so the README preview isn't a placeholder.

## Citation / Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [JaidedAI EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [OpenCV](https://opencv.org/)

## License

Released under the [MIT License](LICENSE).
