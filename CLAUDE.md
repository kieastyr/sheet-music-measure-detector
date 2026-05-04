# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLOv8-based pipeline to detect measures in sheet music PDFs. Detects two classes: **systems** (class 0, full-width staff groups) and **barlines** (class 1, vertical lines between measures). Measures are then derived by intersecting system rows with barline columns.

## Common Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Run inference on a PDF
python main.py path/to/sheet.pdf --model runs/detect/train4/weights/best.pt

# Train the model
python train.py

# Data preparation pipeline (run in order):
python scripts/pdf_to_images.py path/to/sheet.pdf -o datasets/measure-detection/images/train
python scripts/auto_label2.py        # Auto-generate YOLO labels via CV
python scripts/split_data.py         # Move 20% of labeled data to val set
```

## Architecture

**Pipeline flow:**
1. `scripts/pdf_to_images.py` — converts PDF pages to PNG images at 300 DPI
2. `scripts/auto_label2.py` (or `label_structure.py`) — auto-generates YOLO `.txt` labels using classical CV (morphological ops to detect staff lines and barlines)
3. `scripts/split_data.py` — splits `datasets/measure-detection/images/train` into train/val (80/20)
4. `train.py` — trains YOLOv8n on `data.yaml`, uses MPS on Apple Silicon, exports to ONNX
5. `main.py` — inference: converts PDF page-by-page, runs YOLO, then uses `find_staves_in_system()` to further subdivide systems into individual staves, producing a measure grid

**Dataset structure** (`data.yaml`):
```
datasets/measure-detection/
  images/train/   # PNG pages
  images/val/
  labels/train/   # YOLO format .txt files (class cx cy w h, normalized)
  labels/val/
```

**Key inference logic in `main.py`:**
- Systems filtered at conf > 0.6 with IoU-based NMS
- Barlines filtered at conf > 0.2
- `find_staves_in_system()` uses morphological horizontal line detection to split a system box into individual stave rows
- Final measures are the grid cells: each (barline column) x (stave row) within a system
- Output debug images saved to `runs/predict/<pdf_stem>/`

**`scripts/auto_label2.py` vs `scripts/label_structure.py`:** These files are near-identical auto-labelers using the same CV approach (deskew → detect horizontal/vertical lines → group into staves/systems → write YOLO labels). `auto_label2.py` is the current version used for labeling.

## Dependencies

Python 3.14, `.venv` virtualenv. Key packages: `ultralytics` (YOLOv8), `opencv-python`, `pdf2image`, `torch`, `numpy`. Requires `poppler` system dependency for `pdf2image`.
