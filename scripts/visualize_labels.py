"""
Visualize dataset images with their YOLO label overlays.

Usage:
    python scripts/visualize_labels.py [--split raw|train|val]

Output: runs/debug_labels/<split>/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

DATASET_ROOT = Path("datasets/measure-detection")

CLASS_COLORS = {
    0: (255, 80, 80),  # system  — blue
    1: (80, 200, 80),  # staff   — green
    2: (80, 80, 255),  # barline — red
}
CLASS_NAMES = {0: "system", 1: "staff", 2: "barline"}


def draw_yolo_labels(img, label_path):
    h, w = img.shape[:2]
    debug = img.copy()

    if not label_path.exists():
        cv2.putText(
            debug, "NO LABEL", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3
        )
        return debug

    counts = {0: 0, 1: 0, 2: 0}
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, cx, cy, bw, bh = (
            int(parts[0]),
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
        )

        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)

        color = CLASS_COLORS.get(cls, (200, 200, 200))
        thickness = 3 if cls == 0 else 2
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, thickness)

        if cls != 2:  # skip label text for barlines (too many)
            cv2.putText(
                debug,
                CLASS_NAMES.get(cls, str(cls)),
                (x1 + 4, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )

        counts[cls] = counts.get(cls, 0) + 1

    # Stats overlay (top-left)
    for cls_id, name in CLASS_NAMES.items():
        cv2.putText(
            debug,
            f"{name}: {counts.get(cls_id, 0)}",
            (16, 50 + cls_id * 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            CLASS_COLORS[cls_id],
            2,
        )

    return debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", default="raw0", choices=["raw0", "raw", "train", "val"]
    )
    args = parser.parse_args()

    img_dir = DATASET_ROOT / "images" / args.split
    lbl_dir = DATASET_ROOT / "labels" / args.split
    out_dir = Path("runs/debug_labels") / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        print(f"No images found in {img_dir}")
        return

    print(f"Processing {len(img_files)} images from '{args.split}' → {out_dir}")

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] {img_path.name} (read error)")
            continue

        label_path = lbl_dir / (img_path.stem + ".txt")
        debug = draw_yolo_labels(img, label_path)

        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), debug)
        has_label = "ok" if label_path.exists() else "NO LABEL"
        print(f"  {img_path.name}  [{has_label}]")

    print(f"Done. Saved to {out_dir}")


if __name__ == "__main__":
    main()
