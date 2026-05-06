import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.pdf_to_images import convert_pdf_to_images, get_pdf_page_count
from scripts.sheet_cv import deskew_image, detect_systems_cv, detect_barlines_in_system


def calculate_iou(box1, box2):
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    return intersection_area / float(area1 + area2 - intersection_area)



def validate_system_box(img, box):
    """Rejects system detections that contain no horizontal staff content."""
    x1, y1, x2, y2 = map(int, box)
    region = img[max(0, y1):y2, max(0, x1):x2]
    if region.size == 0:
        return False
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    w = region.shape[1]
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.15), 1))
    horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
    row_density = np.sum(horiz_lines / 255, axis=1)
    staff_rows = np.sum(row_density > w * 0.1)
    return staff_rows >= 4


def merge_close_barlines(barlines, min_dist_px):
    """Merges barlines closer than min_dist_px, keeping the highest confidence one per cluster."""
    if not barlines:
        return []
    barlines = sorted(barlines, key=lambda x: x["box"][0])
    clusters = [[barlines[0]]]
    for b in barlines[1:]:
        if b["box"][0] - clusters[-1][-1]["box"][0] < min_dist_px:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    return [max(c, key=lambda x: x["conf"]) for c in clusters]


def merge_overlapping_staves(staves):
    """Merges staves with >30% vertical overlap into a union bounding box."""
    if not staves:
        return []
    staves = sorted(staves, key=lambda x: x["box"][1])
    merged = [{"box": list(staves[0]["box"]), "conf": staves[0]["conf"], "class_id": staves[0]["class_id"]}]
    for s in staves[1:]:
        y1, y2 = s["box"][1], s["box"][3]
        my1, my2 = merged[-1]["box"][1], merged[-1]["box"][3]
        overlap_h = max(0.0, min(y2, my2) - max(y1, my1))
        if overlap_h / min(y2 - y1, my2 - my1) > 0.3:
            merged[-1]["box"] = [
                min(merged[-1]["box"][0], s["box"][0]),
                min(my1, y1),
                max(merged[-1]["box"][2], s["box"][2]),
                max(my2, y2),
            ]
            merged[-1]["conf"] = max(merged[-1]["conf"], s["conf"])
        else:
            merged.append({"box": list(s["box"]), "conf": s["conf"], "class_id": s["class_id"]})
    return merged


def combine_systems(yolo_systems, cv_extents, img_w, img_h, y_iou_thresh=0.5):
    """YOLOとCVのsystem候補を統合。Y軸IoU NMSでより小さい（細かい）方を優先する。"""
    cv_systems = [
        {"box": [0.0, float(y1), float(img_w), float(y2)], "conf": 0.7, "class_id": 0, "source": "cv"}
        for y1, y2 in cv_extents
    ]
    all_candidates = yolo_systems + cv_systems
    # Y方向スパンが小さい順（細かい方優先）でNMS
    all_candidates.sort(key=lambda s: s["box"][3] - s["box"][1])
    kept = []
    for cand in all_candidates:
        y1_c, y2_c = cand["box"][1], cand["box"][3]
        suppressed = False
        for k in kept:
            y1_k, y2_k = k["box"][1], k["box"][3]
            overlap = max(0.0, min(y2_c, y2_k) - max(y1_c, y1_k))
            union = max(y2_c, y2_k) - min(y1_c, y1_k)
            if union > 0 and overlap / union > y_iou_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    return sorted(kept, key=lambda s: s["box"][1])


def run_prediction(pdf_path, model_path):
    temp_dir = Path("temp_prediction_images")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    total_pages = get_pdf_page_count(pdf_path)
    output_base = Path("runs/predict") / Path(pdf_path).stem
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "img").mkdir(parents=True, exist_ok=True)
    (output_base / "lbl").mkdir(parents=True, exist_ok=True)

    for page_num in range(1, total_pages + 1):
        print(f"Processing Page {page_num}/{total_pages}...")

        saved_paths = convert_pdf_to_images(
            pdf_path, temp_dir, first_page=page_num, last_page=page_num
        )
        if not saved_paths:
            continue

        img_path = Path(saved_paths[0])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = deskew_image(img)

        # Predict with low conf first to get all candidates
        results = model.predict(source=img, conf=0.1, imgsz=1280, verbose=False)[0]

        detections = []
        for box in results.boxes:
            detections.append(
                {
                    "box": box.xyxy[0].tolist(),
                    "class_id": int(box.cls[0]),
                    "conf": float(box.conf[0]),
                }
            )

        # 1. Filter Systems: High confidence and Non-Overlap
        raw_systems = [d for d in detections if d["class_id"] == 0 and d["conf"] > 0.3]
        raw_systems.sort(key=lambda x: x["conf"], reverse=True)

        systems = []
        for s in raw_systems:
            overlap = False
            for kept_s in systems:
                if calculate_iou(s["box"], kept_s["box"]) > 0.3:
                    overlap = True
                    break
            if not overlap and validate_system_box(img, s["box"]):
                systems.append(s)

        # CVによるsystem検出と統合（より細分化された方を優先）
        cv_groups, _ = detect_systems_cv(img)
        cv_extents = [(g[0][0], g[-1][1]) for g in cv_groups]
        systems = combine_systems(systems, cv_extents, img.shape[1], img.shape[0])

        # 2. Filter Staves: Moderate confidence
        staves_all = [d for d in detections if d["class_id"] == 1 and d["conf"] > 0.4]

        # 3. Filter Barlines: YOLOの最低閾値と合わせる
        barlines = [d for d in detections if d["class_id"] == 2 and d["conf"] > 0.1]

        print(
            f"  - Page {page_num}: Found {len(systems)} systems, {len(staves_all)} staves, {len(barlines)} barlines."
        )

        systems.sort(key=lambda x: x["box"][1])
        debug_img = img.copy()
        img_h, img_w = img.shape[:2]

        # systemのx範囲を画像の左右端に揃える
        for s in systems:
            s["box"][0] = 0.0
            s["box"][2] = float(img_w)

        # 隣接system間のY境界をmidpointで分割（重なり防止）
        if len(systems) > 1:
            if staves_all:
                median_stave_h = float(np.median([s["box"][3] - s["box"][1] for s in staves_all]))
                half_margin = max(5, int(median_stave_h / (1.6 * 4)))
            else:
                half_margin = int(img_h * 0.01)
            for i in range(len(systems) - 1):
                mid = int((systems[i]["box"][3] + systems[i + 1]["box"][1]) / 2)
                systems[i]["box"][3] = float(min(img_h, mid + half_margin))
                systems[i + 1]["box"][1] = float(max(0, mid - half_margin))

        label_lines = []
        final_barlines = []  # YOLO + CVフォールバックを集約

        def to_yolo(cls_id, box):
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w  = (x2 - x1) / img_w
            h  = (y2 - y1) / img_h
            return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

        tol_y = img_h * 0.003  # 縦方向トレランス（画像高さの 0.3%）
        tol_x = img_w * 0.004  # 横方向トレランス（画像幅の 0.4%）

        for sys_idx, sys in enumerate(systems):
            s_x1, s_y1, s_x2, s_y2 = map(int, sys["box"])
            cv2.rectangle(debug_img, (s_x1, s_y1), (s_x2, s_y2), (255, 0, 0), 3)
            cv2.putText(
                debug_img,
                f"Sys {sys_idx}",
                (s_x1 + 10, s_y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 0, 0),
                3,
            )

            label_lines.append(to_yolo(0, sys["box"]))

            # Find staves belonging to this system, remove vertical overlaps
            sys_staves = [
                s for s in staves_all
                if s["box"][1] >= s_y1 - tol_y and s["box"][3] <= s_y2 + tol_y
            ]
            sys_staves = merge_overlapping_staves(sys_staves)

            # Assign barlines whose vertical center falls within this system
            sys_barlines = [
                b
                for b in barlines
                if b["box"][0] >= s_x1 - tol_x
                and b["box"][2] <= s_x2 + tol_x
                and s_y1 <= (b["box"][1] + b["box"][3]) / 2 <= s_y2
            ]
            sys_barlines.sort(key=lambda x: x["box"][0])
            min_barline_dist = int((s_x2 - s_x1) * 0.02)
            sys_barlines = merge_close_barlines(sys_barlines, min_barline_dist)

            # YOLOで十分検出できない場合はCVフォールバック
            if len(sys_barlines) < 2:
                cv_xs = detect_barlines_in_system(img, s_x1, s_y1, s_x2, s_y2)
                cv_bls = [
                    {"box": [float(x - 2), float(s_y1), float(x + 2), float(s_y2)], "conf": 0.5, "class_id": 2, "source": "cv"}
                    for x in cv_xs
                ]
                if len(cv_bls) > len(sys_barlines):
                    sys_barlines = merge_close_barlines(cv_bls, min_barline_dist)
                    print(f"    (sys {sys_idx}: CV fallback → {len(sys_barlines)} barlines)")

            final_barlines.extend(sys_barlines)

            # Draw staves for debug
            for st in sys_staves:
                st_x1, st_y1_, st_x2, st_y2_ = map(int, st["box"])
                cv2.rectangle(debug_img, (st_x1, st_y1_), (st_x2, st_y2_), (0, 200, 0), 2)

            # Draw barlines for debug (YOLO=明るいオレンジ, CV=暗いオレンジ)
            for b in sys_barlines:
                bx1, by1, bx2, by2 = map(int, b["box"])
                color = (0, 80, 200) if b.get("source") == "cv" else (0, 180, 255)
                cv2.line(
                    debug_img,
                    (bx1, max(s_y1, by1)),
                    (bx1, min(s_y2, by2)),
                    color,
                    2,
                )

            # Construct Measure Grid
            for col_idx in range(len(sys_barlines) - 1):
                bx1 = int(sys_barlines[col_idx]["box"][0])
                bx2 = int(sys_barlines[col_idx + 1]["box"][0])

                for row_idx, st in enumerate(sys_staves):
                    m_y1, m_y2 = int(st["box"][1]), int(st["box"][3])
                    pad = (m_y2 - m_y1) // 4
                    m_y1_p = max(0, m_y1 - pad)
                    m_y2_p = min(img.shape[0], m_y2 + pad)

                    cv2.rectangle(
                        debug_img, (bx1, m_y1_p), (bx2, m_y2_p), (0, 165, 255), 2
                    )
                    label = f"{sys_idx}-{col_idx}-{row_idx}"
                    cv2.putText(
                        debug_img,
                        label,
                        (bx1 + 10, m_y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2,
                    )

        # ラベル出力: staveは全件、barlineはsystem割り当て済みのもの（YOLO+CV）
        for st in staves_all:
            label_lines.append(to_yolo(1, st["box"]))
        for b in final_barlines:
            label_lines.append(to_yolo(2, b["box"]))

        cv2.imwrite(str(output_base / "img" / img_path.name), debug_img)

        label_path = output_base / "lbl" / (img_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines))

        img_path.unlink()

    shutil.rmtree(temp_dir)
    print(f"Results saved to {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Sheet Music Measure Detector")
    parser.add_argument("pdf_path", nargs="?", help="Path to the sheet music PDF")
    parser.add_argument(
        "--model",
        default="runs/detect/train4s/weights/best.pt",
        help="Path to the YOLO model",
    )

    args = parser.parse_args()

    if not args.pdf_path:
        print("Usage: python main.py <path_to_pdf>")
        return

    if not os.path.exists(args.pdf_path):
        print(f"Error: File {args.pdf_path} not found.")
        return

    run_prediction(args.pdf_path, args.model)


if __name__ == "__main__":
    main()
