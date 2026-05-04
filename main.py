import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from scripts.pdf_to_images import convert_pdf_to_images, get_pdf_page_count


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


def find_staves_in_system(sys_img):
    if sys_img.size == 0:
        return []
    
    gray = cv2.cvtColor(sys_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    h, w = thresh.shape
    # Stronger horizontal kernel to find staff lines
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w * 0.2), 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
    
    y_profile = np.sum(horizontal_lines / 255, axis=1)
    # Threshold for finding a line
    y_indices = np.where(y_profile > w * 0.15)[0]
    
    lines_y = []
    current_line = []
    for y in y_indices:
        if not current_line or y - current_line[-1] <= 3:
            current_line.append(y)
        else:
            lines_y.append(int(np.mean(current_line)))
            current_line = [y]
    if current_line:
        lines_y.append(int(np.mean(current_line)))
    
    if len(lines_y) < 4:
        return [(0, h)]

    staves = []
    all_diffs = np.diff(lines_y)
    # Reasonable staff space range
    valid_diffs = [d for d in all_diffs if h * 0.005 < d < h * 0.1]
    if not valid_diffs:
        return [(0, h)]
    
    est_space = np.median(valid_diffs)
    curr_staff = [lines_y[0]]
    for y in lines_y[1:]:
        if abs((y - curr_staff[-1]) - est_space) < est_space * 0.5:
            curr_staff.append(y)
        else:
            if len(curr_staff) >= 4:
                staves.append((curr_staff[0], curr_staff[-1]))
            curr_staff = [y]
    if len(curr_staff) >= 4:
        staves.append((curr_staff[0], curr_staff[-1]))
        
    return staves if staves else [(0, h)]


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

    for page_num in range(1, total_pages + 1):
        print(f"Processing Page {page_num}/{total_pages}...")
        
        saved_paths = convert_pdf_to_images(pdf_path, temp_dir, first_page=page_num, last_page=page_num)
        if not saved_paths: continue
            
        img_path = Path(saved_paths[0])
        img = cv2.imread(str(img_path))
        if img is None: continue

        # Predict with low conf first to get all candidates
        results = model.predict(source=img, conf=0.1, imgsz=1280, verbose=False)[0]
        
        detections = []
        for box in results.boxes:
            detections.append({
                'box': box.xyxy[0].tolist(),
                'class_id': int(box.cls[0]),
                'conf': float(box.conf[0])
            })
            
        # 1. Filter Systems: High confidence and Non-Overlap
        raw_systems = [d for d in detections if d['class_id'] == 0 and d['conf'] > 0.6]
        raw_systems.sort(key=lambda x: x['conf'], reverse=True)
        
        systems = []
        for s in raw_systems:
            overlap = False
            for kept_s in systems:
                if calculate_iou(s['box'], kept_s['box']) > 0.3:
                    overlap = True
                    break
            if not overlap:
                systems.append(s)
        
        # 2. Filter Barlines: Moderate confidence
        barlines = [d for d in detections if d['class_id'] == 1 and d['conf'] > 0.2]
        
        print(f"  - Page {page_num}: Found {len(systems)} systems, {len(barlines)} barlines.")
        
        systems.sort(key=lambda x: x['box'][1])
        debug_img = img.copy()
        
        for sys_idx, sys in enumerate(systems):
            s_x1, s_y1, s_x2, s_y2 = map(int, sys['box'])
            cv2.rectangle(debug_img, (s_x1, s_y1), (s_x2, s_y2), (255, 0, 0), 3)
            cv2.putText(debug_img, f"Sys {sys_idx}", (s_x1 + 10, s_y1 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            # Find barlines strictly or nearly in this system vertically
            sys_barlines = [b for b in barlines if b['box'][0] >= s_x1 - 10 and b['box'][2] <= s_x2 + 10 
                            and b['box'][1] < s_y2 and b['box'][3] > s_y1]
            sys_barlines.sort(key=lambda x: x['box'][0])
            
            # Find staff lines (parts)
            sys_region = img[s_y1:s_y2, s_x1:s_x2]
            staves = find_staves_in_system(sys_region)
            
            # Draw barlines for debug
            for b in sys_barlines:
                bx1, by1, bx2, by2 = map(int, b['box'])
                cv2.line(debug_img, (bx1, max(s_y1, by1)), (bx1, min(s_y2, by2)), (0, 0, 255), 2)

            # Construct Measure Grid
            for col_idx in range(len(sys_barlines) - 1):
                bx1 = int(sys_barlines[col_idx]['box'][0])
                bx2 = int(sys_barlines[col_idx+1]['box'][0])
                
                for row_idx, (st_y1, st_y2) in enumerate(staves):
                    m_y1, m_y2 = s_y1 + st_y1, s_y1 + st_y2
                    # Use smaller vertical padding for better accuracy
                    pad = (m_y2 - m_y1) // 4
                    m_y1_p = max(0, m_y1 - pad)
                    m_y2_p = min(img.shape[0], m_y2 + pad)
                    
                    cv2.rectangle(debug_img, (bx1, m_y1_p), (bx2, m_y2_p), (0, 255, 0), 2)
                    label = f"{sys_idx}-{col_idx}-{row_idx}"
                    cv2.putText(debug_img, label, (bx1 + 10, m_y1 + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imwrite(str(output_base / img_path.name), debug_img)
        img_path.unlink()

    shutil.rmtree(temp_dir)
    print(f"Results saved to {output_base}")


def main():
    parser = argparse.ArgumentParser(description="Sheet Music Measure Detector")
    parser.add_argument("pdf_path", nargs="?", help="Path to the sheet music PDF")
    parser.add_argument(
        "--model", default="runs/detect/train4/weights/best.pt", help="Path to the YOLO model"
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
