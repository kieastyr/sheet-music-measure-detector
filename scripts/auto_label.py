import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

def detect_measures(image_path, debug_out=None):
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive Thresholding for better binarization in various lighting
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 15, 2)

    h, w = binary.shape

    # 1. Extract horizontal lines (Staves)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 1))
    detected_staves = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 2. Extract vertical lines (Barlines)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 100))
    detected_barlines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # 3. Find individual staff regions
    cnts_h, _ = cv2.findContours(detected_staves, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    staff_lines = []
    for c in cnts_h:
        x, y, w_rect, h_rect = cv2.boundingRect(c)
        if w_rect > w * 0.3: # Minimum width for a staff line
            staff_lines.append((y, y + h_rect))
    staff_lines.sort()

    # Group 5 lines (or similar) into a single staff
    refined_staves = []
    if staff_lines:
        curr_top, curr_bot = staff_lines[0]
        for i in range(1, len(staff_lines)):
            if staff_lines[i][0] - curr_bot < 20: # Distance between lines in a staff
                curr_bot = staff_lines[i][1]
            else:
                refined_staves.append((curr_top, curr_bot))
                curr_top, curr_bot = staff_lines[i]
        refined_staves.append((curr_top, curr_bot))

    # 4. Find barlines and their x-coordinates
    cnts_v, _ = cv2.findContours(detected_barlines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    barline_segments = []
    for c in cnts_v:
        x, y, w_rect, h_rect = cv2.boundingRect(c)
        if h_rect > 20: # Minimum height for a barline segment
            barline_segments.append((x, y, x + w_rect, y + h_rect))

    # 5. Build measure boxes for each staff
    measures = []
    for s_top, s_bot in refined_staves:
        # Extend the vertical search slightly to catch barlines
        search_top, search_bot = s_top - 5, s_bot + 5
        
        x_coords = []
        for x1, y1, x2, y2 in barline_segments:
            # Check if this barline segment crosses the current staff
            if not (y2 < search_top or y1 > search_bot):
                x_coords.append((x1 + x2) // 2)
        
        x_coords = sorted(list(set(x_coords)))
        
        # Clean up x_coords that are too close (likely same barline)
        unique_x = []
        if x_coords:
            curr_x = x_coords[0]
            for i in range(1, len(x_coords)):
                if x_coords[i] - curr_x > 15:
                    unique_x.append(curr_x)
                    curr_x = x_coords[i]
            unique_x.append(curr_x)

        # Create measure boxes from pairs of vertical lines
        for i in range(len(unique_x) - 1):
            mx1, mx2 = unique_x[i], unique_x[i+1]
            # Ensure measure has reasonable width
            if mx2 - mx1 > 30:
                measures.append([mx1, s_top, mx2 - mx1, s_bot - s_top])

    # Save debug image
    if debug_out:
        debug_img = img.copy()
        for (x, y, mw, mh) in measures:
            cv2.rectangle(debug_img, (x, y), (x + mw, y + mh), (0, 255, 0), 1)
        cv2.imwrite(str(debug_out), debug_img)

    return measures

def save_yolo_labels(measures, img_shape, output_path):
    h, w = img_shape[:2]
    with open(output_path, 'w') as f:
        for (mx, my, mw, mh) in measures:
            # YOLO format: class x_center y_center width height (normalized 0.0 to 1.0)
            x_center = (mx + mw / 2) / w
            y_center = (my + mh / 2) / h
            norm_w = mw / w
            norm_h = mh / h
            f.write(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

def main():
    input_dir = Path("datasets/measure-detection/images/raw")
    label_dir = Path("datasets/measure-detection/labels/train")
    debug_dir = Path("datasets/measure-detection/debug")
    
    label_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(input_dir.glob("*.png"))
    print(f"Found {len(image_files)} images. Processing...")

    for img_path in tqdm(image_files):
        # 1. Detect measures
        debug_path = debug_dir / img_path.name
        boxes = detect_measures(img_path, debug_out=debug_path)
        
        # 2. Save YOLO labels
        img = cv2.imread(str(img_path))
        if img is not None:
            label_path = label_dir / img_path.with_suffix(".txt").name
            save_yolo_labels(boxes, img.shape, label_path)

if __name__ == "__main__":
    main()
