import cv2
import numpy as np
from pathlib import Path
import math


def deskew_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)
    img_w = img.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * 0.05), 1))
    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    lines = cv2.HoughLinesP(
        horiz, 1, np.pi / 180, 100, minLineLength=int(img_w * 0.1), maxLineGap=20
    )
    if lines is None:
        return img
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if -10 < angle < 10:
            angles.append(angle)
    if not angles:
        return img
    median_angle = np.median(angles)
    if abs(median_angle) < 0.05:
        return img
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def generate_structural_labels(image_path, output_txt_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    img = deskew_image(img)
    img_h, img_w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines for systems
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * 0.1), 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    # Detect vertical lines for barlines
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(img_h * 0.02)))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

    # 1. Find Staff Lines -> Systems
    y_profile = np.sum(horizontal_lines / 255, axis=1)
    y_indices = np.where(y_profile > img_w * 0.1)[0]

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
        return

    # Group into staves, then systems
    staves = []
    if lines_y:
        all_diffs = np.diff(lines_y)
        valid_diffs = [d for d in all_diffs if img_h * 0.002 < d < img_h * 0.02]
        if not valid_diffs:
            return
        est_space = np.median(valid_diffs)

        curr_staff = [lines_y[0]]
        for y in lines_y[1:]:
            if abs((y - curr_staff[-1]) - est_space) < est_space * 0.4:
                curr_staff.append(y)
            else:
                if len(curr_staff) >= 4:
                    staves.append((curr_staff[0], curr_staff[-1]))
                curr_staff = [y]
        if len(curr_staff) >= 4:
            staves.append((curr_staff[0], curr_staff[-1]))

    if not staves:
        return

    systems = []
    curr_sys = [staves[0]]
    for i in range(1, len(staves)):
        if staves[i][0] - curr_sys[-1][1] > img_h * 0.05:
            systems.append(curr_sys)
            curr_sys = [staves[i]]
        else:
            curr_sys.append(staves[i])
    if curr_sys:
        systems.append(curr_sys)

    yolo_labels = []
    # Class 0: system, Class 1: staff, Class 2: barline

    for sys in systems:
        sys_y_min, sys_y_max = sys[0][0], sys[-1][1]
        # Add padding to system
        padding = (sys_y_max - sys_y_min) * 0.2
        sys_y_min_p = max(0, int(sys_y_min - padding))
        sys_y_max_p = min(img_h, int(sys_y_max + padding))

        # Save System Label (class 0)
        cx = 0.5
        cy = (sys_y_min_p + sys_y_max_p) / 2 / img_h
        w = 1.0
        h = (sys_y_max_p - sys_y_min_p) / img_h
        yolo_labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Save Staff Labels (class 1) — one per stave in the system
        for st_y1, st_y2 in sys:
            st_pad = (st_y2 - st_y1) * 0.3
            st_y1_p = max(0, int(st_y1 - st_pad))
            st_y2_p = min(img_h, int(st_y2 + st_pad))
            st_cy = (st_y1_p + st_y2_p) / 2 / img_h
            st_h = (st_y2_p - st_y1_p) / img_h
            yolo_labels.append(f"1 0.500000 {st_cy:.6f} 1.000000 {st_h:.6f}")

        # Find Barlines within this system
        sys_vert = vertical_lines[sys_y_min:sys_y_max, :]
        x_profile = np.sum(sys_vert / 255, axis=0)
        x_peaks = np.where(x_profile > (sys_y_max - sys_y_min) * 0.2)[0]

        barlines_x = []
        curr_x_block = []
        for x in x_peaks:
            if not curr_x_block or x - curr_x_block[-1] < 5:
                curr_x_block.append(x)
            else:
                barlines_x.append(int(np.mean(curr_x_block)))
                curr_x_block = [x]
        if curr_x_block:
            barlines_x.append(int(np.mean(curr_x_block)))

        for bx in barlines_x:
            # Barline Label (class 2)
            bcx = bx / img_w
            bcy = (sys_y_min_p + sys_y_max_p) / 2 / img_h
            bw = 10 / img_w  # Fixed small width for barline
            bh = (sys_y_max_p - sys_y_min_p) / img_h
            yolo_labels.append(f"2 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f}")

    with open(output_txt_path, "w") as f:
        f.write("\n".join(yolo_labels))


def main():
    img_dir = Path("datasets/measure-detection/images/train")
    lbl_dir = Path("datasets/measure-detection/labels/train")
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_file in sorted(img_dir.glob("*.png")):
        # Extract page number
        try:
            page_num = int(img_file.stem.split("_page_")[-1])
            if 2 <= page_num <= 23 or True:
                print(f"Labeling structural: {img_file.name}")
                output_txt = lbl_dir / (img_file.stem + ".txt")
                generate_structural_labels(img_file, output_txt)
        except ValueError:
            continue


if __name__ == "__main__":
    main()
