"""楽譜解析で共通利用するCV（画像処理）ユーティリティ。"""

import math
import cv2
import numpy as np


def deskew_image(img):
    """水平スタッフラインを基準に画像の傾きを補正する。"""
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
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def detect_systems_cv(img):
    """形態素演算でスタッフライン→段落（system）を検出する。

    Returns:
        sys_groups: list of lists of (y1, y2) stave tuples.
                    外側リストが system、内側リストが各 stave の縦範囲。
        est_space:  推定譜線間隔（px, float）。検出失敗時は 0.0。
    """
    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * 0.1), 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(img_h * 0.02)))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

    y_profile = np.sum(horizontal_lines / 255, axis=1)
    y_indices = np.where(y_profile > img_w * 0.1)[0]

    lines_y = []
    current_line = []
    for y in y_indices:
        if not current_line or y - current_line[-1] <= int(img_h * 0.001):
            current_line.append(y)
        else:
            lines_y.append(int(np.mean(current_line)))
            current_line = [y]
    if current_line:
        lines_y.append(int(np.mean(current_line)))

    if len(lines_y) < 4:
        return [], 0.0

    all_diffs = np.diff(lines_y)
    valid_diffs = [d for d in all_diffs if img_h * 0.002 < d < img_h * 0.02]
    if not valid_diffs:
        return [], 0.0
    est_space = float(np.median(valid_diffs))

    staves = []
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
        return [], est_space

    sys_groups = []
    curr_sys = [staves[0]]
    for i in range(1, len(staves)):
        if staves[i][0] - curr_sys[-1][1] > img_h * 0.05:
            sys_groups.append(curr_sys)
            curr_sys = [staves[i]]
        else:
            curr_sys.append(staves[i])
    if curr_sys:
        sys_groups.append(curr_sys)

    # バーラインで繋がっている隣接グループを結合
    merged = [sys_groups[0]]
    for i in range(1, len(sys_groups)):
        gap_top = merged[-1][-1][1]
        gap_bot = sys_groups[i][0][0]
        if gap_bot > gap_top:
            gap_vert = vertical_lines[gap_top:gap_bot, :]
            col_density = np.sum(gap_vert / 255, axis=0)
            if np.any(col_density > (gap_bot - gap_top) * 0.3):
                merged[-1] = merged[-1] + sys_groups[i]
                continue
        merged.append(sys_groups[i])

    return merged, est_space


def detect_barlines_in_system(img, s_x1, s_y1, s_x2, s_y2):
    """system 領域内でCV形態素演算によりバーラインを検出する。

    Returns:
        list of int: 画像座標系でのバーライン x 位置リスト。
    """
    region = img[s_y1:s_y2, s_x1:s_x2]
    if region.size == 0:
        return []
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    sys_h, sys_w = region.shape[:2]
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(sys_h * 0.2))))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)
    x_profile = np.sum(vertical_lines / 255, axis=0)
    x_peaks = np.where(x_profile > sys_h * 0.15)[0]
    barlines_x = []
    curr_x_block = []
    for x in x_peaks:
        if not curr_x_block or x - curr_x_block[-1] < max(3, int(sys_w * 0.002)):
            curr_x_block.append(x)
        else:
            barlines_x.append(int(np.mean(curr_x_block)))
            curr_x_block = [x]
    if curr_x_block:
        barlines_x.append(int(np.mean(curr_x_block)))
    return [s_x1 + bx for bx in barlines_x]
