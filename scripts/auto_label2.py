import math
from pathlib import Path

import cv2
import numpy as np


def deskew_image(img):
    """画像から水平な直線を検出し、傾きを自動補正（回転）する関数"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 線の検出を安定させるため、文字などをぼかす
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY_INV)

    # 横線だけを大まかに抽出（画像幅の5%程度をカーネル幅とする）
    img_w = img.shape[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * 0.05), 1))
    horiz = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # ハフ変換で直線を検出
    # (画像, 距離分解能, 角度分解能, 閾値, 最小の線の長さ, 線と線の最大の間隔)
    lines = cv2.HoughLinesP(
        horiz, 1, np.pi / 180, 100, minLineLength=int(img_w * 0.1), maxLineGap=20
    )

    if lines is None:
        print("傾き補正：基準となる横線が見つかりませんでした。")
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 2点間の角度を計算（度数法）
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # 縦線や極端な斜め線を除外。ほぼ水平（-10度 〜 10度）の線だけを集める
        if -10 < angle < 10:
            angles.append(angle)

    if not angles:
        print("傾き補正：水平に近い線が見つかりませんでした。")
        return img

    # 外れ値の影響を避けるため、平均ではなく「中央値」を採用
    median_angle = np.median(angles)
    print(f"自動傾き補正：画像を {median_angle:.2f} 度 回転させます。")

    # 傾きがごくわずか（0.05度未満など）ならそのまま返す
    if abs(median_angle) < 0.05:
        return img

    # 画像を回転
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

    # 【重要】回転してできた余白は、黒ではなく白(255,255,255)で埋める
    rotated_img = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    return rotated_img


def extract_measures_v4(image_path, output_txt_path):
    # 画像の読み込み
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("画像を読み込めませんでした。")
        return

    # 読み込んだ画像を、処理する前に真っ直ぐに補正する！
    img = deskew_image(original_img)
    img_h, img_w = img.shape[:2]
    print(f"入力画像サイズ: {img_w} x {img_h}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # ==========================================
    # 1. 線の抽出
    # ==========================================
    horiz_kernel_w = int(img_w * 0.1)  # 横線抽出用のカーネル幅（画像幅の10%）
    # 解像度に合わせて高さを算出（3600pxなら約3〜5px程度になります）
    horiz_kernel_h = max(
        1, int(img_h * 0.0003)
    )  # 横線抽出用のカーネル高さ（画像高さの0.05%）
    horiz_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (horiz_kernel_w, horiz_kernel_h)
    )
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    vert_kernel_h = int(img_h * 0.02)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_kernel_h))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vert_kernel)

    # ==========================================
    # 2. 五線（パート）の等間隔性を利用した正確な検出
    # ==========================================
    y_profile = np.sum(horizontal_lines / 255, axis=1)
    y_indices = np.where(y_profile > img_w * 0.1)[0]

    # 2-1. 太い線を1本の中心Y座標にまとめる
    lines_y = []
    line_thickness_thresh = max(3, int(img_h * 0.002))  # 線幅の許容

    current_line = []
    for y in y_indices:
        if not current_line:
            current_line.append(y)
        elif y - current_line[-1] <= line_thickness_thresh:
            current_line.append(y)
        else:
            lines_y.append(int(np.mean(current_line)))
            current_line = [y]
    if current_line:
        lines_y.append(int(np.mean(current_line)))

    if len(lines_y) < 5:
        print("線が少なすぎて五線を検出できません。")
        return

    # 2-2. 五線の線間隔（staff_space）を全体から推定
    all_diffs = np.diff(lines_y)
    # 画像高さの0.2%〜2%の範囲を妥当な線間隔と仮定
    valid_diffs = [d for d in all_diffs if img_h * 0.002 < d < img_h * 0.02]

    if not valid_diffs:
        print("五線の間隔を推定できませんでした。")
        return

    estimated_space = np.median(valid_diffs)
    print(f"推定された五線の線間隔: 約 {estimated_space:.1f} px")

    # 2-3. 等間隔に並ぶ線をグループ化してパート（五線）を特定
    staves = []
    current_staff = [lines_y[0]]

    for y in lines_y[1:]:
        gap = y - current_staff[-1]

        # 間隔が推定間隔の±40%以内なら同じ五線の一部とみなす
        if abs(gap - estimated_space) < estimated_space * 0.4:
            current_staff.append(y)
        else:
            # 別の線に移る前に、まとまりが4本以上ならパートとして登録
            if len(current_staff) >= 4:
                staves.append((current_staff[0], current_staff[-1]))
            current_staff = [y]

    if len(current_staff) >= 4:
        staves.append((current_staff[0], current_staff[-1]))

    print(f"検出されたパート数: {len(staves)}")

    # ==========================================
    # 3. パートを「段（システム）」にグループ化
    # ==========================================
    if not staves:
        print("パートが検出されませんでした。")
        return

    systems = []
    current_system = [staves[0]]
    system_gap_thresh = int(img_h * 0.05)  # パート間の最大ギャップ（画像高さの5%）

    for i in range(1, len(staves)):
        gap = staves[i][0] - staves[i - 1][1]
        if gap > system_gap_thresh:
            systems.append(current_system)
            current_system = [staves[i]]
        else:
            current_system.append(staves[i])
    if current_system:
        systems.append(current_system)

    print(f"検出された段（システム）数: {len(systems)}")
    for idx, sys in enumerate(systems):
        print(f"  - システム {idx + 1}: {len(sys)} パート")

    # ==========================================
    # 4. 段ごとに小節線を検出し、矩形化
    # ==========================================
    yolo_annotations = []
    class_id = 0

    barline_thickness_thresh = int(img_w * 0.005)
    min_measure_width = int(img_w * 0.02)

    for sys_idx, system_staves in enumerate(systems):
        sys_y_min = system_staves[0][0]
        sys_y_max = system_staves[-1][1]
        sys_h = sys_y_max - sys_y_min

        sys_vert_lines = vertical_lines[sys_y_min:sys_y_max, :]
        x_profile = np.sum(sys_vert_lines / 255, axis=0)
        x_peaks = np.where(x_profile > sys_h * 0.2)[0]

        barlines_x = []
        current_x_block = []
        for x in x_peaks:
            if not current_x_block:
                current_x_block.append(x)
            elif x - current_x_block[-1] < barline_thickness_thresh:
                current_x_block.append(x)
            else:
                barlines_x.append(int(np.mean(current_x_block)))
                current_x_block = [x]
        if current_x_block:
            barlines_x.append(int(np.mean(current_x_block)))

        for staff_y_min, staff_y_max in system_staves:
            staff_h = staff_y_max - staff_y_min

            # 【余白の調整】五線の高さの1.5倍
            pad_y = int(staff_h * 1.5)  # 1.5倍に変更して余裕を持たせる
            y_min_padded = max(0, staff_y_min - pad_y)
            y_max_padded = min(img_h, staff_y_max + pad_y)

            for i in range(len(barlines_x) - 1):
                x_min = barlines_x[i]
                x_max = barlines_x[i + 1]

                if x_max - x_min < min_measure_width:
                    continue

                center_x = ((x_min + x_max) / 2.0) / img_w
                center_y = ((y_min_padded + y_max_padded) / 2.0) / img_h
                width = (x_max - x_min) / img_w
                height = (y_max_padded - y_min_padded) / img_h

                yolo_annotations.append(
                    f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                )

                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
                cv2.rectangle(
                    img,
                    (x_min, y_min_padded),
                    (x_max, y_max_padded),
                    colors[sys_idx % 3],
                    5,
                )

    with open(output_txt_path, "w") as f:
        f.write("\n".join(yolo_annotations))

    print(f"出力したアノテーション（小節）数: {len(yolo_annotations)}")
    debug_dir = Path("datasets/measure-detection/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(debug_dir / f"{image_path.stem}_debug_output.jpg", img)


def main():
    input_dir = Path("datasets/measure-detection/images/raw")
    label_dir = Path("datasets/measure-detection/labels/train")

    label_dir.mkdir(parents=True, exist_ok=True)
    # 実行例（ファイル名は適宜変更してください）
    for img_file in sorted(input_dir.glob("*.png")):
        print(f"Processing: {img_file.name}")
        output_txt = label_dir / (img_file.stem + ".txt")
        extract_measures_v4(img_file, output_txt)


if __name__ == "__main__":
    main()
