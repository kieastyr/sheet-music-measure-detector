import cv2
import numpy as np
from pathlib import Path

try:
    from scripts.sheet_cv import deskew_image, detect_systems_cv, detect_barlines_in_system
except ImportError:
    from sheet_cv import deskew_image, detect_systems_cv, detect_barlines_in_system


def generate_structural_labels(image_path, output_txt_path, debug_output_path=None):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    img = deskew_image(img)
    img_h, img_w = img.shape[:2]

    sys_groups, est_space = detect_systems_cv(img)
    if not sys_groups or est_space == 0.0:
        return

    # x範囲検出用に horizontal_lines を再計算
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(img_w * 0.1), 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel)

    x_margin_px = int(img_w * 0.01)
    half_staff = int(est_space) // 2
    padding = int(est_space * 4 * 2.0)

    # Pass 1: 隣接 system 間の共有境界を midpoint ± half_staff に設定
    bounds = [
        [int(g[0][0]), int(g[-1][1]), int(g[0][0]), int(g[-1][1])]
        for g in sys_groups
    ]
    for i in range(len(bounds) - 1):
        mid = (bounds[i][1] + bounds[i + 1][0]) // 2
        bounds[i][3] = int(mid + half_staff)
        bounds[i + 1][2] = int(mid - half_staff)
    # Pass 2: 外側エッジにのみ padding を付与
    bounds[0][2] = int(max(0, bounds[0][0] - padding))
    bounds[-1][3] = int(min(img_h, bounds[-1][1] + padding))

    yolo_labels = []
    sys_barlines_all = []  # debug描画用に保持

    for sys_staves, bound in zip(sys_groups, bounds):
        sys_y_min, sys_y_max, sys_y_min_p, sys_y_max_p = bound

        # x 範囲をスタッフラインの水平分布から検出
        sys_horiz = horizontal_lines[sys_y_min:sys_y_max, :]
        x_profile_sys = np.sum(sys_horiz / 255, axis=0)
        x_active = np.where(x_profile_sys > 0)[0]
        if len(x_active) > 0:
            x_min = max(0, x_active[0] - x_margin_px)
            x_max = min(img_w, x_active[-1] + x_margin_px)
        else:
            x_min, x_max = 0, img_w

        sys_cx = (x_min + x_max) / 2 / img_w
        sys_w_norm = (x_max - x_min) / img_w

        # System label (class 0) — 全幅固定
        cy = (sys_y_min_p + sys_y_max_p) / 2 / img_h
        h = (sys_y_max_p - sys_y_min_p) / img_h
        yolo_labels.append(f"0 0.500000 {cy:.6f} 1.000000 {h:.6f}")

        # Staff labels (class 1) — system 内の各 stave
        for st_y1, st_y2 in sys_staves:
            st_pad = (st_y2 - st_y1) * 0.3
            st_y1_p = max(0, int(st_y1 - st_pad))
            st_y2_p = min(img_h, int(st_y2 + st_pad))
            st_cy = (st_y1_p + st_y2_p) / 2 / img_h
            st_h = (st_y2_p - st_y1_p) / img_h
            yolo_labels.append(f"1 {sys_cx:.6f} {st_cy:.6f} {sys_w_norm:.6f} {st_h:.6f}")

        # Barline labels (class 2)
        barlines_x = detect_barlines_in_system(img, 0, sys_y_min, img_w, sys_y_max)
        sys_barlines_all.append(barlines_x)
        for bx in barlines_x:
            bcx = bx / img_w
            bcy = (sys_y_min_p + sys_y_max_p) / 2 / img_h
            bh = (sys_y_max_p - sys_y_min_p) / img_h
            yolo_labels.append(f"2 {bcx:.6f} {bcy:.6f} 0.004000 {bh:.6f}")

    with open(output_txt_path, "w") as f:
        f.write("\n".join(yolo_labels))

    if debug_output_path is not None:
        debug_img = img.copy()
        for sys_staves, bound, barlines_x in zip(sys_groups, bounds, sys_barlines_all):
            sys_y_min, sys_y_max, sys_y_min_p, sys_y_max_p = bound

            sys_horiz = horizontal_lines[sys_y_min:sys_y_max, :]
            x_profile_sys = np.sum(sys_horiz / 255, axis=0)
            x_active = np.where(x_profile_sys > 0)[0]
            if len(x_active) > 0:
                dbg_x_min = max(0, x_active[0] - x_margin_px)
                dbg_x_max = min(img_w, x_active[-1] + x_margin_px)
            else:
                dbg_x_min, dbg_x_max = 0, img_w - 1

            cv2.rectangle(debug_img, (0, sys_y_min_p), (img_w - 1, sys_y_max_p), (255, 0, 0), 3)

            for st_y1, st_y2 in sys_staves:
                st_pad = (st_y2 - st_y1) * 0.3
                st_y1_p = max(0, int(st_y1 - st_pad))
                st_y2_p = min(img_h, int(st_y2 + st_pad))
                cv2.rectangle(debug_img, (dbg_x_min, st_y1_p), (dbg_x_max, st_y2_p), (0, 200, 0), 2)

            for bx in barlines_x:
                cv2.line(debug_img, (bx, sys_y_min_p), (bx, sys_y_max_p), (0, 0, 255), 2)

        Path(debug_output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_output_path), debug_img)


def main():
    img_dir = Path("datasets/measure-detection/images/raw0")
    lbl_dir = Path("datasets/measure-detection/labels/raw0")
    debug_dir = Path("datasets/debug")
    lbl_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    for img_file in sorted(img_dir.glob("*.png")):
        try:
            page_num = int(img_file.stem.split("_page_")[-1])
            if 2 <= page_num <= 23 or True:
                print(f"Labeling structural: {img_file.name}")
                output_txt = lbl_dir / (img_file.stem + ".txt")
                debug_img_path = debug_dir / img_file.name
                generate_structural_labels(
                    img_file, output_txt, debug_output_path=debug_img_path
                )
        except ValueError:
            continue


if __name__ == "__main__":
    main()
