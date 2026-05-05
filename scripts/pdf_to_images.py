import os
from pdf2image import convert_from_path, pdfinfo_from_path
from pathlib import Path
import argparse
from tqdm import tqdm


def get_pdf_page_count(pdf_path):
    info = pdfinfo_from_path(pdf_path)
    return info["Pages"]


def convert_pdf_to_images(
    pdf_path, output_dir, dpi=300, first_page=None, last_page=None
):
    """
    PDFの指定されたページ範囲を画像に変換して保存します。
    """
    pdf_name = Path(pdf_path).stem

    # ページ範囲の決定
    if first_page is None:
        first_page = 1
    if last_page is None:
        last_page = get_pdf_page_count(pdf_path)

    images = convert_from_path(
        pdf_path, dpi=dpi, first_page=first_page, last_page=last_page
    )

    os.makedirs(output_dir, exist_ok=True)

    saved_paths = []
    for i, image in enumerate(images):
        page_num = first_page + i
        output_path = os.path.join(output_dir, f"{pdf_name}_page_{page_num:03d}.png")
        image.save(output_path, "PNG")
        saved_paths.append(output_path)

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PDF pages to images for YOLOv8"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--output",
        "-o",
        default="datasets/measure-detection/images/raw0",
        help="Output directory",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for image conversion")

    args = parser.parse_args()
    convert_pdf_to_images(args.pdf_path, args.output, args.dpi)
