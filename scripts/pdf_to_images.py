import os
from pdf2image import convert_from_path
from pathlib import Path
import argparse
from tqdm import tqdm

def convert_pdf_to_images(pdf_path, output_dir, dpi=300):
    """
    PDFの各ページを画像に変換して保存します。
    """
    pdf_name = Path(pdf_path).stem
    images = convert_from_path(pdf_path, dpi=dpi)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, image in tqdm(enumerate(images), total=len(images), desc="Converting PDF to images"):
        output_path = os.path.join(output_dir, f"{pdf_name}_page_{i+1:03d}.png")
        image.save(output_path, "PNG")
        tqdm.write(f"Saved: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF pages to images for YOLOv8")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--output", "-o", default="datasets/measure-detection/images/raw", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for image conversion")
    
    args = parser.parse_args()
    convert_pdf_to_images(args.pdf_path, args.output, args.dpi)
