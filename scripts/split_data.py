import os
import random
import shutil


def split_dataset():
    src_img_dir = "datasets/measure-detection/images/raw"
    src_lbl_dir = "datasets/measure-detection/labels/raw"
    train_img_dir = "datasets/measure-detection/images/train"
    train_lbl_dir = "datasets/measure-detection/labels/train"
    val_img_dir = "datasets/measure-detection/images/val"
    val_lbl_dir = "datasets/measure-detection/labels/val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    images = [
        f for f in os.listdir(src_img_dir) if f.endswith(".png") or f.endswith(".jpg")
    ]

    # Filter only images that have a matching label
    valid_images = []
    for img in images:
        name = os.path.splitext(img)[0]
        if os.path.exists(os.path.join(src_lbl_dir, name + ".txt")):
            valid_images.append(img)

    print(f"Found {len(valid_images)} valid image-label pairs.")

    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_images)

    val_size = int(len(valid_images) * 0.2)
    val_images = set(valid_images[:val_size])
    train_images = valid_images[val_size:]

    print(f"Copying {len(train_images)} to train, {len(val_images)} to val.")

    for img in valid_images:
        name = os.path.splitext(img)[0]
        lbl = name + ".txt"
        dst_img_dir = val_img_dir if img in val_images else train_img_dir
        dst_lbl_dir = val_lbl_dir if img in val_images else train_lbl_dir
        shutil.copy2(os.path.join(src_img_dir, img), os.path.join(dst_img_dir, img))
        shutil.copy2(os.path.join(src_lbl_dir, lbl), os.path.join(dst_lbl_dir, lbl))


if __name__ == "__main__":
    split_dataset()
