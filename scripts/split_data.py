import os
import random
import shutil

def split_dataset():
    img_dir = "datasets/measure-detection/images/train"
    lbl_dir = "datasets/measure-detection/labels/train"
    val_img_dir = "datasets/measure-detection/images/val"
    val_lbl_dir = "datasets/measure-detection/labels/val"

    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    images = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    
    # Filter only matching labels
    valid_images = []
    for img in images:
        name = os.path.splitext(img)[0]
        lbl = name + ".txt"
        if os.path.exists(os.path.join(lbl_dir, lbl)):
            valid_images.append(img)
    
    print(f"Found {len(valid_images)} valid image-label pairs.")

    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_images)
    
    val_size = int(len(valid_images) * 0.2)
    val_images = valid_images[:val_size]
    
    print(f"Moving {len(val_images)} images to validation set.")

    for img in val_images:
        name = os.path.splitext(img)[0]
        lbl = name + ".txt"
        
        # Move image
        shutil.move(os.path.join(img_dir, img), os.path.join(val_img_dir, img))
        # Move label
        shutil.move(os.path.join(lbl_dir, lbl), os.path.join(val_lbl_dir, lbl))

if __name__ == "__main__":
    split_dataset()
