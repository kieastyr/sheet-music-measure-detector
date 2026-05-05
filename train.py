from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
import torch


def _inject_nonuniform_scale():
    """Albumentations をパッチして非均一スケール augmentation (0.8x–2.0x) を注入する。
    x/y を独立にサンプリングするため縦横比は維持されない。
    albumentations >= 1.3 の A.Affine(scale=dict) を使用。
    """
    try:
        import albumentations as A

        def patched_init(self, p=1.0, **kwargs):
            self.p = p
            self.transform = None
            self.contains_spatial = True
            try:
                self.transform = A.Compose(
                    [A.Affine(scale={"x": (0.8, 2.0), "y": (0.8, 2.0)}, p=0.8)],
                    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
                )
                print("albumentations: 非均一スケール aug (x/y 各 0.8x–2.0x) を設定しました")
            except Exception as e:
                print(f"albumentations: transform の設定に失敗しました: {e}")

        Albumentations.__init__ = patched_init
    except ImportError:
        print("albumentations が未インストールのため非均一スケール aug をスキップします")


def train_measure_detector():
    # Load a pretrained YOLOv8s model
    model = YOLO("yolov8s.pt")

    # Determine device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on device: {device}")

    _inject_nonuniform_scale()

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        plots=True,
        device=device,
        degrees=1.0,
    )
    
    # Evaluate model performance on the validation set
    metrics = model.val()
    
    # Export the model
    model.export(format="onnx")

if __name__ == "__main__":
    train_measure_detector()
