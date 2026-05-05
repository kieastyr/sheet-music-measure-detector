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
                    [
                        A.Affine(scale={"x": (0.8, 1.4), "y": (0.8, 2.0)}, p=0.8),
                        # # ページの反り・波打ちを模倣（sigma 大 = 大きくなだらかな波）
                        # A.ElasticTransform(alpha=40, sigma=25, p=0.4),
                        # スキャン品質のばらつきを模倣（楽譜は白黒なので contrast が特に重要）
                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.5),
                        # スキャンノイズ
                        A.GaussNoise(p=0.3),
                        # ピンぼけ・解像度低下
                        A.Blur(blur_limit=3, p=0.2),
                    ],
                    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
                )
                print("albumentations: scale / brightness-contrast / noise / blur aug を設定しました")
            except Exception as e:
                print(f"albumentations: transform の設定に失敗しました: {e}")

        Albumentations.__init__ = patched_init
    except ImportError:
        print("albumentations が未インストールのため非均一スケール aug をスキップします")


def train_measure_detector():
    # Load a pretrained YOLO26m model
    # model = YOLO("runs/detect/train1s/weights/best.pt")
    model = YOLO("yolo26l.pt")

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
        patience=20,   # 20エポック改善なしで早期終了
        imgsz=1280,
        batch=4,       # OOM 対策（VRAM に余裕があれば 8 に上げると速い）
        plots=True,
        device=device,
        degrees=1.0,   # barline は縦線なので大きい回転は有害
        mosaic=1.0,    # 4枚合成で位置バリエーション（デフォルト値、明示）
        scale=0.3,     # Albumentations の非均一 scale と重複するため抑え気味
        copy_paste=0.3,   # インスタンスを別画像にコピーして増強   
    )
    
    # Evaluate model performance on the validation set
    metrics = model.val()
    
    # Export the model
    model.export(format="onnx")

if __name__ == "__main__":
    train_measure_detector()
