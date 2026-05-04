from ultralytics import YOLO
import torch

def train_measure_detector():
    # Load a pretrained YOLOv8n model
    model = YOLO("yolov8n.pt")

    # Determine device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Train the model
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        plots=True,
        device=device,
    )
    
    # Evaluate model performance on the validation set
    metrics = model.val()
    
    # Export the model
    model.export(format="onnx")

if __name__ == "__main__":
    train_measure_detector()
