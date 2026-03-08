from ultralytics import YOLO

def train_measure_detector():
    # Load a pretrained YOLOv8n model (Nano is good for testing)
    model = YOLO("yolov8n.pt")

    # Train the model
    # epochs: Number of training epochs
    # imgsz: Input image size (depends on resolution of score images)
    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        plots=True,
    )
    
    # Evaluate model performance on the validation set
    metrics = model.val()
    
    # Export the model to ONNX format for deployment
    model.export(format="onnx")

if __name__ == "__main__":
    train_measure_detector()
