import torch
from ultralytics import YOLO
import shutil
import os

def main():
    # -------------------------------
    # Cleanup previous training runs
    # -------------------------------
    # Remove old training artifacts to avoid mixing results
    # WARNING: This deletes ALL previous runs
    if os.path.exists("runs"):
        shutil.rmtree("runs")

    # -------------------------------
    # Hardware sanity check
    # -------------------------------
    # Confirm whether CUDA (GPU) is available
    print("CUDA available:", torch.cuda.is_available())

    # -------------------------------
    # Model initialization
    # -------------------------------
    # Load a pretrained YOLO model as a starting point
    # Transfer learning speeds up convergence and improves accuracy
    model = YOLO("yolo11n.pt")

    # -------------------------------
    # Training configuration
    # -------------------------------
    model.train(
        # Dataset configuration file (classes, paths, splits)
        data="dataset/1/GARBAGE CLASSIFICATION/data.yaml",

        # Number of training epochs
        # Lower value reduces overfitting for small datasets
        epochs=10,

        # Early stopping:
        # Training stops if validation loss does not improve
        # for 2 consecutive epochs
        patience=2,

        # Dropout regularization:
        # Randomly disables 10% of neurons during training
        # Helps prevent overfitting and co-adaptation
        dropout=0.1,

        # Input image resolution
        # Higher values = better accuracy, slower training
        imgsz=640,

        # Device selection:
        # -1 lets Ultralytics automatically choose GPU if available
        device=-1,

        # Name of the experiment (used in runs/ directory)
        name="garbage_cls"
    )
    # https://www.ultralytics.com/glossary/overfitting 
    # https://www.ultralytics.com/blog/what-is-overfitting-in-computer-vision-how-to-prevent-it

    # -------------------------------
    # Save best trained model
    # -------------------------------
    # Ultralytics automatically saves the best model based on validation metrics
    best_model = "runs/detect/garbage_cls/weights/best.pt"

    # Ensure destination directory exists
    os.makedirs("models", exist_ok=True)

    # Copy best model to a stable, version-controlled location
    shutil.copy(best_model, "models/yolo11n-garbage.pt")

    print("✅ Model saved: models/yolo11n-garbage.pt")

# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":
    main()
