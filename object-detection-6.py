import torch
import cv2
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import yaml

# Path configuration
CUSTOM_DATASET_DIR = "custom_dataset"
IMAGE_DIR = os.path.join(CUSTOM_DATASET_DIR, "images")
LABEL_DIR = os.path.join(CUSTOM_DATASET_DIR, "labels")
MODEL_DIR = "models"
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "custom_yolov5.pt")
ODF_FILE_PATH = "O.D.F"

# Ensure directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Parse O.D.F file and generate labels
def parse_odf():
    if not os.path.exists(ODF_FILE_PATH):
        print(f"Error: {ODF_FILE_PATH} not found.")
        return

    with open(ODF_FILE_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                image_name = line
                label_name = os.path.splitext(image_name)[0] + ".txt"

                # Move image to dataset folder
                src_image_path = Path(image_name)
                dst_image_path = Path(IMAGE_DIR) / src_image_path.name
                if src_image_path.exists():
                    os.rename(src_image_path, dst_image_path)

                    # Generate label file (placeholder YOLO format)
                    with open(Path(LABEL_DIR) / label_name, "w") as label_file:
                        class_id = 0  # Assign class 0 to all items (customize as needed)
                        label_file.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                else:
                    print(f"Warning: {src_image_path} not found.")

# Load YOLOv5 model
if os.path.exists(CURRENT_MODEL_PATH):
    print("Loading custom trained model...")
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=CURRENT_MODEL_PATH)
else:
    print("No custom model found. Using pre-trained model.")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Training function
def train_model():
    print("Starting training...")
    data_yaml = {
        "train": IMAGE_DIR,
        "val": IMAGE_DIR,  # Using the same data for simplicity; consider a separate validation set
        "nc": len(get_classes()),
        "names": get_classes()
    }
    
    yaml_path = os.path.join(CUSTOM_DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f)

    # Train YOLOv5 model
    torch.hub.load('ultralytics/yolov5', 'train',
                   data=yaml_path, epochs=10, weights='yolov5s', save_dir=MODEL_DIR)

# Helper to get unique classes from labels
def get_classes():
    classes = set()
    for label_file in Path(LABEL_DIR).glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                class_id = int(line.split()[0])
                classes.add(class_id)
    return sorted(classes)

# Parse O.D.F file
parse_odf()

# Webcam setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from webcam.")
        break

    # Convert the frame to RGB (YOLOv5 expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(rgb_frame)

    # Parse results
    detections = results.pandas().xyxy[0]  # Bounding boxes, confidence, and class

    for _, detection in detections.iterrows():
        x1, y1, x2, y2, confidence, class_name = (
            int(detection["xmin"]),
            int(detection["ymin"]),
            int(detection["xmax"]),
            int(detection["ymax"]),
            detection["confidence"],
            detection["name"],
        )

        # Draw bounding box
        color = (0, 255, 0) if class_name in ["apple", "banana", "orange", "kiwi"] else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Add label
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame
    cv2.imshow('Object Detection', frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Train model on new data if added
if len(list(Path(IMAGE_DIR).glob("*.jpg"))) > 0:
    train_model()
    print("Model training completed.")
