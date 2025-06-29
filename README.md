# 🧠📸 Real-Time Custom Object Detection with YOLOv5

A complete pipeline for training a **custom YOLOv5 model** 🏋️‍♂️, parsing annotated data 🔖, and performing **real-time object detection** 🎯 via your webcam 🎥 — all in one script!

---

## 🚀 Features

- 📁 Auto-organizes your dataset from a simple `O.D.F` list
- 🏷️ Automatically generates YOLO-format labels
- 🤖 Trains YOLOv5 on your custom images
- 🧪 Uses the trained model for real-time webcam detection
- ✅ Fallback to pre-trained YOLOv5s if no custom model exists
- 🔄 Detects common fruits and supports custom classes

---

## 📂 Directory Structure

<pre> ```bash custom_dataset/ ├── images/ # Your dataset images ├── labels/ # YOLOv5 formatted label files ├── data.yaml # Auto-generated training configuration models/ └── custom_yolov5.pt # Trained model saved here O.D.F # List of image filenames to include in the dataset ``` </pre>

---

## 📦 Requirements

- Python 3.8+
- OpenCV
- PyTorch
- `torchvision`
- `ultralytics/yolov5` (via `torch.hub`)
- `PyYAML`

Install dependencies:

```bash
pip install torch torchvision opencv-python pyyaml

---

## 📜 How It Works

📄 O.D.F Parsing:
List your image files in O.D.F. The script moves them to the dataset folder and creates default labels (you can customize later).

📦 Model Loading:
If a custom_yolov5.pt model exists, it’s loaded. If not, the default YOLOv5s model is used.

🎥 Webcam Detection:
Real-time object detection is run using your webcam. Results are shown live with class names and confidence scores.

🧠 Training:
If new images are added, the model is retrained on the updated dataset using YOLOv5's training loop.

---

## controls: 

Press q to quit the webcam window.

To retrain, just add new images to O.D.F and run again.

---

##label Format (YOLO)

All values normalized between 0 and 1.

---

## 🛠️ Customization

To support multiple classes, modify label generation in parse_odf()

To add class names, adjust the get_classes() function to return a proper names list

---

## 📌 Notes

For best results, organize a proper train/val split

Add real annotations instead of placeholder boxes for serious training

---

## ❤️ Inspired by

Ultralytics YOLOv5

Big thanks to the community!

