# 🧠 Object Detection & Recognition using YOLOv5

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv5](https://img.shields.io/badge/YOLO-v5-brightgreen)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-blue)

> A complete real-time object detection system using the YOLOv5 architecture, trained on the COCO dataset and implemented with PyTorch, OpenCV, and Jupyter Notebooks.

---

## 🗓 Project Timeline
**Duration:** January 2025 – February 2025  
**Author:** Anuj Mishra  
**Version:** 1.0.0  
**Status:** ✅ Production-ready

--

## 📚 Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Features](#-key-features)
- [Project Structure](#-project-structure)
- [Sample Outputs](#-sample-outputs)
- [Setup Instructions](#-setup-instructions)
- [Run Detection](#-run-inference)
- [Training a Model](#-training-details)
- [Working with Custom Datasets](#-custom-datasets)
- [Evaluation Metrics](#-evaluation-metrics)
- [Model Comparison](#-yolov5-variant-comparison)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview

YOLOv5 is one of the fastest and most accurate object detection models. This project demonstrates:
- Training YOLOv5 from scratch
- Evaluating detection accuracy
- Inference on webcam, videos, and images
- Transfer learning on custom datasets

---

## 🧰 Tech Stack

| Category         | Tools & Frameworks                      |
|------------------|------------------------------------------|
| Language         | Python 3.8+                             |
| Deep Learning    | PyTorch, YOLOv5                         |
| Data Handling    | NumPy, Pandas                           |
| Visualization    | OpenCV, Matplotlib                      |
| Development      | Jupyter Notebook, Git                   |
| Deployment-ready | CLI Inference Scripts                   |

---

## ✨ Key Features

- ✅ 80-class detection using COCO pre-trained weights
- 🧠 Trained YOLOv5s using transfer learning
- 🔬 Visualized bounding boxes and confidence scores
- 🧪 Evaluated with IoU, mAP@0.5, precision, recall
- 🔄 Easily switch between `image`, `video`, or `webcam` input
- 🗃️ Dataset-ready for custom training via `YAML` configuration
- 📈 Learning curves and loss graphs for model convergence

---

## 🗂 Project Structure

```
yolov5-object-detection/
├── yolov5/                # YOLOv5 cloned repo (Ultralytics)
├── data/                  # Dataset configs & YAML files
├── notebooks/             # EDA, experiments, training notebooks
├── outputs/               # Predicted images, bounding boxes
├── detect.py              # CLI for object detection
├── train.py               # Training script using COCO/custom data
├── custom.yaml            # Custom dataset YAML config
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🖼 Sample Outputs

| Original Image | Detected Result |
|----------------|-----------------|
| ![](outputs/sample1.jpg) | ![](outputs/result1.jpg) |
| ![](outputs/sample2.jpg) | ![](outputs/result2.jpg) |

---

## 📦 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/yolov5-object-detection.git
cd yolov5-object-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Clone YOLOv5 repo
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

---

## ▶️ Run Inference

Detect objects on webcam, image, or video using:

```bash
python detect.py \
  --weights yolov5s.pt \
  --img 640 \
  --conf 0.25 \
  --source path/to/image.jpg    # Can also be 0 for webcam or video.mp4
```

---

## 🧠 Training Details

```bash
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data coco.yaml \
  --weights yolov5s.pt \
  --cache
```

- **Architecture:** YOLOv5s
- **Epochs:** 100
- **Optimizer:** SGD
- **Losses:** GIoU, obj, cls
- **Device:** CUDA / GPU enabled

---

## 🛠 Custom Datasets

To train on your own data:
1. Label images using [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/tzutalin/labelImg)
2. Create a `data/custom.yaml` file:
```yaml
train: ../data/train/images
val: ../data/val/images
nc: 5
names: ['car', 'dog', 'helmet', 'phone', 'person']
```
3. Train using:
```bash
python train.py --img 640 --batch 16 --epochs 50 --data custom.yaml --weights yolov5s.pt
```

---

## 📊 Evaluation Metrics

| Metric       | Description |
|--------------|-------------|
| **mAP@0.5**  | Mean Average Precision @ IoU 0.5 |
| **IoU**      | Intersection over Union |
| **Precision**| TP / (TP + FP) |
| **Recall**   | TP / (TP + FN) |

---

## 📈 YOLOv5 Variant Comparison

| Model     | Speed (FPS) | mAP@0.5 | Size     |
|-----------|-------------|---------|----------|
| YOLOv5s   | ✅ Fastest   | 36.7    | 14.0 MB  |
| YOLOv5m   | Medium      | 44.5    | 41.0 MB  |
| YOLOv5l   | Slower      | 47.7    | 77.0 MB  |
| YOLOv5x   | Slowest     | 50.1    | 160.0 MB |

> Use YOLOv5s for real-time deployment, YOLOv5x for highest accuracy.

---

## 🙌 Acknowledgements

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [COCO Dataset](https://cocodataset.org)
- [Roboflow](https://roboflow.com)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [LabelImg](https://github.com/tzutalin/labelImg)

---

## 📬 Contact

**Anuj Mishra**  
📫 [LinkedIn](https://www.linkedin.com/in/anujmishra05)  
🌐 [Portfolio](https://professional-portfolio-plum.vercel.app/)  
🐙 [GitHub](https://github.com/Anujmishra2005)

---

## 🤝 Contributing

We welcome contributions!

```bash
# Fork this repository
# Create a new branch
git checkout -b feature/your-feature

# Commit your changes
git commit -m "Added a new feature"

# Push to your fork
git push origin feature/your-feature

# Open a Pull Request
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

> If you found this project helpful, don’t forget to ⭐ it on GitHub and share it with others!
