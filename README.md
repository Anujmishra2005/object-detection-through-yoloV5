# 🧠 Object Detection & Recognition using YOLOv5

A solo end-to-end deep learning project demonstrating real-time object detection and classification using the **YOLOv5 (You Only Look Once)** architecture. This project is built to show how to **train, evaluate, and deploy** an object detection system using **PyTorch**, **OpenCV**, and **Jupyter Notebook**, with the **COCO dataset** as the training source.

---

## 📅 Project Timeline
**Duration:** January 2025 – February 2025  
**Author:** Anuj Mishra  
**Status:** Completed ✅

---

## 🚀 Overview

This project leverages **YOLOv5s** for detecting and recognizing objects across **80 object categories** from the COCO dataset. It includes:
- Data preprocessing
- Model training
- Evaluation with industry-standard metrics
- Real-time inference
- Customizability for future datasets

---

## 🧰 Tech Stack

| Category              | Tools & Frameworks                                           |
|-----------------------|--------------------------------------------------------------|
| Deep Learning         | PyTorch, YOLOv5                                              |
| Computer Vision       | OpenCV, Roboflow, COCO Dataset                               |
| Visualization         | Matplotlib, Seaborn                                          |
| Development Tools     | Jupyter Notebook, Python (venv), Git                         |
| Deployment Readiness  | CLI Interface, Webcam/Video/Image Input Support              |

---

## ✨ Key Features

- 🧹 **Data Preprocessing**: COCO dataset cleaned, normalized, and augmented.
- 🏷️ **Multi-Class Detection**: Supports detection of 80 classes like `person`, `car`, `dog`, etc.
- 🎯 **Hyperparameter Tuning**: Custom learning rate, batch size, and training epochs.
- ⚡ **Optimized Training**: Utilized GPU acceleration with PyTorch’s AMP for mixed precision.
- 📸 **Real-Time Detection**: Detects objects from webcam, video streams, and static images.
- 📊 **Evaluation Metrics**: Supports `mAP@0.5`, `IoU`, `Precision`, and `Recall`.
- 🧪 **Visual Debugging**: Draws bounding boxes with confidence scores using OpenCV.
- 📁 **Custom Dataset Support**: Ready-to-train on your own data via Roboflow or LabelImg.

---

## 🖼 Sample Outputs

| Input Image | YOLOv5 Prediction |
|-------------|-------------------|
| ![sample1](outputs/sample1.jpg) | ![result1](outputs/result1.jpg) |
| ![sample2](outputs/sample2.jpg) | ![result2](outputs/result2.jpg) |

---

## 🗂 Project Structure

```
yolov5-object-detection/
├── yolov5/                  # YOLOv5 repo from Ultralytics
├── data/                    # Contains dataset config & YAML
├── notebooks/               # Jupyter notebooks for EDA and experiments
├── outputs/                 # Inference results and prediction images
├── detect.py                # Script to run detection on inputs
├── train.py                 # YOLOv5 model training script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🧪 Evaluation Metrics

| Metric       | Description                                      |
|--------------|--------------------------------------------------|
| **mAP@0.5**  | Mean Average Precision at 0.5 IoU                |
| **IoU**      | Intersection over Union for overlap accuracy     |
| **Precision**| Fraction of correct positive predictions         |
| **Recall**   | Fraction of actual positives correctly detected  |

---

## 📦 Setup Instructions

```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/yolov5-object-detection.git
cd yolov5-object-detection

# Step 2: Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # On Windows use: venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Clone YOLOv5 from Ultralytics
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

---

## ▶️ Run Inference

Detect objects from an image, video, or webcam using the command line:

```bash
# Run detection on image/video/webcam
python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source input.jpg
```

**Parameters:**
- `--weights`: Path to YOLOv5 trained model (default: `yolov5s.pt`)
- `--img`: Input image resolution (e.g., 640)
- `--conf`: Confidence threshold (default: 0.25)
- `--source`: Input source (image path, video file, webcam index)

---

## 📁 Dataset Used

- **Dataset:** [COCO 2017](https://www.kaggle.com/datasets/andrewmvd/coco2017)
- **Source:** Kaggle / COCO Official
- **Content:** 200K+ labeled images across 80 categories

E.g.:
- Person
- Bicycle
- Car
- Dog
- Truck
- Airplane
- TV, Laptop, Bottle, Chair... etc.

---

## 🧠 Training Details

- **Architecture:** YOLOv5s (smallest variant)
- **Batch Size:** 16
- **Epochs:** 100
- **Optimizer:** SGD
- **Learning Rate:** 0.01
- **Loss Functions:** GIoU loss, Objectness loss, Classification loss

---

## 🙌 Acknowledgements

Special thanks to:
- [Ultralytics – YOLOv5](https://github.com/ultralytics/yolov5)
- [COCO Dataset](https://cocodataset.org/)
- [Roboflow](https://roboflow.com)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

## 💬 Contact

📧 **Anuj Mishra**  
🔗 [LinkedIn](https://www.linkedin.com/in/anujmishra05/)  
🌐 [Portfolio Website](https://professional-portfolio-plum.vercel.app/)  
🐙 [GitHub](https://github.com/Anujmishra2005)

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# Step 1: Fork this repo
# Step 2: Create a feature branch
git checkout -b feature/your-feature

# Step 3: Make your changes and commit
git commit -m "Added new feature X"

# Step 4: Push to GitHub
git push origin feature/your-feature

# Step 5: Open a Pull Request!
```

---

## ⭐️ Show Your Support

If you found this project helpful, please consider **starring** 🌟 the repository. It helps others discover the project and motivates further improvements.

---
