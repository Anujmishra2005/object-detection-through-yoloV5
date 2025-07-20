# 🧠 Object Detection and Recognition using YOLOv5

A solo project focused on real-time object detection using **YOLOv5** (You Only Look Once), trained on the **COCO 2017** dataset. This project demonstrates how to **train, evaluate, and deploy** a deep learning model for object detection using PyTorch, OpenCV, and Jupyter Notebook.

---

## 📅 Project Duration
**January 2025 – February 2025**

---

## 🛠️ Tech Stack
- 🔍 **Computer Vision**
- 🧠 **Machine Learning**
- 🗣️ **Natural Language Processing**
- ⚙️ **YOLOv5 (v5s)**
- 🔥 **PyTorch**
- 🖼️ **COCO Dataset (via Kaggle)**
- 📷 **OpenCV**
- 📓 **Jupyter Notebook**

---

## 🌟 Key Features
- ✅ **Cleaned and preprocessed** the COCO dataset for optimal training.
- 📌 **Trained YOLOv5s** on 80 object categories with real-world variability.
- 🎯 **Fine-tuned hyperparameters**: learning rate, batch size, epochs, etc.
- ⚡ **Optimized GPU training** using PyTorch with AMP (Automatic Mixed Precision).
- 📷 **Real-time object detection** on images, videos, and webcam feed.
- 📈 **Evaluated with industry-standard metrics**: mAP@0.5, IoU, Precision, Recall.
- 🧾 **Visualized bounding boxes** and labels using OpenCV and Matplotlib.
- ☁️ **Ready for custom dataset training** using tools like Roboflow or LabelImg.

---

## 📂 Dataset

Used the [COCO 2017 dataset](https://www.kaggle.com/datasets/andrewmvd/coco2017) (Common Objects in Context), containing:
- 200,000+ labeled images
- 80 distinct object categories
- Diverse contexts, angles, and lighting conditions

---

## 🖼️ Sample Outputs

| Input Image | YOLOv5 Prediction |
|-------------|-------------------|
| ![sample1](outputs/sample1.jpg) | ![result1](outputs/result1.jpg) |
| ![sample2](outputs/sample2.jpg) | ![result2](outputs/result2.jpg) |

---

## 📦 Installation Guide

```bash
# Clone your repo
git clone https://github.com/your-username/yolov5-object-detection.git
cd yolov5-object-detection

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # use venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Clone YOLOv5 from Ultralytics
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source your_input_image_or_video.mp4
```

---

## 📊 Evaluation Metrics
- `mAP@0.5` (Mean Average Precision)
- `IoU` (Intersection over Union)
- `Precision` and `Recall`

---

## 🗂️ Project Structure

```
yolov5-object-detection/
├── data/
├── notebooks/
├── outputs/
├── detect.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 🙏 Acknowledgements
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [COCO Dataset](https://cocodataset.org)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

## 📬 Contact

- 🔗 [LinkedIn – Anuj Mishra](https://www.linkedin.com/in/anujmishra05/)
- 🌐 [Portfolio Website](https://professional-portfolio-plum.vercel.app/)

---

## 🌟 Contributing

Contributions are welcome! Here's how you can contribute:

```bash
# Fork the repository
git checkout -b feature/your-feature-name

# Make changes and commit
git commit -m "Add your feature"

# Push your branch
git push origin feature/your-feature-name

# Create a Pull Request
```

---

⭐️ **Star this repository** if you found it helpful!
