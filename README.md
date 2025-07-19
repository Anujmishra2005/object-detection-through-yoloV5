# 🧠 Object Detection and Recognition using YOLOv5

An individual project focused on real-time object detection using the YOLOv5 (You Only Look Once) model, trained on the COCO dataset. This project demonstrates how to train, evaluate, and deploy a custom object detection model using PyTorch, OpenCV, and Jupyter Notebook.

## 🗓 Project Duration
**Jan 2025 – Feb 2025**

## 🔧 Tech Stack
- **Machine Learning**
- **Computer Vision**
- **Natural Language Processing**
- **PyTorch**
- **YOLOv5**
- **COCO Dataset (Kaggle)**
- **OpenCV**
- **Jupyter Notebook**

## 📌 Features
- 🧹 Cleaned and preprocessed data to enhance model accuracy.
- 🧠 Trained YOLOv5 model on the COCO dataset to detect **80 object classes**.
- 🚀 Optimized model using PyTorch for faster inference and better precision.
- 🎯 Real-time object detection with high accuracy using bounding boxes.
- 📊 Evaluated model using standard performance metrics (mAP, IoU, Precision, Recall).

## 🗂 Dataset
The [COCO dataset](https://www.kaggle.com/datasets/andrewmvd/coco2017) (Common Objects in Context) is used for training and validation. It contains over 200,000 labeled images across 80 object categories.

## 🖼 Sample Output

| Image | Prediction |
|-------|------------|
| ![sample1](outputs/sample1.jpg) | ![result1](outputs/result1.jpg) |
| ![sample2](outputs/sample2.jpg) | ![result2](outputs/result2.jpg) |

## 📦 Installation

git clone https://github.com/your-username/yolov5-object-detection.git
cd yolov5-object-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

## 🚀 How to Run

python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source your_input_image_or_video.mp4

## 📊 Evaluation Metrics
- mAP (mean Average Precision)
- IoU (Intersection over Union)
- Precision & Recall

## 📁 Project Structure

yolov5-object-detection/
├── data/
├── notebooks/
├── outputs/
├── detect.py
├── train.py
├── requirements.txt
└── README.md

## 🤝 Acknowledgements
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [COCO Dataset](https://cocodataset.org)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

## 📬 Contact
Feel free to reach out to me on [LinkedIn](https://www.linkedin.com/in/anujmishra05/) or check out my [Portfolio](https://professional-portfolio-plum.vercel.app/)

⭐️ Star this repo if you found it useful!
