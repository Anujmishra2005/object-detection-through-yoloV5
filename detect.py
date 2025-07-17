import torch
import cv2

model = torch.hub.load('yolov5', 'custom', path='runs/train/exp/weights/best.pt', source='local')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    results.render()

    cv2.imshow('YOLOv5 Face Detection', results.imgs[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()