from flask import Flask, render_template, Response, request
import cv2
import torch

app = Flask(__name__)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Or use custom model path

# Globals
cap = None
streaming = False

# Generate frames from camera
def gen_frames():
    global cap
    while streaming and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO detection
        results = model(frame)
        results.render()

        # Convert RGB to BGR (to avoid blue tint)
        annotated_frame = results.ims[0]
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap, streaming
    if not streaming:
        cap = cv2.VideoCapture(0)
        streaming = True
    return '', 204

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap, streaming
    streaming = False
    if cap and cap.isOpened():
        cap.release()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
