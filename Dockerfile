# Use Python base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download YOLOv5 repo
RUN python -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)"

# Expose port
EXPOSE 8080

# Run app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app.main:app"]
