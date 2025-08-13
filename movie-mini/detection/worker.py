import cv2
import redis
import json
import requests
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
API_URL = os.getenv("API_URL", "http://api:8000")
GO2RTC_API = os.getenv("GO2RTC_API", "http://go2rtc:1984")

# Initialize Redis
redis_client = redis.from_url(REDIS_URL)

# Initialize YOLO models
print("Loading YOLO models...")
yolo_model = YOLO('yolov8n.pt')  # Lightweight model for better performance

# YOLO class names for human detection
HUMAN_CLASSES = [0]  # person class
FACE_MODEL = None  # Will use YOLO for face detection as well

print("YOLO models loaded successfully!")

def detect_humans(frame):
    """Detect humans in frame using YOLOv8"""
    results = yolo_model(frame, classes=HUMAN_CLASSES, conf=0.5)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                detection = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence),
                    "class": "person",
                    "class_id": class_id
                }
                detections.append(detection)
    
    return detections

def detect_faces(frame):
    """Detect faces in frame using OpenCV Haar Cascade (fallback) or YOLO"""
    # Using OpenCV's built-in face detection as YOLO doesn't have face class
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    detections = []
    for (x, y, w, h) in faces:
        detection = {
            "bbox": [int(x), int(y), int(x + w), int(y + h)],
            "confidence": 0.8,  # Default confidence for Haar cascade
            "class": "face",
            "class_id": -1  # Custom face class
        }
        detections.append(detection)
    
    return detections

def process_stream_frame(stream_url, detection_type):
    """Process single frame from stream for detection"""
    try:
        # Open video stream
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            print(f"Failed to open stream: {stream_url}")
            return []
        
        # Read frame
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"Failed to read frame from stream: {stream_url}")
            return []
        
        # Perform detection based on type
        if detection_type == "human":
            detections = detect_humans(frame)
        elif detection_type == "face":
            detections = detect_faces(frame)
        else:
            detections = []
        
        return detections
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return []

def send_detection_result(camera_id, detection_type, detections):
    """Send detection results back to API"""
    try:
        result_data = {
            "camera_id": camera_id,
            "detection_type": detection_type,
            "detections": detections,
            "timestamp": datetime.now().isoformat()
        }
        
        response = requests.post(
            f"{API_URL}/detection_result",
            json=result_data,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✅ Sent {len(detections)} {detection_type} detections for {camera_id}")
        else:
            print(f"❌ Failed to send results: {response.status_code}")
            
    except Exception as e:
        print(f"Error sending results: {e}")

def process_detection_task(task_data):
    """Process a single detection task"""
    try:
        camera_id = task_data["camera_id"]
        stream_url = task_data["stream_url"]
        detection_type = task_data["detection_type"]
        
        print(f"🔍 Processing {detection_type} detection for {camera_id}")
        
        # Process frame and get detections
        detections = process_stream_frame(stream_url, detection_type)
        
        # Send results back to API
        send_detection_result(camera_id, detection_type, detections)
        
    except Exception as e:
        print(f"Error processing detection task: {e}")

def main():
    """Main worker loop"""
    print("🚀 Detection worker started!")
    print(f"Connecting to Redis: {REDIS_URL}")
    print(f"API URL: {API_URL}")
    
    while True:
        try:
            # Check for new tasks in Redis queue
            task_data = redis_client.brpop("detection_tasks", timeout=5)
            
            if task_data:
                # Parse task data
                task_json = task_data[1].decode('utf-8')
                task = json.loads(task_json)
                
                # Process the detection task
                process_detection_task(task)
            else:
                # No tasks, wait a bit
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("🛑 Worker stopped by user")
            break
        except Exception as e:
            print(f"❌ Worker error: {e}")
            time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    main()
