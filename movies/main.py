"""
Main Gradio demo for real-time face and person detection
"""
import gradio as gr
import cv2
import numpy as np
import base64
import requests
import json
from ultralytics import YOLO
import torch
from threading import Thread, Event
import queue
import time
import logging
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
cap = None
detection_model = None
face_cascade = None
stream_active = False
frame_queue = queue.Queue(maxsize=2)

def initialize_models():
    """Initialize YOLOv8 model and face cascade"""
    global detection_model, face_cascade
    
    try:
        # Load YOLOv8 model
        device = "cuda" if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        logger.info(f"Loading YOLOv8 model on {device}...")
        detection_model = YOLO(settings.YOLO_MODEL)
        detection_model.to(device)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        logger.info("Models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        return False

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        class_name = det['class']
        confidence = det['confidence']
        
        # Choose color based on class
        if class_name == 'person':
            color = (0, 255, 0)  # Green for person
        elif class_name == 'face':
            color = (255, 0, 0)  # Blue for face
        else:
            color = (0, 0, 255)  # Red for others
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence
        label = f"{class_name}: {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
        
        # Draw label background
        cv2.rectangle(frame, (x1, label_y - label_size[1] - 5), 
                     (x1 + label_size[0], label_y + 5), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return frame

def detect_objects_in_frame(frame):
    """Detect persons and faces in a single frame"""
    detections = []
    
    try:
        # Detect persons using YOLOv8
        if detection_model and settings.DETECT_PERSON:
            results = detection_model(frame, conf=settings.CONFIDENCE_THRESHOLD, 
                                    iou=settings.IOU_THRESHOLD, verbose=False)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # Person class
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].tolist()
                            detections.append({
                                'class': 'person',
                                'confidence': conf,
                                'bbox': xyxy
                            })
        
        # Detect faces using Haar Cascade
        if face_cascade is not None and settings.DETECT_FACE:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                detections.append({
                    'class': 'face',
                    'confidence': 0.95,
                    'bbox': [float(x), float(y), float(x+w), float(y+h)]
                })
    
    except Exception as e:
        logger.error(f"Detection error: {e}")
    
    return detections

def camera_stream_worker():
    """Worker thread for capturing camera frames"""
    global cap, stream_active, frame_queue
    
    while stream_active:
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Put frame in queue (drop old frames if queue is full)
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except:
                        pass
                frame_queue.put(frame)
            else:
                logger.warning("Failed to read frame from camera")
                time.sleep(0.1)
        else:
            time.sleep(0.1)

def start_camera():
    """Initialize and start camera capture"""
    global cap, stream_active
    
    try:
        if cap is not None:
            cap.release()
        
        logger.info(f"Connecting to camera: {settings.RTSP_URL}")
        cap = cv2.VideoCapture(settings.RTSP_URL)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return False
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
        
        # Start stream worker thread
        stream_active = True
        thread = Thread(target=camera_stream_worker, daemon=True)
        thread.start()
        
        logger.info("Camera started successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error starting camera: {e}")
        return False

def stop_camera():
    """Stop camera capture"""
    global cap, stream_active
    
    stream_active = False
    time.sleep(0.5)  # Wait for thread to stop
    
    if cap is not None:
        cap.release()
        cap = None
    
    logger.info("Camera stopped")

def process_frame():
    """Process a single frame from camera"""
    global frame_queue
    
    try:
        # Get latest frame from queue
        frame = None
        while not frame_queue.empty():
            frame = frame_queue.get_nowait()
        
        if frame is None:
            return None, "No frame available", 0, 0
        
        # Make a copy for processing
        display_frame = frame.copy()
        
        # Detect objects
        detections = detect_objects_in_frame(frame)
        
        # Count detections
        person_count = sum(1 for d in detections if d['class'] == 'person')
        face_count = sum(1 for d in detections if d['class'] == 'face')
        
        # Draw detections on frame
        display_frame = draw_detections(display_frame, detections)
        
        # Convert BGR to RGB for display
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        status = f"Detected {person_count} person(s) and {face_count} face(s)"
        
        return display_frame, status, person_count, face_count
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        return None, f"Error: {str(e)}", 0, 0

def toggle_stream(is_streaming):
    """Toggle camera stream on/off"""
    if is_streaming:
        if start_camera():
            return True, "Camera started successfully"
        else:
            return False, "Failed to start camera"
    else:
        stop_camera()
        return False, "Camera stopped"

def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    with gr.Blocks(title="Movies - Real-time Detection Demo") as demo:
        gr.Markdown("# 🎬 Movies - Real-time Face & Person Detection")
        gr.Markdown("Detect faces and persons in real-time from RTSP camera stream")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Video display
                video_output = gr.Image(
                    label="Camera Feed",
                    type="numpy",
                    height=540
                )
                
                # Status display
                status_text = gr.Textbox(
                    label="Status",
                    value="Camera not started",
                    interactive=False
                )
            
            with gr.Column(scale=1):
                # Controls
                gr.Markdown("### Controls")
                
                stream_toggle = gr.Checkbox(
                    label="Stream Active",
                    value=False
                )
                
                refresh_btn = gr.Button(
                    "Capture Frame",
                    variant="primary"
                )
                
                gr.Markdown("### Detection Stats")
                
                person_count = gr.Number(
                    label="Persons Detected",
                    value=0,
                    interactive=False
                )
                
                face_count = gr.Number(
                    label="Faces Detected",
                    value=0,
                    interactive=False
                )
                
                gr.Markdown("### Settings")
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=settings.CONFIDENCE_THRESHOLD,
                    step=0.05,
                    label="Confidence Threshold"
                )
                
                auto_refresh = gr.Checkbox(
                    label="Auto Refresh",
                    value=True
                )
                
                refresh_rate = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="Refresh Rate (seconds)"
                )
        
        # Event handlers
        def update_confidence(value):
            settings.CONFIDENCE_THRESHOLD = value
            return f"Confidence threshold set to {value}"
        
        confidence_slider.change(
            update_confidence,
            inputs=[confidence_slider],
            outputs=[status_text]
        )
        
        stream_toggle.change(
            toggle_stream,
            inputs=[stream_toggle],
            outputs=[stream_toggle, status_text]
        )
        
        refresh_btn.click(
            process_frame,
            outputs=[video_output, status_text, person_count, face_count]
        )
        
        # Auto-refresh functionality
        def auto_refresh_frame(should_refresh, rate):
            if should_refresh and stream_active:
                frame, status, persons, faces = process_frame()
                return frame, status, persons, faces
            return None, "Auto-refresh disabled or stream inactive", 0, 0
        
        # Set up periodic refresh
        demo.load(
            auto_refresh_frame,
            inputs=[auto_refresh, refresh_rate],
            outputs=[video_output, status_text, person_count, face_count],
            every=0.5
        )
    
    return demo

def main():
    """Main entry point"""
    logger.info("Starting Movies Detection Demo...")
    
    try:
        # Initialize models
        if not initialize_models():
            logger.error("Failed to initialize models")
            print("❌ Failed to initialize models. Please check your installation.")
            return
        
        print("✅ Models loaded successfully!")
        print("🚀 Starting Gradio interface...")
        
        # Create and launch Gradio interface
        demo = create_gradio_interface()
        
        print(f"🌐 Opening web interface at: http://localhost:{settings.GRADIO_PORT}")
        print("📹 Make sure your camera is accessible before starting stream")
        print("⏹️  Press Ctrl+C to stop the application")
        
        demo.launch(
            server_name=settings.GRADIO_HOST,
            server_port=settings.GRADIO_PORT,
            share=settings.GRADIO_SHARE,
            show_error=True,
            quiet=False,
            debug=False
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        print("\n👋 Application stopped by user")
        stop_camera()
    except Exception as e:
        logger.error(f"Error launching demo: {e}")
        print(f"❌ Error: {e}")
        print("💡 Try running with: python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")
        stop_camera()

if __name__ == "__main__":
    main()
