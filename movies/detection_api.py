"""
FastAPI service for YOLOv8 face and person detection
"""
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
from PIL import Image
import io
import base64
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Movies Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models
class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    
class DetectionResponse(BaseModel):
    success: bool
    detections: List[Detection]
    total_persons: int
    total_faces: int
    message: str = ""

# Global model variables
person_model = None
face_model = None

def load_models():
    """Load YOLOv8 models for person and face detection"""
    global person_model, face_model
    
    try:
        # Set device
        device = settings.DEVICE if torch.cuda.is_available() and settings.USE_GPU else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load person detection model (standard YOLOv8)
        logger.info("Loading person detection model...")
        person_model = YOLO(settings.YOLO_MODEL)
        person_model.to(device)
        
        # For face detection, we'll use the same model but filter for persons
        # In production, you'd use a specialized face detection model
        logger.info("Loading face detection model...")
        # Note: yolov8n-face.pt would need to be trained specifically for faces
        # For now, we'll use the standard model
        face_model = YOLO(settings.YOLO_MODEL)
        face_model.to(device)
        
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    if not load_models():
        logger.error("Failed to load models on startup")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Movies Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": person_model is not None and face_model is not None
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect persons and faces in uploaded image
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        detections = []
        total_persons = 0
        total_faces = 0
        
        # Detect persons
        if settings.DETECT_PERSON and person_model:
            results = person_model(image, conf=settings.CONFIDENCE_THRESHOLD, iou=settings.IOU_THRESHOLD)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        # Class 0 is person in COCO dataset
                        if cls == 0:
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].tolist()
                            
                            detections.append(Detection(
                                class_name="person",
                                confidence=conf,
                                bbox=xyxy
                            ))
                            total_persons += 1
        
        # For face detection, we need a specialized model
        # Here we're using a simple approach with cascade classifier
        if settings.DETECT_FACE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                detections.append(Detection(
                    class_name="face",
                    confidence=0.9,  # Cascade doesn't provide confidence
                    bbox=[float(x), float(y), float(x+w), float(y+h)]
                ))
                total_faces += 1
        
        return DetectionResponse(
            success=True,
            detections=detections[:settings.MAX_DETECTIONS],
            total_persons=total_persons,
            total_faces=total_faces,
            message=f"Detected {total_persons} persons and {total_faces} faces"
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return DetectionResponse(
            success=False,
            detections=[],
            total_persons=0,
            total_faces=0,
            message=str(e)
        )

@app.post("/detect_frame")
async def detect_frame(frame_data: Dict[str, Any]):
    """
    Detect objects in a base64 encoded frame
    Used for real-time detection from video stream
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(frame_data.get("image", ""))
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Invalid image data")
        
        detections = []
        
        # Detect persons
        if settings.DETECT_PERSON and person_model:
            results = person_model(image, conf=settings.CONFIDENCE_THRESHOLD, iou=settings.IOU_THRESHOLD, verbose=False)
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # Person class
                            conf = float(box.conf[0])
                            xyxy = box.xyxy[0].tolist()
                            detections.append({
                                "class": "person",
                                "confidence": conf,
                                "bbox": xyxy
                            })
        
        # Detect faces
        if settings.DETECT_FACE:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                detections.append({
                    "class": "face",
                    "confidence": 0.9,
                    "bbox": [float(x), float(y), float(x+w), float(y+h)]
                })
        
        return JSONResponse(content={
            "success": True,
            "detections": detections
        })
        
    except Exception as e:
        logger.error(f"Frame detection error: {e}")
        return JSONResponse(content={
            "success": False,
            "detections": [],
            "error": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
