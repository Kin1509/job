"""
Configuration settings for the movies project
"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Camera settings
    RTSP_URL: str = "rtsp://admin:UNV123456%23@192.168.21.43:554/ch01"
    CAMERA_WIDTH: int = 1280
    CAMERA_HEIGHT: int = 720
    CAMERA_FPS: int = 25
    
    # YOLOv8 model settings
    YOLO_MODEL: str = "yolov8n.pt"  # nano model for faster inference
    YOLO_FACE_MODEL: str = "yolov8n-face.pt"  # face detection model
    CONFIDENCE_THRESHOLD: float = 0.5
    IOU_THRESHOLD: float = 0.45
    
    # API settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    
    # Gradio settings
    GRADIO_HOST: str = "0.0.0.0"
    GRADIO_PORT: int = 7860
    GRADIO_SHARE: bool = False
    
    # Detection settings
    DETECT_PERSON: bool = True
    DETECT_FACE: bool = True
    MAX_DETECTIONS: int = 100
    
    # Performance settings
    USE_GPU: bool = True
    DEVICE: str = "cuda:0"  # or "cpu"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
