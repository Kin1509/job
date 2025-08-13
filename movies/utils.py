"""
Utility functions for video processing and detection
"""
import cv2
import numpy as np
import base64
import io
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def encode_frame_to_base64(frame):
    """
    Encode OpenCV frame to base64 string
    """
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    except Exception as e:
        logger.error(f"Error encoding frame: {e}")
        return None

def decode_base64_to_frame(base64_string):
    """
    Decode base64 string to OpenCV frame
    """
    try:
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Error decoding base64: {e}")
        return None

def resize_frame(frame, max_width=1280, max_height=720):
    """
    Resize frame while maintaining aspect ratio
    """
    height, width = frame.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized
    
    return frame

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union

def non_max_suppression(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to remove overlapping detections
    """
    if len(detections) == 0:
        return []
    
    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while detections:
        current = detections.pop(0)
        keep.append(current)
        
        # Remove overlapping detections
        detections = [
            d for d in detections
            if calculate_iou(current['bbox'], d['bbox']) < iou_threshold
        ]
    
    return keep

def format_detection_stats(detections):
    """
    Format detection statistics for display
    """
    stats = {}
    for det in detections:
        class_name = det['class']
        if class_name not in stats:
            stats[class_name] = 0
        stats[class_name] += 1
    
    return stats

def validate_rtsp_url(url):
    """
    Validate RTSP URL format
    """
    if not url:
        return False
    
    if not url.startswith('rtsp://'):
        return False
    
    return True

def test_camera_connection(url, timeout=5):
    """
    Test if camera connection is available
    """
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout * 1000)
        
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        
        return False
    except Exception as e:
        logger.error(f"Error testing camera connection: {e}")
        return False

def get_video_info(url):
    """
    Get video stream information
    """
    try:
        cap = cv2.VideoCapture(url)
        
        if not cap.isOpened():
            return None
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
        
    except Exception as e:
        logger.error(f"Error getting video info: {e}")
        return None
