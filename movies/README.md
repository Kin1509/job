# Movies - Real-time Face & Person Detection System

A real-time video streaming and object detection system using YOLOv8 for detecting persons and faces from RTSP camera feeds.

## Features

- 🎥 **Real-time RTSP Camera Streaming** - Connect to IP cameras via RTSP protocol
- 👤 **Person Detection** - Detect persons using YOLOv8 model
- 😊 **Face Detection** - Detect faces using Haar Cascade classifier
- 🖥️ **Gradio Web Interface** - User-friendly web UI for live monitoring
- 🚀 **FastAPI Backend** - RESTful API for detection services
- 📊 **Real-time Statistics** - Live count of detected persons and faces
- ⚙️ **Configurable Parameters** - Adjust confidence thresholds and detection settings

## Project Structure

```
movies/
├── streamcam.py        # Original camera streaming script
├── main.py            # Gradio demo application
├── detection_api.py   # FastAPI detection service
├── config.py          # Configuration settings
├── utils.py           # Utility functions
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

1. **Clone the repository**
```bash
cd d:\job\movies
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download YOLOv8 model** (will auto-download on first run)
The system will automatically download the YOLOv8 nano model (`yolov8n.pt`) when you first run the application.

## Configuration

Edit `config.py` or create a `.env` file to customize settings:

```python
# Camera settings
RTSP_URL = "rtsp://admin:UNV123456%23@192.168.21.43:554/ch01"
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# API settings
API_PORT = 8001

# Gradio settings
GRADIO_PORT = 7860
```

## Usage

### Option 1: Run Gradio Demo (Recommended)

```bash
python main.py
```

This will:
- Load YOLOv8 models
- Start the Gradio web interface at http://localhost:7860
- Connect to the RTSP camera
- Display real-time detection results

### Option 2: Run FastAPI Service

```bash
python detection_api.py
```

This will start the API service at http://localhost:8001

API Endpoints:
- `GET /` - API info
- `GET /health` - Health check
- `POST /detect` - Upload image for detection
- `POST /detect_frame` - Detect objects in base64 encoded frame

### Option 3: Simple Camera Stream

```bash
python streamcam.py
```

This runs the basic camera streaming without detection.

## Web Interface Features

The Gradio interface provides:

1. **Live Camera Feed** - Real-time video display with bounding boxes
2. **Stream Control** - Start/stop camera streaming
3. **Detection Stats** - Live count of persons and faces
4. **Settings Panel**:
   - Confidence threshold adjustment
   - Auto-refresh toggle
   - Refresh rate control

## API Usage Example

```python
import requests
import base64

# Encode image to base64
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode()

# Send to API
response = requests.post(
    "http://localhost:8001/detect_frame",
    json={"image": image_base64}
)

detections = response.json()["detections"]
```

## System Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster inference)
- Network access to RTSP camera
- Windows/Linux/macOS

## Performance Tips

1. **Use GPU acceleration** - Install CUDA and PyTorch with GPU support
2. **Adjust model size** - Use `yolov8n.pt` for speed, `yolov8x.pt` for accuracy
3. **Optimize confidence threshold** - Lower values detect more objects but may have false positives
4. **Reduce camera resolution** - Lower resolution improves processing speed

## Troubleshooting

### Camera Connection Issues
- Verify RTSP URL is correct
- Check network connectivity
- Ensure camera credentials are valid
- Try using VLC to test the RTSP stream

### Detection Not Working
- Ensure models are downloaded
- Check GPU/CUDA availability
- Verify Python dependencies are installed
- Review logs for error messages

### Low FPS
- Reduce camera resolution
- Use smaller YOLOv8 model (nano)
- Increase refresh rate in settings
- Check CPU/GPU usage

## License

This project is for educational and development purposes.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Gradio](https://gradio.app/)
- [FastAPI](https://fastapi.tiangolo.com/)
- OpenCV community
