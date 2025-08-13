# Movie-Mini Streaming System

A simplified video streaming system for camera management with AI detection, optimized for Windows.

## 🎯 Features

- **Camera Stream Management**: Add, list, and delete RTSP camera streams
- **Multi-Protocol Support**: RTSP → RTMP/HLS/WebRTC conversion
- **AI Detection**: Human detection (YOLOv8) và Face detection (OpenCV)
- **Web Interface**: Modern UI for stream management và detection control
- **RESTful API**: Complete API for stream operations
- **Windows Compatible**: Optimized for Windows Docker Desktop

## 🏗️ Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   FastAPI   │────│   go2rtc    │────│    NGINX    │    │   Redis     │
│  (Port 8000)│    │ (Port 1984) │    │ (Port 8080) │    │ (Port 6379) │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                     │                     │              │
      │                     │                     │              │
   Stream API          Stream Convert        Stream Serve    Task Queue
      │                                                          │
      └──────────────────────────────────────────────────────────┘
                                │
                    ┌─────────────────────┐
                    │  Detection Worker   │
                    │   (YOLOv8 + OpenCV) │
                    └─────────────────────┘
```

## 🚀 Cách Chạy Trên Windows

### Prerequisites
- **Docker Desktop for Windows** (đã cài đặt và đang chạy)
- **Git** (để clone project)
- **RTSP camera URLs** để test

### 🔧 Installation

1. **Clone project**:
```bash
git clone <repository-url>
cd movie-mini
```

2. **Tạo thư mục temp** (Windows compatibility):
```bash
mkdir temp
```

3. **Khởi chạy hệ thống** (sử dụng script Windows):
```bash
start-windows.bat
```

**Hoặc chạy manual:**
```bash
docker-compose up -d --build
```

4. **Truy cập web interface**:
```
http://localhost:8000
```

### 🛑 Dừng Hệ Thống

```bash
stop-windows.bat
```

**Hoặc manual:**
```bash
docker-compose down
```

## 📡 API Endpoints

### Stream Management
- `POST /start_streams` - Thêm camera streams với detection options
- `GET /streams` - List active streams
- `DELETE /streams` - Xóa streams
- `GET /health` - Health check

### AI Detection
- `POST /detect` - Start detection task cho camera
- `GET /detection_results/{camera_id}` - Xem kết quả detection

### Example: Add Stream với Detection
```bash
curl -X POST "http://localhost:8000/start_streams" \
-H "Content-Type: application/json" \
-d '[{
  "camera_id": "cam001",
  "url": "rtsp://admin:password@192.168.1.100/stream",
  "detect_human": true,
  "detect_face": true
}]'
```

## 🎬 Stream URLs

Sau khi thêm camera, bạn có thể xem streams qua:

- **HLS**: `http://localhost:8080/api/stream.m3u8?src={camera_id}`
- **WebRTC**: `http://localhost:8080/webrtc.html?src={camera_id}`
- **RTSP**: `rtsp://localhost:8554/{camera_id}`

## 🤖 AI Detection Features

### Human Detection
- **Model**: YOLOv8n (lightweight)
- **Class**: Person (class ID: 0)
- **Confidence**: 0.5 threshold
- **Output**: Bounding boxes với confidence scores

### Face Detection  
- **Model**: OpenCV Haar Cascade
- **Method**: `detectMultiScale`
- **Output**: Face bounding boxes

### Detection Workflow
1. Thêm camera với detection enabled
2. Worker tự động process frames từ stream
3. Kết quả gửi về API qua Redis
4. Xem results trong web interface

## 🛠️ Development

### Local Development (Windows)
```bash
# API Service
cd api
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Detection Worker
cd detection  
pip install -r requirements.txt
python worker.py
```

### Project Structure
```
movie-mini/
├── docker-compose.yml       # Service orchestration (5 services)
├── start-windows.bat        # Windows start script
├── stop-windows.bat         # Windows stop script
├── .env.example            # Environment configuration
├── README.md               # This documentation
├── temp/                   # Windows temp directory
├── api/                    # FastAPI application
│   ├── main.py            # Main API (550 lines) với detection
│   ├── Dockerfile         # API container
│   └── requirements.txt   # Python dependencies
├── detection/              # YOLOv8 Detection Worker
│   ├── worker.py          # Detection processing (178 lines)
│   ├── Dockerfile         # Worker container
│   └── requirements.txt   # YOLO + OpenCV dependencies
├── go2rtc/                # Stream conversion
│   └── go2rtc.yaml        # Stream converter config
└── nginx/                 # Web server
    └── nginx.conf         # Proxy và serve streams
```

## 🔧 Configuration

### Environment Variables
- `IP`: Server IP (default: host.docker.internal for Windows)
- `NGINX_PORT`: NGINX port (default: 8080)
- `RTSP_PORT`: RTSP output port (default: 8554)
- `GO2RTC_PORT`: go2rtc API port (default: 1984)
- `REDIS_URL`: Redis connection string

### Windows-Specific Settings
- Network: `host.docker.internal` cho inter-service communication
- Volumes: Windows-compatible path mounting
- Temp directory: `./temp` cho temporary files

## 🐛 Troubleshooting

### Common Issues

1. **Docker not running**:
   - Start Docker Desktop
   - Run `start-windows.bat`

2. **Port conflicts**:
   - Check ports 8000, 8080, 1984, 6379 không bị sử dụng
   - Modify ports trong `docker-compose.yml` nếu cần

3. **RTSP connection failed**:
   - Verify camera URL và credentials
   - Check network connectivity
   - Test với VLC player trước

4. **Detection not working**:
   - Check Redis connection trong `/health`
   - View detection worker logs: `docker-compose logs detection-worker`

## 📝 Usage Examples

### 1. Add Camera với Human Detection
```javascript
// Via Web Interface
Camera ID: cam001
RTSP URL: rtsp://admin:password@192.168.1.100/stream  
☑️ Detect Human
```

### 2. Manual Detection Start
```javascript
// Select camera từ dropdown
// Choose "Human Detection" 
// Click "Start Detection"
```

### 3. View Detection Results
```javascript
// Select camera
// Click "View Results"
// Auto-refresh mỗi 10 giây
```

## 🎉 What's New vs Original

**Added Features:**
- ✅ Windows compatibility
- ✅ YOLOv8 human detection  
- ✅ OpenCV face detection
- ✅ Redis task queue
- ✅ Detection results API
- ✅ Enhanced web UI
- ✅ Windows batch scripts

**Simplified Architecture:**
- ❌ No video recording/segmentation
- ❌ No PTZ control
- ❌ No complex video processing
- ❌ No MongoDB (uses in-memory storage)
- ❌ No bounding box rendering on video

Perfect cho testing và development camera streaming với AI detection trên Windows!
