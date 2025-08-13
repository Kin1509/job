from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import requests
import redis
import json
import uuid
from datetime import datetime

app = FastAPI(title="Movie-Mini Streaming API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GO2RTC_API = os.getenv("GO2RTC_API", "http://go2rtc:1984")
NGINX_PORT = os.getenv("NGINX_PORT", "8080")
IP = os.getenv("IP", "host.docker.internal")
RTSP_PORT = os.getenv("RTSP_PORT", "8554")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# Redis connection
try:
    redis_client = redis.from_url(REDIS_URL)
except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None

# Pydantic models
class StreamRequest(BaseModel):
    camera_id: str
    url: str
    detect_human: bool = False
    detect_face: bool = False

class StreamInfo(BaseModel):
    camera_id: str
    input: str
    rtmp_url: str
    hls_url: str
    webrtc_url: str
    detect_human: bool = False
    detect_face: bool = False

class StreamResponse(BaseModel):
    success: bool
    streams: List[StreamInfo]

class DeleteStreamsRequest(BaseModel):
    ids: List[str]

class DeleteStreamsResponse(BaseModel):
    success: bool
    message: str
    deleted_streams: List[str]

class DetectionRequest(BaseModel):
    camera_id: str
    detection_type: str  # "human" or "face"

class DetectionResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None

class DetectionResult(BaseModel):
    camera_id: str
    detection_type: str
    detections: List[Dict[str, Any]]
    timestamp: str

# In-memory storage for detection results (in production, use database)
detection_results = {}

# Default camera configuration
DEFAULT_CAMERA = {
    "camera_id": "cam001",
    "url": "rtsp://admin:UNV123456%23@192.168.21.43:554/ch01",
    "detect_human": True,
    "detect_face": True
}

async def initialize_default_camera():
    """Initialize default camera on startup"""
    try:
        print("🎬 Initializing default camera...")
        
        # Check if go2rtc is available
        response = requests.get(f"{GO2RTC_API}/api/streams", timeout=5)
        
        # Add default camera
        stream_request = StreamRequest(**DEFAULT_CAMERA)
        await start_streams([stream_request])
        
        print(f"✅ Default camera '{DEFAULT_CAMERA['camera_id']}' added successfully!")
        
    except Exception as e:
        print(f"⚠️ Could not add default camera: {e}")

@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    # Wait a bit for go2rtc to be ready
    import asyncio
    await asyncio.sleep(5)
    
    # Initialize default camera
    await initialize_default_camera()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    redis_status = "connected" if redis_client else "disconnected"
    return {
        "status": "healthy", 
        "service": "movie-mini",
        "redis": redis_status
    }

@app.post("/start_streams", response_model=StreamResponse)
async def start_streams(streams: List[StreamRequest]):
    """Start new camera streams with optional detection"""
    try:
        stream_infos = []
        
        for stream in streams:
            stream_name = stream.camera_id
            
            print(f"🎬 Adding stream: {stream_name} -> {stream.url}")
            
            # Validate RTSP URL
            if not stream.url.startswith(('rtsp://', 'rtmp://', 'http://')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid stream URL format: {stream.url}"
                )
            
            # Configure stream in go2rtc
            go2rtc_config = {
                "streams": {
                    stream_name: stream.url
                }
            }
            
            print(f"📡 Configuring go2rtc with: {go2rtc_config}")
            
            # Add stream to go2rtc
            try:
                response = requests.patch(
                    f"{GO2RTC_API}/api/config",
                    json=go2rtc_config,
                    timeout=10
                )
                
                print(f"📊 go2rtc response: {response.status_code} - {response.text}")
                
            except requests.exceptions.RequestException as e:
                print(f"❌ go2rtc connection error: {e}")
                raise HTTPException(
                    status_code=503,
                    detail=f"Cannot connect to go2rtc service: {str(e)}"
                )
            
            if response.status_code == 200:
                # Create stream info
                stream_info = StreamInfo(
                    camera_id=stream.camera_id,
                    input=stream.url,
                    rtmp_url=f"rtsp://{IP}:{RTSP_PORT}/{stream.camera_id}",
                    hls_url=f"http://{IP}:{NGINX_PORT}/api/stream.m3u8?src={stream.camera_id}",
                    webrtc_url=f"http://{IP}:{NGINX_PORT}/webrtc.html?src={stream.camera_id}",
                    detect_human=stream.detect_human,
                    detect_face=stream.detect_face
                )
                stream_infos.append(stream_info)
                
                print(f"✅ Stream {stream.camera_id} configured successfully")
                
                # Start detection tasks if requested
                if stream.detect_human and redis_client:
                    task_data = {
                        "camera_id": stream.camera_id,
                        "stream_url": stream.url,
                        "detection_type": "human"
                    }
                    redis_client.lpush("detection_tasks", json.dumps(task_data))
                    print(f"🤖 Human detection task queued for {stream.camera_id}")
                
                if stream.detect_face and redis_client:
                    task_data = {
                        "camera_id": stream.camera_id,
                        "stream_url": stream.url,
                        "detection_type": "face"
                    }
                    redis_client.lpush("detection_tasks", json.dumps(task_data))
                    print(f"👤 Face detection task queued for {stream.camera_id}")
                    
            else:
                error_msg = f"go2rtc returned {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to configure stream {stream.camera_id}: {error_msg}"
                )
        
        return StreamResponse(success=True, streams=stream_infos)
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error starting streams: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/streams", response_model=StreamResponse)
async def list_streams():
    """List all active streams"""
    try:
        response = requests.get(f"{GO2RTC_API}/api/streams")
        go2rtc_streams = response.json()
        
        stream_infos = []
        for camera_id, stream_data in go2rtc_streams.items():
            input_url = ""
            if stream_data.get("producers"):
                input_url = stream_data["producers"][0].get("url", "")
            
            stream_info = StreamInfo(
                camera_id=camera_id,
                input=input_url,
                rtmp_url=f"rtsp://{IP}:{RTSP_PORT}/{camera_id}",
                hls_url=f"http://{IP}:{NGINX_PORT}/api/stream.m3u8?src={camera_id}",
                webrtc_url=f"http://{IP}:{NGINX_PORT}/webrtc.html?src={camera_id}",
                detect_human=False,  # Default values
                detect_face=False
            )
            stream_infos.append(stream_info)
        
        return StreamResponse(success=True, streams=stream_infos)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing streams: {str(e)}")

@app.delete("/streams", response_model=DeleteStreamsResponse)
async def delete_streams(request: DeleteStreamsRequest):
    """Delete specified streams"""
    try:
        deleted_streams = []
        
        for camera_id in request.ids:
            # Remove stream from go2rtc
            response = requests.delete(f"{GO2RTC_API}/api/streams?src={camera_id}")
            
            if response.status_code == 200:
                deleted_streams.append(camera_id)
                # Clean up detection results
                if camera_id in detection_results:
                    del detection_results[camera_id]
        
        return DeleteStreamsResponse(
            success=True,
            message=f"Deleted {len(deleted_streams)} streams",
            deleted_streams=deleted_streams
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting streams: {str(e)}")

@app.post("/detect", response_model=DetectionResponse)
async def start_detection(request: DetectionRequest):
    """Start human or face detection for a camera"""
    try:
        if not redis_client:
            raise HTTPException(status_code=500, detail="Redis not available")
        
        # Get stream URL from go2rtc
        response = requests.get(f"{GO2RTC_API}/api/streams")
        streams = response.json()
        
        if request.camera_id not in streams:
            raise HTTPException(status_code=404, detail="Camera stream not found")
        
        stream_data = streams[request.camera_id]
        stream_url = ""
        if stream_data.get("producers"):
            stream_url = stream_data["producers"][0].get("url", "")
        
        # Create detection task
        task_id = str(uuid.uuid4())
        task_data = {
            "task_id": task_id,
            "camera_id": request.camera_id,
            "stream_url": stream_url,
            "detection_type": request.detection_type
        }
        
        # Add task to Redis queue
        redis_client.lpush("detection_tasks", json.dumps(task_data))
        
        return DetectionResponse(
            success=True,
            message=f"Detection task started for {request.camera_id}",
            task_id=task_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting detection: {str(e)}")

@app.get("/detection_results/{camera_id}")
async def get_detection_results(camera_id: str):
    """Get latest detection results for a camera"""
    try:
        if camera_id in detection_results:
            return {
                "success": True,
                "results": detection_results[camera_id]
            }
        else:
            return {
                "success": True,
                "results": []
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting results: {str(e)}")

@app.post("/detection_result")
async def receive_detection_result(result: DetectionResult):
    """Receive detection results from worker (internal endpoint)"""
    try:
        camera_id = result.camera_id
        if camera_id not in detection_results:
            detection_results[camera_id] = []
        
        # Keep only last 10 results per camera
        detection_results[camera_id].append(result.dict())
        if len(detection_results[camera_id]) > 10:
            detection_results[camera_id] = detection_results[camera_id][-10:]
        
        return {"success": True, "message": "Result received"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error receiving result: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve main page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Movie-Mini Streaming</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .stream-item { border: 1px solid #ddd; padding: 20px; margin: 10px 0; border-radius: 8px; background: #fafafa; }
            .detection-item { border: 1px solid #28a745; padding: 15px; margin: 5px 0; border-radius: 5px; background: #f8fff9; }
            button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
            button:hover { background: #0056b3; }
            button.detect { background: #28a745; }
            button.detect:hover { background: #1e7e34; }
            button.danger { background: #dc3545; }
            button.danger:hover { background: #c82333; }
            input, select { padding: 8px; margin: 5px; border: 1px solid #ddd; border-radius: 5px; width: 250px; }
            .checkbox-group { margin: 10px 0; }
            .checkbox-group label { margin-right: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 Movie-Mini Streaming System</h1>
            <p>Camera streaming với AI detection (Human & Face)</p>
            
            <div>
                <h3>📹 Add New Stream</h3>
                <input type="text" id="cameraId" placeholder="Camera ID (e.g., cam001)" />
                <input type="text" id="rtspUrl" placeholder="RTSP URL" />
                <div class="checkbox-group">
                    <label><input type="checkbox" id="detectHuman"> Detect Human</label>
                    <label><input type="checkbox" id="detectFace"> Detect Face</label>
                </div>
                <button onclick="addStream()">Add Stream</button>
            </div>
            
            <div>
                <h3>🔴 Active Streams</h3>
                <button onclick="loadStreams()">Refresh Streams</button>
                <div id="streamsList"></div>
            </div>
            
            <div>
                <h3>🤖 Detection Controls</h3>
                <select id="detectionCameraId">
                    <option value="">Select Camera</option>
                </select>
                <select id="detectionType">
                    <option value="human">Human Detection</option>
                    <option value="face">Face Detection</option>
                </select>
                <button class="detect" onclick="startDetection()">Start Detection</button>
                <button onclick="loadDetectionResults()">View Results</button>
                <div id="detectionResults"></div>
            </div>
        </div>
        
        <script>
            async function addStream() {
                const cameraId = document.getElementById('cameraId').value;
                const rtspUrl = document.getElementById('rtspUrl').value;
                const detectHuman = document.getElementById('detectHuman').checked;
                const detectFace = document.getElementById('detectFace').checked;
                
                if (!cameraId || !rtspUrl) {
                    alert('Please fill in both Camera ID and RTSP URL');
                    return;
                }
                
                try {
                    const response = await fetch('/start_streams', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify([{ 
                            camera_id: cameraId, 
                            url: rtspUrl,
                            detect_human: detectHuman,
                            detect_face: detectFace
                        }])
                    });
                    
                    const result = await response.json();
                    if (result.success) {
                        alert('Stream added successfully!');
                        loadStreams();
                        updateDetectionCameraList();
                        document.getElementById('cameraId').value = '';
                        document.getElementById('rtspUrl').value = '';
                        document.getElementById('detectHuman').checked = false;
                        document.getElementById('detectFace').checked = false;
                    } else {
                        alert('Failed to add stream');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function loadStreams() {
                try {
                    const response = await fetch('/streams');
                    const result = await response.json();
                    
                    const streamsList = document.getElementById('streamsList');
                    streamsList.innerHTML = '';
                    
                    if (result.success && result.streams.length > 0) {
                        result.streams.forEach(stream => {
                            const streamDiv = document.createElement('div');
                            streamDiv.className = 'stream-item';
                            streamDiv.innerHTML = `
                                <h4>📹 ${stream.camera_id}</h4>
                                <p><strong>Input:</strong> ${stream.input}</p>
                                <p><strong>HLS:</strong> <a href="${stream.hls_url}" target="_blank">${stream.hls_url}</a></p>
                                <p><strong>WebRTC:</strong> <a href="${stream.webrtc_url}" target="_blank">${stream.webrtc_url}</a></p>
                                <p><strong>Detection:</strong> Human: ${stream.detect_human ? '✅' : '❌'}, Face: ${stream.detect_face ? '✅' : '❌'}</p>
                                <button class="danger" onclick="deleteStream('${stream.camera_id}')">Delete Stream</button>
                            `;
                            streamsList.appendChild(streamDiv);
                        });
                    } else {
                        streamsList.innerHTML = '<p>No active streams</p>';
                    }
                } catch (error) {
                    console.error('Error loading streams:', error);
                }
            }
            
            async function deleteStream(cameraId) {
                if (!confirm(`Delete stream ${cameraId}?`)) return;
                
                try {
                    const response = await fetch('/streams', {
                        method: 'DELETE',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ ids: [cameraId] })
                    });
                    
                    const result = await response.json();
                    if (result.success) {
                        alert('Stream deleted successfully!');
                        loadStreams();
                        updateDetectionCameraList();
                    } else {
                        alert('Failed to delete stream');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function startDetection() {
                const cameraId = document.getElementById('detectionCameraId').value;
                const detectionType = document.getElementById('detectionType').value;
                
                if (!cameraId) {
                    alert('Please select a camera');
                    return;
                }
                
                try {
                    const response = await fetch('/detect', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            camera_id: cameraId,
                            detection_type: detectionType
                        })
                    });
                    
                    const result = await response.json();
                    if (result.success) {
                        alert(`${detectionType} detection started for ${cameraId}`);
                    } else {
                        alert('Failed to start detection');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function loadDetectionResults() {
                const cameraId = document.getElementById('detectionCameraId').value;
                if (!cameraId) {
                    alert('Please select a camera');
                    return;
                }
                
                try {
                    const response = await fetch(`/detection_results/${cameraId}`);
                    const result = await response.json();
                    
                    const resultsDiv = document.getElementById('detectionResults');
                    resultsDiv.innerHTML = '';
                    
                    if (result.success && result.results.length > 0) {
                        result.results.forEach(detection => {
                            const detectionDiv = document.createElement('div');
                            detectionDiv.className = 'detection-item';
                            detectionDiv.innerHTML = `
                                <h5>🔍 ${detection.detection_type.toUpperCase()} Detection</h5>
                                <p><strong>Camera:</strong> ${detection.camera_id}</p>
                                <p><strong>Time:</strong> ${detection.timestamp}</p>
                                <p><strong>Detections:</strong> ${detection.detections.length} objects found</p>
                                <details>
                                    <summary>View Details</summary>
                                    <pre>${JSON.stringify(detection.detections, null, 2)}</pre>
                                </details>
                            `;
                            resultsDiv.appendChild(detectionDiv);
                        });
                    } else {
                        resultsDiv.innerHTML = '<p>No detection results found</p>';
                    }
                } catch (error) {
                    console.error('Error loading detection results:', error);
                }
            }
            
            async function updateDetectionCameraList() {
                try {
                    const response = await fetch('/streams');
                    const result = await response.json();
                    
                    const select = document.getElementById('detectionCameraId');
                    select.innerHTML = '<option value="">Select Camera</option>';
                    
                    if (result.success && result.streams.length > 0) {
                        result.streams.forEach(stream => {
                            const option = document.createElement('option');
                            option.value = stream.camera_id;
                            option.textContent = stream.camera_id;
                            select.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Error updating camera list:', error);
                }
            }
            
            // Load data on page load
            loadStreams();
            updateDetectionCameraList();
            
            // Auto-refresh detection results every 10 seconds
            setInterval(() => {
                const cameraId = document.getElementById('detectionCameraId').value;
                if (cameraId) {
                    loadDetectionResults();
                }
            }, 10000);
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
