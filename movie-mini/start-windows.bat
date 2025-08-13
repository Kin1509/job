@echo off
echo Starting Movie-Mini Streaming System for Windows...

REM Create temp directory if not exists
if not exist "temp" mkdir temp

REM Check if Docker is running
docker version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Building and starting services...
docker-compose up -d --build

echo.
echo ========================================
echo Movie-Mini Streaming System Started!
echo ========================================
echo.
echo Web Interface: http://localhost:8000
echo NGINX Streams: http://localhost:8080
echo go2rtc API: http://localhost:1984
echo Redis: localhost:6379
echo.
echo Press any key to view logs...
pause

echo.
echo Showing service logs (Press Ctrl+C to exit):
docker-compose logs -f
