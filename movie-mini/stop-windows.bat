@echo off
echo Stopping Movie-Mini Streaming System...

docker-compose down

echo.
echo ========================================
echo Movie-Mini System Stopped!
echo ========================================
echo.
echo To clean up completely (remove volumes):
echo docker-compose down -v
echo.
pause
