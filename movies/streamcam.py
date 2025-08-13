import cv2

# Link RTSP với '#' đã được mã hóa thành %23
url = "rtsp://admin:UNV123456%23@192.168.21.43:554/ch01"

# Mở luồng video
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("Không thể kết nối tới camera. Kiểm tra lại đường dẫn hoặc mạng.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Mất kết nối hoặc không nhận được dữ liệu từ camera.")
        break
    
    # Hiển thị khung hình
    cv2.imshow("Camera Stream", frame)
    
    # Nhấn ESC (27) để thoát
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
