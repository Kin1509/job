import torch
import clip
from PIL import Image
import gradio as gr
from pyngrok import ngrok, conf
import time
import psutil
import torch.cuda as cuda

# Cấu hình ngrok
conf.get_default().auth_token = "317ev8ZpU7k891jccnXABQOPzLw_3Tvcts8KSEC8Drs4c4Uzb"

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50", device=device)
model.eval()

def get_memory_usage():
    """Lấy thông bộ nhớ CPU/GPU theo GB"""
    # RAM (GB)
    ram_gb = psutil.virtual_memory().used / (1024 ** 3)
    
    # VRAM (GB)
    if device == "cuda":
        vram_gb = cuda.memory_allocated() / (1024 ** 3)
    else:
        vram_gb = 0
    
    return ram_gb, vram_gb

def compare_images(img1, img2, threshold=0.85):
    """Tính toán độ tương đồng và thời gian xử lý"""
    
    
    # Lấy thông số bộ nhớ trước khi xử lý
    ram_before, vram_before = get_memory_usage()
    
    # Tiền xử lý ảnh
    img1_pre = preprocess(img1).unsqueeze(0).to(device)
    img2_pre = preprocess(img2).unsqueeze(0).to(device)
    start_time = time.time()
    # Tính toán độ tương đồng
    with torch.no_grad():
        features1 = model.encode_image(img1_pre)
        features2 = model.encode_image(img2_pre)
        similarity = torch.cosine_similarity(features1, features2).item()
    
    # Tính toán thời gian và bộ nhớ
    latency_ms = (time.time() - start_time) * 1000
    ram_after, vram_after = get_memory_usage()
    
    # Tạo báo cáo
    report = f"""
    ## 📊 Kết quả so sánh ảnh
    - **Độ tương đồng**: {similarity*100:.2f}%
    - **Thời gian xử lý**: {latency_ms:.2f} ms
    
    ## 🖥️ Bộ nhớ sử dụng
    - **RAM**: {max(ram_before, ram_after):.2f} GB
    - **VRAM**: {max(vram_before, vram_after):.2f} GB
    """
    return report

# Tạo giao diện
demo = gr.Interface(
    fn=compare_images,
    inputs=[
        gr.Image(type="pil", label="Ảnh 1"),
        gr.Image(type="pil", label="Ảnh 2"),
        gr.Slider(0.5, 1.0, value=0.85, label="Ngưỡng tương đồng")
    ],
    outputs=gr.Markdown(label="Kết quả"),
    title="🖼️ So sánh ảnh bằng AI"
)

# Chạy ứng dụng
if __name__ == "__main__":
    public_url = ngrok.connect(7860)
    print("👉 Public URL:", public_url)
    demo.launch(server_name="0.0.0.0", server_port=7860)