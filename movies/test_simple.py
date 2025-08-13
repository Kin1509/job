"""
Simple test script to verify all components work
"""
import cv2
import torch
from ultralytics import YOLO
import gradio as gr
import numpy as np

def test_components():
    """Test all components individually"""
    print("🔧 Testing components...")
    
    # Test OpenCV
    try:
        print("✅ OpenCV version:", cv2.__version__)
    except Exception as e:
        print("❌ OpenCV error:", e)
        return False
    
    # Test PyTorch
    try:
        print("✅ PyTorch version:", torch.__version__)
        print("✅ CUDA available:", torch.cuda.is_available())
    except Exception as e:
        print("❌ PyTorch error:", e)
        return False
    
    # Test YOLOv8
    try:
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 loaded successfully")
    except Exception as e:
        print("❌ YOLOv8 error:", e)
        return False
    
    # Test Gradio
    try:
        import gradio
        print("✅ Gradio version:", gradio.__version__)
    except Exception as e:
        print("❌ Gradio error:", e)
        return False
    
    return True

def simple_demo():
    """Simple Gradio demo without camera"""
    def process_image(image):
        if image is None:
            return None, "No image provided"
        
        try:
            # Load YOLOv8
            model = YOLO('yolov8n.pt')
            
            # Run detection
            results = model(image, conf=0.5)
            
            # Draw results
            annotated = results[0].plot()
            
            # Count persons
            person_count = 0
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        if int(box.cls[0]) == 0:  # Person class
                            person_count += 1
            
            return annotated, f"Detected {person_count} person(s)"
            
        except Exception as e:
            return image, f"Error: {str(e)}"
    
    # Create interface
    interface = gr.Interface(
        fn=process_image,
        inputs=gr.Image(type="numpy"),
        outputs=[
            gr.Image(type="numpy"),
            gr.Textbox(label="Status")
        ],
        title="🎬 Movies - Simple Detection Test",
        description="Upload an image to test person detection"
    )
    
    return interface

def main():
    """Main function"""
    print("🎬 Movies - Component Test")
    print("=" * 40)
    
    # Test components
    if not test_components():
        print("\n❌ Some components failed. Please fix dependencies first.")
        return
    
    print("\n✅ All components working!")
    print("🚀 Starting simple demo...")
    
    try:
        demo = simple_demo()
        demo.launch(
            server_port=7861,  # Different port to avoid conflicts
            share=False,
            show_error=True
        )
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    main()
