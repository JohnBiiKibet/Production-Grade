import pytest
import sys

def test_dependency_compatibility():
    """Verify that Gradio and huggingface_hub can co-exist without ImportError."""
    try:
        import huggingface_hub
        import gradio as gr
        # This is the specific line that was failing in your logs
        from huggingface_hub import whoami
        print(f"✅ Success: Gradio {gr.__version__} and HF Hub {huggingface_hub.__version__} are compatible.")
    except ImportError as e:
        pytest.fail(f"❌ Dependency Conflict Detected: {e}")

def test_model_initialization():
    """Ensure the YOLO model can load in the CI environment."""
    from ultralytics import YOLO
    try:
        model = YOLO('yolov8n.pt')
        assert model is not None
        print("✅ Success: YOLOv8 model initialized.")
    except Exception as e:
        pytest.fail(f"❌ Model Loading Failed: {e}")
