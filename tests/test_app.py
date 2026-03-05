import pytest
import sys

def test_imports():
    """Check if core dependencies are correctly installed and compatible."""
    try:
        import gradio
        import ultralytics
        from huggingface_hub import whoami
        print("✅ Core imports successful!")
    except ImportError as e:
        pytest.fail(f"Dependency Error: {e}. Check requirements.txt version pinning.")

def test_cv_model_loading():
    """Verify the YOLO model can initialize without crashing the container."""
    from ultralytics import YOLO
    try:
        model = YOLO('yolov8n.pt')
        assert model is not None
        print("✅ CV Model initialized successfully!")
    except Exception as e:
        pytest.fail(f"Model Loading Error: {e}")
