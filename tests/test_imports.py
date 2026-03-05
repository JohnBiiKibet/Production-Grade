import pytest

def test_streamlit_imports():
    """Verify core Streamlit and MLOps dependencies are present."""
    try:
        import streamlit as st
        import psutil
        from PIL import Image
        print(f"✅ Success: Streamlit {st.__version__} is ready.")
    except ImportError as e:
        pytest.fail(f"❌ Streamlit Dependency Missing: {e}")

def test_ml_imports():
    """Verify YOLO and CV dependencies are present."""
    try:
        from ultralytics import YOLO
        import cv2
        print("✅ Success: YOLO and OpenCV are ready.")
    except ImportError as e:
        pytest.fail(f"❌ ML Dependency Missing: {e}")
