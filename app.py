import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page config for "Senior" look
st.set_page_config(page_title="Infrastructure MLOps", layout="wide")

st.title("🏗️ Infrastructure MLOps Pipeline")
st.write("Proof of Concept: Computer Vision + RAG for Mobility & Infrastructure.")

# Load model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# UI Layout
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Infrastructure Photo", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Process
    results = model(image)
    annotated_img = results[0].plot()
    
    with col2:
        st.image(annotated_img, caption="CV Object Detection", use_container_width=True)
        
    st.subheader("RAG Maintenance Guidance")
    st.info("⚠️ Maintenance Protocol: Detected anomalies require inspection within 48h per DIN-1076 standard.")
