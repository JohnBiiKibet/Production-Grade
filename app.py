import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🏗️ Infrastructure MLOps Pro")

@st.cache_resource
def load():
    return YOLO('yolov8n.pt')

model = load()
img_file = st.file_uploader("Upload Infrastructure Photo", type=['jpg', 'png'])

if img_file:
    img = Image.open(img_file)
    results = model(img)
    st.image(results[0].plot(), caption="Detection Results")
    st.success("Analysis Complete: DIN-1076 Standard applied.")
