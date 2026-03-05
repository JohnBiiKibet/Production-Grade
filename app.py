import gradio as gr
from ultralytics import YOLO
import os

# Load a lightweight model for infrastructure object detection
# Note: yolov8n.pt will automatically download to the container on first run
model = YOLO('yolov8n.pt') 

def infra_analysis(image):
    if image is None:
        return None, "Please upload an image."
        
    # 1. Computer Vision: Identify objects
    results = model(image)
    # Using results[0].plot() returns a numpy array which Gradio handles well
    annotated_img = results[0].plot()
    
    # 2. RAG Logic: Simulated retrieval
    advice = "⚠️ Maintenance Protocol: Detected anomalies require inspection within 48h per DIN-1076 standard."
    
    return annotated_img, advice

# Senior UI Design
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏗️ Infrastructure MLOps Pipeline")
    gr.Markdown("Proof of Concept: Computer Vision + RAG for Mobility & Infrastructure.")
    
    with gr.Row():
        # type="numpy" is essential for YOLO processing
        input_img = gr.Image(label="Upload Infrastructure Photo", type="numpy")
        output_img = gr.Image(label="CV Object Detection")
        
    output_text = gr.Textbox(label="RAG Maintenance Guidance")
    btn = gr.Button("Analyze System", variant="primary")
    
    btn.click(fn=infra_analysis, inputs=input_img, outputs=[output_img, output_text])

# CORRECTED FOR HUGGING FACE SPACES
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False  # Hugging Face doesn't allow share=True
    )
