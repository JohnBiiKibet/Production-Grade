import gradio as gr
from ultralytics import YOLO
import cv2

# Senior Logic: CV Model (Loading a lightweight infra-ready model)
model = YOLO('yolov8n.pt') 

def analyze_infrastructure(image):
    # 1. Computer Vision Detection
    results = model(image)
    annotated_img = results[0].plot()
    detections = [model.names[int(c)] for c in results[0].boxes.cls]

    # 2. Simulated RAG Logic (Bridge to Mobility Manuals)
    # In a full project, this queries a Vector DB
    query = ", ".join(detections) if detections else "General Inspection"
    maintenance_advice = f"Maintence Protocol for {query}: Inspect every 6 months as per DIN-1076."

    return annotated_img, maintenance_advice

demo = gr.Interface(
    fn=analyze_infrastructure,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(label="CV Detection"), gr.Textbox(label="RAG Maintenance Advice")],
    title="Infrastructure Data Science Prototype"
)

if __name__ == "__main__":
    demo.launch()
