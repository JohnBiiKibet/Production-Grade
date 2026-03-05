import gradio as gr
from ultralytics import YOLO

model = YOLO('yolov8n.pt') 

def infra_analysis(image):
    if image is None: return None, "Upload an image."
    results = model(image)
    return results[0].plot(), "⚠️ Maintenance Protocol: Inspection required within 48h."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏗️ Infrastructure MLOps Pipeline")
    with gr.Row():
        input_img = gr.Image(label="Upload Photo", type="numpy")
        output_img = gr.Image(label="Detection")
    output_text = gr.Textbox(label="Guidance")
    btn = gr.Button("Analyze System", variant="primary")
    btn.click(fn=infra_analysis, inputs=input_img, outputs=[output_img, output_text])

if __name__ == "__main__":
    # share=False is critical on Hugging Face
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
