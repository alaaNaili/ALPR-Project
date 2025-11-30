import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import sys

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.alpr_pipeline import ALPRPipeline

# Initialize pipeline (only once)
print("Initializing ALPR system...")
pipeline = ALPRPipeline()
print("System ready!")


def process_image(image):
    """
    Process uploaded image

    Args:
        image: numpy array from Gradio

    Returns:
        annotated image and results text
    """
    if image is None:
        return None, "Please upload an image"

    try:
        # Convert RGB to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect plates
        detections = pipeline.detect_plates(image_bgr)

        if len(detections) == 0:
            return image, "❌ No license plates detected"

        # Process each detection
        results_text = f"✅ Found {len(detections)} license plate(s)\n\n"

        for idx, det in enumerate(detections, 1):
            # Crop plate
            x1, y1, x2, y2 = det['bbox']
            plate_img = image_bgr[y1:y2, x1:x2]

            # Recognize text
            recognition = pipeline.recognize_text(plate_img)

            # Add to results
            results_text += f"📋 Plate {idx}:\n"
            results_text += f"   Text: {recognition['text']}\n"
            results_text += f"   Detection Confidence: {det['confidence']:.2%}\n"
            results_text += f"   OCR Confidence: {recognition['confidence']:.2%}\n\n"

            # Draw on image
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{recognition['text']} ({det['confidence']:.2f})"
            cv2.putText(image_bgr, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Convert back to RGB for display
        annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        return annotated_rgb, results_text

    except Exception as e:
        return image, f"❌ Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="License Plate Recognition System") as demo:
    gr.Markdown(
        """
        # 🚗 Automatic License Plate Recognition (ALPR)

        Upload an image to detect and recognize license plates.

        **Features:**
        - 🎯 YOLOv8 Detection
        - 🔤 EasyOCR Recognition (Chinese + English)
        - 📊 Confidence Scores
        """
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Image",
                type="numpy",
                height=400
            )

            with gr.Row():
                clear_btn = gr.ClearButton()
                submit_btn = gr.Button("🚀 Detect & Recognize", variant="primary")

        with gr.Column():
            output_image = gr.Image(
                label="Results",
                type="numpy",
                height=400
            )

            output_text = gr.Textbox(
                label="Detection Details",
                lines=10,
                max_lines=15
            )

    # Examples
    gr.Markdown("### 📸 Try these examples:")
    gr.Examples(
        examples=[
            "data/processed/unified_dataset/test/images/ccpd_test_000001.jpg",
            "data/processed/unified_dataset/test/images/ccpd_test_000010.jpg",
            "data/processed/unified_dataset/test/images/alpr_test_000001.jpg",
        ],
        inputs=input_image,
        label="Sample Images"
    )

    # Button actions
    submit_btn.click(
        fn=process_image,
        inputs=input_image,
        outputs=[output_image, output_text]
    )

    clear_btn.add([input_image, output_image, output_text])

    gr.Markdown(
        """
        ---
        ### ℹ️ About
        This system uses:
        - **YOLOv8** for license plate detection
        - **EasyOCR** for text recognition
        - Trained on **CCPD2019** + **Kaggle ALPR** datasets
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",  # localhost
        server_port=7860,
        share=True,  # Set to True to get public link
        show_error=True
    )