import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment variable for OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.config import Config

# Import YOLO and EasyOCR
from ultralytics import YOLO
import easyocr


class ALPRPipeline:
    """Complete ALPR Pipeline: Detection → Recognition"""

    def __init__(self, detector_path=None, use_gpu=True):
        print(f"\n{'=' * 60}")
        print("INITIALIZING ALPR PIPELINE")
        print(f"{'=' * 60}\n")

        self.use_gpu = use_gpu

        # Load detector
        if detector_path is None:
            detector_path = str(Config.DETECTOR_DIR / "yolov8_detector5" / "weights" / "best.pt")

        print(f"📥 Loading detector from: {detector_path}")
        self.detector = YOLO(detector_path)
        print(f"✅ Detector loaded")

        # Load OCR
        print(f"\n📥 Loading EasyOCR (this may take a minute on first run)...")
        self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=use_gpu)
        print(f"✅ EasyOCR loaded")

        print(f"\n{'=' * 60}")
        print("✅ ALPR Pipeline Ready!")
        print(f"{'=' * 60}\n")

    def detect_plates(self, image, conf_threshold=0.25):
        """Detect license plates in image"""
        results = self.detector(image, conf=conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf)
                })

        return detections

    def recognize_text(self, image):
        """Recognize text using EasyOCR"""
        results = self.reader.readtext(image)

        if not results:
            return {'text': '', 'confidence': 0.0}

        # Get best result
        best_result = max(results, key=lambda x: x[2])

        return {
            'text': best_result[1],
            'confidence': float(best_result[2])
        }

    def process_image(self, image_path, save_output=True, output_dir=None):
        """Process single image: detect + recognize"""
        print(f"\n{'=' * 60}")
        print(f"Processing: {os.path.basename(image_path)}")
        print(f"{'=' * 60}\n")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load: {image_path}")
            return None

        h, w = image.shape[:2]
        print(f"📐 Image size: {w}x{h}")

        # Detect plates
        print(f"🔍 Detecting plates...")
        detections = self.detect_plates(image)
        print(f"   Found {len(detections)} plate(s)")

        # Process each detection
        results = []
        for idx, det in enumerate(detections, 1):
            print(f"\n📋 Plate {idx}/{len(detections)}")
            print(f"   BBox: {det['bbox']}")
            print(f"   Detection confidence: {det['confidence']:.3f}")

            # Crop plate
            x1, y1, x2, y2 = det['bbox']
            plate_img = image[y1:y2, x1:x2]

            # Recognize text
            print(f"   🔤 Recognizing text...")
            recognition = self.recognize_text(plate_img)

            print(f"   Text: '{recognition['text']}'")
            print(f"   OCR confidence: {recognition['confidence']:.3f}")

            results.append({
                'detection': det,
                'recognition': recognition,
                'plate_image': plate_img
            })

        # Draw results
        annotated = self._draw_results(image.copy(), results)

        # Save if requested
        if save_output and output_dir:
            self._save_results(image_path, results, annotated, output_dir)

        return {
            'image_path': image_path,
            'num_plates': len(results),
            'results': results,
            'annotated_image': annotated
        }

    def _draw_results(self, image, results):
        """Draw bounding boxes and text"""
        for result in results:
            bbox = result['detection']['bbox']
            text = result['recognition']['text']
            conf = result['detection']['confidence']

            x1, y1, x2, y2 = bbox

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw text
            label = f"{text} ({conf:.2f})"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image

    def _save_results(self, image_path, results, annotated, output_dir):
        """Save results to disk"""
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Save annotated image
        out_path = os.path.join(output_dir, f"{base_name}_result.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"\n💾 Saved: {out_path}")

        # Save JSON
        json_data = []
        for idx, r in enumerate(results, 1):
            json_data.append({
                'plate_id': idx,
                'bbox': r['detection']['bbox'],
                'confidence': r['detection']['confidence'],
                'text': r['recognition']['text'],
                'ocr_confidence': r['recognition']['confidence']
            })

        json_path = os.path.join(output_dir, f"{base_name}_data.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='ALPR Pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--output', type=str, default='results/alpr_output', help='Output directory')

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ALPRPipeline()

    # Process image
    result = pipeline.process_image(args.image, save_output=True, output_dir=args.output)

    if result:
        # Show result
        cv2.imshow('Result', result['annotated_image'])
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()