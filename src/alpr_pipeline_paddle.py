import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import cv2
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from ultralytics import YOLO
from paddleocr import PaddleOCR


class ALPRPipelinePaddle:
    """ALPR Pipeline with PaddleOCR (Better for Chinese plates)"""

    def __init__(self, detector_path=None):
        print(f"\n{'=' * 60}")
        print("INITIALIZING ALPR PIPELINE (PaddleOCR)")
        print(f"{'=' * 60}\n")

        # Load detector
        if detector_path is None:
            detector_path = str(Config.DETECTOR_DIR / "yolov8_detector5" / "weights" / "best.pt")

        print(f"📥 Loading detector: {detector_path}")
        self.detector = YOLO(detector_path)
        print(f"✅ Detector loaded")

        # Load PaddleOCR
        print(f"\n📥 Loading PaddleOCR...")
        self.reader = PaddleOCR(lang='ch')
        print(f"✅ PaddleOCR loaded")

        print(f"\n{'=' * 60}")
        print("✅ Pipeline Ready!")
        print(f"{'=' * 60}\n")

    def detect_plates(self, image, conf_threshold=0.25):
        """Detect license plates"""
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
        """Recognize text with PaddleOCR"""
        # Preprocess
        processed = self._preprocess_for_ocr(image)

        try:
            # Standard OCR call
            result = self.reader.ocr(processed)

            # Handle empty results
            if not result or not result[0]:
                return {'text': '', 'confidence': 0.0}

            # Navigate through nested structure
            lines = result[0]

            if not lines:
                return {'text': '', 'confidence': 0.0}

            # Extract all text lines
            all_texts = []
            for item in lines:
                try:
                    # PaddleOCR format: [bbox, (text, score)]
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text_info = item[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            score = text_info[1]
                            all_texts.append((text, score))
                except Exception:
                    continue

            if not all_texts:
                return {'text': '', 'confidence': 0.0}

            # Get the best result (highest confidence)
            best_text, best_score = max(all_texts, key=lambda x: x[1])

            return {
                'text': str(best_text),
                'confidence': float(best_score)
            }

        except Exception as e:
            print(f"OCR Error: {e}")
            return {'text': '', 'confidence': 0.0}

    def _preprocess_for_ocr(self, image):
        """Preprocess image for better OCR"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize if too small (OCR works better with larger images)
        height = gray.shape[0]
        if height < 60:
            scale = 60 / height
            width = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (width, 60), interpolation=cv2.INTER_CUBIC)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Convert back to BGR for PaddleOCR
        result = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

        return result

    def process_image(self, image_path):
        """Process single image"""
        # Handle both string paths and Path objects
        image_path = str(image_path)

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load: {image_path}")
            return None

        # Detect plates
        detections = self.detect_plates(image)

        if not detections:
            print("No plates detected")
            return []

        results = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']

            # Add padding around detection
            h, w = image.shape[:2]
            pad_x = int((x2 - x1) * 0.15)  # Increased padding for better OCR
            pad_y = int((y2 - y1) * 0.15)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            # Crop plate
            plate_img = image[y1:y2, x1:x2]

            # Recognize text
            recognition = self.recognize_text(plate_img)

            results.append({
                'detection': det,
                'recognition': recognition,
                'plate_image': plate_img
            })

        return results


def test_paddle():
    """Quick test of PaddleOCR pipeline"""
    pipeline = ALPRPipelinePaddle()

    # Use absolute path
    test_image = project_root / "data" / "processed" / "unified_dataset" / "test" / "images" / "ccpd_test_000002.jpg"

    print(f"\nTesting on: {test_image}")

    if not test_image.exists():
        print(f"❌ Test image not found: {test_image}")
        return

    results = pipeline.process_image(test_image)

    if results:
        print(f"\n✅ Found {len(results)} plate(s)")
        for i, r in enumerate(results, 1):
            print(f"\nPlate {i}:")
            print(f"  Text: '{r['recognition']['text']}'")
            print(f"  Detection confidence: {r['detection']['confidence']:.2%}")
            print(f"  OCR confidence: {r['recognition']['confidence']:.2%}")
    else:
        print("\n❌ No plates detected or recognition failed")


if __name__ == "__main__":
    test_paddle()