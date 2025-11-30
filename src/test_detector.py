import os
import sys
from pathlib import Path
import cv2

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
from src.config import Config


def test_detector(image_path):
    """Test detector on single image"""

    # Load model
    model_path = Config.DETECTOR_DIR / "yolov8_detector5" / "weights" / "best.pt"
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Run detection
    print(f"Processing: {image_path}")
    results = model(image_path, conf=0.25)

    # Show results
    for result in results:
        result.show()

    print("Done!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_detector.py <image_path>")
    else:
        test_detector(sys.argv[1])