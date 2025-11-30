import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import cv2
import json
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.alpr_pipeline import ALPRPipeline
from src.ccpd_parser import CCPDParser


def evaluate_on_test_set(num_samples=100):
    """Evaluate ALPR system on test images"""

    print("=" * 60)
    print("ALPR SYSTEM EVALUATION")
    print("=" * 60)

    # Initialize
    pipeline = ALPRPipeline()
    parser = CCPDParser()

    # Get test images
    test_dir = project_root / "data" / "processed" / "unified_dataset" / "test" / "images"
    test_images = [f for f in test_dir.glob("ccpd_test_*.jpg")][:num_samples]

    print(f"\nEvaluating on {len(test_images)} images...\n")

    # Metrics
    total_images = 0
    images_with_detection = 0
    correct_detections = 0
    total_plates = 0
    correct_recognitions = 0
    partial_recognitions = 0

    results = []

    for img_path in tqdm(test_images, desc="Processing"):
        total_images += 1

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Get ground truth from filename
        filename = img_path.stem.replace("ccpd_test_", "")
        # Find original CCPD filename from the image
        # For CCPD images, we can parse the original filename

        # Detect
        detections = pipeline.detect_plates(image)

        if len(detections) > 0:
            images_with_detection += 1

            for det in detections:
                total_plates += 1

                # Crop and recognize
                x1, y1, x2, y2 = det['bbox']
                plate_img = image[y1:y2, x1:x2]
                recognition = pipeline.recognize_text(plate_img)

                result = {
                    'image': img_path.name,
                    'detected': True,
                    'bbox': det['bbox'],
                    'confidence': det['confidence'],
                    'recognized_text': recognition['text'],
                    'ocr_confidence': recognition['confidence']
                }
                results.append(result)
        else:
            results.append({
                'image': img_path.name,
                'detected': False
            })

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nDetection Performance:")
    print(f"  Total images: {total_images}")
    print(f"  Images with detection: {images_with_detection}")
    print(f"  Detection rate: {images_with_detection / total_images * 100:.1f}%")
    print(f"  Total plates detected: {total_plates}")

    print(f"\nRecognition Performance:")
    print(f"  Note: Ground truth comparison requires manual verification")
    print(f"  Total recognitions: {len([r for r in results if r.get('recognized_text')])}")

    # Show some examples
    print(f"\nSample Results:")
    for i, r in enumerate(results[:10], 1):
        if r.get('detected'):
            print(f"\n{i}. {r['image']}")
            print(f"   Detected: ✅")
            print(f"   Text: {r['recognized_text']}")
            print(f"   Confidence: Det={r['confidence']:.2f}, OCR={r['ocr_confidence']:.2f}")
        else:
            print(f"\n{i}. {r['image']}")
            print(f"   Detected: ❌")

    # Save results
    output_file = project_root / "results" / "evaluation_results.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_on_test_set(num_samples=100)