import os
import re
import cv2
import numpy as np
from pathlib import Path


class CCPDParser:
    """
    Parse CCPD2019 filename encoding

    Filename format example:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

    Structure:
    1. Area code (025): Province/region code
    2. Tilt angles (95_113): Horizontal and vertical tilt
    3. Bounding box (154&383_386&473): x1&y1_x2&y2
    4. Four corners (386&473_177&454_154&383_363&402): Four corner points
    5. License plate number (0_0_22_27_27_33_16): Character indices
    6. Brightness (37): Image brightness
    7. Blur level (15): Blur degree
    """

    # Chinese province abbreviations (31 provinces)
    PROVINCES = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
                 "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤",
                 "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]

    # License plate characters (excluding I and O to avoid confusion)
    ALPHABETS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                 'W', 'X', 'Y', 'Z']

    # All characters that can appear on a Chinese license plate
    ADS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
           'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
           'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5',
           '6', '7', '8', '9']

    def __init__(self):
        pass

    def parse_filename(self, filename):
        """
        Parse CCPD filename and extract all information

        Args:
            filename: CCPD image filename (with or without extension)

        Returns:
            dict: Parsed information including bbox, corners, plate number, etc.
        """
        # Remove extension if present
        filename = os.path.splitext(os.path.basename(filename))[0]

        try:
            # Split filename by '-'
            parts = filename.split('-')

            if len(parts) < 7:
                return None

            # Parse each component
            area_code = int(parts[0])
            tilt_angles = [int(x) for x in parts[1].split('_')]
            bbox_coords = [int(x) for x in parts[2].replace('&', '_').split('_')]
            corners_coords = [int(x) for x in parts[3].replace('&', '_').split('_')]
            plate_indices = [int(x) for x in parts[4].split('_')]
            brightness = int(parts[5])
            blur = int(parts[6])

            # Convert plate indices to characters
            plate_number = self._decode_plate_number(plate_indices)

            # Organize bounding box
            bbox = {
                'x1': bbox_coords[0],
                'y1': bbox_coords[1],
                'x2': bbox_coords[2],
                'y2': bbox_coords[3],
                'width': bbox_coords[2] - bbox_coords[0],
                'height': bbox_coords[3] - bbox_coords[1]
            }

            # Organize corners (top-right, bottom-right, bottom-left, top-left)
            corners = {
                'top_right': (corners_coords[0], corners_coords[1]),
                'bottom_right': (corners_coords[2], corners_coords[3]),
                'bottom_left': (corners_coords[4], corners_coords[5]),
                'top_left': (corners_coords[6], corners_coords[7])
            }

            result = {
                'filename': filename,
                'area_code': area_code,
                'province': self.PROVINCES[area_code] if area_code < len(self.PROVINCES) else '?',
                'tilt_horizontal': tilt_angles[0],
                'tilt_vertical': tilt_angles[1],
                'bbox': bbox,
                'corners': corners,
                'plate_number': plate_number,
                'plate_indices': plate_indices,
                'brightness': brightness,
                'blur': blur
            }

            return result

        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return None

    def _decode_plate_number(self, indices):
        """
        Decode license plate number from indices

        Chinese license plate format: Province + Letter + 5 characters
        Example: 皖A12345
        """
        if len(indices) < 7:
            return "INVALID"

        try:
            province = self.PROVINCES[indices[0]]
            letter = self.ALPHABETS[indices[1]]
            remaining = ''.join([self.ADS[idx] for idx in indices[2:]])

            plate_number = f"{province}{letter}{remaining}"
            return plate_number
        except:
            return "DECODE_ERROR"

    def parse_and_visualize(self, image_path, save_path=None):
        """
        Parse filename and visualize annotations on image

        Args:
            image_path: Path to CCPD image
            save_path: Optional path to save annotated image

        Returns:
            Annotated image (numpy array)
        """
        # Parse filename
        info = self.parse_filename(image_path)

        if info is None:
            print(f"Failed to parse: {image_path}")
            return None

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None

        img_height, img_width = img.shape[:2]

        # Draw bounding box
        bbox = info['bbox']
        cv2.rectangle(img,
                      (bbox['x1'], bbox['y1']),
                      (bbox['x2'], bbox['y2']),
                      (0, 255, 0), 2)

        # Draw corners
        corners = info['corners']
        corner_points = [
            corners['top_left'],
            corners['top_right'],
            corners['bottom_right'],
            corners['bottom_left']
        ]

        # Draw corner points
        for point in corner_points:
            cv2.circle(img, point, 3, (0, 0, 255), -1)

        # Draw corner lines
        for i in range(4):
            pt1 = corner_points[i]
            pt2 = corner_points[(i + 1) % 4]
            cv2.line(img, pt1, pt2, (255, 0, 0), 1)

        # Add text annotations
        text_y = 30
        cv2.putText(img, f"Plate: {info['plate_number']}",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 25
        cv2.putText(img, f"Province: {info['province']} (Code: {info['area_code']})",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        text_y += 20
        cv2.putText(img, f"Brightness: {info['brightness']}, Blur: {info['blur']}",
                    (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, img)
            print(f"Saved annotated image to: {save_path}")

        return img

    def convert_to_yolo_format(self, filename, img_width, img_height):
        """
        Convert CCPD annotation to YOLO format

        Args:
            filename: CCPD filename
            img_width: Image width
            img_height: Image height

        Returns:
            str: YOLO format annotation string
        """
        info = self.parse_filename(filename)

        if info is None:
            return None

        bbox = info['bbox']

        # Calculate center point and dimensions (normalized)
        x_center = ((bbox['x1'] + bbox['x2']) / 2) / img_width
        y_center = ((bbox['y1'] + bbox['y2']) / 2) / img_height
        width = bbox['width'] / img_width
        height = bbox['height'] / img_height

        # YOLO format: class_id x_center y_center width height
        yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

        return yolo_line

    def batch_parse(self, image_folder, max_samples=None):
        """
        Parse multiple CCPD images from a folder

        Args:
            image_folder: Path to folder containing CCPD images
            max_samples: Maximum number of samples to parse (None = all)

        Returns:
            list: List of parsed information dicts
        """
        if not os.path.exists(image_folder):
            print(f"Folder not found: {image_folder}")
            return []

        image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

        if max_samples:
            image_files = image_files[:max_samples]

        results = []
        failed = 0

        print(f"Parsing {len(image_files)} images from {image_folder}...")

        for img_file in image_files:
            info = self.parse_filename(img_file)
            if info:
                results.append(info)
            else:
                failed += 1

        print(f"✅ Successfully parsed: {len(results)}")
        print(f"❌ Failed to parse: {failed}")

        return results


def main():
    """Test CCPD parser"""
    print("=" * 60)
    print("CCPD FILENAME PARSER TEST")
    print("=" * 60 + "\n")

    # Initialize parser
    parser = CCPDParser()

    # Test with a sample filename
    sample_filename = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"

    print(f"Testing with sample filename:")
    print(f"{sample_filename}\n")

    info = parser.parse_filename(sample_filename)

    if info:
        print("✅ Successfully parsed!")
        print(f"\n📋 Parsed Information:")
        print(f"   Province: {info['province']} (Area code: {info['area_code']})")
        print(f"   License Plate: {info['plate_number']}")
        print(
            f"   Bounding Box: ({info['bbox']['x1']}, {info['bbox']['y1']}) to ({info['bbox']['x2']}, {info['bbox']['y2']})")
        print(f"   Size: {info['bbox']['width']} x {info['bbox']['height']}")
        print(f"   Tilt: H={info['tilt_horizontal']}°, V={info['tilt_vertical']}°")
        print(f"   Brightness: {info['brightness']}")
        print(f"   Blur: {info['blur']}")

        # Test YOLO conversion
        print(f"\n📝 YOLO Format:")
        yolo_line = parser.convert_to_yolo_format(sample_filename, 720, 1160)
        print(f"   {yolo_line}")
    else:
        print("❌ Failed to parse")

    # Test with actual images if available
    print("\n" + "=" * 60)
    print("Testing with actual CCPD images...")
    print("=" * 60 + "\n")

    # Get project root
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file))
    ccpd_base_path = os.path.join(project_root, "data", "raw", "CCPD2019", "ccpd_base")

    if os.path.exists(ccpd_base_path):
        # Parse first 10 images
        results = parser.batch_parse(ccpd_base_path, max_samples=10)

        if results:
            print(f"\n📊 Sample Results (first 3):")
            for idx, info in enumerate(results[:3], 1):
                print(f"\n{idx}. {info['filename'][:50]}...")
                print(f"   Plate: {info['plate_number']}")
                print(f"   Province: {info['province']}")
                print(f"   BBox: {info['bbox']['width']}x{info['bbox']['height']} pixels")

            # Try to visualize one image
            print(f"\n📸 Visualizing first image...")
            results_dir = os.path.join(project_root, "results")
            os.makedirs(results_dir, exist_ok=True)

            first_image = os.path.join(ccpd_base_path, results[0]['filename'] + '.jpg')
            save_path = os.path.join(results_dir, "ccpd_parsed_sample.jpg")

            annotated = parser.parse_and_visualize(first_image, save_path)

            if annotated is not None:
                print(f"✅ Saved annotated image to: {save_path}")
    else:
        print(f"❌ CCPD base folder not found at: {ccpd_base_path}")

    print("\n" + "=" * 60)
    print("✅ Parser test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()