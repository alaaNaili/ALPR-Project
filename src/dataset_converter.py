import os
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from ccpd_parser import CCPDParser


class DatasetConverter:
    """
    Convert CCPD2019 and Kaggle ALPR datasets to unified YOLO format
    and create train/val/test splits
    """

    def __init__(self, project_root=None):
        # Auto-detect project root
        if project_root is None:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))

        self.project_root = project_root

        # Source paths
        self.ccpd_path = os.path.join(project_root, "data", "raw", "CCPD2019")
        self.alpr_path = os.path.join(project_root, "data", "raw", "kaggle_alpr")

        # Output path
        self.output_path = os.path.join(project_root, "data", "processed", "unified_dataset")

        # Initialize CCPD parser
        self.ccpd_parser = CCPDParser()

        print(f"📁 Project root: {project_root}")
        print(f"📁 CCPD path: {self.ccpd_path}")
        print(f"📁 ALPR path: {self.alpr_path}")
        print(f"📁 Output path: {self.output_path}\n")

    def create_output_structure(self):
        """Create output folder structure for YOLO format dataset"""
        splits = ['train', 'val', 'test']

        for split in splits:
            images_dir = os.path.join(self.output_path, split, 'images')
            labels_dir = os.path.join(self.output_path, split, 'labels')

            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)

            print(f"✅ Created: {split}/images and {split}/labels")

    def convert_ccpd_split(self, split_name='train', max_images=None):
        """
        Convert CCPD split to YOLO format

        Args:
            split_name: 'train', 'val', or 'test'
            max_images: Maximum images to convert (None = all)

        Returns:
            int: Number of images processed
        """
        print(f"\n{'=' * 60}")
        print(f"Converting CCPD {split_name.upper()} split...")
        print(f"{'=' * 60}\n")

        # Read split file
        split_file = os.path.join(self.ccpd_path, "splits", f"{split_name}.txt")

        if not os.path.exists(split_file):
            print(f"❌ Split file not found: {split_file}")
            return 0

        with open(split_file, 'r') as f:
            image_paths = [line.strip() for line in f.readlines()]

        if max_images:
            image_paths = image_paths[:max_images]

        print(f"📊 Total images in split: {len(image_paths)}")

        # Output directories
        output_images_dir = os.path.join(self.output_path, split_name, 'images')
        output_labels_dir = os.path.join(self.output_path, split_name, 'labels')

        converted = 0
        failed = 0

        for img_rel_path in tqdm(image_paths, desc=f"Converting {split_name}"):
            try:
                # Build full path
                img_full_path = os.path.join(self.ccpd_path, img_rel_path)

                if not os.path.exists(img_full_path):
                    failed += 1
                    continue

                # Read image to get dimensions
                img = cv2.imread(img_full_path)
                if img is None:
                    failed += 1
                    continue

                img_height, img_width = img.shape[:2]

                # Parse filename
                filename = os.path.basename(img_rel_path)
                yolo_annotation = self.ccpd_parser.convert_to_yolo_format(
                    filename, img_width, img_height
                )

                if yolo_annotation is None:
                    failed += 1
                    continue

                # Create unique filename (prefix with 'ccpd_')
                new_filename = f"ccpd_{split_name}_{converted:06d}.jpg"

                # Copy image
                output_img_path = os.path.join(output_images_dir, new_filename)
                shutil.copy2(img_full_path, output_img_path)

                # Save label
                label_filename = new_filename.replace('.jpg', '.txt')
                output_label_path = os.path.join(output_labels_dir, label_filename)

                with open(output_label_path, 'w') as f:
                    f.write(yolo_annotation + '\n')

                converted += 1

            except Exception as e:
                failed += 1
                continue

        print(f"\n✅ Successfully converted: {converted}")
        print(f"❌ Failed: {failed}")

        return converted

    def convert_alpr_split(self, split_ratio={'train': 0.7, 'val': 0.15, 'test': 0.15}):
        """
        Convert Kaggle ALPR dataset and split into train/val/test

        Args:
            split_ratio: Dictionary with train/val/test ratios

        Returns:
            dict: Number of images in each split
        """
        print(f"\n{'=' * 60}")
        print(f"Converting Kaggle ALPR dataset...")
        print(f"{'=' * 60}\n")

        alpr_images_dir = os.path.join(self.alpr_path, "images")
        alpr_labels_dir = os.path.join(self.alpr_path, "labels")

        # Get all image files
        image_files = sorted([f for f in os.listdir(alpr_images_dir)
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        print(f"📊 Total ALPR images: {len(image_files)}")
        print(f"📊 Split ratios: Train={split_ratio['train']}, Val={split_ratio['val']}, Test={split_ratio['test']}")

        # Calculate split indices
        total = len(image_files)
        train_end = int(total * split_ratio['train'])
        val_end = train_end + int(total * split_ratio['val'])

        splits = {
            'train': image_files[:train_end],
            'val': image_files[train_end:val_end],
            'test': image_files[val_end:]
        }

        results = {}

        for split_name, files in splits.items():
            print(f"\n📁 Processing {split_name} split ({len(files)} images)...")

            output_images_dir = os.path.join(self.output_path, split_name, 'images')
            output_labels_dir = os.path.join(self.output_path, split_name, 'labels')

            converted = 0
            failed = 0

            for idx, img_file in enumerate(tqdm(files, desc=f"Converting {split_name}")):
                try:
                    # Source paths
                    src_img_path = os.path.join(alpr_images_dir, img_file)

                    # Get corresponding label file
                    label_file = os.path.splitext(img_file)[0] + '.txt'
                    src_label_path = os.path.join(alpr_labels_dir, label_file)

                    if not os.path.exists(src_label_path):
                        failed += 1
                        continue

                    # Create new filename
                    ext = os.path.splitext(img_file)[1]
                    new_filename = f"alpr_{split_name}_{idx:06d}{ext}"

                    # Copy image
                    dst_img_path = os.path.join(output_images_dir, new_filename)
                    shutil.copy2(src_img_path, dst_img_path)

                    # Copy label
                    label_filename = new_filename.replace(ext, '.txt')
                    dst_label_path = os.path.join(output_labels_dir, label_filename)
                    shutil.copy2(src_label_path, dst_label_path)

                    converted += 1

                except Exception as e:
                    failed += 1
                    continue

            print(f"✅ Converted: {converted}")
            print(f"❌ Failed: {failed}")

            results[split_name] = converted

        return results

    def create_dataset_yaml(self, dataset_stats):
        """
        Create dataset.yaml file for YOLO training

        Args:
            dataset_stats: Dictionary with dataset statistics
        """
        yaml_path = os.path.join(self.output_path, "dataset.yaml")

        # Get absolute paths
        train_path = os.path.abspath(os.path.join(self.output_path, "train", "images"))
        val_path = os.path.abspath(os.path.join(self.output_path, "val", "images"))
        test_path = os.path.abspath(os.path.join(self.output_path, "test", "images"))

        yaml_content = f"""# ALPR Unified Dataset
# Combined CCPD2019 and Kaggle ALPR datasets

# Paths
path: {os.path.abspath(self.output_path)}
train: train/images
val: val/images
test: test/images

# Classes
nc: 1  # number of classes
names: ['license_plate']  # class names

# Dataset Statistics
# Train: {dataset_stats['train']} images
# Val: {dataset_stats['val']} images
# Test: {dataset_stats['test']} images
# Total: {sum(dataset_stats.values())} images
"""

        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        print(f"\n✅ Created dataset.yaml at: {yaml_path}")

    def save_dataset_info(self, dataset_stats):
        """Save dataset information to JSON"""
        info = {
            'dataset_name': 'ALPR_Unified',
            'created_from': ['CCPD2019', 'Kaggle_ALPR'],
            'format': 'YOLO',
            'num_classes': 1,
            'class_names': ['license_plate'],
            'splits': dataset_stats,
            'total_images': sum(dataset_stats.values()),
            'output_path': self.output_path
        }

        info_path = os.path.join(self.output_path, "dataset_info.json")

        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)

        print(f"✅ Saved dataset info to: {info_path}")

    def convert_full_dataset(self,
                             ccpd_train_limit=None,
                             ccpd_val_limit=None,
                             ccpd_test_limit=None,
                             include_alpr=True):
        """
        Convert complete dataset (CCPD + ALPR) to unified format

        Args:
            ccpd_train_limit: Max images from CCPD train (None = all)
            ccpd_val_limit: Max images from CCPD val (None = all)
            ccpd_test_limit: Max images from CCPD test (None = all)
            include_alpr: Whether to include Kaggle ALPR dataset
        """
        print("=" * 60)
        print("UNIFIED DATASET CONVERSION")
        print("=" * 60 + "\n")

        # Create output structure
        self.create_output_structure()

        # Track statistics
        stats = {'train': 0, 'val': 0, 'test': 0}

        # Convert CCPD splits
        stats['train'] += self.convert_ccpd_split('train', ccpd_train_limit)
        stats['val'] += self.convert_ccpd_split('val', ccpd_val_limit)
        stats['test'] += self.convert_ccpd_split('test', ccpd_test_limit)

        # Convert ALPR dataset
        if include_alpr and os.path.exists(self.alpr_path):
            alpr_stats = self.convert_alpr_split()
            for split in ['train', 'val', 'test']:
                stats[split] += alpr_stats[split]

        # Create dataset configuration files
        self.create_dataset_yaml(stats)
        self.save_dataset_info(stats)

        # Print final summary
        print("\n" + "=" * 60)
        print("CONVERSION COMPLETE!")
        print("=" * 60)
        print(f"\n📊 Final Dataset Statistics:")
        print(f"   Train:      {stats['train']:,} images")
        print(f"   Validation: {stats['val']:,} images")
        print(f"   Test:       {stats['test']:,} images")
        print(f"   Total:      {sum(stats.values()):,} images")
        print(f"\n📁 Output location: {self.output_path}")
        print(f"📄 Dataset config: {os.path.join(self.output_path, 'dataset.yaml')}")
        print("\n✅ Ready for training!")


def main():
    """Main conversion process"""

    print("\n" + "=" * 60)
    print("DATASET CONVERTER - CCPD2019 + Kaggle ALPR → Unified YOLO")
    print("=" * 60 + "\n")

    converter = DatasetConverter()

    # Ask user for conversion parameters
    print("⚙️  Conversion Options:\n")
    print("1. Full dataset (may take time and disk space)")
    print("2. Sample dataset (faster, for testing)")
    print("3. Custom limits\n")

    choice = input("Choose option (1/2/3) [default: 2]: ").strip() or "2"

    if choice == "1":
        # Full dataset
        print("\n🚀 Converting FULL dataset...")
        converter.convert_full_dataset(
            ccpd_train_limit=None,
            ccpd_val_limit=None,
            ccpd_test_limit=None,
            include_alpr=True
        )

    elif choice == "2":
        # Sample dataset
        print("\n🚀 Converting SAMPLE dataset (10k train, 5k val, 5k test)...")
        converter.convert_full_dataset(
            ccpd_train_limit=10000,
            ccpd_val_limit=5000,
            ccpd_test_limit=5000,
            include_alpr=True
        )

    else:
        # Custom
        print("\n⚙️  Enter custom limits (press Enter for no limit):\n")

        train_limit = input("CCPD train images [default: 10000]: ").strip()
        train_limit = int(train_limit) if train_limit else 10000

        val_limit = input("CCPD val images [default: 5000]: ").strip()
        val_limit = int(val_limit) if val_limit else 5000

        test_limit = input("CCPD test images [default: 5000]: ").strip()
        test_limit = int(test_limit) if test_limit else 5000

        include_alpr = input("Include Kaggle ALPR? (y/n) [default: y]: ").strip().lower() or 'y'
        include_alpr = include_alpr == 'y'

        print(f"\n🚀 Converting with custom limits...")
        converter.convert_full_dataset(
            ccpd_train_limit=train_limit,
            ccpd_val_limit=val_limit,
            ccpd_test_limit=test_limit,
            include_alpr=include_alpr
        )


if __name__ == "__main__":
    main()