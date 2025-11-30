import os
from pathlib import Path


class Config:
    """Centralized configuration for ALPR project"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()

    # Data paths
    DATA_ROOT = PROJECT_ROOT / "data"
    PROCESSED_DATA = DATA_ROOT / "processed"
    UNIFIED_DATASET = PROCESSED_DATA / "unified_dataset"

    # Model paths
    MODELS_DIR = PROJECT_ROOT / "models"
    DETECTOR_DIR = MODELS_DIR / "detector"

    # Output paths
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Create directories if they don't exist
    for dir_path in [MODELS_DIR, DETECTOR_DIR, RESULTS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Detection configuration
    DETECTION = {
        'dataset_yaml': str(UNIFIED_DATASET / "dataset.yaml"),
        'model_size': 'yolov8n',
        'epochs': 20,
        'batch_size': 16,
        'img_size': 640,
        'device': 0,
        'save_dir': str(DETECTOR_DIR),
        'name': 'yolov8_detector',
    }

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("ALPR PROJECT CONFIGURATION")
        print("=" * 60)
        print(f"\nProject Root: {cls.PROJECT_ROOT}")
        print(f"Dataset: {cls.UNIFIED_DATASET}")
        print(f"Models: {cls.MODELS_DIR}")
        print(f"Results: {cls.RESULTS_DIR}")
        print(f"\nDetection Model: {cls.DETECTION['model_size']}")
        print(f"Epochs: {cls.DETECTION['epochs']}")
        print(f"Batch Size: {cls.DETECTION['batch_size']}")
        print("=" * 60)


if __name__ == "__main__":
    Config.print_config()

    # Verify dataset exists
    dataset_path = Config.DETECTION['dataset_yaml']
    if os.path.exists(dataset_path):
        print(f"\n✅ Dataset found: {dataset_path}")
    else:
        print(f"\n❌ Dataset NOT found: {dataset_path}")