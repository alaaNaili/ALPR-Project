import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config


def main():
    print("\n" + "=" * 60)
    print("YOLOV8 LICENSE PLATE DETECTOR TRAINING")
    print("=" * 60 + "\n")

    # Print configuration
    Config.print_config()

    # Check GPU
    print("\n🔍 Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  No GPU detected. Training will use CPU (slower)")
        response = input("\nContinue training on CPU? (y/n): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return

    # Verify dataset exists
    dataset_yaml = Config.DETECTION['dataset_yaml']
    if not os.path.exists(dataset_yaml):
        print(f"\n❌ Dataset not found: {dataset_yaml}")
        print("Please run dataset_converter.py first.")
        return

    # Show training configuration
    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    print(f"Model: {Config.DETECTION['model_size']}")
    print(f"Epochs: {Config.DETECTION['epochs']}")
    print(f"Batch Size: {Config.DETECTION['batch_size']}")
    print(f"Image Size: {Config.DETECTION['img_size']}")
    print("=" * 60)

    response = input("\nStart training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return

    # Load model
    print(f"\n📥 Loading YOLOv8 model: {Config.DETECTION['model_size']}")
    model = YOLO(f"{Config.DETECTION['model_size']}.pt")

    # Start training
    print("\n🚀 Starting training...\n")
    print("=" * 60)

    results = model.train(
        data=dataset_yaml,
        epochs=Config.DETECTION['epochs'],
        batch=8,  # Reduced from 16
        imgsz=Config.DETECTION['img_size'],
        device=Config.DETECTION['device'],
        project=Config.DETECTION['save_dir'],
        name=Config.DETECTION['name'],
        patience=20,
        save=True,
        plots=True,
        verbose=True,
        workers=0,  # This fixes Windows multiprocessing issue
        cache=False,
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Results saved to: {results.save_dir}")
    print(f"📊 Best model: {results.save_dir}/weights/best.pt")
    print(f"📊 Last model: {results.save_dir}/weights/last.pt")
    print("\n✅ Your detector is ready to use!")


if __name__ == "__main__":
    main()