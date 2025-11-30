import os
import json
import xml.etree.ElementTree as ET


def inspect_label_files(label_dir):
    """
    Quick inspection of label files to understand the annotation format.
    """

    if not os.path.isdir(label_dir):
        print(f"[Error] Label directory does not exist: {label_dir}")
        return

    files = os.listdir(label_dir)

    if len(files) == 0:
        print("[Error] No label files found in the directory.")
        return

    print("\n-------------------------------")
    print(" Exploring annotation structure ")
    print("-------------------------------\n")

    # Count file extensions
    ext_count = {}
    for f in files:
        ext = os.path.splitext(f)[1]
        ext_count[ext] = ext_count.get(ext, 0) + 1

    print("Detected file types:")
    for ext, n in ext_count.items():
        print(f"  {ext}: {n}")

    # Try reading the first few files
    sample_set = files[:5]
    print(f"\nInspecting {len(sample_set)} sample files...\n")

    for idx, fname in enumerate(sample_set, start=1):
        fpath = os.path.join(label_dir, fname)
        ext = os.path.splitext(fname)[1]

        print(f"[{idx}] File: {fname}")
        print(f"     Type: {ext}")

        try:
            if ext == ".txt":
                with open(fpath, "r") as f:
                    lines = f.read().strip().split("\n")
                print(f"     Line count: {len(lines)}")
                for l in lines[:3]:
                    print(f"        > {l}")
                if len(lines) > 3:
                    print("        ...")

            elif ext == ".xml":
                tree = ET.parse(fpath)
                root = tree.getroot()
                tags = [child.tag for child in root]
                print(f"     Root tag: {root.tag}")
                print(f"     First-level tags: {tags}")

            elif ext == ".json":
                with open(fpath, "r") as f:
                    data = json.load(f)
                preview = json.dumps(data, indent=2)
                print("     JSON preview:")
                print(preview[:250], "...\n")

            else:
                # For rare / unexpected formats
                with open(fpath, "r", errors="ignore") as f:
                    snippet = f.read(300)
                print("     Content preview (first 300 chars):")
                print(snippet)

        except Exception as e:
            print(f"     [Could not parse file: {e}]")

        print()

    # Attempt to analyze structure of .txt annotation files
    if ".txt" in ext_count:
        sample_txt = os.path.join(label_dir, files[0])
        with open(sample_txt, "r") as f:
            first_line = f.readline().strip()
        parts = first_line.split()

        print("----------------------------")
        print(" Annotation format analysis")
        print("----------------------------")
        print(f"Sample line: {first_line}")
        print(f"Value count: {len(parts)}")

        if len(parts) == 5:
            print("Format resembles YOLO annotations:")
            print("  class_id x_center y_center width height (normalized)")
        elif len(parts) == 4:
            print("Possible format:")
            print("  xmin ymin xmax ymax OR center_x center_y width height (non-normalized)")
        else:
            print("Unusual annotation format — needs manual review.")

    print("\nInspection completed.\n")


def check_matching_files(label_dir, image_dir):
    """
    Ensures that each image has a corresponding label file (same base name).
    """

    if not os.path.isdir(image_dir):
        print(f"[Error] Image directory does not exist: {image_dir}")
        return

    print("------------------------------")
    print(" Matching images and labels")
    print("------------------------------\n")

    labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}
    images = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}

    print(f"Label files: {len(labels)}")
    print(f"Image files: {len(images)}")

    common = labels & images
    missing_labels = images - labels
    missing_images = labels - images

    print(f"\nMatching pairs: {len(common)}")

    if missing_labels:
        print(f"Images with NO label: {len(missing_labels)}")
        print("Examples:", list(missing_labels)[:3])

    if missing_images:
        print(f"Labels with NO image: {len(missing_images)}")
        print("Examples:", list(missing_images)[:3])

    if len(common) == len(labels) == len(images):
        print("\nEverything is paired correctly ✔")

    print()


def main():
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    label_dir = os.path.join(root_path, "data", "raw", "kaggle_alpr", "labels")
    image_dir = os.path.join(root_path, "data", "raw", "kaggle_alpr", "images")

    print("Using paths:")
    print(f"  - Labels: {label_dir}")
    print(f"  - Images: {image_dir}\n")

    inspect_label_files(label_dir)
    check_matching_files(label_dir, image_dir)


if __name__ == "__main__":
    main()
