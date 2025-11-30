import os
from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt


class DatasetManager:
    """
    Utility class used during early exploration of the datasets (CCPD, Kaggle ALPR).
    Not used during training, but extremely useful for inspection, debugging and file checks.
    """

    def __init__(self, base_path: str | None = None):
        # Resolve project root automatically if user doesn’t pass a custom base path
        if base_path is None:
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent
            base_path = project_root / "data" / "raw"

        self.base_path = Path(base_path)
        self.ccpd_path = self.base_path / "CCPD2019"
        self.alpr_path = self.base_path / "kaggle_alpr"
        self.splits_path = self.ccpd_path / "splits"

        print("\n[DatasetManager] Initialized with:")
        print(f"  ├─ Base path : {self.base_path}")
        print(f"  ├─ CCPD path : {self.ccpd_path}")
        print(f"  └─ ALPR path : {self.alpr_path}\n")

    # -------------------------------------------------------------------------
    #                          DATASET CHECKS
    # -------------------------------------------------------------------------
    def check_dataset_exists(self):
        """Check presence of CCPD2019 and Kaggle ALPR datasets."""
        print("\n[CHECK] Verifying dataset structure...\n")

        # -------- CCPD ----------------------------------
        if self.ccpd_path.exists():
            print(f"✔ CCPD2019 found at: {self.ccpd_path}")

            subfolders = [d for d in self.ccpd_path.iterdir() if d.is_dir() and d.name != "splits"]
            print(f"  └─ Subfolders: {[sf.name for sf in subfolders]}")

            for folder in subfolders:
                jpg_count = len([f for f in folder.glob("*.jpg")])
                print(f"     · {folder.name}: {jpg_count} images")
        else:
            print(f"✘ CCPD2019 NOT FOUND at: {self.ccpd_path}")

        print()

        # -------- CCPD splits ---------------------------
        if self.splits_path.exists():
            split_files = list(self.splits_path.glob("*.txt"))
            print(f"✔ Found splits folder with {len(split_files)} files:")
            for f in split_files:
                print(f"     - {f.name}")
        else:
            print("✘ CCPD splits folder NOT FOUND")

        print()

        # -------- Kaggle ALPR ---------------------------
        if self.alpr_path.exists():
            print(f"✔ Kaggle ALPR found at: {self.alpr_path}")

            images_dir = self.alpr_path / "images"
            labels_dir = self.alpr_path / "labels"

            if images_dir.exists():
                img_count = len(list(images_dir.iterdir()))
                print(f"     · images/: {img_count} files")
            else:
                print("     · images/ MISSING")

            if labels_dir.exists():
                lbl_files = list(labels_dir.iterdir())
                print(f"     · labels/: {len(lbl_files)} files")
                if lbl_files:
                    print(f"       Label ext: {lbl_files[0].suffix}")
            else:
                print("     · labels/ MISSING")
        else:
            print(f"✘ Kaggle ALPR NOT FOUND at: {self.alpr_path}")

        print()

    # -------------------------------------------------------------------------
    #                          SPLIT EXPLORATION
    # -------------------------------------------------------------------------
    def explore_splits(self):
        if not self.splits_path.exists():
            print("✘ Cannot explore splits: folder not found.")
            return

        print("Inspecting CCPD split files:\n")

        for file in sorted(self.splits_path.glob("*.txt")):
            lines = file.read_text().splitlines()
            print(f" {file.name}")
            print(f"   Total entries   : {len(lines)}")
            print(f"   Sample entries  :")
            for l in lines[:3]:
                print(f"       · {l}")
            print()

    def load_split(self, split="train", subset="ccpd_base"):
        """
        Loads paths from one CCPD split file.
        Returns a list of valid image file paths.
        """
        candidate_names = [
            f"{split}.txt",
            f"{subset}_{split}.txt",
            f"{split}_{subset}.txt",
        ]

        for file_name in candidate_names:
            file_path = self.splits_path / file_name
            if file_path.exists():
                print(f"✔ Using split file: {file_name}")

                entries = file_path.read_text().splitlines()
                img_paths = []

                for entry in entries:
                    entry = entry.strip()

                    if "/" in entry or "\\" in entry:
                        path = self.ccpd_path / entry
                    else:
                        path = self.ccpd_path / subset / entry

                    if path.exists():
                        img_paths.append(path)

                print(f"   Loaded {len(img_paths)} valid image paths")
                return img_paths

        print(f"✘ No split file found for: split={split}, subset={subset}")
        return []

    def create_split_info_file(self):
        """Generate a text report listing how many images each split file contains."""
        if not self.splits_path.exists():
            print("✘ Cannot create summary: splits folder missing.")
            return

        project_root = self.base_path.parent.parent
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        output_file = results_dir / "splits_summary.txt"

        with output_file.open("w") as out:
            out.write("CCPD2019 Split Summary\n")
            out.write("=" * 50 + "\n\n")

            for file in sorted(self.splits_path.glob("*.txt")):
                count = len(file.read_text().splitlines())
                out.write(f"{file.name}: {count} images\n")

        print(f"✔ Split summary saved to: {output_file}\n")

    # -------------------------------------------------------------------------
    #                          DATASET INFO
    # -------------------------------------------------------------------------
    def get_ccpd_subset_info(self, subset="ccpd_base"):
        """
        Returns info about a CCPD subset (image count + example names).
        """
        folder = self.ccpd_path / subset
        if not folder.exists():
            print(f"✘ Subset not found: {subset}")
            return None

        images = sorted(folder.glob("*.jpg"))
        return {
            "path": str(folder),
            "name": subset,
            "image_count": len(images),
            "sample_images": [img.name for img in images[:5]],
        }

    # -------------------------------------------------------------------------
    #                      SAMPLE VISUALIZATION HELPERS
    # -------------------------------------------------------------------------
    def _prepare_results_dir(self):
        project_root = self.base_path.parent.parent
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        return results_dir

    def display_samples_from_split(self, split="train", subset="ccpd_base", num_samples=6):
        paths = self.load_split(split, subset)
        if not paths:
            print("✘ No images to display.")
            return

        samples = random.sample(paths, min(num_samples, len(paths)))
        results_dir = self._prepare_results_dir()

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()

        for ax, img_path in zip(axes, samples):
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
                ax.set_title(img_path.name[:25], fontsize=8)
            ax.axis("off")

        plt.suptitle(f"Samples from {split} split ({subset})", fontsize=14)
        plt.tight_layout()

        out_path = results_dir / f"samples_{split}_{subset}.png"
        plt.savefig(out_path, dpi=150)
        print(f"✔ Saved visualization to: {out_path}\n")
        plt.show()

    def display_samples_from_ccpd(self, subset="ccpd_base", num_samples=6):
        folder = self.ccpd_path / subset
        if not folder.exists():
            print("✘ Subset does not exist.")
            return

        images = list(folder.glob("*.jpg"))[:num_samples]
        results_dir = self._prepare_results_dir()

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()

        for ax, img_file in zip(axes, images):
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            ax.set_title(img_file.name[:22], fontsize=8)
            ax.axis("off")

        plt.suptitle(f"CCPD subset: {subset}", fontsize=14)
        plt.tight_layout()

        out_path = results_dir / f"samples_{subset}.png"
        plt.savefig(out_path, dpi=150)
        print(f"✔ Saved sample preview to: {out_path}")
        plt.show()

    def display_samples_from_alpr(self, num_samples=6):
        img_folder = self.alpr_path / "images"
        if not img_folder.exists():
            print("✘ Kaggle ALPR images folder not found.")
            return

        image_files = list(img_folder.iterdir())[:num_samples]
        results_dir = self._prepare_results_dir()

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))
        axes = axes.flatten()

        for ax, img_file in zip(axes, image_files):
            img = cv2.imread(str(img_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax.imshow(img)
            ax.set_title(img_file.name[:25], fontsize=8)
            ax.axis("off")

        plt.suptitle("Kaggle ALPR Sample Images", fontsize=14)
        plt.tight_layout()

        out_path = results_dir / "samples_alpr.png"
        plt.savefig(out_path, dpi=150)
        print(f"✔ Saved ALPR preview to: {out_path}")
        plt.show()


# -------------------------------------------------------------------------
#                            CLI EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" DATASET MANAGER")
    print("=" * 60)

    dm = DatasetManager()
    dm.check_dataset_exists()

    if dm.splits_path.exists():
        dm.explore_splits()
        dm.create_split_info_file()
        dm.display_samples_from_split("train", "ccpd_base")

    if dm.alpr_path.exists():
        dm.display_samples_from_alpr()
