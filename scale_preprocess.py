import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

# ==========================================
# CONFIGURATION SETTINGS
# ==========================================

INPUT_FOLDER = r"C:\Skola\4 letny semester\HSU\data"  # Replace with your input folder path
OUTPUT_FOLDER = r"C:\Skola\4 letny semester\HSU\224x224_pre" # Replace with your output folder path

TARGET_WIDTH = 224  # Set the desired width
TARGET_HEIGHT = 224 # Set the desired height
CLEAR_OUTPUT_FOLDER = True

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLIT_FOLDERS = {"train", "val", "validation", "test"}

# ==========================================


def infer_three_class_label(input_path, input_dir):
    relative = Path(input_path).relative_to(input_dir)
    parts = [part.lower() for part in relative.parts]
    filename = Path(input_path).name.lower()

    if any(part in {"normal", "normal2"} for part in parts) or "normal" in filename:
        return "NORMAL"
    if "bacteria" in filename or "bacterial" in filename or any("bacteria" in part or "bacterial" in part for part in parts):
        return "PNEUMONIA_BACTERIAL"
    if "virus" in filename or "viral" in filename or any("virus" in part or "viral" in part for part in parts):
        return "PNEUMONIA_VIRUS"
    if filename.startswith("im-") or filename.startswith("normal2-im-"):
        return "NORMAL"
    return None


def destination_path(input_path, input_dir, output_dir):
    relative = Path(input_path).relative_to(input_dir)
    parts = list(relative.parts)
    split_name = parts[0] if parts and parts[0].lower() in SPLIT_FOLDERS else None
    label = infer_three_class_label(input_path, input_dir)
    if label is None:
        return None
    if split_name:
        return Path(output_dir) / split_name / label / Path(input_path).name
    return Path(output_dir) / label / Path(input_path).name


def unique_output_path(path):
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def preprocess_images_recursively(input_dir, output_dir, width, height):
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    if CLEAR_OUTPUT_FOLDER and os.path.isdir(output_dir):
        print(f"Clearing output directory: {output_dir}")
        shutil.rmtree(output_dir)

    processed_count = 0
    skipped_count = 0
    class_counts = Counter()

    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            input_path = os.path.join(root, filename)
            output_path = destination_path(input_path, input_dir, output_dir)
            if output_path is None:
                skipped_count += 1
                print(f"Skipped (unknown class/subtype): {os.path.relpath(input_path, input_dir)}")
                continue

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path = unique_output_path(output_path)

            try:
                with Image.open(input_path) as img:
                    img = img.convert("RGB")
                    resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
                    resized_img.save(output_path)

                display_path = os.path.relpath(output_path, output_dir)
                print(f"Successfully resized: {display_path}")
                processed_count += 1
                class_counts[output_path.parent.name] += 1
            except Exception as exc:
                skipped_count += 1
                print(f"Skipped (image error): {os.path.relpath(input_path, input_dir)} [{exc}]")

    print(f"\nDone! Successfully processed {processed_count} images across all subfolders.")
    print(f"Skipped: {skipped_count}")
    for label, count in sorted(class_counts.items()):
        print(f"{label}: {count}")


if __name__ == "__main__":
    print("Starting recursive image preprocessing...")
    print(f"Target size: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    preprocess_images_recursively(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_WIDTH, TARGET_HEIGHT)