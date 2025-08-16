import os
import shutil
import zipfile

# Check if kagglehub is installed
try:
    import kagglehub
except ImportError:
    raise ImportError("Please install kagglehub: pip install kagglehub")

# Dataset name from Kaggle
DATASET_NAME = "ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy"

# Target directory for dataset
TARGET_DIR = os.path.join("data", "raw")

def download_and_extract():
    print(f"📥 Downloading dataset: {DATASET_NAME} ...")
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"✅ Download complete. Dataset files are located at: {path}")

    # Create raw data directory if not exists
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Copy files from KaggleHub path to TARGET_DIR
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            shutil.copytree(item_path, os.path.join(TARGET_DIR, item), dirs_exist_ok=True)
        elif item.lower().endswith(".zip"):
            with zipfile.ZipFile(item_path, "r") as zip_ref:
                zip_ref.extractall(TARGET_DIR)
        else:
            shutil.copy2(item_path, TARGET_DIR)

    print("✅ Files copied and extracted to:", TARGET_DIR)

    # Count images
    image_count = sum(
        len(files) for _, _, files in os.walk(TARGET_DIR)
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files)
    )
    print(f"📸 Total images downloaded: {image_count}")

if __name__ == "__main__":
    download_and_extract()
