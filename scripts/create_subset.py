import os
import pandas as pd
import shutil

LABELS_FILE = "data/raw/labels.csv"
RAW_DIR = "data/raw"
SAMPLE_DIR = "data/sample"
SAMPLE_SIZE = 5000  # Change this for a bigger/smaller sample

# Path to where images are stored in the downloaded dataset
IMAGE_BASE_DIR = os.path.expanduser(
    r"~\.cache\kagglehub\datasets\ascanipek\eyepacs-aptos-messidor-diabetic-retinopathy\versions\4\augmented_resized_V2\train"
)

def create_subset():
    if not os.path.exists(LABELS_FILE):
        raise FileNotFoundError(f"❌ {LABELS_FILE} not found in {RAW_DIR}")

    df = pd.read_csv(LABELS_FILE)

    # Ensure filenames are strings and have an extension
    df["filename"] = df["filename"].astype(str)
    if not df["filename"].str.contains(r"\.").any():
        df["filename"] = df["filename"] + ".png"  # change to .jpg if needed

    # Random sample
    sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)

    os.makedirs(SAMPLE_DIR, exist_ok=True)

    for _, row in sample_df.iterrows():
        filename = row["filename"]
        label = str(row["label"])
        src_path = os.path.join(IMAGE_BASE_DIR, label, filename)

        if not os.path.exists(src_path):
            print(f"⚠️ Missing file: {src_path}")
            continue

        dest_dir = os.path.join(SAMPLE_DIR, label)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src_path, os.path.join(dest_dir, filename))

    # Save subset labels.csv
    sample_df.to_csv(os.path.join(SAMPLE_DIR, "labels.csv"), index=False)

    print(f"🎯 Subset creation complete. Sample dataset is ready at: {SAMPLE_DIR}")
    print(f"📸 Total images in subset: {len(sample_df)}")

if __name__ == "__main__":
    create_subset()
