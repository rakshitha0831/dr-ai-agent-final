import os
import pandas as pd

# Path to the dataset split you want to use
DATASET_DIR = os.path.expanduser(
    r"~\.cache\kagglehub\datasets\ascanipek\eyepacs-aptos-messidor-diabetic-retinopathy\versions\4\augmented_resized_V2\train"
)
OUTPUT_CSV = "data/raw/labels.csv"

def generate_labels():
    data = []
    for label in os.listdir(DATASET_DIR):
        label_dir = os.path.join(DATASET_DIR, label)
        if os.path.isdir(label_dir):
            for img in os.listdir(label_dir):
                if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                    data.append({"filename": img, "label": label})
    
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ labels.csv created with {len(df)} entries at {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_labels()
