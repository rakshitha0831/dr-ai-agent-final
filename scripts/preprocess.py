import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Base directory for the small subset
BASE_DIR = os.path.join("data", "sample")
CSV_PATH = os.path.join(BASE_DIR, "labels.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def infer_layout_and_fix_csv(df: pd.DataFrame):
    """
    Supports two layouts:
    A) data/sample/<class>/<filename>
    B) data/sample/<filename>  (flat)
    This function rewrites df['filename'] to be *relative to BASE_DIR* and returns:
      directory_to_pass_to_flow, fixed_dataframe
    """
    # Ensure strings
    df["label"] = df["label"].astype(str)
    df["filename"] = df["filename"].astype(str)

    # If no extension present, assume .png (change to .jpg if needed)
    if not df["filename"].str.contains(r"\.", regex=True).any():
        df["filename"] = df["filename"] + ".png"

    # Check if files are in class subfolders
    sample_has_class_dirs = any(
        os.path.isdir(os.path.join(BASE_DIR, d)) and d.isdigit()
        for d in os.listdir(BASE_DIR)
    )

    if sample_has_class_dirs:
        # Expect files at data/sample/<label>/<filename>
        # If filename already includes a subfolder, keep it; else prepend label/
        def ensure_label_prefix(row):
            fn = row["filename"].replace("\\", "/")
            if "/" in fn:  # already has a subfolder like "0/xxx.png"
                return fn
            return f"{row['label']}/{fn}"

        df["filename"] = df.apply(ensure_label_prefix, axis=1)
        flow_dir = BASE_DIR
    else:
        # Flat layout: files are directly in data/sample
        # Ensure filename has no path components
        df["filename"] = df["filename"].apply(lambda p: os.path.basename(str(p)))
        flow_dir = BASE_DIR

    # Filter rows to only those that actually exist
    def exists(rel_path):
        return os.path.exists(os.path.join(flow_dir, rel_path.replace("/", os.sep)))

    before = len(df)
    df = df[df["filename"].apply(exists)].copy()
    after = len(df)

    if after == 0:
        raise FileNotFoundError(
            "No valid images found after filename fix. "
            "Check that labels.csv uses the correct filenames and that images exist."
        )

    if after < before:
        print(f"Filtered out {before - after} rows with missing files. Using {after} images.")

    return flow_dir, df

def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"labels.csv not found at {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    flow_dir, df = infer_layout_and_fix_csv(df)

    # Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    # Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=flow_dir,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        directory=flow_dir,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train_gen, val_gen

if __name__ == "__main__":
    train_gen, val_gen = load_data()
    print(f"✅ Preprocessing complete. Train batches: {len(train_gen)}, Validation batches: {len(val_gen)}")
