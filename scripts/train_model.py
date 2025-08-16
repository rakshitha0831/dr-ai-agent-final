import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ==============================
# CONFIG
# ==============================
DATA_DIR = os.path.join("data", "sample")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
MODEL_SAVE_PATH = os.path.join("saved_models", "dr_model.h5")

# ==============================
# LOAD DATA
# ==============================
def load_data():
    print("📂 Loading data...")
    if not os.path.exists(LABELS_CSV):
        raise FileNotFoundError(f"❌ {LABELS_CSV} not found.")

    df = pd.read_csv(LABELS_CSV)

    # Convert labels to strings for categorical mode
    df['label'] = df['label'].astype(str)

    # Include subfolder in path
    df['filename'] = df.apply(lambda row: os.path.join(row['label'], row['filename']), axis=1)

    print("🔍 Sample data with paths:")
    print(df.head())

    # Train-validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train_generator, val_generator

# ==============================
# BUILD MODEL (Functional API)
# ==============================
def build_model():
    print("🛠 Building model (Functional API)...")
    inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', name="last_conv_layer")(x)  # Name for Grad-CAM
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(5, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ==============================
# TRAIN MODEL
# ==============================
if __name__ == "__main__":
    train_gen, val_gen = load_data()

    model = build_model()

    print("🚀 Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
