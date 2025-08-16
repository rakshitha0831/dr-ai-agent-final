import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "..", "saved_models")
SAMPLE_DIR = os.path.join(BASE_DIR, "..", "data", "sample", "0")  # Change folder if needed
IMG_SIZE = (224, 224)  # Match your training size

# =========================
# Get Latest Model
# =========================
def get_latest_model_path():
    model_files = [
        os.path.join(SAVED_MODELS_DIR, f)
        for f in os.listdir(SAVED_MODELS_DIR)
        if f.endswith(".h5")
    ]
    if not model_files:
        raise FileNotFoundError(f"❌ No .h5 model found in {SAVED_MODELS_DIR}")
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

MODEL_PATH = get_latest_model_path()
print(f"📂 Loading trained model: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ Model loaded successfully.")

# =========================
# Get Test Image
# =========================
img_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not img_files:
    raise FileNotFoundError(f"❌ No image files found in {SAMPLE_DIR}")

test_img_path = os.path.join(SAMPLE_DIR, img_files[0])
print(f"🖼 Using test image: {test_img_path}")

# =========================
# Load and Preprocess Image
# =========================
img = image.load_img(test_img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# =========================
# Predict
# =========================
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]
confidence = np.max(predictions) * 100

print(f"🔍 Predicted class: {predicted_class} ({confidence:.2f}% confidence)")
