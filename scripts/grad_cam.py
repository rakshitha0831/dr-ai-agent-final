import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Conv2D

# ==============================
# CONFIG
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "saved_models")
SAMPLE_DIR = os.path.join(BASE_DIR, "..", "data", "sample", "0")  # folder with images
IMG_SIZE = (224, 224)

# ==============================
# GET LATEST MODEL
# ==============================
def get_latest_model_path():
    model_files = [
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if f.endswith(".h5")
    ]
    if not model_files:
        raise FileNotFoundError(f"❌ No .h5 model found in {MODEL_DIR}")
    return max(model_files, key=os.path.getmtime)

MODEL_PATH = get_latest_model_path()
print(f"📂 Loading trained model: {MODEL_PATH}")
model = load_model(MODEL_PATH, compile=False)
print("✅ Model loaded successfully.")

# ==============================
# FIND LAST CONV LAYER
# ==============================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("No Conv2D layers found in the model.")

last_conv_layer_name = find_last_conv_layer(model)
print(f"🧠 Using last conv layer: {last_conv_layer_name}")

# ==============================
# LOAD TEST IMAGE
# ==============================
img_files = [f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not img_files:
    raise FileNotFoundError(f"❌ No image files found in {SAMPLE_DIR}")

TEST_IMAGE_PATH = os.path.join(SAMPLE_DIR, img_files[0])
print(f"🖼 Using test image: {TEST_IMAGE_PATH}")

img = image.load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

preds = model.predict(img_array)
pred_class_idx = np.argmax(preds[0])
confidence = preds[0][pred_class_idx] * 100
print(f"🔍 Predicted class: {pred_class_idx} ({confidence:.2f}% confidence)")

# ==============================
# GRAD-CAM FUNCTION
# ==============================
def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)
    return heatmap

# ==============================
# OVERLAY HEATMAP
# ==============================
def overlay_heatmap(heatmap, img_path, alpha=0.4, cmap='jet'):
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap(cmap)
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    return image.array_to_img(superimposed_img)

# ==============================
# GENERATE & SAVE GRAD-CAM
# ==============================
heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
output_path = os.path.join(BASE_DIR, "grad_cam_result.jpg")
superimposed_img = overlay_heatmap(heatmap, TEST_IMAGE_PATH)
superimposed_img.save(output_path)

print(f"✅ Grad-CAM saved to {output_path}")
plt.imshow(superimposed_img)
plt.axis("off")
plt.show()
