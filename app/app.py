# app.py — Final Enterprise-Ready Version
import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import re
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
from tensorflow.keras.layers import Conv2D, SeparableConv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --------------------------
# CONFIGURATION
# --------------------------
st.set_page_config(
    page_title="Diabetic Retinopathy AI Screening",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.path.join("saved_models", "dr_model.h5")
DATA_DIR = os.path.join("data", "sample")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
GUIDELINES_PATH = os.path.join("guidelines", "dr_guidelines.txt")

class_labels = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# --------------------------
# SAFE LAST CONV LAYER FINDER
# --------------------------
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (Conv2D, SeparableConv2D)):
            return layer.name
    return None

LAST_CONV_LAYER = get_last_conv_layer_name(model)
if LAST_CONV_LAYER is None:
    st.error("No convolutional layer found in the model.")
    st.stop()

# --------------------------
# HELPER FUNCTIONS
# --------------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def grad_cam(input_model, image_array, layer_name):
    grad_model = tf.keras.models.Model(
        [input_model.inputs], [input_model.get_layer(layer_name).output, input_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap + 1e-10)
    return heatmap

def overlay_gradcam(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.size[0], image.size[1]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(np.array(image), 1 - alpha, heatmap, alpha, 0)

def load_guidelines():
    with open(GUIDELINES_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    sections = re.split(r"\n(?=[A-Z][A-Za-z\s\(\)]+:)", content)
    guideline_dict = {}
    for section in sections:
        lines = section.strip().split("\n", 1)
        if len(lines) == 2:
            header = lines[0].strip().rstrip(":")
            text = lines[1].strip()
            guideline_dict[header.lower()] = text
    return guideline_dict

def get_answer(user_question):
    QUESTION_MAP = {
        "definition": "definition",
        "stages": "stages of dr",
        "symptoms": "symptoms",
        "risk factors": "risk factors",
        "screening": "screening recommendations",
        "treatment": "treatment options",
        "prevention": "prevention",
        "emergency": "emergency signs (seek immediate care)"
    }
    q = user_question.lower().strip()
    if q in QUESTION_MAP:
        return GUIDELINES.get(QUESTION_MAP[q], "❌ No information found.")
    for keyword, section in QUESTION_MAP.items():
        if keyword in q:
            return GUIDELINES.get(section, "❌ No information found.")
    return "❌ No relevant section found."

def evaluate_model():
    labels_df = pd.read_csv(LABELS_CSV)
    labels_df["filename"] = labels_df.apply(
        lambda row: os.path.join(str(row["label"]), row["filename"]), axis=1
    )
    labels_df['label'] = labels_df['label'].astype(str)
    val_df = labels_df.sample(frac=0.2, random_state=42)

    val_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=DATA_DIR,
        x_col="filename",
        y_col="label",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    loss, acc = model.evaluate(val_generator, verbose=0)
    y_true = np.array(val_generator.classes)
    y_prob = model.predict(val_generator, verbose=0)
    y_pred = np.array(np.argmax(y_prob, axis=1))

    tn, fp, fn, tp = confusion_matrix(
        (y_true > 0).astype(int), (y_pred > 0).astype(int)
    ).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return acc, sensitivity, specificity, y_true, y_pred, y_prob

GUIDELINES = load_guidelines()

# --------------------------
# SIDEBAR
# --------------------------
st.sidebar.title("📌 Navigation")
selected_tab = st.sidebar.radio("Go to:", ["🩺 DR Detection", "📚 Guidelines Chatbot", "📊 Model Insights"])
st.sidebar.markdown("### ℹ️ About")
st.sidebar.info("AI-powered screening tool for Diabetic Retinopathy with guidelines and live model insights.")

# --------------------------
# PAGE CONTENT
# --------------------------
if selected_tab == "🩺 DR Detection":
    st.markdown("## 🩺 Diabetic Retinopathy Detection")
    uploaded_file = st.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = preprocess_image(image)
        prediction = model.predict(img_array, verbose=0)[0]
        pred_class = np.argmax(prediction)
        confidence = float(np.max(prediction) * 100)

        st.metric(label="Condition", value=class_labels[pred_class], delta=f"{confidence:.2f}% confidence")
        st.progress(confidence / 100)

        heatmap = grad_cam(model, img_array, layer_name=LAST_CONV_LAYER)
        overlay = overlay_gradcam(image, heatmap)

        col1, col2 = st.columns(2)
        col1.image(image, caption="Original Fundus Image", use_container_width=True)
        col2.image(overlay, caption="AI Focused Regions (Grad-CAM)", use_container_width=True)

elif selected_tab == "📚 Guidelines Chatbot":
    st.markdown("## 📚 Diabetic Retinopathy Guidelines Chatbot")
    user_query = st.text_input("Ask about DR (e.g., 'symptoms', 'treatment', 'prevention')")
    if st.button("Get Answer"):
        answer = get_answer(user_query)
        st.markdown(f"**Answer:**\n\n{answer}")

elif selected_tab == "📊 Model Insights":
    st.markdown("## 📊 Model Performance Insights")
    acc, sensitivity, specificity, y_true, y_pred, y_prob = evaluate_model()

    # KPI Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc*100:.2f}%", delta=None)
    col2.metric("Sensitivity", f"{sensitivity:.2f}")
    col3.metric("Specificity", f"{specificity:.2f}")

    # 2×2 Grid for plots
    plot1, plot2 = st.columns(2)
    plot3, plot4 = st.columns(2)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix", fontsize=14)
    plot1.pyplot(fig_cm)

    # ROC Curve
    fpr, tpr, _ = roc_curve((y_true > 0).astype(int), y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
    ax_roc.plot([0, 1], [0, 1], 'k--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    ax_roc.set_title("ROC Curve", fontsize=14)
    plot2.pyplot(fig_roc)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve((y_true > 0).astype(int), y_prob[:, 1])
    fig_pr, ax_pr = plt.subplots(figsize=(5, 4))
    ax_pr.plot(recall, precision, label="PR Curve", linewidth=2)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="lower left")
    ax_pr.set_title("Precision-Recall Curve", fontsize=14)
    plot3.pyplot(fig_pr)

    # Class Distribution Plot
    fig_dist, ax_dist = plt.subplots(figsize=(5, 4))
    class_counts = pd.Series(y_true).value_counts().sort_index()
    bars = ax_dist.bar(class_labels, class_counts, color=sns.color_palette("pastel"))
    ax_dist.set_title("Class Distribution", fontsize=14)
    ax_dist.set_xlabel("Class")
    ax_dist.set_ylabel("Count")
    for bar in bars:
        height = bar.get_height()
        ax_dist.annotate(f'{int(height)}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)
    plot4.pyplot(fig_dist)
