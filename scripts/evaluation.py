# scripts/evaluation.py — Final Version (labels.csv based)

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

# --------------------------
# CONFIG
# --------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "dr_model.h5")
DATA_DIR = os.path.join(BASE_DIR, "data", "sample")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PLOTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --------------------------
# LOAD MODEL
# --------------------------
print(f"📂 Loading trained model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

# --------------------------
# LOAD LABELS & PREPARE DATA
# --------------------------
labels_df = pd.read_csv(LABELS_CSV)

labels_df["filename"] = labels_df.apply(
    lambda row: os.path.join(str(row["label"]), row["filename"]), axis=1
)
labels_df['label'] = labels_df['label'].astype(str)

val_df = labels_df.sample(frac=0.2, random_state=42)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=DATA_DIR,
    x_col="filename",
    y_col="label",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# --------------------------
# EVALUATION
# --------------------------
loss, acc = model.evaluate(val_generator, verbose=0)
print(f"✅ Validation Accuracy: {acc * 100:.2f}%")
print(f"✅ Validation Loss: {loss:.4f}")

pred_probs = model.predict(val_generator, verbose=0)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = np.array(val_generator.classes)
class_labels = list(val_generator.class_indices.keys())

# --------------------------
# CLASSIFICATION REPORT
# --------------------------
print("\n📄 Classification Report:")
print(classification_report(true_classes, pred_classes, target_names=class_labels))

# --------------------------
# CONFUSION MATRIX
# --------------------------
cm = confusion_matrix(true_classes, pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.show()
plt.close()

# --------------------------
# BINARY ROC & PR CURVES
# --------------------------
print("\n📊 Generating Binary ROC & PR Curves...")
binary_true = (true_classes > 0).astype(int)
binary_scores = pred_probs[:, 1] if pred_probs.shape[1] > 1 else pred_probs[:, 0]

fpr, tpr, _ = roc_curve(binary_true, binary_scores)
roc_auc_val = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_val:.2f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Binary ROC Curve — DR vs No DR")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "binary_roc.png"))
plt.show()
plt.close()

precision, recall, _ = precision_recall_curve(binary_true, binary_scores)
plt.figure()
plt.plot(recall, precision, label="Precision-Recall curve", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Binary Precision-Recall Curve — DR vs No DR")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "binary_pr.png"))
plt.show()
plt.close()

# --------------------------
# MULTI-CLASS ROC CURVES
# --------------------------
print("\n📊 Generating Multi-Class ROC Curves...")
y_bin = label_binarize(true_classes, classes=list(range(len(class_labels))))
plt.figure(figsize=(7, 6))
for i, label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_bin[:, i], pred_probs[:, i])
    roc_auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_val:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curves")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "multiclass_roc.png"))
plt.show()
plt.close()

# --------------------------
# MULTI-CLASS PR CURVES
# --------------------------
print("\n📊 Generating Multi-Class Precision-Recall Curves...")
plt.figure(figsize=(7, 6))
for i, label in enumerate(class_labels):
    precision, recall, _ = precision_recall_curve(y_bin[:, i], pred_probs[:, i])
    plt.plot(recall, precision, label=f"{label}")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multi-Class Precision-Recall Curves")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "multiclass_pr.png"))
plt.show()
plt.close()

print(f"\n📁 All plots saved to: {PLOTS_DIR}")
