# DR AI Agent – Diabetic Retinopathy Detection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

The **DR AI Agent** is an end-to-end deep learning solution designed to detect diabetic retinopathy (DR) from retinal fundus images. It implements a complete machine learning lifecycle — from raw data acquisition to model deployment — with an emphasis on clinical usability, explainability, and scalability.

## Dataset Details

* **Source:** Kaggle EyePACS Dataset
* **Full Dataset:** 88k+ labeled images across 5 DR severity levels
* **Prototype Subset:** 5,000 balanced images for rapid iteration and testing
* **Classes:** 0 – No DR, 1 – Mild, 2 – Moderate, 3 – Severe, 4 – Proliferative DR
* **Preprocessing:** 224×224 resizing, normalization, augmentation (rotation, zoom, flip)
* **Note:** Training on the full dataset significantly boosts model accuracy and robustness.

## Pipeline Flow (End-to-End)

1. **Data Acquisition** – Fetch dataset using `download_data.py`.
2. **Subset Creation** – Generate a smaller, balanced dataset with `create_subset.py` for fast prototyping.
3. **Label Mapping** – Create `labels.csv` mapping images to DR severity using `generate_labels.py`.
4. **Data Preprocessing** – Apply resizing, normalization, and augmentations with `preprocess.py`.
5. **Model Development** – Train an EfficientNetB0 classifier using `train_model.py`, fine-tuning with class weights.
6. **Model Evaluation** – Assess performance via `evaluation.py`, producing classification reports, ROC curves, PR curves, and confusion matrices.
7. **Model Explainability** – Generate Grad-CAM heatmaps with `grad_cam.py` to highlight decision-driving retinal regions.
8. **Inference Pipeline** – Deploy `inference.py` for on-demand predictions from uploaded images.
9. **Interactive Deployment** – Use `app.py` (Streamlit) for real-time clinician and research use.
10. **Planned Extensions** – Integrate LLM-based APIs for retrieval-augmented chatbot support, enabling intelligent Q&A over patient history and DR guidelines.

## Model Architecture

* **Base:** EfficientNetB0 (pre-trained on ImageNet)
* **Structure:** GAP → Dense(128, ReLU) → Dropout(0.3) → Dense(5, Softmax)
* **Trainable Parameters:** ~5M

## Training Configuration

* Optimizer: Adam (LR = 1e-4)
* Loss: Categorical Crossentropy
* Batch Size: 32
* Epochs: 20
* Hardware: NVIDIA RTX 3090 GPU

## Evaluation – Prototype Subset (5,000 Images)

* Validation Accuracy: **53.30%**
* Validation Loss: **1.2056**

| Class | Precision | Recall | F1-Score |
| ----- | --------- | ------ | -------- |
| 0     | 0.58      | 0.96   | 0.72     |
| 1     | 0.34      | 0.25   | 0.29     |
| 2     | 0.40      | 0.16   | 0.23     |
| 3     | 0.00      | 0.00   | 0.00     |
| 4     | 1.00      | 0.01   | 0.02     |

Detailed plots (ROC, PR curves, confusion matrices) are stored in the `reports/` directory.

## Explainability

* **Grad-CAM** visualizations provide pixel-level interpretability, helping clinicians verify and trust model predictions.

## Tech Stack & Libraries

* **Languages:** Python 3.10+
* **Frameworks:** TensorFlow, Keras
* **Data Processing:** Pandas, NumPy, OpenCV
* **Visualization:** Matplotlib, Seaborn
* **UI Framework:** Streamlit
* **Version Control:** Git
* **Extensions:** LLM API integration for enhanced retrieval and natural language Q&A

## Deployment

* Local execution: `streamlit run app.py`
* Docker-ready for containerized deployment
* Cloud-compatible: AWS EC2, SageMaker, GCP AI Platform

## Future Enhancements

* Train on the **full EyePACS dataset** to boost accuracy.
* Improve sensitivity and recall for severe/proliferative DR.
* Expand classification to include other retinal diseases.
* Integrate **LLM-powered chatbot** for clinician and patient-facing interactions.

## Author

**Rakshitha Kamarathi Venkatesh**
📧 Email: [rakshitha0897@gmail.com](mailto:rakshitha0897@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/rakshitha-venkatesh-6824b7306/)

