# 🦴 KneeVision AI — Knee Osteoarthritis Classifier

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-98%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-blue)

**Deep learning system for automated knee osteoarthritis severity grading from X-ray images**

[Live Demo](#) · [LinkedIn](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/) · [GitHub](https://github.com/HafsaIbrahim5)

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [KL Grading Scale](#-kellgren-lawrence-kl-grading-scale)
- [Models](#-models)
- [Dataset](#-dataset)
- [App Features](#-app-features)
- [Project Structure](#-project-structure)
- [Setup & Run](#-setup--run)
- [Results](#-results)
- [Disclaimer](#-disclaimer)

---

## 🎯 Overview

KneeVision AI is a computer vision project that automates the grading of knee osteoarthritis (OA) severity from plain radiographs. The system uses two state-of-the-art deep learning architectures — **EfficientNetB2** and **Xception** — fine-tuned via transfer learning on a labelled knee X-ray dataset.

The best model achieves **98% validation accuracy**, demonstrating strong potential as a clinical decision-support tool.

---

## 🩻 Kellgren-Lawrence (KL) Grading Scale

| Grade | Name | Description |
|-------|------|-------------|
| ✅ Grade 0 | Normal | No radiographic features of OA present |
| 🔵 Grade 1 | Doubtful | Doubtful joint space narrowing, possible osteophytic lipping |
| 🟡 Grade 2 | Mild | Definite osteophytes, possible joint space narrowing |
| 🟠 Grade 3 | Moderate | Multiple osteophytes, definite narrowing, sclerosis |
| 🔴 Grade 4 | Severe | Large osteophytes, marked narrowing, severe sclerosis |

---

## 🤖 Models

### EfficientNetB2 ⭐ Best
| Metric | Value |
|--------|-------|
| Accuracy | **98.0%** |
| Precision | 97.6% |
| Recall | 97.8% |
| F1-Score | 97.7% |
| Parameters | 7.7M |
| Input Size | 224×224 |

### Xception
| Metric | Value |
|--------|-------|
| Accuracy | **96.5%** |
| Precision | 96.1% |
| Recall | 96.3% |
| F1-Score | 96.2% |
| Parameters | 22.9M |
| Input Size | 128×128 |

---

## 📂 Dataset

- **Source**: Knee X-Ray Image Dataset (OAI-derived)
- **Total Images**: 3,282
- **Classes**: 5 (KL Grade 0–4)
- **Split**: 80% train / 20% validation

| Class | Images |
|-------|--------|
| Grade 0 — Normal | 514 |
| Grade 1 — Doubtful | 791 |
| Grade 2 — Mild | 696 |
| Grade 3 — Moderate | 663 |
| Grade 4 — Severe | 618 |

---

## ✨ App Features

| Feature | Description |
|---------|-------------|
| 🔬 AI Diagnosis | Upload X-ray → instant KL grade prediction |
| 📊 Confidence Scores | Per-class probability bars + radar chart |
| 📉 Performance Dashboard | Accuracy / Precision / Recall / F1 comparison |
| 🔥 Confusion Matrix | Interactive heatmap visualization |
| 📈 ROC Curves | One-vs-rest ROC-AUC per class |
| ⬇️ Download Report | Exportable diagnosis report (TXT) |
| 💡 Clinical Suggestion | Grade-based action recommendation |
| 🌙 Dark Medical UI | Professional dark theme with cyan accents |

---

## 🗂 Project Structure

```
knee_xray_app/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── model.EfficientNetB2.keras    # ← Place your trained model here
├── model.Xception.keras          # ← Place your trained model here
│
└── notebooks/
    ├── EfficientNetB2.ipynb      # Training notebook
    └── Xception.ipynb            # Training notebook
```

> **Note**: The app runs in **Demo Mode** if model files are not present. To enable real predictions, place your trained `.keras` model files in the root directory.

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/HafsaIbrahim5/knee-xray-classifier.git
cd knee-xray-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Add trained models
Place your saved models in the root directory:
```
model.EfficientNetB2.keras
model.Xception.keras
```

### 4. Run the app
```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

---

## 📊 Results

The **EfficientNetB2** model outperforms Xception across all metrics while using significantly fewer parameters (7.7M vs 22.9M), making it the preferred model for deployment.

Data augmentation techniques (Random Flip, Rotation, Zoom, Crop) were applied to improve generalization and prevent overfitting.

---

## ⚠️ Disclaimer

This application is developed for **research and educational purposes only**. The AI predictions are **not a substitute** for professional medical diagnosis. Always consult a qualified radiologist or orthopedic specialist for clinical assessment.

---

## 👩‍💻 Author

**Hafsa Ibrahim** — AI & Machine Learning Engineer

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/hafsa-ibrahim-ai-mi/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/HafsaIbrahim5)

---

## 📚 References

- Kellgren, J.H. & Lawrence, J.S. (1957). Radiological assessment of osteoarthritis.
- Tan, M. & Le, Q.V. (2019). [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946).
- Chollet, F. (2017). [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357).
- Osteoarthritis Initiative (OAI) Dataset — UCSF.
