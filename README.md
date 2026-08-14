# 💵 Fake Currency Detection using Computer Vision & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An educational, portfolio-grade Computer Vision (CV) and Machine Learning (ML) system designed for analyzing currency note images and classifying them into **Genuine** or **Potentially Counterfeit** categories.

---

> [!IMPORTANT]
> **Educational Disclaimer**: This system is built strictly for learning, experimentation, and portfolio demonstration. It does not provide legally binding or authoritative currency authentication. Predictions represent statistical visual feature patterns rather than official verification procedures.

---

## 🌟 Key Features

1. **Synthetic Currency Dataset & Sample Generator**: Zero-friction setup generating benchmark realistic genuine & counterfeit currency note images.
2. **Multi-Domain Feature Extractor**:
   - **Color Features**: Mean, std, and 16-bin 3D histograms for BGR, HSV, and LAB color spaces.
   - **Texture Features**: Uniform Local Binary Patterns (LBP) and Gray-Level Co-occurrence Matrix (GLCM) metrics (contrast, dissimilarity, homogeneity, energy, correlation).
   - **Edge Features**: Canny edge density ratio, Sobel gradient magnitudes, and Laplacian sharpness variance.
   - **Shape/Contour Features**: Contour counts, area ratio, perimeter, and bounding rectangle metrics.
3. **Machine Learning & Neural Network Benchmarking**: Evaluates Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Support Vector Machine (RBF-SVM), and Neural Networks (MLP/CNN).
4. **Zero Data Leakage Pipeline**: Strict 70% Train, 15% Validation, and 15% Isolated Test dataset split.
5. **Interactive Web Dashboard**: Modern emerald dark-mode glassmorphic interface powered by Flask, featuring real-time image drag-and-drop, channel previews (Canny, CLAHE, Heatmaps), AI mentor lessons, benchmark comparisons, and REST API tester.

---

## 🏗️ Project Architecture

```text
┌─────────────────────────┐
│ Currency Image Input    │  (Web Upload / Sample Selector / Synthetic Dataset)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Security & Validation   │  (Format checks, File size limits, Safe filename, MIME)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Image Preprocessing     │  (Resize 400x200, Gaussian Blur, CLAHE Contrast)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Feature Extraction      │  (Color Histograms + LBP Texture + Canny Edges + Contours)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Model Inference Engine  │  (Logistic Regression, SVM, Random Forest, Neural Network)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ Explainability & Output │  (Prediction label, Confidence score, Heatmaps, REST API)
└─────────────────────────┘
```

---

## 📁 Directory Structure

```text
Fake Currency Detection/
├── dataset/
│   ├── genuine/            # Sample genuine note images
│   └── counterfeit/        # Sample counterfeit note images
├── data/
│   ├── raw/                # Raw uploads / generated samples
│   ├── processed/          # Preprocessed images
│   └── features/           # Extracted feature arrays
├── notebooks/
│   ├── 01_python_image_basics.ipynb
│   ├── 02_opencv_fundamentals.ipynb
│   ├── 03_eda_and_dataset_prep.ipynb
│   ├── 04_feature_extraction_and_ml.ipynb
│   └── 05_cnn_deep_learning.ipynb
├── src/
│   ├── __init__.py
│   ├── security.py          # Input validation, file safety
│   ├── preprocessing.py     # Resizing, blurring, thresholding, contrast
│   ├── features.py          # Color, texture (LBP/GLCM), edge (Canny/Sobel) extractors
│   ├── dataset.py           # Synthetic dataset builder & loader with data split
│   ├── train.py             # Baseline ML & CNN model training
│   ├── evaluate.py          # Metrics, confusion matrix, ROC curve, error analysis
│   ├── predict.py           # Single-image inference pipeline
│   ├── explainability.py   # Visual feature heatmaps generator
│   └── tracker.py           # Experiment logger & history tracker
├── models/                  # Saved .pkl and .json experiment logs
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_pipeline.py
├── app/
│   ├── app.py              # Flask server (Web Dashboard + REST API)
│   ├── templates/
│   │   └── index.html      # Modern web interface
│   └── static/
│       ├── css/
│       │   └── style.css   # Dark glassmorphism styling
│       └── js/
│           └── main.js     # Drag-and-drop, Canvas previews, API caller
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚡ Quick Start & Installation

### 1. Clone & Set Up Environment
```bash
cd "Fake Currency Detection"
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate Benchmark Dataset & Train Models
```bash
python -m src.dataset
python -m src.train
python -m src.evaluate
```

### 4. Run Test Suite
```bash
python -m unittest discover tests/
```

### 5. Launch Interactive Web Dashboard
```bash
python app/app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📡 REST API Usage

### `POST /api/predict`
Upload a currency image file to get real-time inspection results.

#### Example Request (cURL):
```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -F "file=@dataset/genuine/genuine_001.png"
```

#### Example JSON Response:
```json
{
  "success": true,
  "prediction": "genuine",
  "readable_label": "Genuine Note",
  "is_genuine": true,
  "confidence": 0.945,
  "genuine_probability": 0.945,
  "counterfeit_probability": 0.055,
  "metrics_summary": {
    "edge_density": 0.0842,
    "contrast_variation": 48.25
  },
  "disclaimer": "IMPORTANT: This is an experimental machine-learning classification..."
}
```

---

## 🎓 Educational Levels Covered

| Level | Topic | Description & OpenCV Methods |
| :--- | :--- | :--- |
| **Level 1** | Python & Image Basics | Image matrices, RGB vs BGR, Grayscale conversion |
| **Level 2** | OpenCV Fundamentals | `cv2.GaussianBlur()`, `cv2.createCLAHE()`, `cv2.Canny()` |
| **Level 3** | Dataset & EDA | Class balance, image dimensions, zero data leakage split |
| **Level 4** | Feature Extraction | Color stats, Local Binary Patterns (LBP), GLCM texture |
| **Level 5** | Baseline ML | Logistic Regression, Decision Tree, Random Forest, SVM |
| **Level 6** | Deep Learning | Convolutional Neural Networks (`Conv2D`, `ReLU`, `MaxPool`) |
| **Level 7** | Evaluation & Security | Confusion Matrix, ROC-AUC, MIME file validation |

---

## 📜 License
This project is open-source under the MIT License. Built for educational & portfolio demonstration.
