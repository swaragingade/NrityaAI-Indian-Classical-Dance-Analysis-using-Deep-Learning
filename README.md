# 💃 NrityaAI — Indian Classical Dance Analysis using Deep Learning

A real-time AI system for **Indian Classical Dance Style Classification** and **Pose Correction** using MediaPipe BlazePose, a CNN-LSTM model, and joint-angle geometric reasoning.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red?logo=streamlit&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-BlazePose-purple)
![Accuracy](https://img.shields.io/badge/Macro%20F1-99.65%25-brightgreen)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Step-by-Step Usage](#-step-by-step-usage)
- [API Reference](#-api-reference)
- [Model Details](#-model-details)
- [Results](#-results)
- [Contributors](#-contributors)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎭 **Dance Style Classification** | Classifies Bharatanatyam, Kathak, and Odissi with 99.65% macro F1 |
| 🦴 **Real-Time Pose Correction** | Joint-angle deviation engine with directional feedback (raise/lower, bend/extend) |
| 📹 **Video Upload Analysis** | Upload MP4/AVI and get style prediction + pose corrections |
| 🎥 **Live Webcam Mode** | Real-time skeleton overlay at 25–30 FPS with live pose score |
| 📊 **Pose Quality Scoring** | 0–100 score based on deviation from 24 reference poses |
| 🛡️ **Confidence Thresholding** | Unrecognised dance forms routed to "Other" bucket at < 60% confidence |
| 🔒 **Fully Local** | All inference runs on CPU — no cloud, no data upload |

---

## 🏗️ Architecture

The system follows a 4-layer pipeline:

```
User Input (Video / Webcam)
        ↓
[Presentation Layer]     →  Streamlit Dashboard
        ↓
[Application Layer]      →  FastAPI + MediaPipe BlazePose (33 keypoints/frame)
        ↓
[AI Processing Layer]    →  CNN-LSTM (style classification + pose score)
                         →  Joint-Angle Scorer (pose correction feedback)
        ↓
[Data Layer]             →  Reference Pose Database (24 poses × 3 styles)
```

**9-Stage Runtime Workflow:**

1. User Input → 2. Frame Extraction (10 FPS) → 3. BlazePose Keypoint Extraction → 4. Sequence Formation (60-frame windows) → 5. CNN-LSTM Classification → 6. Confidence Gate (≥60%) → 7. Joint-Angle Scoring → 8. JSON Response → 9. Dashboard Display

---

## 🛠️ Tech Stack

| Layer | Technology | Link |
|---|---|---|
| Pose Estimation | MediaPipe BlazePose | [mediapipe.dev](https://developers.google.com/mediapipe) |
| Deep Learning | TensorFlow + Keras | [tensorflow.org](https://www.tensorflow.org) |
| Video Processing | OpenCV | [opencv.org](https://opencv.org) |
| Backend API | FastAPI + Uvicorn | [fastapi.tiangolo.com](https://fastapi.tiangolo.com) |
| Frontend | Streamlit | [streamlit.io](https://streamlit.io) |
| Evaluation | scikit-learn | [scikit-learn.org](https://scikit-learn.org) |
| Dataset Collection | yt-dlp | [github.com/yt-dlp](https://github.com/yt-dlp/yt-dlp) |
| Data Processing | NumPy + Pandas | [numpy.org](https://numpy.org) |

---

## 📁 Project Structure

```
nrityaai/
├── data/
│   ├── raw/                    # Raw videos organised by class
│   │   ├── bharatanatyam/
│   │   ├── kathak/
│   │   └── odissi/
│   ├── keypoints/              # Extracted keypoint CSVs
│   └── reference_poses/        # 24 reference pose JSONs
│       ├── bharatanatyam/      # 11 poses (Ardhamandalam, Nataraj, etc.)
│       ├── kathak/             # 7 poses (Aamad, Chakkar, Tatkar, etc.)
│       └── odissi/             # 6 poses (Tribhanga, Chowk, Abhanga, etc.)
├── src/
│   ├── utils.py
│   ├── extract_keypoints.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── pose_correction.py
├── api/
│   └── main.py                 # FastAPI backend
├── app/
│   └── streamlit_app.py        # Streamlit frontend
├── models/
│   ├── confusion_matrix.png
│   └── training_history.png
├── reports/
│   └── NrityaAI_IEEE_Draft.md
├── requirements.txt
└── run_demo.py                 # End-to-end smoke test
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.10+
- pip
- Webcam (for live mode)

### 1. Clone the repository

```bash
git clone https://github.com/swaragingade/NrityaAI-Indian-Classical-Dance-Analysis-using-Deep-Learning.git
cd NrityaAI-Indian-Classical-Dance-Analysis-using-Deep-Learning
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the pipeline (no real data needed)

```bash
python run_demo.py
```

This generates synthetic data, trains the model for 5 epochs, evaluates it, and tests pose correction end-to-end.

---

## 🚀 Step-by-Step Usage

### Step 1 — Collect dance videos

```bash
yt-dlp -o "data/raw/bharatanatyam/%(title)s.%(ext)s" --format "mp4" "https://www.youtube.com/watch?v=<VIDEO_ID>"
yt-dlp -o "data/raw/kathak/%(title)s.%(ext)s" --format "mp4" "https://www.youtube.com/watch?v=<VIDEO_ID>"
yt-dlp -o "data/raw/odissi/%(title)s.%(ext)s" --format "mp4" "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

### Step 2 — Extract keypoints

```bash
python src/extract_keypoints.py --folder data/raw/bharatanatyam --out_folder data/keypoints/bharatanatyam --label bharatanatyam
```

### Step 3 — Train the model

```bash
python src/train.py
```

Best model saved to `models/best_model.h5`.

### Step 4 — Evaluate

```bash
python src/evaluate.py
```

### Step 5 — Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Docs at: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 6 — Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Opens at: [http://localhost:8501](http://localhost:8501)

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status + model loaded flag |
| `POST` | `/analyze-video` | Upload MP4/AVI → style label + pose corrections |
| `POST` | `/analyze-frame` | Send 33×4 keypoints (webcam) → correction + score |

```bash
# Example: health check
curl http://localhost:8000/health

# Example: analyze a frame
curl -X POST http://localhost:8000/analyze-frame \
  -H "Content-Type: application/json" \
  -d '{"keypoints": [[0.5,0.3,0.0,0.99], ...], "style": "bharatanatyam"}'
```

---

## 🧠 Model Details

```
Input Shape: (60, 33, 4)
  → TimeDistributed Conv1D(64, kernel=3, relu)
  → TimeDistributed Conv1D(128, kernel=3, relu)
  → TimeDistributed GlobalAveragePooling1D
  → LSTM(256, return_sequences=True) + Dropout(0.3)
  → LSTM(128) + Dropout(0.3)
  ├── Style Head  → Dense(3, softmax)    # Bharatanatyam / Kathak / Odissi
  └── Score Head  → Dense(1, sigmoid)    # Pose quality 0–100
```

**Training:** 40,977 augmented samples · Adam optimizer · Multi-task loss (CE + MSE) · Early stopping (patience: 10)

**Joint Angles Tracked:** Left/Right Elbow, Left/Right Knee, Left/Right Hip

---

## 📊 Results

| Metric | Value |
|---|---|
| Macro F1-Score | **99.65%** |
| Bharatanatyam F1 | 100% |
| Kathak F1 | 99.2% |
| Odissi F1 | 99.3% |
| Live Webcam FPS | 25–30 FPS |
| API Response Time | < 50ms |

---

## 📄 License

MIT
