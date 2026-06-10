# NrityaAI 💃

An AI system for **Indian Classical Dance Style Classification** and **Real-Time Pose Correction**. It uses MediaPipe BlazePose for keypoint extraction, a CNN+LSTM model for classifying Bharatanatyam, Kathak, and Odissi, and joint-angle deviation scoring for pose correction feedback.

---

## Project Structure

```
nrityaai/
├── data/
│   ├── raw/               # raw videos organised by class
│   │   ├── bharatanatyam/
│   │   ├── kathak/
│   │   └── odissi/
│   ├── keypoints/         # extracted keypoint CSVs
│   └── reference_poses/   # ideal pose JSONs for correction
├── src/
│   ├── utils.py           # constants, helpers, synthetic data
│   ├── extract_keypoints.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   └── pose_correction.py
├── api/
│   └── main.py            # FastAPI backend
├── app/
│   └── streamlit_app.py   # Streamlit frontend
├── models/                # saved checkpoints + plots
├── requirements.txt
├── run_demo.py            # end-to-end smoke test
└── README.md
```

---

## Setup

### 1. Create a virtual environment (Python 3.10+)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify everything works (no real data needed)

```bash
python run_demo.py
```

This generates synthetic data, trains the model for 5 epochs, evaluates it, and tests pose correction — confirming the entire pipeline works end-to-end.

---

## Step-by-Step Usage

### Step 1 — Collect real dance videos (optional)

Use `yt-dlp` to download Creative Commons or freely available dance performances:

```bash
# Bharatanatyam
yt-dlp -o "data/raw/bharatanatyam/%(title)s.%(ext)s" \
  --format "mp4" \
  "https://www.youtube.com/watch?v=<VIDEO_ID>"

# Kathak
yt-dlp -o "data/raw/kathak/%(title)s.%(ext)s" \
  --format "mp4" \
  "https://www.youtube.com/watch?v=<VIDEO_ID>"

# Odissi
yt-dlp -o "data/raw/odissi/%(title)s.%(ext)s" \
  --format "mp4" \
  "https://www.youtube.com/watch?v=<VIDEO_ID>"
```

> Tip: Use `--match-filter "license=creativecommon"` to stay within legal bounds.

### Step 2 — Extract keypoints

```bash
# Single video
python src/extract_keypoints.py \
  --video data/raw/bharatanatyam/sample.mp4 \
  --out data/keypoints/bharatanatyam/sample.csv

# Entire folder
python src/extract_keypoints.py \
  --folder data/raw/kathak \
  --out_folder data/keypoints/kathak \
  --label kathak
```

### Step 3 — Train the model

```bash
python src/train.py
```

The best model is saved to `models/best_model.h5`. Training curves are saved to `models/training_history.png`.

### Step 4 — Evaluate

```bash
python src/evaluate.py
```

Outputs `models/eval_results.txt` and `models/confusion_matrix.png`.

### Step 5 — Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 6 — Run the frontend

```bash
streamlit run app/streamlit_app.py
```

Opens at [http://localhost:8501](http://localhost:8501).

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service status + model loaded flag |
| POST | `/analyze-video` | Upload `.mp4`/`.avi`, returns style + pose corrections |
| POST | `/analyze-frame` | Send 33×4 keypoints (webcam frame), returns correction |

### Example: analyze a frame

```bash
curl -X POST http://localhost:8000/analyze-frame \
  -H "Content-Type: application/json" \
  -d '{"keypoints": [[0.5,0.3,0.0,0.99], ...], "style": "bharatanatyam"}'
```

---

## Dataset Sources

| Source | URL | License |
|--------|-----|---------|
| YouTube — Bharatanatyam | `youtube.com/results?search_query=bharatanatyam+performance` | Varies |
| YouTube — Kathak | `youtube.com/results?search_query=kathak+dance+performance` | Varies |
| YouTube — Odissi | `youtube.com/results?search_query=odissi+dance+performance` | Varies |
| Wikimedia Commons | `commons.wikimedia.org/wiki/Category:Indian_classical_dance` | CC |

> Always verify licenses before using any video for training.

---

## Model Architecture

```
Input  (60, 33, 4)
  → TimeDistributed Conv1D(64, relu)
  → TimeDistributed Conv1D(128, relu)
  → TimeDistributed GlobalAveragePooling1D
  → LSTM(256, return_sequences=True) + Dropout(0.3)
  → LSTM(128) + Dropout(0.3)
  → style : Dense(3, softmax)    # classification
  → score : Dense(1, sigmoid)    # pose quality 0–1
```

---

## Expected Evaluation Metrics

With a well-curated dataset of ~50+ videos per class:

| Metric | Expected |
|--------|----------|
| Accuracy | 85–95% |
| Macro F1 | 0.85–0.93 |
| Bharatanatyam F1 | 0.88+ |
| Kathak F1 | 0.85+ |
| Odissi F1 | 0.83+ |

> Results will be lower on the synthetic demo data — that's expected.

---

## Tech Stack

- **MediaPipe** — BlazePose 33-keypoint extraction
- **TensorFlow / Keras** — CNN+LSTM model
- **OpenCV** — video decoding, webcam capture
- **FastAPI + Uvicorn** — REST API
- **Streamlit** — web frontend
- **scikit-learn** — evaluation metrics
- **yt-dlp** — dataset collection helper

---

## License

MIT
