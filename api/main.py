"""
api/main.py — FastAPI backend for NrityaAI.

Endpoints:
    GET  /health          → service status
    POST /analyze-video   → classify uploaded dance video
    POST /analyze-frame   → real-time pose correction for a single frame
"""

import io
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Resolve project root so imports work regardless of CWD
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from utils import CLASSES, WINDOW_SIZE, N_KEYPOINTS, N_FEATURES, load_model_once, normalize_keypoints
from feature_engineering import load_csv_sequence, create_windows
from pose_correction import compare_poses, overall_score, get_correction_text, load_reference_pose

# ── State ─────────────────────────────────────────────────────────────────────

_app_state: dict[str, Any] = {"model": None}


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Keras model once at startup."""
    model = load_model_once()
    _app_state["model"] = model
    if model is None:
        print("[api] WARNING: No model loaded — prediction endpoints degraded.")
    else:
        print("[api] Model loaded successfully.")
    yield
    _app_state["model"] = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NrityaAI",
    description="Indian Classical Dance Classification & Pose Correction API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class FrameRequest(BaseModel):
    """Request body for /analyze-frame."""
    keypoints: list[list[float]]   # 33 × 4 (x, y, z, visibility)
    style: str = "bharatanatyam"   # optional — for reference pose lookup


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _predict_from_windows(windows: np.ndarray) -> dict:
    """
    Run model inference on windowed keypoint data.

    Args:
        windows: Float32 array of shape (N, WINDOW_SIZE, 33, 4).

    Returns:
        Dict with predicted_style, confidence, pose_score.
    """
    model = _app_state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    preds = model.predict(windows, verbose=0)
    # Dual-output: [style_probs (N,3), score_probs (N,1)]
    if isinstance(preds, (list, tuple)):
        style_probs, score_probs = preds[0], preds[1]
    else:
        style_probs = preds
        score_probs = np.full((len(preds), 1), 0.5)

    # Aggregate across windows — mean probability
    mean_style = style_probs.mean(axis=0)          # (3,)
    mean_score = float(score_probs.mean()) * 100   # 0–100

    best_idx = int(np.argmax(mean_style))
    confidence = float(mean_style[best_idx]) * 100

    return {
        "predicted_style": CLASSES[best_idx],
        "confidence": round(confidence, 2),
        "pose_score": round(mean_score, 2),
        "class_probabilities": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(CLASSES, mean_style)
        },
    }


def _get_corrections(
    user_frame: np.ndarray,
    style: str,
) -> tuple[list[str], float]:
    """
    Load the first available reference pose for *style* and compute corrections.

    Args:
        user_frame: (N_KEYPOINTS, 4) keypoint array.
        style:      Dance style name.

    Returns:
        (correction_texts, score)
    """
    try:
        ref = load_reference_pose(style, "main_pose")
    except FileNotFoundError:
        return [], 100.0

    corrections = compare_poses(user_frame, ref)
    score = overall_score(corrections)
    texts = get_correction_text(corrections)
    return texts, score


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health():
    """Return service health and whether the model is loaded."""
    return {"status": "ok", "model_loaded": _app_state["model"] is not None}


@app.post("/analyze-video", tags=["Inference"])
async def analyze_video(file: UploadFile = File(...)):
    """
    Accept an uploaded dance video, extract keypoints, classify style,
    and return pose corrections.

    Args:
        file: Video file (.mp4 or .avi).

    Returns:
        JSON with predicted_style, confidence, pose_score, corrections.
    """
    # Validate file type
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".avi"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use .mp4 or .avi.",
        )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        # Extract keypoints
        try:
            from extract_keypoints import extract_from_video
        except ImportError:
            raise HTTPException(status_code=500, detail="mediapipe not available.")

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as csv_tmp:
            csv_path = Path(csv_tmp.name)

        n_frames = extract_from_video(tmp_path, csv_path)
        if n_frames == 0:
            raise HTTPException(
                status_code=422,
                detail="No pose detected in video. Ensure the full body is visible.",
            )

        # Load and window
        seq = load_csv_sequence(csv_path)          # (T, 33, 4)
        seq = normalize_keypoints(seq)
        windows = create_windows(seq)              # (W, 60, 33, 4)

        if windows.shape[0] == 0:
            raise HTTPException(status_code=422, detail="Video too short.")

        prediction = _predict_from_windows(windows)

        # Corrections using last frame
        last_frame = seq[-1]
        corrections, _ = _get_corrections(last_frame, prediction["predicted_style"])
        prediction["corrections"] = corrections

        return JSONResponse(content=prediction)

    finally:
        tmp_path.unlink(missing_ok=True)
        try:
            csv_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/analyze-frame", tags=["Inference"])
async def analyze_frame(req: FrameRequest):
    """
    Accept a single frame's keypoints (from live webcam) and return
    pose score + corrections.

    Args:
        req: JSON body with 'keypoints' (33×4 list) and optional 'style'.

    Returns:
        JSON with pose_score and corrections list.
    """
    if len(req.keypoints) != N_KEYPOINTS:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {N_KEYPOINTS} keypoints, got {len(req.keypoints)}.",
        )

    frame = np.array(req.keypoints, dtype=np.float32)  # (33, 4)

    style = req.style if req.style in CLASSES else CLASSES[0]
    corrections, score = _get_corrections(frame, style)

    return JSONResponse(content={
        "pose_score": round(score, 2),
        "corrections": corrections,
    })
