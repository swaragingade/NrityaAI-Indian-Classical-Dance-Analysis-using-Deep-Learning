"""
utils.py — Shared constants, helpers, and synthetic data generation for NrityaAI.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = ['bharatanatyam', 'kathak', 'odissi']
WINDOW_SIZE = 60
STEP_SIZE = 30
N_KEYPOINTS = 33
N_FEATURES = 4          # x, y, z, visibility

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
KEYPOINTS_DIR = DATA_DIR / "keypoints"
REFERENCE_DIR = DATA_DIR / "reference_poses"
MODELS_DIR = ROOT_DIR / "models"

# ── Keypoint helpers ──────────────────────────────────────────────────────────

def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Center and scale keypoints to the range [-1, 1].

    Args:
        keypoints: Array of shape (..., N_KEYPOINTS, N_FEATURES) or
                   (..., N_KEYPOINTS * N_FEATURES).

    Returns:
        Normalized array of the same shape.
    """
    flat = keypoints.reshape(-1)
    min_val = flat.min()
    max_val = flat.max()
    if max_val - min_val < 1e-8:
        return np.zeros_like(keypoints, dtype=np.float32)
    scaled = 2.0 * (keypoints - min_val) / (max_val - min_val) - 1.0
    return scaled.astype(np.float32)


# ── Model loading ─────────────────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def load_model_once(model_path: Optional[Path] = None):
    """
    Load the Keras model exactly once and cache it in memory.

    Args:
        model_path: Path to the .h5 model file.  Defaults to
                    models/best_model.h5 relative to project root.

    Returns:
        Loaded Keras model, or None if the file does not exist.
    """
    global _MODEL_CACHE
    if model_path is None:
        # prefer .keras (Keras 3 native format), fall back to .h5
        model_path = MODELS_DIR / "best_model.keras"
        if not model_path.exists():
            model_path = MODELS_DIR / "best_model.h5"
        if not model_path.exists():
            model_path = MODELS_DIR / "demo_model.keras"

    model_path = Path(model_path)
    key = str(model_path)

    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if not model_path.exists():
        print(f"[utils] Model file not found: {model_path}")
        return None

    try:
        import os
        if "KERAS_BACKEND" not in os.environ:
            os.environ["KERAS_BACKEND"] = "torch"
        import keras
        model = keras.models.load_model(str(model_path))
        _MODEL_CACHE[key] = model
        print(f"[utils] Model loaded from {model_path}")
        return model
    except Exception as exc:
        print(f"[utils] Failed to load model: {exc}")
        return None


# ── Synthetic data generation ─────────────────────────────────────────────────

def generate_synthetic_keypoints(
    n_frames: int = 200,
    label_idx: int = 0,
    noise_std: float = 0.05,
) -> np.ndarray:
    """
    Generate a synthetic keypoint sequence for one video.

    Each class gets a slightly different bias vector so the model can
    learn a separable (though trivial) pattern during smoke-tests.

    Args:
        n_frames:  Number of frames to generate.
        label_idx: Class index (0=bharatanatyam, 1=kathak, 2=odissi).
        noise_std: Gaussian noise standard deviation.

    Returns:
        Array of shape (n_frames, N_KEYPOINTS, N_FEATURES) in [0, 1].
    """
    rng = np.random.default_rng(seed=label_idx * 42)
    bias = np.zeros(N_FEATURES)
    bias[0] = 0.1 * label_idx          # x offset per class

    frames = []
    for _ in range(n_frames):
        kp = rng.random((N_KEYPOINTS, N_FEATURES)).astype(np.float32)
        kp += bias
        kp += rng.normal(0, noise_std, kp.shape).astype(np.float32)
        kp = np.clip(kp, 0.0, 1.0)
        frames.append(kp)
    return np.stack(frames)


def generate_synthetic_dataset(
    videos_per_class: int = 5,
    frames_per_video: int = 200,
    save: bool = True,
) -> tuple:
    """
    Generate a complete synthetic dataset and optionally save CSVs to
    data/keypoints/<class>/<idx>.csv.

    Args:
        videos_per_class: Number of fake videos per dance style.
        frames_per_video: Frames in each fake video.
        save:            Whether to write CSVs to disk.

    Returns:
        (sequences, labels) where
            sequences: list of np.ndarray, each (n_frames, N_KEYPOINTS, N_FEATURES)
            labels:    list of int class indices
    """
    sequences, labels = [], []
    columns = ["frame"]
    for k in range(N_KEYPOINTS):
        columns += [f"kp{k}_x", f"kp{k}_y", f"kp{k}_z", f"kp{k}_v"]

    for label_idx, cls in enumerate(CLASSES):
        out_dir = KEYPOINTS_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        for vid_idx in range(videos_per_class):
            seq = generate_synthetic_keypoints(frames_per_video, label_idx)
            sequences.append(seq)
            labels.append(label_idx)

            if save:
                rows = []
                for f_idx, frame in enumerate(seq):
                    row = [f_idx] + frame.flatten().tolist()
                    rows.append(row)
                df = pd.DataFrame(rows, columns=columns)
                csv_path = out_dir / f"synthetic_{vid_idx:03d}.csv"
                df.to_csv(csv_path, index=False)

        print(f"[utils] Generated {videos_per_class} synthetic sequences for '{cls}'")

    # Also generate reference pose JSONs
    _generate_reference_poses()

    return sequences, labels


def _generate_reference_poses():
    """Create placeholder reference pose JSON files for each class."""
    rng = np.random.default_rng(seed=99)
    for cls in CLASSES:
        out_dir = REFERENCE_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for pose_name in ["neutral", "main_pose"]:
            path = out_dir / f"{pose_name}.json"
            if not path.exists():
                keypoints = []
                for _ in range(N_KEYPOINTS):
                    keypoints.append({
                        "x": float(rng.random()),
                        "y": float(rng.random()),
                        "z": float(rng.random()),
                    })
                with open(path, "w") as f:
                    json.dump({"keypoints": keypoints}, f, indent=2)
    print("[utils] Reference pose JSONs created.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating synthetic dataset …")
    seqs, lbls = generate_synthetic_dataset(videos_per_class=3, frames_per_video=120)
    print(f"Total sequences: {len(seqs)}, labels: {lbls}")
    print(f"First sequence shape: {seqs[0].shape}")
    print("Done.")
