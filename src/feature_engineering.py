"""
feature_engineering.py — Joint angle computation, sliding-window creation,
and dataset loading for NrityaAI.

Usage (standalone):
    python src/feature_engineering.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from any CWD
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    CLASSES, WINDOW_SIZE, STEP_SIZE, N_KEYPOINTS, N_FEATURES,
    KEYPOINTS_DIR, normalize_keypoints,
)

# ── Joint definitions ─────────────────────────────────────────────────────────

# Each entry: (joint_name, idx_a, idx_b, idx_c)
# Angle is computed at vertex b, between rays b→a and b→c.
JOINT_TRIPLETS = [
    ("left_elbow",  11, 13, 15),
    ("right_elbow", 12, 14, 16),
    ("left_knee",   23, 25, 27),
    ("right_knee",  24, 26, 28),
    ("left_hip",    11, 23, 25),
    ("right_hip",   12, 24, 26),
]
N_JOINTS = len(JOINT_TRIPLETS)   # 6


# ── Angle computation ─────────────────────────────────────────────────────────

def _vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the vector from b to a."""
    return a - b


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the angle (degrees) at vertex *b* formed by points a-b-c.

    Uses arctan2 for numerical stability.

    Args:
        a, b, c: 3-D coordinate arrays of shape (3,).

    Returns:
        Angle in degrees [0, 180].
    """
    v1 = _vec(a, b)
    v2 = _vec(c, b)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def compute_angles(frame_keypoints: np.ndarray) -> np.ndarray:
    """
    Compute joint angles for a single frame.

    Args:
        frame_keypoints: Array of shape (N_KEYPOINTS, N_FEATURES) where
                         columns are [x, y, z, visibility].

    Returns:
        Array of shape (N_JOINTS,) — one angle per joint in degrees.
    """
    angles = np.zeros(N_JOINTS, dtype=np.float32)
    xyz = frame_keypoints[:, :3]   # drop visibility for geometry

    for i, (_, ia, ib, ic) in enumerate(JOINT_TRIPLETS):
        angles[i] = angle_between(xyz[ia], xyz[ib], xyz[ic])

    return angles


# ── Sliding window ────────────────────────────────────────────────────────────

def create_windows(
    sequence: np.ndarray,
    window: int = WINDOW_SIZE,
    step: int = STEP_SIZE,
) -> np.ndarray:
    """
    Slice a temporal sequence into overlapping fixed-length windows.

    Args:
        sequence: Array of shape (T, ...) where T is the time dimension.
        window:   Number of frames per window.
        step:     Frame stride between consecutive windows.

    Returns:
        Array of shape (n_windows, window, ...).
        Returns empty array with correct shape if sequence is too short.
    """
    T = sequence.shape[0]
    if T < window:
        # Pad with zeros if the sequence is shorter than one window
        pad_len = window - T
        pad = np.zeros((pad_len, *sequence.shape[1:]), dtype=sequence.dtype)
        sequence = np.concatenate([sequence, pad], axis=0)
        T = window

    windows = []
    start = 0
    while start + window <= T:
        windows.append(sequence[start: start + window])
        start += step

    return np.stack(windows) if windows else np.empty((0, window, *sequence.shape[1:]))


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_csv_sequence(csv_path: Path) -> np.ndarray:
    """
    Load a keypoint CSV and return an array of shape (T, N_KEYPOINTS, N_FEATURES).

    Args:
        csv_path: Path to a keypoint CSV produced by extract_keypoints.py.

    Returns:
        Float32 array of shape (T, 33, 4).

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Keypoint CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    # Drop the 'frame' column
    values = df.drop(columns=["frame"]).values.astype(np.float32)
    # Reshape to (T, 33, 4)
    T = values.shape[0]
    return values.reshape(T, N_KEYPOINTS, N_FEATURES)


def load_dataset(
    keypoints_dir: Path | str = KEYPOINTS_DIR,
    use_angles: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load all keypoint CSVs, apply windowing, and return X / y arrays.

    Args:
        keypoints_dir: Root folder with sub-folders per class.
        use_angles:    If True, return angle features (60, 6) instead of
                       raw keypoints (60, 33, 4).

    Returns:
        X         : Float32 array of shape (N_samples, WINDOW_SIZE, 33, 4)
                    or (N_samples, WINDOW_SIZE, 6) if use_angles.
        y         : Int array of shape (N_samples,) with class indices.
        label_map : Dict mapping class name → int index.
    """
    keypoints_dir = Path(keypoints_dir)
    label_map = {cls: i for i, cls in enumerate(CLASSES)}
    X_list, y_list = [], []

    for cls in CLASSES:
        cls_dir = keypoints_dir / cls
        if not cls_dir.exists():
            print(f"[feature_eng] Folder missing, skipping: {cls_dir}")
            continue

        csv_files = sorted(cls_dir.glob("*.csv"))
        if not csv_files:
            print(f"[feature_eng] No CSVs in {cls_dir}")
            continue

        for csv_path in csv_files:
            try:
                seq = load_csv_sequence(csv_path)           # (T, 33, 4)
                seq = normalize_keypoints(seq)

                if use_angles:
                    angle_seq = np.array(
                        [compute_angles(frame) for frame in seq], dtype=np.float32
                    )                                        # (T, 6)
                    windows = create_windows(angle_seq)     # (W, 60, 6)
                else:
                    windows = create_windows(seq)           # (W, 60, 33, 4)

                if windows.shape[0] == 0:
                    continue

                label_idx = label_map[cls]
                X_list.append(windows)
                y_list.extend([label_idx] * windows.shape[0])

            except Exception as exc:
                print(f"[feature_eng] SKIP {csv_path.name}: {exc}")

    if not X_list:
        raise RuntimeError(
            f"No data found in {keypoints_dir}. "
            "Run utils.generate_synthetic_dataset() first."
        )

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    print(f"[feature_eng] Dataset: X={X.shape}, y={y.shape}")
    return X, y, label_map


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Feature Engineering Demo ===")
    # Synthetic single frame
    dummy_frame = np.random.rand(N_KEYPOINTS, N_FEATURES).astype(np.float32)
    angles = compute_angles(dummy_frame)
    print(f"Joint angles for dummy frame: {dict(zip([j[0] for j in JOINT_TRIPLETS], angles.round(2)))}")

    # Windowing demo
    dummy_seq = np.random.rand(150, N_KEYPOINTS, N_FEATURES).astype(np.float32)
    windows = create_windows(dummy_seq)
    print(f"Windows from 150-frame sequence: {windows.shape}")

    # Dataset loading (requires synthetic data)
    print("\nAttempting to load dataset (run utils.py first if empty)…")
    try:
        X, y, lmap = load_dataset()
        print(f"Loaded X={X.shape}, y={y.shape}, label_map={lmap}")
    except RuntimeError as e:
        print(f"[demo] {e}")
        print("Generating synthetic data first…")
        sys.path.insert(0, str(Path(__file__).parent))
        from utils import generate_synthetic_dataset
        generate_synthetic_dataset(videos_per_class=2, frames_per_video=120)
        X, y, lmap = load_dataset()
        print(f"Loaded X={X.shape}, y={y.shape}")
