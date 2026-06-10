"""
generate_reference_poses.py — Extract reference pose JSONs from labeled images.

Bharatanatyam: uses the 9 labeled "* Augmented" folders in data/reference_poses/
Kathak:        uses top-5 highest-visibility images from data/raw/kaggle_images/kathak/
Odissi:        uses top-4 highest-visibility images from data/raw/kaggle_images/odissi/

Output: data/reference_poses/<style>/<pose_name>.json

Usage:
    python scripts/generate_reference_poses.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_poses"
KAGGLE_DIR    = PROJECT_ROOT / "data" / "raw" / "kaggle_images"
MODEL_PATH    = PROJECT_ROOT / "models" / "pose_landmarker.task"

# ── Pose folder → JSON name mapping (bharatanatyam) ───────────────────────────

BHARATANATYAM_POSES = {
    "Ardhamandalam Augmented": "ardhamandalam",
    "Bramha Augmented":        "bramha",
    "Garuda Augmented":        "garuda",
    "Muzhumandi Augmented":    "muzhumandi",
    "Nagabandham Augmented":   "nagabandham",
    "Nataraj Augmented":       "nataraj",
    "Prenkhana Augmented":     "prenkhana",
    "Samapadam Augmented":     "samapadam",
    "Swastika Augmented":      "swastika",
}

# Kathak and Odissi pose names (assigned to top-visibility images in order)
KATHAK_POSES = ["thaat", "aamad", "tatkar", "chakkar", "namaskar"]
ODISSI_POSES = ["tribhanga", "chowk", "samabhanga", "abhanga"]

# ── Joint angle definitions ───────────────────────────────────────────────────

JOINT_TRIPLETS = [
    ("left_elbow",  11, 13, 15),
    ("right_elbow", 12, 14, 16),
    ("left_knee",   23, 25, 27),
    ("right_knee",  24, 26, 28),
    ("left_hip",    11, 23, 25),
    ("right_hip",   12, 24, 26),
]


# ── MediaPipe helpers ─────────────────────────────────────────────────────────

def _make_detector():
    """Create a MediaPipe PoseLandmarker (Tasks API)."""
    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def _run_mediapipe(image_path: Path, detector) -> tuple:
    """Run MediaPipe Pose on a single image. Returns (landmarks, avg_visibility) or (None, 0)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0.0
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    if not result.pose_landmarks:
        return None, 0.0
    lms = result.pose_landmarks[0]   # first (only) pose
    avg_vis = float(np.mean([lm.visibility for lm in lms]))
    return lms, avg_vis


def _angle_between(a, b, c) -> float:
    """Angle in degrees at vertex b."""
    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _build_json(pose_name: str, style: str, source_image: str, landmarks) -> dict:
    """Build the reference pose JSON dict from MediaPipe landmarks."""
    keypoints = {}
    for i, lm in enumerate(landmarks):
        keypoints[str(i)] = {
            "x": round(lm.x, 4),
            "y": round(lm.y, 4),
            "z": round(lm.z, 4),
            "visibility": round(lm.visibility, 4),
        }

    angles = {}
    for joint_name, ia, ib, ic in JOINT_TRIPLETS:
        angles[joint_name] = round(
            _angle_between(landmarks[ia], landmarks[ib], landmarks[ic]), 2
        )

    return {
        "pose_name": pose_name,
        "style": style,
        "source_image": source_image,
        "keypoints": keypoints,
        "reference_angles": angles,
    }


def _best_image(image_paths: list[Path], detector) -> tuple:
    """Return the image with the highest average landmark visibility."""
    best_path, best_lms, best_vis = None, None, -1.0
    for img_path in image_paths:
        lms, vis = _run_mediapipe(img_path, detector)
        if lms and vis > best_vis:
            best_vis, best_lms, best_path = vis, lms, img_path
    return best_path, best_lms


def _top_n_images(image_paths: list[Path], n: int, detector) -> list[tuple]:
    """Return the n images with highest average visibility, as (path, landmarks) pairs."""
    scored = []
    for img_path in image_paths:
        lms, vis = _run_mediapipe(img_path, detector)
        if lms:
            scored.append((vis, img_path, lms))
    scored.sort(reverse=True)
    return [(p, lms) for _, p, lms in scored[:n]]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not _MP_AVAILABLE:
        print("ERROR: mediapipe not installed. Run: pip install mediapipe")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Download it with:")
        print("  curl -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task -o models/pose_landmarker.task")
        sys.exit(1)

    detector = _make_detector()
    total_saved = 0

    # ── Bharatanatyam: labeled folders ────────────────────────────────────────
    print("\n── Bharatanatyam ──────────────────────────────────────────────")
    out_dir = REFERENCE_DIR / "bharatanatyam"
    out_dir.mkdir(parents=True, exist_ok=True)

    for folder_name, pose_name in BHARATANATYAM_POSES.items():
        folder = REFERENCE_DIR / folder_name
        if not folder.exists():
            print(f"  SKIP {folder_name} — folder not found")
            continue

        images = sorted(list(folder.glob("*.jpg")) + list(folder.glob("*.png")))
        if not images:
            print(f"  SKIP {folder_name} — no images")
            continue

        best_path, best_lms = _best_image(images, detector)
        if best_lms is None:
            print(f"  SKIP {pose_name} — MediaPipe detected no pose in any image")
            continue

        data = _build_json(pose_name, "bharatanatyam", best_path.name, best_lms)
        out_path = out_dir / f"{pose_name}.json"
        out_path.write_text(json.dumps(data, indent=2))
        print(f"  ✓ {pose_name}.json  (from {best_path.name}, angles: {data['reference_angles']})")
        total_saved += 1

    # ── Kathak: top-5 from kaggle images ──────────────────────────────────────
    print("\n── Kathak ─────────────────────────────────────────────────────")
    out_dir = REFERENCE_DIR / "kathak"
    out_dir.mkdir(parents=True, exist_ok=True)

    kathak_images = sorted(
        list((KAGGLE_DIR / "kathak").glob("*.jpg")) +
        list((KAGGLE_DIR / "kathak").glob("*.png"))
    )

    top_kathak = _top_n_images(kathak_images, len(KATHAK_POSES), detector)
    if not top_kathak:
        print("  SKIP — no poses detected in kathak kaggle images")
    else:
        for (img_path, lms), pose_name in zip(top_kathak, KATHAK_POSES):
            data = _build_json(pose_name, "kathak", img_path.name, lms)
            out_path = out_dir / f"{pose_name}.json"
            out_path.write_text(json.dumps(data, indent=2))
            print(f"  ✓ {pose_name}.json  (from {img_path.name}, angles: {data['reference_angles']})")
            total_saved += 1

    # ── Odissi: top-4 from kaggle images ──────────────────────────────────────
    print("\n── Odissi ─────────────────────────────────────────────────────")
    out_dir = REFERENCE_DIR / "odissi"
    out_dir.mkdir(parents=True, exist_ok=True)

    odissi_images = sorted(
        list((KAGGLE_DIR / "odissi").glob("*.jpg")) +
        list((KAGGLE_DIR / "odissi").glob("*.png"))
    )

    top_odissi = _top_n_images(odissi_images, len(ODISSI_POSES), detector)
    if not top_odissi:
        print("  SKIP — no poses detected in odissi kaggle images")
    else:
        for (img_path, lms), pose_name in zip(top_odissi, ODISSI_POSES):
            data = _build_json(pose_name, "odissi", img_path.name, lms)
            out_path = out_dir / f"{pose_name}.json"
            out_path.write_text(json.dumps(data, indent=2))
            print(f"  ✓ {pose_name}.json  (from {img_path.name}, angles: {data['reference_angles']})")
            total_saved += 1

    detector.close()

    print(f"\n✅ Done — {total_saved}/18 reference pose JSONs saved to data/reference_poses/")


if __name__ == "__main__":
    main()
