"""
pose_correction.py — Deviation scoring and feedback generation for NrityaAI.

Each reference pose JSON has the form:
    {"keypoints": [{"x": float, "y": float, "z": float}, ...]}  (33 entries)

Usage (standalone):
    python src/pose_correction.py
"""

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import CLASSES, REFERENCE_DIR, N_KEYPOINTS
from feature_engineering import angle_between, JOINT_TRIPLETS

# ── Suggestion templates ──────────────────────────────────────────────────────

_SUGGESTIONS: dict[str, dict[str, str]] = {
    "left_elbow": {
        "too_straight": "Bend your left elbow slightly to match the reference pose.",
        "too_bent": "Extend your left elbow a little more.",
    },
    "right_elbow": {
        "too_straight": "Bend your right elbow slightly to match the reference pose.",
        "too_bent": "Extend your right elbow a little more.",
    },
    "left_knee": {
        "too_straight": "Bend your left knee more for this pose.",
        "too_bent": "Straighten your left knee slightly.",
    },
    "right_knee": {
        "too_straight": "Bend your right knee more for this pose.",
        "too_bent": "Straighten your right knee slightly.",
    },
    "left_hip": {
        "too_straight": "Open your left hip wider.",
        "too_bent": "Bring your left hip slightly inward.",
    },
    "right_hip": {
        "too_straight": "Open your right hip wider.",
        "too_bent": "Bring your right hip slightly inward.",
    },
}

_DEVIATION_THRESHOLD = 15.0   # degrees


# ── Reference pose I/O ────────────────────────────────────────────────────────

def load_reference_pose(style: str, pose_name: str = "main_pose") -> np.ndarray:
    """
    Load a reference pose JSON and return a (33, 3) XYZ array.

    Args:
        style:     One of 'bharatanatyam', 'kathak', 'odissi'.
        pose_name: Filename stem of the reference pose JSON.

    Returns:
        Float32 array of shape (N_KEYPOINTS, 3).

    Raises:
        FileNotFoundError: If the JSON file does not exist.
    """
    path = REFERENCE_DIR / style / f"{pose_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Reference pose not found: {path}")

    with open(path) as f:
        data = json.load(f)

    kps = data["keypoints"]
    if isinstance(kps, dict):
        kps = [kps[str(i)] for i in range(len(kps))]
    arr = np.array([[kp["x"], kp["y"], kp["z"]] for kp in kps], dtype=np.float32)
    return arr


def list_reference_poses(style: str) -> list[str]:
    """
    Return available pose names for a given style.

    Args:
        style: Dance style name.

    Returns:
        List of pose name stems.
    """
    style_dir = REFERENCE_DIR / style
    if not style_dir.exists():
        return []
    return [p.stem for p in style_dir.glob("*.json")]


# ── Comparison logic ──────────────────────────────────────────────────────────

def compare_poses(
    user_keypoints_frame: np.ndarray,
    reference_keypoints_frame: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """
    Compare a user's pose frame against a reference pose and return
    joint-level corrections for joints deviating more than 15°.

    Args:
        user_keypoints_frame:      Shape (33, ≥3) — user's current frame.
        reference_keypoints_frame: Shape (33, 3)  — ideal reference pose.

    Returns:
        Dict keyed by joint name.  Each value contains:
            {
                "deviation_degrees": float,
                "suggestion": str,
            }
        Only joints with deviation > _DEVIATION_THRESHOLD are included.
    """
    user_xyz = np.array(user_keypoints_frame)[:, :3]
    ref_xyz = np.array(reference_keypoints_frame)[:, :3]

    corrections: dict[str, dict[str, Any]] = {}

    for joint_name, ia, ib, ic in JOINT_TRIPLETS:
        user_angle = angle_between(user_xyz[ia], user_xyz[ib], user_xyz[ic])
        ref_angle = angle_between(ref_xyz[ia], ref_xyz[ib], ref_xyz[ic])
        deviation = abs(user_angle - ref_angle)

        if deviation > _DEVIATION_THRESHOLD:
            diff_sign = "too_straight" if user_angle > ref_angle else "too_bent"
            suggestion = _SUGGESTIONS.get(joint_name, {}).get(
                diff_sign, f"Adjust your {joint_name.replace('_', ' ')}."
            )
            corrections[joint_name] = {
                "deviation_degrees": round(deviation, 2),
                "suggestion": suggestion,
            }

    return corrections


def overall_score(corrections: dict[str, dict[str, Any]]) -> float:
    """
    Convert joint corrections into an overall quality score 0–100.

    Each joint contributes an equal share; deviating joints reduce the
    score proportionally to their deviation (capped at 90°).

    Args:
        corrections: Output of compare_poses().

    Returns:
        Float in [0.0, 100.0].
    """
    n_joints = len(JOINT_TRIPLETS)
    if not corrections:
        return 100.0

    penalty = 0.0
    for info in corrections.values():
        dev = min(info["deviation_degrees"], 90.0)
        penalty += dev / 90.0

    score = max(0.0, 100.0 * (1.0 - penalty / n_joints))
    return round(score, 2)


def get_correction_text(corrections: dict[str, dict[str, Any]]) -> list[str]:
    """
    Convert corrections dict into a human-readable list of suggestion strings.

    Args:
        corrections: Output of compare_poses().

    Returns:
        List of strings, one per correctable joint.
        Returns ["Great form! Keep it up."] if no corrections needed.
    """
    if not corrections:
        return ["Great form! Keep it up."]

    lines = []
    for joint, info in corrections.items():
        joint_label = joint.replace("_", " ").title()
        lines.append(
            f"{joint_label}: {info['suggestion']} "
            f"(deviation: {info['deviation_degrees']}°)"
        )
    return lines


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Pose Correction Demo ===")
    from utils import generate_synthetic_dataset, N_KEYPOINTS

    # Ensure reference poses exist
    generate_synthetic_dataset(videos_per_class=1, frames_per_video=60, save=False)

    # Build a dummy "user" frame (33 × 4)
    user_frame = np.random.rand(N_KEYPOINTS, 4).astype(np.float32)

    for style in CLASSES:
        try:
            ref = load_reference_pose(style, "main_pose")
            corrections = compare_poses(user_frame, ref)
            score = overall_score(corrections)
            text = get_correction_text(corrections)
            print(f"\n[{style}] score={score}")
            for t in text:
                print(f"  • {t}")
        except FileNotFoundError as e:
            print(f"  {e}")
