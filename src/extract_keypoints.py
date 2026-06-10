"""
extract_keypoints.py — MediaPipe BlazePose keypoint extraction from video files.

Usage (standalone):
    python src/extract_keypoints.py --video path/to/video.mp4 --out output.csv
    python src/extract_keypoints.py --folder data/raw/bharatanatyam \
                                    --out_folder data/keypoints/bharatanatyam \
                                    --label bharatanatyam
"""

import csv
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[extract_keypoints] mediapipe not installed — extraction disabled.")

MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker.task"

# ── Constants ─────────────────────────────────────────────────────────────────

TARGET_FPS = 10          # sample rate to reduce redundancy
N_KEYPOINTS = 33
N_FEATURES = 4           # x, y, z, visibility

# Build CSV column names once
_COLUMNS = ["frame"]
for _k in range(N_KEYPOINTS):
    _COLUMNS += [f"kp{_k}_x", f"kp{_k}_y", f"kp{_k}_z", f"kp{_k}_v"]


# ── Core extraction ───────────────────────────────────────────────────────────

def extract_from_video(
    video_path: str | Path,
    output_csv_path: str | Path,
    target_fps: int = TARGET_FPS,
    max_seconds: int = 0,
) -> int:
    """
    Extract MediaPipe Pose keypoints from a video and save to CSV.

    Each row in the CSV corresponds to one sampled frame and contains:
        frame, kp0_x, kp0_y, kp0_z, kp0_v, kp1_x, …, kp32_v

    Args:
        video_path:      Path to the input video (.mp4 / .avi).
        output_csv_path: Destination CSV file path.
        target_fps:      Frames per second to sample (default 10).
        max_seconds:     Stop after this many seconds of video (default 30).
                         Set to 0 to process the full video.

    Returns:
        Number of frames written to the CSV.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError:      If mediapipe is not installed.
    """
    video_path = Path(video_path)
    output_csv_path = Path(output_csv_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not _MP_AVAILABLE:
        raise RuntimeError("mediapipe is not installed. Run: pip install mediapipe")

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(round(native_fps / target_fps)))
    max_raw_frames = int(native_fps * max_seconds) if max_seconds > 0 else None

    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    written = 0
    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(_COLUMNS)

        with mp_vision.PoseLandmarker.create_from_options(opts) as detector:
            frame_idx = 0
            sampled_idx = 0

            while True:
                if max_raw_frames and frame_idx >= max_raw_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                    result = detector.detect_for_video(mp_img, timestamp_ms)

                    if result.pose_landmarks:
                        lms = result.pose_landmarks[0]
                        row = [sampled_idx]
                        for lm in lms:
                            row += [lm.x, lm.y, lm.z, lm.visibility]
                        writer.writerow(row)
                        written += 1
                        sampled_idx += 1

                frame_idx += 1

    cap.release()
    print(f"[extract] {written} frames → {output_csv_path}")
    return written


# ── Batch processing ──────────────────────────────────────────────────────────

def process_folder(
    input_folder: str | Path,
    output_folder: str | Path,
    label: str,
    target_fps: int = TARGET_FPS,
) -> list[Path]:
    """
    Batch-extract keypoints for all videos in a folder.

    Processes every .mp4 and .avi file found directly inside *input_folder*
    and writes one CSV per video to *output_folder*.

    Args:
        input_folder:  Folder containing raw dance videos.
        output_folder: Destination folder for CSV files.
        label:         Dance style label (used only for logging).
        target_fps:    Sampling rate forwarded to extract_from_video().

    Returns:
        List of output CSV paths successfully created.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        list(input_folder.glob("*.mp4")) + list(input_folder.glob("*.avi"))
    )

    if not video_files:
        print(f"[extract] No videos found in {input_folder}")
        return []

    print(f"[extract] Processing {len(video_files)} videos for label='{label}'")
    results = []
    for vid_path in video_files:
        out_csv = output_folder / (vid_path.stem + ".csv")
        try:
            extract_from_video(vid_path, out_csv, target_fps)
            results.append(out_csv)
        except Exception as exc:
            print(f"[extract] SKIP {vid_path.name}: {exc}")

    return results


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NrityaAI keypoint extractor")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Single video file path")
    group.add_argument("--folder", help="Folder of videos")

    parser.add_argument("--out", help="Output CSV path (for --video)")
    parser.add_argument("--out_folder", help="Output folder (for --folder)")
    parser.add_argument("--label", default="unknown", help="Dance style label")
    args = parser.parse_args()

    if args.video:
        out = args.out or (Path(args.video).stem + "_keypoints.csv")
        extract_from_video(args.video, out)
    else:
        out_folder = args.out_folder or "data/keypoints/" + args.label
        process_folder(args.folder, out_folder, args.label)
