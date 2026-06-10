"""
extract_keypoints_images.py — MediaPipe keypoint extraction from static images.

Each image produces one row in the output CSV. Images are sorted and treated
as a sequence so the feature engineering pipeline can create windows from them.

Usage:
    python src/extract_keypoints_images.py \
        --folder data/raw/kaggle_images/odissi \
        --out_folder data/keypoints/odissi \
        --label odissi
"""

import csv
import argparse
from pathlib import Path

import cv2

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False
    print("[extract_keypoints_images] mediapipe not installed.")

MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker.task"

N_KEYPOINTS = 33
_COLUMNS = ["frame"]
for _k in range(N_KEYPOINTS):
    _COLUMNS += [f"kp{_k}_x", f"kp{_k}_y", f"kp{_k}_z", f"kp{_k}_v"]


def extract_from_images(
    image_folder: str | Path,
    output_csv_path: str | Path,
) -> int:
    """
    Extract MediaPipe Pose keypoints from all images in a folder.
    Saves one row per image to a single CSV file.

    Returns number of rows written.
    """
    image_folder = Path(image_folder)
    output_csv_path = Path(output_csv_path)

    if not _MP_AVAILABLE:
        raise RuntimeError("mediapipe not installed. Run: pip install mediapipe")

    images = sorted(
        list(image_folder.glob("*.jpg")) +
        list(image_folder.glob("*.jpeg")) +
        list(image_folder.glob("*.png"))
    )
    if not images:
        print(f"[extract_images] No images found in {image_folder}")
        return 0

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
    )

    written = 0
    with open(output_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(_COLUMNS)

        with mp_vision.PoseLandmarker.create_from_options(opts) as detector:
            for frame_idx, img_path in enumerate(images):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)

                if result.pose_landmarks:
                    lms = result.pose_landmarks[0]
                    row = [frame_idx]
                    for lm in lms:
                        row += [lm.x, lm.y, lm.z, lm.visibility]
                    writer.writerow(row)
                    written += 1

    print(f"[extract_images] {written}/{len(images)} images → {output_csv_path}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NrityaAI image keypoint extractor")
    parser.add_argument("--folder", required=True, help="Folder of images")
    parser.add_argument("--out_folder", required=True, help="Output folder for CSV")
    parser.add_argument("--label", default="unknown", help="Dance style label")
    args = parser.parse_args()

    out_csv = Path(args.out_folder) / f"{args.label}_images.csv"
    extract_from_images(args.folder, out_csv)
