"""
run_demo.py — End-to-end smoke test for NrityaAI.

Steps:
  1. Generate synthetic keypoint data
  2. Train CNN+LSTM for 5 epochs
  3. Evaluate on the same data (sanity check)
  4. Test pose correction module
  5. Print pass/fail summary

Run:
    python run_demo.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable
_SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(_SRC))

import numpy as np


def _header(title: str) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def step1_generate_data() -> tuple:
    _header("STEP 1 — Generate Synthetic Dataset")
    from utils import generate_synthetic_dataset
    seqs, labels = generate_synthetic_dataset(
        videos_per_class=4,
        frames_per_video=180,
        save=True,
    )
    print(f"  Sequences generated : {len(seqs)}")
    print(f"  Unique classes      : {sorted(set(labels))}")
    assert len(seqs) == 12, "Expected 12 sequences (4 × 3 classes)"
    print("  [PASS] Data generation")
    return seqs, labels


def step2_load_features() -> tuple:
    _header("STEP 2 — Feature Engineering")
    from feature_engineering import load_dataset
    X, y, label_map = load_dataset()
    print(f"  X shape      : {X.shape}")
    print(f"  y shape      : {y.shape}")
    print(f"  label_map    : {label_map}")
    assert X.ndim == 4, "Expected X with 4 dimensions (N, T, 33, 4)"
    print("  [PASS] Feature engineering")
    return X, y, label_map


def step3_train(X: np.ndarray, y: np.ndarray):
    _header("STEP 3 — Train CNN+LSTM (5 epochs, smoke test)")
    from train import build_model, train_model
    from utils import MODELS_DIR

    model_path = MODELS_DIR / "demo_model.keras"
    model, history = train_model(
        X, y,
        epochs=5,
        batch_size=8,
        val_split=0.2,
        model_path=model_path,
    )
    assert model_path.exists(), f"Model checkpoint not saved at {model_path}"
    train_acc = history.history.get("style_accuracy", [0])[-1]
    print(f"  Final train style_accuracy : {train_acc:.4f}")
    print("  [PASS] Training")
    return model, model_path


def step4_evaluate(X: np.ndarray, y: np.ndarray, model):
    _header("STEP 4 — Evaluate")
    from evaluate import evaluate
    from utils import MODELS_DIR

    results = evaluate(X, y, model=model, save_dir=MODELS_DIR)
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Macro F1  : {results['macro_f1']:.4f}")
    assert 0.0 <= results["accuracy"] <= 1.0
    print("  [PASS] Evaluation")
    return results


def step5_pose_correction():
    _header("STEP 5 — Pose Correction Module")
    from pose_correction import (
        load_reference_pose, compare_poses, overall_score, get_correction_text
    )
    from utils import CLASSES, N_KEYPOINTS

    # Use a deterministic user frame
    rng = np.random.default_rng(42)
    user_frame = rng.random((N_KEYPOINTS, 4)).astype(np.float32)

    for style in CLASSES:
        ref = load_reference_pose(style, "main_pose")
        assert ref.shape == (N_KEYPOINTS, 3), f"Bad ref shape for {style}"
        corrections = compare_poses(user_frame, ref)
        score = overall_score(corrections)
        texts = get_correction_text(corrections)
        print(f"  [{style}] score={score:.1f}  corrections={len(corrections)}")
        assert 0.0 <= score <= 100.0

    print("  [PASS] Pose correction")


def main():
    print("\n" + "🎭" * 30)
    print("  NrityaAI — End-to-End Demo")
    print("🎭" * 30)

    try:
        seqs, labels = step1_generate_data()
        X, y, label_map = step2_load_features()
        model, model_path = step3_train(X, y)
        results = step4_evaluate(X, y, model)
        step5_pose_correction()

        _header("SUMMARY")
        print("  ✅ Synthetic data generation  PASS")
        print("  ✅ Feature engineering         PASS")
        print("  ✅ CNN+LSTM training            PASS")
        print("  ✅ Evaluation pipeline         PASS")
        print("  ✅ Pose correction module      PASS")
        print()
        print("  All steps completed successfully! 🎉")
        print(f"  Demo model saved → {model_path}")
        print()
        print("  Next steps:")
        print("    1. Add real dance videos to data/raw/<class>/")
        print("    2. python src/extract_keypoints.py --folder data/raw/bharatanatyam ...")
        print("    3. python src/train.py")
        print("    4. uvicorn api.main:app --reload")
        print("    5. streamlit run app/streamlit_app.py")

    except Exception as exc:
        print(f"\n  ❌ Demo failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
