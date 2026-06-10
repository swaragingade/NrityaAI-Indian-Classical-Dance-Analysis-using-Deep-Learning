"""
evaluate.py — Model evaluation: classification report, confusion matrix, F1.

Usage (standalone):
    python src/evaluate.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from utils import CLASSES, MODELS_DIR, load_model_once
from feature_engineering import load_dataset

# sklearn is a soft dependency — caught gracefully below
try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
    )
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    print("[evaluate] scikit-learn not installed — metrics unavailable.")


# ── Core evaluation ───────────────────────────────────────────────────────────

def evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model=None,
    model_path: Path | None = None,
    save_dir: Path | None = None,
) -> dict:
    """
    Run inference on *X*, compute classification metrics, save plots and text.

    Args:
        X:          Input array (N, WINDOW_SIZE, 33, 4).
        y:          Ground-truth integer labels (N,).
        model:      Pre-loaded Keras model.  Loaded from disk if None.
        model_path: Path to .keras or .h5 file (used only when model is None).
        save_dir:   Directory for output files.  Defaults to models/.

    Returns:
        Dict with keys: accuracy, macro_f1, report_str, predictions.

    Raises:
        RuntimeError: If model cannot be loaded or sklearn is missing.
    """
    if not _SKLEARN:
        raise RuntimeError("scikit-learn is required for evaluation.")

    if save_dir is None:
        save_dir = MODELS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load model if not provided
    if model is None:
        model = load_model_once(model_path)
    if model is None:
        raise RuntimeError(
            "Model not loaded. Train first with: python src/train.py"
        )

    # Predict
    preds = model.predict(X, verbose=0)
    # preds is a list [style_probs, score_probs] for dual-output model
    if isinstance(preds, (list, tuple)):
        style_probs = preds[0]
    else:
        style_probs = preds

    y_pred = np.argmax(style_probs, axis=1)

    # Metrics
    report = classification_report(y, y_pred, target_names=CLASSES, digits=4)
    macro_f1 = f1_score(y, y_pred, average="macro")
    accuracy = np.mean(y == y_pred)

    print("\n=== Classification Report ===")
    print(report)
    print(f"Macro F1 : {macro_f1:.4f}")
    print(f"Accuracy : {accuracy:.4f}")

    # Save text report
    results_path = save_dir / "eval_results.txt"
    with open(results_path, "w") as f:
        f.write("=== NrityaAI Evaluation Results ===\n\n")
        f.write(report)
        f.write(f"\nMacro F1  : {macro_f1:.4f}\n")
        f.write(f"Accuracy  : {accuracy:.4f}\n")
    print(f"[evaluate] Results saved → {results_path}")

    # Confusion matrix plot
    _save_confusion_matrix(y, y_pred, save_dir)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "report_str": report,
        "predictions": y_pred,
        "probabilities": style_probs,
    }


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path,
) -> None:
    """Render and save a seaborn confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.title("Confusion Matrix — NrityaAI")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    out_path = save_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"[evaluate] Confusion matrix → {out_path}")


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== NrityaAI — Evaluation ===")

    # Load or generate data
    try:
        X, y, _ = load_dataset()
    except RuntimeError:
        print("Generating synthetic data…")
        from utils import generate_synthetic_dataset
        generate_synthetic_dataset(videos_per_class=3, frames_per_video=150)
        X, y, _ = load_dataset()

    # Evaluate
    try:
        results = evaluate(X, y)
        print(f"\nAccuracy : {results['accuracy']:.4f}")
        print(f"Macro F1 : {results['macro_f1']:.4f}")
    except RuntimeError as e:
        print(f"[evaluate] {e}")
