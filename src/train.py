"""
train.py — CNN+LSTM model definition and training for NrityaAI.

Usage (standalone):
    python src/train.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    CLASSES, WINDOW_SIZE, N_KEYPOINTS, N_FEATURES,
    MODELS_DIR, KEYPOINTS_DIR,
)
from feature_engineering import load_dataset

# ── Model definition ──────────────────────────────────────────────────────────

def build_model(
    window_size: int = WINDOW_SIZE,
    n_keypoints: int = N_KEYPOINTS,
    n_features: int = N_FEATURES,
    n_classes: int = len(CLASSES),
) -> "tf.keras.Model":
    """
    Build the dual-output CNN+LSTM model.

    Architecture:
        Input (window_size, n_keypoints, n_features)
        → TimeDistributed Conv1D(64, relu)
        → TimeDistributed Conv1D(128, relu)
        → TimeDistributed GlobalAveragePooling1D
        → LSTM(256, return_sequences=True) + Dropout(0.3)
        → LSTM(128) + Dropout(0.3)
        → style  : Dense(n_classes, softmax)   [classification]
        → score  : Dense(1, sigmoid)           [pose quality 0-1]

    Args:
        window_size:  Temporal window length (frames).
        n_keypoints:  Number of pose keypoints (default 33).
        n_features:   Features per keypoint (x, y, z, visibility = 4).
        n_classes:    Number of dance style classes.

    Returns:
        Compiled tf.keras.Model with two outputs: 'style' and 'score'.
    """
    import keras
    from keras import layers, Model

    inp = layers.Input(shape=(window_size, n_keypoints, n_features), name="keypoints")

    # TimeDistributed CNN — treat each frame as a 1-D "channel" sequence
    x = layers.TimeDistributed(
        layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
        name="td_conv1",
    )(inp)
    x = layers.TimeDistributed(
        layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
        name="td_conv2",
    )(x)
    x = layers.TimeDistributed(
        layers.GlobalAveragePooling1D(), name="td_gap"
    )(x)                                         # (batch, window_size, 128)

    # LSTM stack
    x = layers.LSTM(256, return_sequences=True, name="lstm1")(x)
    x = layers.Dropout(0.3, name="drop1")(x)
    x = layers.LSTM(128, name="lstm2")(x)
    x = layers.Dropout(0.3, name="drop2")(x)

    # Dual outputs
    style_out = layers.Dense(n_classes, activation="softmax", name="style")(x)
    score_out = layers.Dense(1, activation="sigmoid", name="score")(x)

    model = Model(inputs=inp, outputs=[style_out, score_out])
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss={"style": "categorical_crossentropy", "score": "mse"},
        loss_weights={"style": 1.0, "score": 0.5},
        metrics={"style": "accuracy"},
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    val_split: float = 0.2,
    model_path: Path | None = None,
) -> tuple:
    """
    Train the CNN+LSTM model and save the best checkpoint.

    Dummy pose-quality scores (0.5 constant) are used when ground-truth
    scores are unavailable — replace with real scores if you have them.

    Args:
        X:          Input array of shape (N, WINDOW_SIZE, 33, 4).
        y:          Integer label array of shape (N,).
        epochs:     Maximum training epochs.
        batch_size: Mini-batch size.
        val_split:  Fraction of data reserved for validation.
        model_path: Where to save the best model (.h5). Defaults to
                    models/best_model.h5.

    Returns:
        (model, history) — trained Keras model and History object.
    """
    import keras

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if model_path is None:
        model_path = MODELS_DIR / "best_model.keras"

    n_classes = len(CLASSES)
    y_cat = keras.utils.to_categorical(y, num_classes=n_classes)

    # Dummy pose quality scores — replace with real labels if available
    pose_scores = np.full((len(X), 1), 0.5, dtype=np.float32)

    # Balance classes by oversampling minorities to match the largest class
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    X_parts, y_parts = [], []
    for cls, cnt in zip(classes, counts):
        idx = np.where(y == cls)[0]
        if cnt < max_count:
            idx = np.random.choice(idx, size=max_count, replace=True)
        X_parts.append(X[idx])
        y_parts.append(y[idx])
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    y_cat = keras.utils.to_categorical(y, num_classes=n_classes)
    pose_scores = np.full((len(X), 1), 0.5, dtype=np.float32)
    print(f"[train] Balanced dataset: {len(X)} samples ({max_count} per class)")

    # Augment: horizontal flip + gaussian noise → 3× dataset size
    X_flip = X.copy()
    X_flip[:, :, :, 0] = 1.0 - X_flip[:, :, :, 0]   # mirror x-coords
    X_noise = np.clip(X + np.random.normal(0, 0.01, X.shape).astype(np.float32), -1.0, 1.0)
    X = np.concatenate([X, X_flip, X_noise], axis=0)
    y_cat = np.concatenate([y_cat, y_cat, y_cat], axis=0)
    pose_scores = np.full((len(X), 1), 0.5, dtype=np.float32)
    perm = np.random.permutation(len(X))
    X, y_cat, pose_scores = X[perm], y_cat[perm], pose_scores[perm]
    print(f"[train] After augmentation: {len(X)} samples")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_style_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_style_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X,
        {"style": y_cat, "score": pose_scores},
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    _plot_history(history)
    print(f"[train] Model saved → {model_path}")
    return model, history


def _plot_history(history) -> None:
    """Save training/validation accuracy and loss curves to models/."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Style accuracy
    axes[0].plot(history.history.get("style_accuracy", []), label="train_acc")
    axes[0].plot(history.history.get("val_style_accuracy", []), label="val_acc")
    axes[0].set_title("Style Classification Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Total loss
    axes[1].plot(history.history.get("loss", []), label="train_loss")
    axes[1].plot(history.history.get("val_loss", []), label="val_loss")
    axes[1].set_title("Total Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    out_path = MODELS_DIR / "training_history.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"[train] Training curves saved → {out_path}")


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== NrityaAI — Training ===")

    # Try to load existing data; fall back to synthetic
    try:
        X, y, _ = load_dataset()
    except RuntimeError:
        print("No dataset found — generating synthetic data…")
        from utils import generate_synthetic_dataset
        generate_synthetic_dataset(videos_per_class=5, frames_per_video=200)
        X, y, _ = load_dataset()

    print(f"Training on X={X.shape}, y={y.shape}")
    model, hist = train_model(X, y, epochs=100)
    print("Training complete.")
