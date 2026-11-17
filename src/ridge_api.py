"""Small CPU-side Ridge wrapper for the published frozen fusion features.

The full image encoders are intentionally not included in this GitHub demo.
The demo data contains precomputed 5,181-dimensional fusion vectors produced by
 the corrected 2026-10-21 pipeline. This module keeps training and inference
reproducible without distributing patient photographs or multi-GB encoders.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


TARGET_NAMES = ("color", "area")


def composite(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    return np.c_[scores, scores.mean(axis=1)]


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict:
    y_true = composite(y_true) if y_true.shape[1] == 2 else y_true
    prediction = composite(prediction) if prediction.shape[1] == 2 else prediction
    out = {}
    for j, name in enumerate(("color", "area", "overall")):
        rho = spearmanr(y_true[:, j], prediction[:, j]).statistic
        out[name] = {"spearman": float(rho) if np.isfinite(rho) else 0.0,
                     "mae": float(np.abs(y_true[:, j] - prediction[:, j]).mean())}
    return out


def predict_saved_model(model_path: str | Path, X: np.ndarray) -> np.ndarray:
    """Apply the corrected source Ridge state to already transformed features."""
    with Path(model_path).open("rb") as handle:
        state = pickle.load(handle)
    linear = state["linear"]
    coefficient = np.asarray(linear["coefficient"], dtype=np.float64)
    intercept = np.asarray(linear["intercept"], dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != coefficient.shape[0]:
        raise ValueError(f"Expected X with {coefficient.shape[0]} columns, got {X.shape}")
    return composite(np.clip(X @ coefficient + intercept, 0.0, 10.0))


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float = 100.0) -> dict:
    """Fit the same centered multi-output Ridge equation used by the pipeline."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean = X.mean(axis=0)
    y_mean = y.mean(axis=0)
    centered = X - x_mean
    gram = centered.T @ centered
    penalty = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    coefficient = np.linalg.solve(gram + penalty, centered.T @ (y - y_mean))
    intercept = y_mean - x_mean @ coefficient
    return {"coefficient": coefficient, "intercept": intercept,
            "alpha": float(alpha), "x_mean": x_mean, "y_mean": y_mean}


def predict_ridge(state: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return np.clip(X @ state["coefficient"] + state["intercept"], 0.0, 10.0)


def save_ridge(path: str | Path, state: dict) -> None:
    with Path(path).open("wb") as handle:
        pickle.dump(state, handle, protocol=5)
