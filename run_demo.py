# -*- coding: utf-8 -*-
"""One-click CPU demo for DeepPWS (no encoders, no photographs needed).

Part 1 (inference): the shipped Shanghai source Ridge scores 20 demo records
                     shipped as pre-transformed 5,181-dim fusion vectors.
Part 2 (train demo): refit the same multi-output Ridge equation on a
                     patient-grouped holdout split (API demonstration only —
                     20 records is far too few for meaningful accuracy).

For the full image pipeline (synthetic photographs -> registration ->
frozen encoders -> fusion -> scores) see README.md, section "Web workstation"
and extract/ — they need the downloadable encoder checkpoints.

Usage:  python run_demo.py [--no-refit]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from ridge_api import fit_ridge, metrics, predict_ridge, predict_saved_model  # noqa: E402

DEMO_NPZ = ROOT / "data" / "demo" / "demo_features.npz"
MODEL_PATH = ROOT / "weights" / "shanghai_source.pkl"
REPORT_PATH = ROOT / "outputs" / "demo_report.json"
ALPHA = 100.0
TEST_GROUP_EVERY = 4


def grouped_holdout(split_group: np.ndarray):
    """Deterministic split at the patient-group level (never split a patient)."""
    groups = np.unique(split_group)
    test_groups = set(groups[::TEST_GROUP_EVERY].tolist())
    is_test = np.array([g in test_groups for g in split_group])
    return np.flatnonzero(~is_test), np.flatnonzero(is_test)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refit", action="store_true",
                    help="skip the grouped-holdout training demo")
    args = ap.parse_args()

    data = np.load(DEMO_NPZ, allow_pickle=False)
    X = data["X"].astype(np.float64)
    y = data["y"].astype(np.float64)
    groups = data["split_group"]
    print(f"demo data: X={X.shape}, y={y.shape}, "
          f"{np.unique(groups).size} patient groups, targets=[color, area]")

    # --- Part 1: inference with the shipped source model ------------------
    shipped = predict_saved_model(MODEL_PATH, X)
    shipped_metrics = metrics(y, shipped)
    print(f"\n[inference] shipped source model on all {len(X)} demo records")
    for name, value in shipped_metrics.items():
        print(f"  {name:>7}: spearman={value['spearman']:+.4f}"
              f"  mae={value['mae']:.4f}")
    good = shipped[:, 2] >= 5.0
    print(f"[report] {int(good.sum())}/{len(good)} records scored as "
          f"treatment success (overall >= 5)")

    report = {
        "demo_records": int(len(X)),
        "patient_groups": int(np.unique(groups).size),
        "shipped_model": str(MODEL_PATH.relative_to(ROOT)),
        "inference_metrics": shipped_metrics,
    }

    # --- Part 2: grouped train / test demo --------------------------------
    if not args.no_refit:
        train, test = grouped_holdout(groups)
        assert not set(groups[train]) & set(groups[test]), "group leak"
        state = fit_ridge(X[train], y[train], alpha=ALPHA)
        prediction = predict_ridge(state, X[test])
        train_metrics = metrics(y[test], prediction)
        print(f"\n[train] grouped holdout Ridge (alpha={ALPHA:g}): "
              f"{len(train)} train / {len(test)} test records")
        for name, value in train_metrics.items():
            print(f"  {name:>7}: spearman={value['spearman']:+.4f}"
                  f"  mae={value['mae']:.4f}")
        report["train_demo"] = {"alpha": ALPHA,
                                "train_records": int(len(train)),
                                "test_records": int(len(test)),
                                "metrics": train_metrics}

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
