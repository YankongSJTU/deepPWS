# -*- coding: utf-8 -*-
"""Release smoke tests (CPU-only, no encoder checkpoints required).

Run from the repository root:  python tests/test_release.py
or via pytest.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ridge_api import fit_ridge, metrics, predict_ridge, predict_saved_model  # noqa: E402


def test_demo_features_layout():
    data = np.load(ROOT / "data" / "demo" / "demo_features.npz",
                   allow_pickle=False)
    assert data["X"].shape == (20, 5181)
    assert data["y"].shape == (20, 2)
    assert len(data["split_group"]) == 20


def test_shipped_model_scores_in_range():
    data = np.load(ROOT / "data" / "demo" / "demo_features.npz",
                   allow_pickle=False)
    pred = predict_saved_model(ROOT / "weights" / "shanghai_source.pkl",
                               data["X"])
    assert pred.shape == (20, 3)
    assert pred.min() >= 0.0 and pred.max() <= 10.0
    m = metrics(data["y"], pred)
    assert -1.0 <= m["overall"]["spearman"] <= 1.0


def test_ridge_refit_api():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    y = np.c_[np.abs(X[:, 0]) * 3 + 1, np.abs(X[:, 1]) * 2 + 1]
    state = fit_ridge(X[:30], y[:30], alpha=1.0)
    pred = predict_ridge(state, X[30:])
    assert pred.shape == (10, 2)          # colour, area (metrics adds overall)
    assert pred.min() >= 0.0 and pred.max() <= 10.0
    m = metrics(y[30:], pred)
    assert np.isfinite(m["color"]["mae"]) and m["color"]["mae"] < 3.0


def test_web_weights_checksums():
    from inference import FusionModel
    model = FusionModel(ROOT / "web" / "models")
    data = np.load(ROOT / "data" / "demo" / "demo_features.npz",
                   allow_pickle=False)
    x = data["X"][0]
    features = {"before": x[:768], "after": x[768:1536],
                "aligned_lab": x[2304:2304 + 2709]}
    clinical = {"age": 20, "skin_type": "III", "lesion_site": "cheek",
                "nerve_zone": "V2", "phenotype": "red"}
    result = model.predict(features, clinical)
    scores = result["scores"]
    assert set(scores) == {"colour", "area", "overall"}
    assert all(0.0 <= v <= 10.0 for v in scores.values())
    assert abs(scores["overall"]
               - (scores["colour"] + scores["area"]) / 2) < 1e-9


def test_fake_data_generator_deterministic():
    gen = ROOT / "data" / "demo" / "generate_fake_data.py"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(gen), "--out", tmp,
                        "--patients", "2", "--seed", "0"], check=True,
                       capture_output=True)
        manifest = json.loads(
            (Path(tmp) / "demo_records.json").read_text(encoding="utf-8"))
        assert len(manifest) == 2
        assert {"patient_id", "before", "after", "color_score",
                "area_score"} <= set(manifest[0])
        assert (Path(tmp) / "images" / "demo_p01_before.jpg").exists()
        assert (Path(tmp) / "images" / "demo_p01_after.jpg").exists()


def test_pooled_clinical_bundle():
    from clinical_pooled_inference import PooledClinical
    model = PooledClinical(ROOT / "weights" / "clinical_pooled")
    rng = np.random.default_rng(0)
    out = model.predict(rng.normal(size=768).astype(np.float32),
                        rng.normal(size=512).astype(np.float32),
                        rng.normal(size=(16, 768)).astype(np.float32))
    assert set(out["colour"]["probabilities"]) == {"粉红", "红色", "紫红"}
    assert abs(sum(out["colour"]["probabilities"].values()) - 1) < 1e-4
    assert set(out["nerve"]["probabilities"]) == {"V1", "V2", "V3"}
    assert 1 <= out["age_years"] <= 61
    assert abs(sum(out["fitzpatrick"]["probabilities"].values()) - 1) < 1e-6
    assert out["model_version"].startswith("2026.11.17")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if fails else 0)
