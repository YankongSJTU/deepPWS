# -*- coding: utf-8 -*-
"""Pooled-cohort clinical-attribute & demographic inference (NumPy only).

Loads the serving bundle under weights/clinical_pooled/ (manifest.json +
per-member npz) exported by clinical_boost_20261113/export_pooled_models.py
and reproduces the torch forward pass without torch:

    from clinical_pooled_inference import PooledClinical
    model = PooledClinical("weights/clinical_pooled")
    out = model.predict(clip_before_768, farl_before_512, farl_local_16x768)
    out["colour"]   # {'probabilities': {label: p}, 'prediction': label}
    out["nerve"]    # {'probabilities': {V1: p, ...}, 'prediction': [V1, V2]}
    out["age_years"], out["fitzpatrick"]  # demographic estimates

Protocol: pooled multicentre baseline cohort (1,003 patients), frozen
before-image encoders; out-of-fold metrics in the manuscript.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _linear(x, w, prefix):
    return x @ w[f"{prefix}.weight"].T + w[f"{prefix}.bias"]


def _norm(x, w, prefix):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return ((x - mean) / np.sqrt(var + 1e-5) * w[f"{prefix}.weight"]
            + w[f"{prefix}.bias"])


def _gelu(x):
    from scipy.special import erf
    return (0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))).astype(np.float32)


class PooledClinical:
    def __init__(self, root):
        self.root = Path(root)
        self.meta = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8"))
        self.prep = {}
        for spec_list in self.meta["members"].values():
            for spec in spec_list:
                if spec["pca_dim"] not in self.prep:
                    self.prep[spec["pca_dim"]] = dict(
                        np.load(self.root / f"prep_pca{spec['pca_dim']}.npz"))
        self.members = {}                      # (variant, seed) -> weights
        for task, spec_list in self.meta["members"].items():
            for spec in spec_list:
                for seed in spec["seeds"]:
                    key = (spec["variant"], seed)
                    if key not in self.members:
                        self.members[key] = dict(np.load(
                            self.root / f"{spec['variant']}_s{seed}.npz"))
        self.member_map = {t: [s for s in self.meta["members"][t]]
                           for t in self.meta["members"]}
        age = dict(np.load(self.root / "age_ridge.npz"))
        self.age = age
        fitz = dict(np.load(self.root / "fitz_logistic.npz"))
        self.fitz = fitz
        self.label_maps = self.meta["label_maps"]

    # ------------------------------------------------------------------ fwd
    def _features(self, pca_dim, clip_g, farl_g, local):
        p = self.prep[pca_dim]
        g = np.concatenate([np.atleast_2d(clip_g),
                            np.atleast_2d(farl_g)], 1).astype(np.float32)
        gb = (g - p["global_mean"]) / p["global_std"]
        pc = (local.reshape(1, -1) - p["pca_mean"]) @ p["pca_components"].T
        return np.c_[gb, (pc - p["pc_mean"]) / p["pc_std"]].astype(np.float32)

    def _single_head(self, w, task, x):
        p = f"heads.{task}"
        h = _gelu(_norm(_linear(x, w, f"{p}.0"), w, f"{p}.1"))
        return _linear(h, w, f"{p}.4")

    def _joint_head(self, w, task, x):
        h = _gelu(_norm(_linear(x, w, "trunk.0"), w, "trunk.1"))
        return _linear(h, w, f"heads.{task}")

    def predict(self, clip_before, farl_before, farl_local):
        out = {}
        for task, spec_list in self.member_map.items():
            probs = []
            for spec in spec_list:
                x = self._features(spec["pca_dim"], clip_before,
                                   farl_before, farl_local)
                for seed in spec["seeds"]:
                    w = self.members[(spec["variant"], seed)]
                    logits = (self._single_head(w, task, x)
                              if spec["family"] == "single"
                              else self._joint_head(w, task, x))
                    probs.append(np.exp(logits) / np.exp(logits).sum()
                                 if task in ("colour", "side")
                                 else 1.0 / (1.0 + np.exp(-logits)))
            p = np.mean(probs, 0)[0]
            labels = self.label_maps[task]
            if task in ("colour", "side"):
                out[task] = {"probabilities": dict(zip(labels, map(float, p))),
                             "prediction": labels[int(p.argmax())]}
            else:
                thr = self.meta["threshold"]
                out[task] = {"probabilities": dict(zip(labels, map(float, p))),
                             "prediction": [labels[i] for i, v in
                                            enumerate(p) if v >= thr]}
        g = np.concatenate([np.atleast_2d(clip_before),
                            np.atleast_2d(farl_before)], 1).astype(np.float32)
        gs = (g - self.age["mean"]) / self.age["std"]
        out["age_years"] = float(np.clip(gs @ self.age["coefficient"]
                                         + self.age["intercept"],
                                         *self.age["age_range"])[0])
        z = (g @ self.fitz["coefficient"].T + self.fitz["intercept"])[0, 0]
        p_iv = 1.0 / (1.0 + np.exp(-z))
        out["fitzpatrick"] = {
            "probabilities": {"III": float(1.0 - p_iv), "IV": float(p_iv)},
            "prediction": "IV" if p_iv >= 0.5 else "III"}
        out["model_version"] = self.meta["version"]
        return out
