# DeepPWS — offline AI-assisted documentation for PWS laser therapy

DeepPWS scores port-wine-stain (PWS) laser-treatment response from one
before/after facial photograph pair plus optional clinical fields, and
proposes clinical attributes and demographic estimates — **fully offline**,
with every model output treated as a suggestion pending physician
confirmation. This repository packages the complete system of the manuscript
*(DeepPWS: frozen multimodal fusion with a regularised Ridge readout)*:
training code, testing code, portable feature extractors, the deployed web
workstation, small trained weights, and synthetic (fake) demo data — no
patient data is included.

## Results (retrospective, three centres)

| protocol | colour ρ | area ρ | overall ρ |
|---|---|---|---|
| Shanghai 774-fold patient-group LOOCV (813 records) | 0.745 | 0.740 | **0.757** (MAE 1.40) |
| direct transfer — Chongqing (n=101) | | | 0.725 |
| direct transfer — Zhejiang (n=79) | | | 0.563 |
| Zhejiang + 50-shot adaptation | | | 0.621 |

The readout is a multi-output Ridge (~10⁴ parameters) over **frozen**
features: CLIP ViT-L/14 (2,304-d) + FaRL ViT-B/16 (1,536-d) + aligned Δa*
residual (789-d) + 3D/UV geometry (384-d) + one-hot clinical covariates →
5,013 + C dimensions. See `docs/MODEL_CARD.md`.

## Quick start

```bash
pip install -r requirements.txt
bash run.sh            # demo + smoke tests, CPU-only, no checkpoints needed
```

`run.sh` (1) generates the synthetic demo photographs if absent, (2) scores
20 de-identified demo records with the shipped Shanghai Ridge
(`outputs/demo_report.json`), (3) runs the test suite. To exercise the
training API on the demo split: `python run_demo.py`.

## Repository layout

```
run_demo.py            one-click CPU demo (shipped weights + optional refit)
run.sh                 demo | web | extract entry points
weights/               small trained weights (hash-verified where noted)
  shanghai_source.pkl          final 5,181-feature Ridge (training-format pickle)
  fusion_source.npz + model.json   serving-format NumPy weights (same model)
  clinical_pooled/             LATEST clinical/demographic heads: pooled
                               multicentre baseline cohort (1,003 patients),
                               5-seed variant ensembles + age ridge +
                               Fitzpatrick logistic, NumPy-only serving
  clinical_weights/            earlier derivation-protocol heads (webapp)
  demographic_weights/         earlier derivation-protocol demographic heads
src/                   clean inference library
  ridge_api.py         train/score the Ridge equation (CPU, no torch)
  inference.py         exact NumPy serving path (sha256-verified weights)
  clinical_pooled_inference.py   pooled-cohort attribute + demographic heads
  clinical_inference.py / demographic_inference.py   (earlier protocol)
train/                 paper pipeline (verbatim research code)
  fusion_loocv.py      patient-group LOOCV + source refit + external transfer
  ridge_aligned_loocv.py  previous aligned-CLIP baseline
  fusion_model.py, aligned_lab_features.py, farl_encoder.py, ...
extract/               portable CLI feature extractors (see extract/README.md)
web/                   the offline web workstation (FastAPI + HTML/JS frontend)
data/demo/             synthetic photographs + demo fusion vectors + manifests
tests/                 release smoke tests
docs/                  MODEL_CARD.md, CHECKPOINTS.md
```

## Clinical-attribute & demographic estimation (pooled cohort)

The latest heads (matching the manuscript) are trained on the **pooled
multicentre baseline cohort** — one first-visit photograph + clinical form
per patient, all centres, no follow-up pairing required (1,003 patients) —
evaluated by patient-grouped 10-fold cross-validation with five-seed
ensembles:

| task | pooled OOF | Shanghai | Chongqing | Zhejiang |
|---|---|---|---|---|
| colour type (macro-F1) | 0.706 | 0.685 | 0.592 | 0.775 |
| laterality | 0.678 | 0.692 | 0.572 | 0.430 |
| trigeminal zone | 0.772 | 0.777 | 0.687 | 0.800 |
| lesion region | 0.404 | 0.405 | 0.217 | 0.333 |
| age (MAE years; median baseline) | 3.01 (8.7) | 2.96 (8.8) | 3.21 (7.9) | 3.28 (9.3) |
| Fitzpatrick III/IV | 0.673 (AUC 0.81) | 0.652 (0.78) | III only | 0.828 (0.87) |

NumPy-only serving (no torch needed once features are extracted):

```python
from clinical_pooled_inference import PooledClinical
model = PooledClinical("weights/clinical_pooled")
out = model.predict(clip_before_768, farl_before_512, farl_local_16x768)
out["colour"]["prediction"]      # 粉红 / 红色 / 紫红
out["nerve"]["prediction"]       # e.g. ["V2"]
out["age_years"], out["fitzpatrick"]["prediction"]
```

The frozen encoders that produce these inputs are the same checkpoints as
the scoring pipeline (docs/CHECKPOINTS.md). The web workstation under
`web/` still bundles the earlier derivation-protocol heads; the pooled
heads are served through the API above.

## The web workstation (`web/`)

FastAPI + Uvicorn backend, static HTML/JS/CSS frontend, SQLite case storage,
single-thread inference pools, offline guard — the deployed hospital
workstation. Launch:

```bash
pip install -r web/requirements-web.txt        # + web/requirements-inference.txt
python web/launch_offline.py                   # http://127.0.0.1:8875
```

Image scoring inside the workstation additionally needs the frozen encoder
checkpoints under `web/runtime/` (CLIP, FaRL, 3DDFA-V2, LAM/FLAME, ≈6.7 GB,
or the 2.2 GB CPU-only TorchScript pair) — exact layout and download sources
in `docs/CHECKPOINTS.md`. Without them the launcher prints a precise
asset checklist and exits; the CPU demo above is unaffected.

## Scoring your own images

1. Obtain the checkpoints (`docs/CHECKPOINTS.md`) and run the extractors:

   ```bash
   bash run.sh extract     # uses the synthetic demo pairs by default
   ```

2. Or programmatically, with precomputed feature blocks:

   ```python
   from inference import FusionModel
   model = FusionModel("weights")            # fusion_source.npz + model.json
   result = model.predict(
       {"before": clip_before_768, "after": clip_after_768,
        "aligned_lab": block_2709},          # Δa*789 + geometry144 + UV240 + FaRL1536
       {"age": 20, "skin_type": "III", "lesion_site": "...",
        "nerve_zone": "V2", "phenotype": "..."})
   result["scores"]   # {'colour': .., 'area': .., 'overall': ..} on 0–10
   ```

3. Attribute/demographic side estimates: `src/clinical_inference.py`,
   `src/demographic_inference.py` (image-only, pre-fill suggestions).

## Training on your own cohort (`train/`)

`train/fusion_loocv.py` reproduces the paper protocol end-to-end: patient
-grouped LOOCV with 3-fold grouped inner α selection, full-source refit and
matched 50-record external transfer over 10 seeds; it writes per-fold
pickles, the source model (`shanghai_source.pkl`) and a protocol digest. It
expects the research-data layout resolved through the `FACEBA_PROJECT_ROOT`
environment variable (`outputs/records.csv`, `data/{center}.score.xls`,
`outputs/raw_features/*.npz`); prepare it with your own de-identified data —
no such data ships here. `web/prepare_bundle.py` converts a refit pickle
into the serving-format `fusion_source.npz` + `model.json`.

## Ethics & safety

Retrospective derivation only; not a medical device and not for diagnostic
or treatment-planning use. Uncalibrated confidences degrade under domain
shift, so the product never auto-writes below 0.9 confidence and manual
entries always override model suggestions. Demo images in `data/demo/` are
synthetic (`generate_fake_data.py`).

## Citation & license

MIT license (see `LICENSE`). If you use DeepPWS, cite the DeepPWS
manuscript. Model details: `docs/MODEL_CARD.md`.
