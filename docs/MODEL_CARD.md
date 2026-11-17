# DeepPWS — Model Card

## Overview

DeepPWS is an offline AI-assisted documentation system for port-wine-stain
(PWS) laser therapy. From one before/after facial photograph pair and
optional clinical fields, it estimates:

1. **Response scores** — colour fading and area reduction on the physician
   0–10 scale (overall = mean of the two), via a regularised multi-output
   Ridge over frozen multimodal features;
2. **Clinical-attribute candidates** — lesion colour type, laterality,
   trigeminal-zone category (V1/V2/V3), coarse anatomical region;
3. **Demographic estimates** — age and Fitzpatrick III/IV, for pre-filling
   blank fields only.

Every model-provided value is a *suggestion pending clinician confirmation*;
manual entries always take priority and nothing is written automatically
below 0.9 confidence.

## Architecture (response scoring)

Frozen feature blocks, concatenated and z-scored (5,013 + C dimensions):

| block | dim | source |
|---|---|---|
| CLIP ViT-L/14 global (before / after / difference) | 2,304 | 3 × 768-d projected embeddings |
| FaRL ViT-B/16 global (before / after / difference) | 1,536 | 3 × 512-d class tokens (LAION-FACE 20M, epoch 16) |
| aligned Δa* residual | 789 | 20 CIELAB statistics + 768-d CLIP encoding of the signed ±40 Δa* map + 1 availability flag |
| reconstructed 3D / UV geometry | 384 | 144 dense geometry (12 fields × 2 moments × 6 regions) + 240 UV texture descriptors (FLAME-topology LAM reconstructions) |
| clinical covariates | C | one-hot age, Fitzpatrick type, site, trigeminal zone, colour phenotype; missing → explicit category |

Readout: multi-output Ridge, min ‖Y − XB − 1bᵀ‖²_F + α‖B‖²_F, α over 25
log-spaced values selected by 3-fold grouped inner CV (~10⁴ trainable
parameters; retrains in seconds on CPU).

## Reported performance (retrospective, three centres)

| protocol | colour ρ | area ρ | overall ρ |
|---|---|---|---|
| Shanghai 774-fold patient-group LOOCV (813 records) | 0.745 | 0.740 | 0.757 (MAE 1.40) |
| direct external transfer — Chongqing (n = 101) | — | — | 0.725 |
| direct external transfer — Zhejiang (n = 79) | — | — | 0.563 |
| Zhejiang + 50-shot local adaptation | — | — | 0.621 |

Clinical-attribute recognition (pooled multicentre baseline cohort,
1,003 patients, patient-grouped 10-fold OOF, five-seed ensembles):
colour type 0.706, laterality 0.678, trigeminal zone 0.772, lesion region
0.404; per-centre colour 0.685/0.592/0.775 (SH/CQ/ZJ). Age estimation MAE
3.01 years pooled (median baseline 8.74); Fitzpatrick III/IV macro-F1
0.673 (AUC 0.808) pooled, 0.828 (AUC 0.872) in Zhejiang.

## Intended use & limitations

- Retrospective derivation; prospective validation not yet performed.
- Scores are physician-scale estimates for documentation support, **not** a
  diagnosis or a treatment-planning device.
- Softmax confidences are uncalibrated and degrade under domain shift; the
  system therefore never auto-writes below 0.9 confidence.
- Operates fully offline; no patient image or feature leaves the workstation.

## Data & privacy

Training data: retrospective paired facial photographs from Shanghai
(derivation), Chongqing and Zhejiang (external) centres; not redistributable.
This repository ships **synthetic demo images only** (see
`data/demo/generate_fake_data.py`).
