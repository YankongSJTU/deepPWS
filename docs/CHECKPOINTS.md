# Encoder checkpoints (not bundled in this repository)

The trained DeepPWS readout weights are tiny and ship inside the repo
(`weights/`, `web/models/`, `web/clinical_models/`, `web/demographic_models/`,
≈ 20 MB total). The frozen image encoders and 3D tools are large public
checkpoints that you must obtain separately and place under `web/runtime/`
(the web workstation) and/or next to the extractors in `extract/`.

## Layout the web workstation expects (`web/runtime/`)

| path | size | what it is | where to get it |
|---|---|---|---|
| `runtime/weights/clip/model.safetensors` (+ tokenizer/config) | 1.7 GB | CLIP ViT-L/14 vision tower | HuggingFace `openai/clip-vit-large-patch14` |
| `runtime/weights/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth` | 652 MB | FaRL ViT-B/16, LAION-FACE 20M epoch-16 | FaRL release (github.com/FacePerceiver/FaRL) |
| `runtime/3DDFA_V2/weights/mb1_120x120.pth` | 13 MB | 3DDFA-V2 landmark model | github.com/cleardusk/3DDFA_V2 |
| `runtime/3DDFA_V2/configs/bfm_noneck_v3.pkl` | 23 MB | BFM config for 3DDFA-V2 | github.com/cleardusk/3DDFA_V2 |
| `runtime/3DDFA_V2/weights/FaceBoxesProd.pth` | 3.9 MB | FaceBoxes detector | github.com/zisianw/FaceBoxes.PyTorch |
| `runtime/model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors` | 799 MB | LAM-20k 3D reconstruction | LAM release (registration required) |
| `runtime/model_zoo/flame_assets/flame2023.pkl` + `FLAME_masks.pkl` + `flame_arkit_bs.npy` | ~57 MB | FLAME topology assets | FLAME project (registration required) |
| `runtime/lam/` | — | LAM inference code | LAM repository |

## CPU-only 3D alternative (`web/standalone_models/`)

`web/backend/feature_worker.py` can run the 3D branch on CPU through
TorchScript exports instead of the CUDA LAM pipeline:

| path | size | produced by |
|---|---|---|
| `standalone_models/lam_encoder.pt` | 1.25 GB | `web/export_standalone_lam.py` |
| `standalone_models/lam_decoder.pt` | 0.98 GB | `web/export_standalone_lam.py` |

If neither `standalone_models/` nor the CUDA LAM runtime is present,
`launch_offline.py` exits at startup with a checklist of the missing assets
(it will not serve half-configured). Only the Ridge demo (`bash run.sh`)
runs checkpoint-free.

## Integrity

Every shipped weight file is hash-verified at load time (`parameter_sha256`
in the accompanying `model.json`), so silent substitution is impossible.
`web/runtime_source.json` records the provenance of the vendored extraction
modules.
