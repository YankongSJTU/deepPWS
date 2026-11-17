# Portable per-record feature extractors

These three standalone scripts extract DeepPWS feature blocks from paired
before/after photographs on a machine with the encoder checkpoints. They are
the same code paths the web workstation runs, packaged as CLI batch tools:

| script | output block | needs |
|---|---|---|
| `local_extract_aligned_lab.py` | 20 CIELAB residual statistics + signed Δa* map (+CLIP re-encode in the full pipeline) | 3DDFA-V2 weights, FaceBoxes |
| `local_extract_farl.py` | FaRL 512-d globals (before/after) + 16×768 local tokens | FaRL ep16 checkpoint, torch (GPU or CPU) |
| `local_extract_3d_uv.py` | FLAME-topology meshes → 144 geometry + 240 UV descriptors | CUDA + LAM-20k + FLAME assets |

Common interface:

```bash
python local_extract_<block>.py --manifest <csv> --data-dir <root> --out <dir>
# manifest columns: record_key, before, after   (image paths relative to --data-dir)
```

One `npz` per record plus `manifest.csv` and `provenance.json`; reruns skip
existing outputs. Checkpoint layout and download sources: `docs/CHECKPOINTS.md`.
Chinese companion guide: `LOCAL_FEATURE_EXTRACTION_README.md`.
