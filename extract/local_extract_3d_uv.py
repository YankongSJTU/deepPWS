"""Portable LAM 3D reconstruction + UV baking for NewShanghai records.

Run on the PC that has the LAM code, checkpoints, and CUDA environment.
The output is directly uploadable to the server feature directory.
"""
from pathlib import Path
import argparse, json
import numpy as np, pandas as pd
from PIL import Image
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file


def load_lam(lam_root: Path):
    import sys
    sys.path.insert(0, str(lam_root))
    from lam.models.modeling_lam import ModelLAM
    cfg = OmegaConf.load(lam_root / 'configs/inference/lam-20k-8gpu.yaml')
    cfg.model.human_model_path = str(lam_root / 'model_zoo/human_parametric_models')
    model = ModelLAM(**cfg.model).cuda().eval()
    ckpt = lam_root / 'model_zoo/lam_models/releases/lam/lam-20k/step_045500/model.safetensors'
    checkpoint = load_file(str(ckpt), device='cpu')
    state = model.state_dict()
    for key, value in checkpoint.items():
        if key in state and state[key].shape == value.shape:
            state[key].copy_(value)
    return model


def reconstruct(model, path: Path):
    image = np.asarray(Image.open(path).convert('RGB').resize((518, 518)), np.float32) / 255
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).cuda()
    flame = {'betas': torch.zeros((1, 300), device='cuda')}
    with torch.inference_mode():
        feats = model.forward_encode_image(tensor)
        query, _ = model.renderer.get_query_points(flame, device=torch.device('cuda'))
        latent = model.forward_transformer(feats, camera_embeddings=None, query_points=query, query_feats=None)
        gaussians, query, _, _ = model.renderer.forward_gs(latent, query, flame, additional_features={})
        gaussian = gaussians[0]
        vertices = gaussian.xyz.detach().cpu().numpy()
        colors = gaussian.shs.detach().cpu().numpy()
        colors = colors[:, 0, :] if colors.ndim == 3 else colors[:, :3]
        colors = 1 / (1 + np.exp(-colors))
    return vertices, colors


def bake_uv(source: Path, destination: Path, size=256):
    import cv2
    z = np.load(source)
    xyz = z['vertices'].astype(np.float32)
    rgb = np.clip(z['colors'].astype(np.float32), 0, 1)
    radius = np.linalg.norm(xyz, axis=1) + 1e-8
    u = np.arctan2(xyz[:, 1], xyz[:, 0]) / (2 * np.pi) + .5
    v = .5 - np.arcsin(np.clip(xyz[:, 2] / radius, -1, 1)) / np.pi
    px = np.clip((u * (size - 1)).round().astype(int), 0, size - 1)
    py = np.clip((v * (size - 1)).round().astype(int), 0, size - 1)
    image = np.zeros((size, size, 3), np.float32)
    weight = np.zeros((size, size), np.float32)
    for x, y, color in zip(px, py, rgb):
        image[y, x] += color
        weight[y, x] += 1
    mask = weight > 0
    image[mask] /= weight[mask, None]
    if mask.any():
        image = np.where(mask[..., None], image, cv2.inpaint((image * 255).astype(np.uint8), 255 - mask.astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255)
    Image.fromarray(np.clip(image * 255, 0, 255).astype(np.uint8)).save(destination)
    return float(mask.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--lam-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--uv-size', type=int, default=256)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('LAM extraction requires CUDA')
    args.out.mkdir(parents=True, exist_ok=True)
    uv_dir = args.out / 'uv'; uv_dir.mkdir(exist_ok=True)
    model = load_lam(args.lam_root)
    table = pd.read_csv(args.manifest)
    rows = []
    for index, row in enumerate(table.itertuples(), 1):
        key = str(row.record_key).replace(':', '_')
        for phase in ('before', 'after'):
            npz = args.out / f'{key}_{phase}.npz'
            try:
                if not npz.exists():
                    vertices, colors = reconstruct(model, args.data_dir / getattr(row, phase))
                    np.savez_compressed(npz, vertices=vertices, colors=colors)
                uv = uv_dir / f'{npz.stem}.png'
                coverage = bake_uv(npz, uv, args.uv_size) if not uv.exists() else np.nan
                rows.append({'record_key': row.record_key, 'phase': phase, 'path': str(npz), 'uv_path': str(uv), 'coverage': coverage, 'completion_fraction': 1 - coverage if np.isfinite(coverage) else np.nan, 'status': 'ok'})
            except Exception as exc:
                rows.append({'record_key': row.record_key, 'phase': phase, 'path': str(npz), 'uv_path': str(uv_dir / f'{npz.stem}.png'), 'coverage': 0, 'completion_fraction': 1, 'status': f'error:{type(exc).__name__}:{exc}'})
        if index % 25 == 0:
            pd.DataFrame(rows).to_csv(args.out / 'manifest.csv', index=False)
            pd.DataFrame(rows).to_csv(args.out / 'uv_manifest.csv', index=False)
            print(f'{index}/{len(table)}', flush=True)
    pd.DataFrame(rows).to_csv(args.out / 'manifest.csv', index=False)
    pd.DataFrame(rows).to_csv(args.out / 'uv_manifest.csv', index=False)
    (args.out / 'provenance.json').write_text(json.dumps({'encoder': 'LAM 20k', 'record_manifest': str(args.manifest), 'uv_size': args.uv_size, 'record_policy': 'numbered sessions remain independent records'}, indent=2))
    print('completed', len(rows), 'phase files')


if __name__ == '__main__':
    main()
