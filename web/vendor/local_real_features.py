"""Extract phase-specific point-cloud descriptors and frozen CLIP UV embeddings.

Uses local weights directly, without Transformers or network access. Geometry
descriptors are deterministic canonical spatial pooling, not learned shape codes.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors import safe_open

BASE = Path(__import__('os').environ['FACEBA_RUNTIME_ROOT'])


class ClipVision:
    def __init__(self, device):
        self.device = device
        self.cfg = json.loads((BASE / 'weights/clip/config.json').read_text())['vision_config']
        self.weights = {}
        with safe_open(str(BASE / 'weights/clip/model.safetensors'), framework='pt') as file:
            for key in file.keys():
                if key.startswith('vision_model.') or key == 'visual_projection.weight':
                    self.weights[key] = file.get_tensor(key).to(device)

    def linear(self, x, name):
        return F.linear(x, self.weights[name + '.weight'], self.weights.get(name + '.bias'))

    def norm(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self.weights[name + '.weight'], self.weights[name + '.bias'], self.cfg['layer_norm_eps'])

    @torch.inference_mode()
    def __call__(self, paths, spatial=False):
        arrays = []
        for path in paths:
            with Image.open(path) as image:
                if spatial:
                    from PIL import ImageOps
                    image = ImageOps.exif_transpose(image).convert('RGB').resize((224, 224), Image.Resampling.BICUBIC)
                    arrays.append(np.asarray(image, dtype=np.float32) / 255)
                    continue
                image = image.convert('RGB')
                width, height = image.size
                scale = 224 / min(width, height)
                image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
                left, top = (image.width - 224) // 2, (image.height - 224) // 2
                arrays.append(np.asarray(image.crop((left, top, left + 224, top + 224)), dtype=np.float32) / 255)
        x = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).to(self.device)
        mean = torch.tensor([.48145466, .4578275, .40821073], device=self.device)[None, :, None, None]
        std = torch.tensor([.26862954, .26130258, .27577711], device=self.device)[None, :, None, None]
        x = F.conv2d((x - mean) / std, self.weights['vision_model.embeddings.patch_embedding.weight'], stride=14).flatten(2).transpose(1, 2)
        token = self.weights['vision_model.embeddings.class_embedding'].expand(len(x), 1, -1)
        x = torch.cat([token, x], dim=1) + self.weights['vision_model.embeddings.position_embedding.weight']
        x = self.norm(x, 'vision_model.pre_layrnorm')
        heads = self.cfg['num_attention_heads']
        for index in range(self.cfg['num_hidden_layers']):
            prefix = f'vision_model.encoder.layers.{index}'
            h = self.norm(x, prefix + '.layer_norm1')
            q, k, v = [self.linear(h, prefix + '.self_attn.' + part + '_proj').reshape(len(x), x.shape[1], heads, -1).transpose(1, 2) for part in ('q', 'k', 'v')]
            h = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape_as(x)
            x = x + self.linear(h, prefix + '.self_attn.out_proj')
            h = self.linear(self.norm(x, prefix + '.layer_norm2'), prefix + '.mlp.fc1')
            h = h * torch.sigmoid(1.702 * h)
            x = x + self.linear(h, prefix + '.mlp.fc2')
        cls = self.linear(self.norm(x[:, 0], 'vision_model.post_layernorm'), 'visual_projection')
        if spatial:
            patches = x[:, 1:].transpose(1, 2).reshape(len(x), 1024, 16, 16)
            patches = F.avg_pool2d(patches, 4).flatten(2).transpose(1, 2)
            return cls.cpu().numpy(), patches.cpu().numpy()
        return F.normalize(cls, dim=-1).cpu().numpy()


def geometry(vertices):
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid point cloud')
    # Canonical bins retain anatomical position; no patient IDs or paired QC.
    bounds = np.array([.25, .25, .25])
    bins = np.clip(((vertices / bounds + 1) * 2).astype(int), 0, 3)
    index = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
    features = []
    for cell in range(64):
        points = vertices[index == cell]
        features.extend([len(points) / len(vertices)])
        features.extend(points.mean(0) if len(points) else np.zeros(3))
        features.extend(points.std(0) if len(points) else np.zeros(3))
    features.extend(np.quantile(vertices, [.05, .25, .5, .75, .95], axis=0).ravel())
    return np.asarray(features, np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    formal = BASE / 'outputs/formal3d_full'
    manifest = pd.read_csv(formal / 'manifest_complete.csv')
    uv = pd.read_csv(formal / 'uv_manifest.csv')
    manifest = manifest.merge(uv[['record_key', 'phase', 'uv_path', 'status']], on=['record_key', 'phase'], suffixes=('', '_uv'), validate='one_to_one')
    if not manifest.status.eq('ok').all() or not manifest.status_uv.eq('ok').all():
        raise ValueError('Failed reconstructions require explicit handling')
    if args.limit:
        manifest = manifest.head(args.limit)
    dest = BASE / 'outputs/real_phase_features'
    dest.mkdir(parents=True, exist_ok=True)
    model = ClipVision('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', model.device, 'rows', len(manifest), flush=True)
    rows = list(manifest.itertuples())
    for start in range(0, len(rows), args.batch):
        batch = rows[start:start + args.batch]
        pending = [row for row in batch if not (dest / (Path(row.path).stem + '.npz')).exists()]
        if pending:
            textures = model([formal / 'uv' / Path(row.uv_path).name for row in pending])
            for row, texture in zip(pending, textures):
                with np.load(formal / Path(row.path).name) as z:
                    shape = geometry(z['vertices'])
                np.savez_compressed(dest / (Path(row.path).stem + '.npz'), geometry=shape, uv=texture)
        if start % 100 == 0:
            print('processed', min(start + args.batch, len(rows)), '/', len(rows), flush=True)
    (dest / 'provenance.json').write_text(json.dumps({'geometry': 'canonical 4x4x4 pooling of real Gaussian vertices plus coordinate quantiles; 463 deterministic descriptors', 'uv': 'frozen CLIP ViT-L/14 visual projection, normalized 768 dimensions; reconstructed spherical UV images', 'phase_isolation': True, 'limitations': 'UV colors originate from Gaussian SH conversion and sparse inpainting, not calibrated photographic texture'}, indent=2))
    print('complete', flush=True)


if __name__ == '__main__':
    main()
