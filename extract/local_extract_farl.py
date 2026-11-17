"""Portable FaRL ViT-B/16 global/local feature extraction.

Use a local FaRL checkpoint compatible with src/farl_encoder.py. Output one
npz per record with the exact fields consumed by farl_ridge.load_farl:
before_global, after_global, before_local, after_local.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np, pandas as pd
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from farl_encoder import FaRLVision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    import torch
    device = args.device if torch.cuda.is_available() else 'cpu'
    args.out.mkdir(parents=True, exist_ok=True)
    encoder = FaRLVision(args.checkpoint, device=device)
    table = pd.read_csv(args.manifest); rows=[]
    for index, row in enumerate(table.itertuples(), 1):
        key = str(row.record_key).replace(':', '_'); dest = args.out / f'{key}.npz'
        if not dest.exists():
            before, before_local = encoder([Image.open(args.data_dir / row.before)])
            after, after_local = encoder([Image.open(args.data_dir / row.after)])
            np.savez_compressed(dest, before_global=before[0], after_global=after[0], before_local=before_local[0], after_local=after_local[0])
        rows.append({'record_key': row.record_key, 'path': str(dest), 'status': 'ok'})
        if index % 25 == 0: pd.DataFrame(rows).to_csv(args.out / 'manifest.csv', index=False); print(f'{index}/{len(table)}', flush=True)
    pd.DataFrame(rows).to_csv(args.out / 'manifest.csv', index=False)
    (args.out / 'protocol.json').write_text(json.dumps({'encoder': 'FaRL ViT-B/16', 'checkpoint': str(args.checkpoint), 'checkpoint_sha256': encoder.sha256, 'global_dim': 512, 'local_dim': [16, 512], 'record_policy': 'numbered sessions remain independent records'}, indent=2))
    print('completed', len(rows), 'records')


if __name__ == '__main__': main()
