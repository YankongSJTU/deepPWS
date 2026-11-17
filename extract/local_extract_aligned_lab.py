"""Portable aligned Lab residual extraction for NewShanghai.

Requires the 3DDFA_V2 code and weights. Output npz contains `stats` with
20 Lab/residual descriptors; the server adds the 768-dim CLIP encoding of the
saved delta-a image plus a one-column availability flag.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np, pandas as pd, cv2
from PIL import Image, ImageOps


def load_landmarks(ddfa_root: Path):
    sys.path.insert(0, str(ddfa_root))
    if not hasattr(np, 'long'): np.long = np.int64
    from FaceBoxes import FaceBoxes
    from TDDFA import TDDFA
    boxes = FaceBoxes()
    model = TDDFA(arch='mobilenet', widen_factor=1., checkpoint_fp=str(ddfa_root / 'weights/mb1_120x120.pth'), bfm_fp=str(ddfa_root / 'configs/bfm_noneck_v3.pkl'), size=120, num_params=62, gpu_mode=False)
    return boxes, model


def read_image(path, size=512):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        width, height = image.size
        image = image.resize((int(width * size / max(width, height)), int(height * size / max(width, height))), Image.Resampling.BICUBIC)
        return np.asarray(image)


def landmarks(rgb, boxes, model):
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    faces = np.asarray(boxes(bgr))
    if len(faces) == 0: raise ValueError('no_face')
    score = (faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1]) * faces[:, 4]
    face = faces[int(score.argmax())]
    params, roi = model(bgr, [face])
    xyz = model.recon_vers(params, roi, dense_flag=False)[0].T
    return xyz[:, :2], float(face[4])


def moments(values):
    return np.r_[values.mean(0), values.std(0), np.quantile(values, [.1, .5, .9], axis=0).ravel()]


def extract(before_path, after_path, boxes, model):
    before_image, after_image = read_image(before_path), read_image(after_path)
    before, before_conf = landmarks(before_image, boxes, model)
    after, after_conf = landmarks(after_image, boxes, model)
    selected = np.arange(17, 68)
    matrix, inliers = cv2.estimateAffinePartial2D(after[selected], before[selected], method=cv2.RANSAC, ransacReprojThreshold=5, maxIters=2000, confidence=.99, refineIters=10)
    if matrix is None: raise ValueError('alignment_failed')
    height, width = before_image.shape[:2]
    warped = cv2.warpAffine(after_image, matrix, (width, height))
    valid = cv2.warpAffine(np.ones(after_image.shape[:2], np.uint8), matrix, (width, height), flags=cv2.INTER_NEAREST) > 0
    face = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(before.round().astype(np.int32)), 1)
    mask = face.astype(bool) & valid
    if mask.sum() < 300: raise ValueError('insufficient_overlap')
    before_lab = cv2.cvtColor(before_image.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    after_lab = cv2.cvtColor(warped.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    delta = after_lab - before_lab
    data = np.c_[delta[mask], np.linalg.norm(delta[mask], axis=1)]
    stats = moments(data)
    delta_a = np.where(mask, delta[:, :, 1], 0)
    signed = np.clip(delta_a / 40, -1, 1)
    pseudo = np.stack([.5 + .5 * signed, .5 - .25 * np.abs(signed), .5 - .5 * signed], axis=2)
    residual = cv2.transform(after[selected][None].astype(np.float32), matrix)[0] - before[selected]
    meta = {'before_face_confidence': before_conf, 'after_face_confidence': after_conf, 'affine_inlier_fraction': float(inliers.mean()), 'overlap_fraction': float(mask.mean()), 'mean_delta_a': float(delta[mask, 1].mean())}
    return stats, Image.fromarray(np.clip(pseudo * 255, 0, 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BICUBIC), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--ddfa-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    boxes, model = load_landmarks(args.ddfa_root)
    table = pd.read_csv(args.manifest); rows=[]
    for index, row in enumerate(table.itertuples(), 1):
        key = str(row.record_key).replace(':', '_'); dest = args.out / f'{key}.npz'; png = args.out / f'{key}.png'; meta_path = args.out / f'{key}.json'
        try:
            if not dest.exists():
                stats, image, meta = extract(args.data_dir / row.before, args.data_dir / row.after, boxes, model)
                image.save(png); np.savez_compressed(dest, stats=stats)
                meta.update(record_key=row.record_key, status='ok'); meta_path.write_text(json.dumps(meta, indent=2))
            rows.append({'record_key': row.record_key, 'path': str(dest), 'delta_image': str(png), 'status': 'ok'})
        except Exception as exc:
            meta_path.write_text(json.dumps({'record_key': row.record_key, 'status': 'failed', 'reason': str(exc)}, indent=2)); rows.append({'record_key': row.record_key, 'path': str(dest), 'delta_image': str(png), 'status': 'failed'})
        if index % 25 == 0: pd.DataFrame(rows).to_csv(args.out / 'manifest.csv', index=False); print(f'{index}/{len(table)}', flush=True)
    pd.DataFrame(rows).to_csv(args.out / 'manifest.csv', index=False)
    (args.out / 'provenance.json').write_text(json.dumps({'method': '3DDFA_V2 68 landmarks, after-to-before RANSAC affine, Lab delta-a residual', 'manifest': str(args.manifest), 'record_policy': 'numbered sessions remain independent records'}, indent=2))
    print('completed', len(rows), 'records')


if __name__ == '__main__': main()
