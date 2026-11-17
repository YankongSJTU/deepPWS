# -*- coding: utf-8 -*-
"""Generate synthetic (fake) demo photographs for the DeepPWS pipeline.

Creates paired before/after face-like images with a fading port-wine-stain
patch plus a clinical-fields JSON, so the full offline pipeline
(registration -> encoders -> fusion -> Ridge) can be exercised without any
real patient data. Deterministic under a fixed seed.

Usage:
    python generate_fake_data.py [--out .] [--patients 6] [--seed 0]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SKIN = [214, 186, 168]
BG = [235, 238, 240]

SITES = ["cheek", "cheek", "nose", "forehead", "jaw", "cheek"]
ZONES = ["V2", "V2", "V1", "V1", "V3", "V2"]
COLOURS = ["red", "purple", "red", "pink", "red", "purple"]


def face_base(size=512, rng=None):
    """Skin-toned oval head with subtle shading on a neutral background."""
    img = Image.new("RGB", (size, size), tuple(BG))
    d = ImageDraw.Draw(img)
    cx, cy = size * 0.5, size * 0.52
    rx, ry = size * 0.34, size * 0.42
    # background shadow
    d.ellipse([cx - rx - 8, cy - ry - 4, cx + rx + 8, cy + ry + 10],
              fill=tuple(int(c * 0.92) for c in BG))
    # head
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=tuple(SKIN))
    # simple shading: darker jaw sides
    shade = Image.new("L", (size, size), 0)
    ds = ImageDraw.Draw(shade)
    ds.ellipse([cx - rx * 0.9, cy, cx + rx * 0.9, cy + ry], fill=26)
    dark = Image.new("RGB", (size, size), tuple(int(c * 0.86) for c in SKIN))
    img = Image.composite(Image.blend(img, dark, 0.5), img, shade)
    # eyes + mouth hints (light) so the image reads as a face
    d = ImageDraw.Draw(img)
    for ex in (cx - rx * 0.45, cx + rx * 0.45):
        d.ellipse([ex - 14, cy - ry * 0.30 - 9, ex + 14, cy - ry * 0.30 + 9],
                  fill=(150, 132, 122))
    d.ellipse([cx - 26, cy + ry * 0.45 - 8, cx + 26, cy + ry * 0.45 + 10],
              fill=(178, 138, 130))
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    if rng is not None:  # sensor noise
        arr = np.asarray(img).astype(np.int16)
        arr += rng.integers(-6, 7, arr.shape, dtype=np.int16)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img


def add_stain(img, site, strength, rng):
    """Blush a port-wine-stain patch; strength 0..1 (1 = untreated red)."""
    size = img.size[0]
    cx, cy = size * 0.5, size * 0.52
    pos = {"cheek": (cx + size * 0.16, cy + size * 0.05),
           "nose": (cx, cy - size * 0.04),
           "forehead": (cx, cy - size * 0.24),
           "jaw": (cx - size * 0.18, cy + size * 0.22)}[site]
    r = size * (0.075 + 0.03 * strength)
    mask = Image.new("L", img.size, 0)
    dm = ImageDraw.Draw(mask)
    pts = []
    for k in range(9):  # irregular blob outline
        ang = k / 9 * 2 * np.pi
        rr = r * (0.78 + 0.3 * abs(np.sin(2 * ang + rng.uniform(0, 3))))
        pts.append((pos[0] + rr * np.cos(ang), pos[1] + rr * np.sin(ang)))
    dm.polygon(pts, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(6))
    red = max(0, 255 - int(70 * (1 - strength)))       # deeper red when strong
    stain_col = (red, int(60 + 40 * (1 - strength)), int(70 + 50 * (1 - strength)))
    layer = Image.new("RGB", img.size, stain_col)
    return Image.composite(Image.blend(img, layer, 0.55), img, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent))
    ap.add_argument("--patients", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    records = []
    for i in range(args.patients):
        pid = f"demo_p{i + 1:02d}"
        site = SITES[i % len(SITES)]
        strength0 = float(rng.uniform(0.55, 1.0))
        response = float(rng.uniform(0.25, 0.95))       # fake treatment effect
        before = add_stain(face_base(512, rng), site, strength0, rng)
        after = add_stain(face_base(512, rng), site,
                          strength0 * (1 - response), rng)
        # mild pose/lighting shift between visits
        after = after.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC,
                             center=(256, 260))
        after = after.point(lambda v: int(v * rng.uniform(0.96, 1.04)))
        before.save(out / "images" / f"{pid}_before.jpg", quality=92)
        after.save(out / "images" / f"{pid}_after.jpg", quality=92)
        records.append({
            "patient_id": pid,
            "before": f"images/{pid}_before.jpg",
            "after": f"images/{pid}_after.jpg",
            "age": int(rng.integers(4, 41)),
            "skin_type": "III" if rng.random() < 0.6 else "IV",
            "site": site,
            "trigeminal_zone": ZONES[i % len(ZONES)],
            "colour_phenotype": COLOURS[i % len(COLOURS)],
            # synthetic physician scores are ONLY for pipeline demonstration
            "color_score": round(10 * response * strength0, 1),
            "area_score": round(10 * response * (1 - 0.15 * rng.random()), 1),
        })
    (out / "demo_records.json").write_text(json.dumps(records, indent=2),
                                           encoding="utf-8")
    print(f"wrote {len(records)} fake pairs -> {out / 'images'}")
    print(f"manifest: {out / 'demo_records.json'}")


if __name__ == "__main__":
    main()
