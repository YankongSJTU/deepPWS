# NewShanghai 本地特征提取与服务器上传说明

## 1. 服务器上的新版记录清单

服务器已经生成：

```text
/export/home/kongyan/project/faceBA/data/newshanghai/record_manifest_generated.csv
```

它包含 1244 条独立记录，记录键形如：

```text
Shanghai:new:298-1
Shanghai:new:298-2
```

个人 PC 上运行脚本时，必须使用同一份 `record_manifest_generated.csv`。建议先从服务器下载到 PC，或者复制一份完全相同的文件。

## 2. 本地脚本

脚本位于服务器项目：

```text
faceba_fusion_server_20261014/scripts/local_extract_3d_uv.py
faceba_fusion_server_20261014/scripts/local_extract_aligned_lab.py
faceba_fusion_server_20261014/scripts/local_extract_farl.py
```

同时需要对应的：

- `local_extract_3d_uv.py`：LAM 代码、LAM 20k checkpoint、FLAME 模型文件；
- `local_extract_aligned_lab.py`：3DDFA_V2 代码和 `mb1_120x120.pth`；
- `local_extract_farl.py`：服务器使用的 FaRL ViT-B/16 checkpoint；
- 三个脚本都需要新版照片目录和新版 manifest。

## 3. 3D/UV 特征

```bash
python local_extract_3d_uv.py \
  --manifest /path/to/record_manifest_generated.csv \
  --data-dir /path/to/data/newshanghai \
  --lam-root /path/to/hugenew \
  --out /path/to/newshanghai_3d \
  --uv-size 256
```

输出目录内部应包含：

```text
newshanghai_3d/
  Shanghai_new_298-1_before.npz
  Shanghai_new_298-1_after.npz
  Shanghai_new_298-2_before.npz
  Shanghai_new_298-2_after.npz
  manifest.csv
  uv_manifest.csv
  uv/*.png
  provenance.json
```

`.npz` 必须包含：

```text
vertices: N×3
colors: N×3
```

服务器上传到：

```text
/export/home/kongyan/project/faceBA/hugenew/outputs/newshanghai_3d/
```

不要直接覆盖旧的：

```text
hugenew/outputs/formal3d_full/
```

## 4. 配准颜色特征

```bash
python local_extract_aligned_lab.py \
  --manifest /path/to/record_manifest_generated.csv \
  --data-dir /path/to/data/newshanghai \
  --ddfa-root /path/to/3DDFA_V2 \
  --out /path/to/newshanghai_aligned_lab
```

输出包括：

```text
newshanghai_aligned_lab/
  Shanghai_new_298-1.npz
  Shanghai_new_298-1.png
  Shanghai_new_298-1.json
  manifest.csv
  provenance.json
```

`.npz` 当前包含：

```text
stats: 25 维 Lab/ΔE 统计量
```

服务器上传到：

```text
/export/home/kongyan/project/faceBA/hugenew/outputs/newshanghai_aligned_lab/
```

说明：最终 Fusion 输入中的配准颜色块还包括 768 维 signed Δa CLIP embedding 和 1 维 availability 标记。你上传 `*.png` 和 `stats` 后，服务器会用已经存在的 CLIP 权重对这些 224×224 配准图编码，再组合为最终 789 维块。

## 5. FaRL 特征

```bash
python local_extract_farl.py \
  --manifest /path/to/record_manifest_generated.csv \
  --data-dir /path/to/data/newshanghai \
  --checkpoint /path/to/farl_vit_b16_checkpoint.pt \
  --out /path/to/newshanghai_farl \
  --device cuda:0
```

每个 `.npz` 必须包含以下字段：

```text
before_global: 512
​​after_global: 512
before_local: 16×512
after_local: 16×512
```

服务器上传到：

```text
/export/home/kongyan/project/faceBA/hugenew/outputs/newshanghai_farl/
```

注意：服务器当前没有发现 FaRL checkpoint。必须使用与原 Fusion 实验相同的 FaRL ViT-B/16 权重；不要用普通 CLIP 权重替代，否则不能与旧模型特征空间对应。

## 6. 已完成的 CLIP 特征

新版 1244 条记录的 CLIP 特征已经在服务器完成：

```text
/export/home/kongyan/project/faceBA/hugenew/outputs/newshanghai_clip/
```

不需要再次上传或提取。其每个 `.npz` 包含：

```text
before: 768
after: 768
before_local: 16×1024
after_local: 16×1024
```

## 7. 上传前检查

在 PC 上运行：

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
m = pd.read_csv('record_manifest_generated.csv')
for folder in ['newshanghai_3d', 'newshanghai_aligned_lab', 'newshanghai_farl']:
    p = Path(folder)
    print(folder, 'exists=', p.exists(), 'files=', len(list(p.glob('*.npz'))))
print('records=', len(m), 'unique_keys=', m.record_key.nunique())
PY
```

服务器端最终要求每个目录都能按 manifest 的 `record_key` 找到对应文件，不能只按基础病人编号找文件。
