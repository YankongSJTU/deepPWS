#!/usr/bin/env bash
# DeepPWS one-click entry points.
#
#   bash run.sh            demo + tests (CPU only, no checkpoints needed)
#   bash run.sh web        launch the offline web workstation (127.0.0.1:8875)
#   bash run.sh extract    run portable feature extractors on the fake demo
#                          images (needs encoder checkpoints, see
#                          docs/CHECKPOINTS.md)
set -euo pipefail
cd "$(dirname "$0")"

python - <<'PY'
import sys
assert sys.version_info >= (3, 10), "Python >= 3.10 required"
PY

mode="${1:-demo}"

case "$mode" in
  demo)
    echo "== [1/3] synthetic demo photographs (fake data) =="
    [ -f data/demo/demo_records.json ] || \
      python data/demo/generate_fake_data.py --out data/demo
    echo "== [2/3] CPU demo: shipped weights on demo fusion features =="
    python run_demo.py
    echo "== [3/3] smoke tests =="
    python tests/test_release.py
    echo "done. see outputs/demo_report.json"
    ;;
  web)
    echo "launching offline workstation on http://127.0.0.1:8875 ..."
    echo "(scoring inside the web app additionally needs the checkpoints in"
    echo " web/runtime/ and web/standalone_models/ — see docs/CHECKPOINTS.md)"
    python web/launch_offline.py
    ;;
  extract)
    echo "portable extraction on fake demo pairs -> outputs/demo_features_raw/"
    for s in aligned_lab farl; do
      python extract/local_extract_$s.py \
        --manifest data/demo/extract_manifest.csv \
        --data-dir data/demo --out outputs/demo_features_raw/$s \
        || exit 1
    done
    echo "3D/UV extraction requires CUDA + LAM; see extract/README.md"
    ;;
  *)
    echo "usage: bash run.sh [demo|web|extract]" >&2
    exit 2
    ;;
esac
