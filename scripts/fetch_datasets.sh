#!/usr/bin/env bash
# Fetch the .bin + y_labels.csv files the official mlcommons/tiny runner
# expects for ic01/kws01/vww01/ad01. Defers to upstream's per-bench helpers
# where they exist; inlines deviations (CIFAR-10 wget, VWW PIL transcode,
# AD curl -L) called out per section.
#
# Required env:
#   TINY     path to your mlcommons/tiny clone
# Optional env:
#   PYTHON   python to use (default: $TINY/.venv/bin/python)
#
# Usage:
#   export TINY=/path/to/your/mlcommons-tiny-clone
#   uv venv "$TINY/.venv"
#   uv pip install --python "$TINY/.venv/bin/python" -e .
#   scripts/fetch_datasets.sh

set -euo pipefail

: "${TINY:?set TINY to your mlcommons/tiny clone, e.g. export TINY=\$HOME/Projects/tiny}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
DATASETS="$REPO/runner/tests/data"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

if [[ ! -d "$TINY/benchmark" ]]; then
    echo "error: TINY=$TINY doesn't look like a mlcommons/tiny clone" >&2
    exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "error: $PYTHON not found. Set up the venv first:" >&2
    echo "  uv venv \"\$TINY/.venv\"" >&2
    echo "  uv pip install --python \"\$TINY/.venv/bin/python\" -e ." >&2
    exit 2
fi

# --- IC01 (image_classification): see $TINY/benchmark/training/image_classification/README.md ---
mkdir -p "$DATASETS/ic01"
cd "$TINY/benchmark/training/image_classification"
[[ -d cifar-10-batches-py ]] || {
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xf cifar-10-python.tar.gz
}
"$PYTHON" perf_samples_loader.py
mv perf_samples/* "$DATASETS/ic01/"
mv y_labels.csv "$DATASETS/ic01/"

# --- KWS01 (keyword_spotting): see $TINY/benchmark/training/keyword_spotting/README.md ---
mkdir -p "$DATASETS/kws01"
cd "$TINY/benchmark/training/keyword_spotting"
"$PYTHON" make_bin_files.py \
    --bin_file_path="$DATASETS/kws01" \
    --feature_type=mfcc \
    --num_bin_files=1000 \
    --target_set=test \
    --tfl_file_name=trained_models/kws_ref_model.tflite \
    --data_dir="$HOME/data"

# --- VWW01 (visual_wake_words): upstream has no bin generator; PIL transcode ---
mkdir -p ~/.cache/vw_coco2014_96
cd ~/.cache/vw_coco2014_96
[[ -d vw_coco2014_96 ]] || {
    wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
    tar -xf vw_coco2014_96.tar.gz
}
mkdir -p "$DATASETS/vww01"
cp "$TINY/benchmark/evaluation/datasets/vww01/y_labels.csv" "$DATASETS/vww01/"
"$PYTHON" - "$DATASETS/vww01" <<'EOF'
import csv, os, sys
import numpy as np
from PIL import Image
SRC = os.path.expanduser("~/.cache/vw_coco2014_96/vw_coco2014_96")
DST = sys.argv[1]
with open(f"{DST}/y_labels.csv") as f:
    for row in csv.reader(f):
        name, _classes, label = row
        sub = "person" if label.strip() == "1" else "non_person"
        stem = os.path.splitext(name)[0]
        candidates = [f"{SRC}/{sub}/COCO_val2014_{stem}.jpg",
                      f"{SRC}/{sub}/COCO_val2014_{stem}.png"]
        src = next((p for p in candidates if os.path.exists(p)), None)
        if not src:
            print(f"missing: {name}", file=sys.stderr); continue
        img = np.array(Image.open(src).convert("RGB"))
        assert img.shape == (96, 96, 3), f"{name}: {img.shape}"
        with open(f"{DST}/{name}", "wb") as out:
            out.write(img.tobytes())
EOF

# --- AD01 (anomaly_detection): see $TINY/benchmark/training/anomaly_detection/README.md ---
# Upstream get_dataset.sh uses curl without -L; pull the eval-test zip with -L.
cd "$TINY/benchmark/training/anomaly_detection"
mkdir -p eval_data && cd eval_data
[[ -d ToyCar/test ]] || {
    curl -sL "https://zenodo.org/records/3841772/files/eval_data_test_ToyCar.zip" \
         -o eval_data_test_ToyCar.zip
    unzip -q eval_data_test_ToyCar.zip
    rm eval_data_test_ToyCar.zip
}
cd ..
"$PYTHON" convert_dataset.py -e
mkdir -p "$DATASETS/ad01"
mv eval_data/ToyCar/test/*_hist_librosa.bin "$DATASETS/ad01/"
cp "$TINY/benchmark/evaluation/datasets/ad01/y_labels.csv" "$DATASETS/ad01/"

echo ""
echo "All datasets fetched. Sanity check:"
for B in ic01 kws01 vww01; do
    COUNT=$(ls "$DATASETS/$B"/*.bin 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(stat -f%z $(ls "$DATASETS/$B"/*.bin 2>/dev/null | head -1) 2>/dev/null)
    echo "  $B: $COUNT files, $SIZE bytes"
done
echo "  (ad01: many files; runner uses the 248 listed in y_labels.csv)"
