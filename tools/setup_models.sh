#!/usr/bin/env bash
# Fetch the two YOLOE checkpoints and the RelateAnything checkpoint.
#
# huggingface.co is NOT reachable from this machine (curl times out) but
# hf-mirror.com is, hence the explicit mirror base URL.  Nothing here sets
# HTTP_PROXY/HTTPS_PROXY.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="$DIR/models"
mkdir -p "$MODELS/relsgg-vits16plus"

YOLOE_BASE="https://github.com/ultralytics/assets/releases/download/v8.3.0"
for f in yoloe-11s-seg-pf.pt yoloe-11s-seg.pt; do
  [ -f "$MODELS/$f" ] || curl -L --retry 3 -o "$MODELS/$f" "$YOLOE_BASE/$f"
  echo "$f $(stat -c%s "$MODELS/$f")"
done

HF="https://hf-mirror.com/maelic/relsgg-vits16plus/resolve/main"
for f in config.json model.pth text_student.pt predicate_embeddings.npz \
         predicate_bank.npz calibration.json thresholds.json relateanything.json \
         tokenizer.json tokenizer_config.json; do
  [ -f "$MODELS/relsgg-vits16plus/$f" ] || \
    curl -L --retry 3 -o "$MODELS/relsgg-vits16plus/$f" "$HF/$f"
  echo "relsgg/$f $(stat -c%s "$MODELS/relsgg-vits16plus/$f")"
done

# relsgg with --no-deps: its pyproject pins transformers~=5.14 / numpy~=2.5 /
# Pillow~=12.3, and a plain install would downgrade a working ultralytics/torch
# environment.  The code runs fine against what is already installed.
PY="${VEASON_PYTHON:-<你的 python>}"
if ! "$PY" -c "import relsgg" 2>/dev/null; then
  "$PY" -m pip install -e "$DIR/third_party/RelateAnything" --no-deps --no-build-isolation
fi
echo "relsgg: $("$PY" -c 'import relsgg; print(relsgg.__file__)')"
