#!/usr/bin/env bash
# One-time, offline-friendly setup for `veason detect` only.
#
# YOLOE's text-prompt path needs two extra pieces that `veason index/regions/seg/
# answer` never touch:
#   1. the `clip` module -- ultralytics wants its own fork, and its auto-install
#      shells out to git.exe, which cannot reach github on this machine (the
#      Windows system proxy at 127.0.0.1:7890 is only read by Python/pip).
#      So: download the tarball through Python's urllib, then pip install it.
#   2. `mobileclip_blt.ts` (572 MB) -- the MobileCLIP text tower YOLOE scores
#      box tokens against; it must sit in models/ultralytics_weights/.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${VEASON_PYTHON:-<你的 python>}"
mkdir -p "$DIR/models/ultralytics_weights"

if ! "$PY" -c "import clip" 2>/dev/null; then
  BUILD="$DIR/build_tmp"
  mkdir -p "$BUILD"
  "$PY" - "$BUILD" <<'PY'
import os, sys, tarfile, urllib.request
build = sys.argv[1]
tgz = os.path.join(build, "clip.tar.gz")
url = "https://codeload.github.com/ultralytics/CLIP/tar.gz/refs/heads/main"
print("downloading", url)
with urllib.request.urlopen(url, timeout=120) as r, open(tgz, "wb") as f:
    while True:
        b = r.read(1 << 20)
        if not b:
            break
        f.write(b)
with tarfile.open(tgz) as tf:
    tf.extractall(build)
print("extracted")
PY
  "$PY" -m pip install "$BUILD/CLIP-main"
fi
echo "clip: $("$PY" -c 'import clip; print(clip.__file__)')"

TS="$DIR/models/ultralytics_weights/mobileclip_blt.ts"
if [ ! -f "$TS" ]; then
  curl -L --retry 3 -o "$TS" \
    "https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts"
fi
echo "mobileclip_blt.ts $(stat -c%s "$TS")"
