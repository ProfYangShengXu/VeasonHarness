#!/usr/bin/env python
"""Round-3 weights: fetch the five new axes' checkpoints with ``urllib``.

Shell ``curl``/``wget`` are broken on this box (HTTP 000); Python's urllib is
not.  Everything lands under ``models/`` (git-ignored).

    python tools/fetch_round3_models.py

Downloads (skipped when the file is already the right size):

    mediapipe/face_landmarker.task                  ~3.6 MB  (Google storage)
    videomae-small-kinetics/*                       ~88  MB  (hf-mirror)
    depth-anything-v2-small/onnx/model.onnx         ~99  MB  (hf-mirror)
    depth-anything-v2-small/preprocessor_config.json
    ultralytics_weights/yolo11n-pose.pt             ~6.3 MB  (GitHub assets)

RapidOCR ships its models inside the wheel; ``pip install rapidocr-onnxruntime``
is the whole setup for the text axis.
"""
from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")

JOBS = [
    ("mediapipe/face_landmarker.task",
     "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
     "face_landmarker/float16/1/face_landmarker.task", 3758596),
    ("videomae-small-kinetics/config.json",
     "https://hf-mirror.com/MCG-NJU/videomae-small-finetuned-kinetics/"
     "resolve/main/config.json", 22864),
    ("videomae-small-kinetics/preprocessor_config.json",
     "https://hf-mirror.com/MCG-NJU/videomae-small-finetuned-kinetics/"
     "resolve/main/preprocessor_config.json", None),
    ("videomae-small-kinetics/pytorch_model.bin",
     "https://hf-mirror.com/MCG-NJU/videomae-small-finetuned-kinetics/"
     "resolve/main/pytorch_model.bin", 88197173),
    ("depth-anything-v2-small/preprocessor_config.json",
     "https://hf-mirror.com/onnx-community/depth-anything-v2-small/"
     "resolve/main/preprocessor_config.json", 461),
    ("depth-anything-v2-small/onnx/model.onnx",
     "https://hf-mirror.com/onnx-community/depth-anything-v2-small/"
     "resolve/main/onnx/model.onnx", 99060839),
    ("ultralytics_weights/yolo11n-pose.pt",
     "https://github.com/ultralytics/assets/releases/download/v8.4.0/"
     "yolo11n-pose.pt", 6255593),
]


def fetch(rel: str, url: str, expect: int | None) -> None:
    dst = os.path.join(MODELS, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and (expect is None or os.path.getsize(dst) == expect):
        print(f"have  {rel}  ({os.path.getsize(dst)} bytes)")
        return
    tmp = dst + ".part"
    print(f"get   {rel}  <- {url}")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst)
    print(f"done  {rel}  ({os.path.getsize(dst)} bytes)")


def main() -> int:
    for rel, url, expect in JOBS:
        fetch(rel, url, expect)
    missing = [rel for rel, _, expect in JOBS
               if not os.path.exists(os.path.join(MODELS, *rel.split("/")))]
    if missing:
        print("MISSING:", missing, file=sys.stderr)
        return 3
    print("all round-3 weights present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
