"""YOLOE detectors: a prompt-free one for indexing, a promptable one for queries.

The index must not know what is being searched for (that would be fitting the
vocabulary to the benchmark expression), so indexing uses the **prompt-free**
checkpoint, which carries the 4,585-class LVIS vocabulary in its weights.
``vrs detect`` is the only place a caller-supplied phrase reaches a model, and
it uses the *promptable* checkpoint with ``set_classes``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from . import paths
from .errors import VrsError

#: Deterministic, so re-indexing the same video yields the same boxes.
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640


def _clip_box(box: Sequence[float], w: int, h: int) -> list[int] | None:
    x1, y1, x2, y2 = (float(v) for v in box)
    x1, y1 = max(0.0, min(x1, w - 1)), max(0.0, min(y1, h - 1))
    x2, y2 = max(0.0, min(x2, w - 1)), max(0.0, min(y2, h - 1))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def _label(names, cls_index: int) -> str:
    """Ultralytics exposes ``names`` as a dict on released checkpoints and as a
    list on freshly built ones; both must resolve to a human-readable label."""
    if isinstance(names, dict):
        return str(names.get(int(cls_index), int(cls_index)))
    if 0 <= int(cls_index) < len(names):
        return str(names[int(cls_index)])
    return str(int(cls_index))


def _result_to_candidates(result, names, frame_shape, source: str) -> tuple[list[dict], list[np.ndarray | None]]:
    """Ultralytics ``Results`` -> region dicts (+ parallel mask list).

    Masks come back in the network's letterbox space and are cropped back to
    the frame with ``scale_masks``; a region whose mask is empty degrades to
    ``has_mask: false`` rather than silently drawing nothing.
    """
    h, w = frame_shape[:2]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)

    masks = None
    if getattr(result, "masks", None) is not None and result.masks is not None:
        from ultralytics.utils.ops import scale_masks

        m = result.masks.data
        if m.ndim == 3 and m.shape[0] == len(boxes):
            # `masks.data` lives in the letterboxed *network* space (384x640 for a
            # 768x432 frame at imgsz=640). Resizing it straight to (h, w) smears
            # the 12px padding into the picture and shifts the mask off its box;
            # scale_masks crops the padding first.
            m = scale_masks(m[:, None], (h, w))[:, 0]
            masks = (m.numpy() > 0.5)

    candidates: list[dict] = []
    mask_list: list[np.ndarray | None] = []
    for i in range(len(boxes)):
        bbox = _clip_box(xyxy[i], w, h)
        if bbox is None:
            continue
        label = _label(names, int(cls[i]))
        mask = masks[i] if masks is not None else None
        if mask is not None and not mask.any():
            mask = None
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        candidates.append(
            {
                "label": str(label),
                "score": round(float(conf[i]), 4),
                "bbox": bbox,
                "area_ratio": round(area / float(w * h), 6),
                "has_mask": bool(mask is not None),
                "source": source,
            }
        )
        mask_list.append(mask)
    return candidates, mask_list


def _set_ultralytics_weights_dir() -> None:
    """Point ultralytics' weight cache at our own folder.

    ``WEIGHTS_DIR`` is a module-level ``Path`` copied at import time, so
    rebinding the attribute (not the setting) is what actually moves the
    lookup for ``mobileclip_blt.ts``.
    """
    try:
        from ultralytics.utils import SETTINGS
        import ultralytics.utils as u
        import ultralytics.utils.downloads as d
        import ultralytics.nn.text_model as t
    except Exception:  # pragma: no cover - ultralytics always present here
        return
    target = str(paths.ULTRALYTICS_WEIGHTS_DIR) + "\\"
    SETTINGS["weights_dir"] = target
    for mod in (u, d, t):
        if hasattr(mod, "WEIGHTS_DIR"):
            mod.WEIGHTS_DIR = Path(target)


def _require(path: Path, what: str, hint: str) -> None:
    if not path.is_file():
        raise VrsError(f"{what} not found: {path}", hint=hint)


class PromptFreeDetector:
    """``YOLOE-11S-seg-pf``: 4,585 built-in labels, no vocabulary input."""

    def __init__(self, weights: Path | None = None, conf: float = DEFAULT_CONF) -> None:
        self.weights = Path(weights or paths.DEFAULT_YOLOE_PF)
        self.conf = conf
        _require(
            self.weights,
            "prompt-free YOLOE weights",
            "download it (see README 'Models'): "
            "curl -L -o models/yoloe-11s-seg-pf.pt "
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt",
        )
        from ultralytics import YOLOE

        self.model = YOLOE(str(self.weights))
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> tuple[list[dict], list[np.ndarray | None]]:
        res = self.model.predict(frame, conf=self.conf, imgsz=DEFAULT_IMGSZ, verbose=False)[0]
        return _result_to_candidates(res, self.names, frame.shape, "yoloe-pf-lvis4585")


class PromptedDetector:
    """``YOLOE-11S-seg`` + caller phrases (open vocabulary at inference time)."""

    def __init__(self, weights: Path | None = None, conf: float = DEFAULT_CONF) -> None:
        self.weights = Path(weights or paths.DEFAULT_YOLOE)
        self.conf = conf
        _require(
            self.weights,
            "promptable YOLOE weights",
            "download it (see README 'Models'): "
            "curl -L -o models/yoloe-11s-seg.pt "
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt",
        )
        _set_ultralytics_weights_dir()
        missing = []
        try:
            import clip  # noqa: F401
        except ImportError:
            missing.append("clip")
        if not (paths.ULTRALYTICS_WEIGHTS_DIR / "mobileclip_blt.ts").is_file():
            missing.append("mobileclip_blt.ts")
        if missing:
            raise VrsError(
                "`vrs detect` needs the YOLOE text encoder, which is not installed "
                f"(missing: {', '.join(missing)})",
                hint="run `bash tools/setup_detect.sh` once (offline install of the "
                "ultralytics CLIP fork + the 572MB MobileCLIP text tower); "
                "`vrs index/regions/seg/answer` do not need it.",
            )
        from ultralytics import YOLOE

        self.model = YOLOE(str(self.weights))

    def detect(self, frame: np.ndarray, phrases: Iterable[str]) -> tuple[list[dict], list[np.ndarray | None]]:
        phrases = [p.strip() for p in phrases if p and p.strip()]
        if not phrases:
            raise VrsError(
                "no phrases given to `detect`",
                hint='usage: vrs detect <index_dir> <frame_id> "red dress" "a person".',
                code=2,
            )
        # set_classes re-parameterises the head; the text encoder runs once per call.
        self.model.set_classes(phrases, self.model.get_text_pe(phrases))
        res = self.model.predict(frame, conf=self.conf, imgsz=DEFAULT_IMGSZ, verbose=False)[0]
        return _result_to_candidates(res, {i: p for i, p in enumerate(phrases)}, frame.shape,
                                     "yoloe-prompted")
