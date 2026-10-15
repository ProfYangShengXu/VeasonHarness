"""Image artifacts: the full frame, a crop, and the segmentation overlay.

``seg`` is the deliverable picture.  It never invents a mask: when the region
carries one it draws a translucent fill plus contour and reports
``mask_source: "yoloe-seg"``; when it does not, it falls back to a box overlay
and *says so* with ``mask_source: "box-only"``.  A silent fallback that still
looks like a segmentation is the failure mode this field exists to prevent.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .errors import VrsError
from .paths import for_agent, for_windows
from .store import IndexStore

#: BGR, high contrast against typical street/room scenes.
_PALETTE = [
    (60, 76, 231), (74, 195, 139), (255, 144, 30), (182, 89, 155),
    (46, 204, 230), (200, 200, 60), (120, 60, 220), (30, 130, 255),
]


def _color(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


def _artifact(store: IndexStore, name: str, img: np.ndarray, bbox: list[int] | None,
              out: Path | None = None) -> dict:
    out = Path(out) if out is not None else store.output_path(name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), img):
        raise VrsError(
            f"could not write image to {out}",
            hint="check the index directory is writable and the drive has space.",
        )
    payload = {"out": for_agent(out), "out_win": for_windows(out)}
    if bbox is not None:
        payload["bbox"] = bbox
    return payload


def show(store: IndexStore, frame_id: int, out: Path | None = None) -> dict:
    img = cv2.imread(str(store.frame_image(frame_id)))
    if img is None:
        raise VrsError(
            f"frame image unreadable: {store.frame_image(frame_id)}",
            hint="re-run `vrs index`; the stored PNG is corrupt.",
        )
    return _artifact(store, f"show_f{frame_id:06d}.png", img, None, out)


def crop(store: IndexStore, frame_id: int, region_idx: int, pad: int = 8,
         out: Path | None = None) -> dict:
    img = cv2.imread(str(store.frame_image(frame_id)))
    region = store.get_region(frame_id, region_idx)
    x1, y1, x2, y2 = region["bbox"]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        raise VrsError(
            f"region {region_idx} crops to nothing on frame {frame_id}",
            hint="the stored bbox is degenerate; re-run `vrs index`.",
        )
    payload = _artifact(store, f"crop_f{frame_id:06d}_r{region_idx}.png", patch,
                        list(region["bbox"]), out)
    payload["region_idx"] = region_idx
    payload["label"] = region["label"]
    return payload


def seg(store: IndexStore, frame_id: int, region_idx: int, out: Path | None = None) -> dict:
    img = cv2.imread(str(store.frame_image(frame_id)))
    if img is None:
        raise VrsError(
            f"frame image unreadable: {store.frame_image(frame_id)}",
            hint="re-run `vrs index`; the stored PNG is corrupt.",
        )
    region = store.get_region(frame_id, region_idx)
    bbox = list(region["bbox"])
    color = _color(region_idx)
    masks = store.load_masks(frame_id)
    mask = masks.get(region_idx)

    if mask is not None and bool(np.asarray(mask).any()):
        m = np.asarray(mask).astype(bool)
        if m.shape[:2] != img.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        tint = img.copy()
        tint[m] = color
        canvas = cv2.addWeighted(img, 0.55, tint, 0.45, 0)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, color, 2, cv2.LINE_AA)
        mask_source = "yoloe-seg"
    else:
        canvas = img.copy()
        cv2.rectangle(canvas, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2, cv2.LINE_AA)
        mask_source = "box-only"

    label = f"#{region_idx} {region['label']}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ty = max(th + 6, bbox[1] - 4)
    cv2.rectangle(canvas, (bbox[0], ty - th - 6), (bbox[0] + tw + 8, ty + 2), color, -1)
    cv2.putText(canvas, label, (bbox[0] + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)

    payload = _artifact(store, f"seg_f{frame_id:06d}_r{region_idx}.png", canvas, bbox, out)
    payload["mask_source"] = mask_source
    payload["region_idx"] = region_idx
    payload["label"] = region["label"]
    return payload
