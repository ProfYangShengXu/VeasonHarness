"""``answer`` and ``verify``: the delivered fact and its independent check.

The contract (Veason-R1 / ReVOS output)::

    {"keyframe_timestamp": <int seconds>, "bbox_2d_list": [[x1, y1, x2, y2], ...]}

Two deliberate choices:

* ``keyframe_timestamp`` is **seconds**, computed from the frame's ``t`` with
  half-up rounding.  ``round()`` is *not* used: Python rounds halves to even,
  so ``round(2.5) == 2`` and a frame at 2.5 s would silently land a second
  early on some frames and not others.
* ``verify`` compares the **snapshot written by the last ``answer``** against
  the region table as it is *now*.  Comparing a freshly recomputed answer to
  itself would be vacuous; the snapshot makes it possible for the check to
  actually fail (rebuild or edit the region table and it does).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from .errors import VrsError
from .paths import for_agent
from .store import IndexStore, sha256_json

EXIT_VERIFY_FAILED = 4


def keyframe_timestamp(t: float) -> int:
    """Seconds since the start of the video, half-up.  Not ``round()``."""
    return int(math.floor(float(t) + 0.5))


def _check_idxs(idxs: Sequence[int]) -> list[int]:
    out = [int(i) for i in idxs]
    if not out:
        raise VrsError(
            "no region_idx given",
            hint="usage: vrs answer <index_dir> <frame_id> <region_idx> [<region_idx> ...]",
            code=2,
        )
    if len(set(out)) != len(out):
        # Duplicates are allowed in the answer, but the snapshot name must be stable.
        raise VrsError(
            f"region_idx repeated: {out}",
            hint="pass each region once; the order you pass becomes the order of bbox_2d_list.",
            code=2,
        )
    return out


def build_answer(store: IndexStore, frame_id: int, idxs: Sequence[int]) -> tuple[dict, dict]:
    """The single place where a region idx becomes delivered coordinates."""
    meta = store.require()
    frame = store.require_frame(frame_id)
    regions = [store.get_region(frame_id, i) for i in idxs]
    payload = {
        "keyframe_timestamp": keyframe_timestamp(frame["t"]),
        "bbox_2d_list": [[int(v) for v in r["bbox"]] for r in regions],
    }
    detail = {
        "frame_id": frame_id,
        "region_idxs": list(idxs),
        "t": frame["t"],
        "fps": meta.get("fps"),
        "labels": [r["label"] for r in regions],
        "regions_sha256": sha256_json({"frame_id": frame_id, "regions": store.require_regions(frame_id)["regions"]}),
    }
    return payload, detail


def answer(store: IndexStore, frame_id: int, idxs: Sequence[int]) -> dict:
    idxs = _check_idxs(idxs)
    payload, detail = build_answer(store, frame_id, idxs)
    snapshot = {**payload, **detail, "answer_schema": 1}
    path = store.save_snapshot(frame_id, idxs, snapshot)
    # stdout stays exactly the official two-key payload; the bookkeeping lives on disk.
    return payload


def _mask_within_bbox(store: IndexStore, frame_id: int, idxs: Sequence[int],
                      tolerance_px: int = 2) -> dict:
    """An *independent* fact: the stored mask must live inside the stored box.

    The box comes from the detector head and the mask from the segmentation
    branch; comparing them catches a whole class of post-processing mistakes
    that the snapshot check cannot see (a mask scaled through its letterbox
    padding, for instance, used to stick 11 px out of its own box).
    """
    import numpy as np

    masks = store.load_masks(frame_id)
    rows = []
    for i in idxs:
        m = masks.get(i)
        if m is None:
            rows.append({"region_idx": i, "has_mask": False, "match": True, "note": "no mask stored"})
            continue
        ys, xs = np.where(np.asarray(m))
        if len(xs) == 0:
            rows.append({"region_idx": i, "has_mask": False, "match": True, "note": "empty mask"})
            continue
        box = store.get_region(frame_id, i)["bbox"]
        mask_box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        outside = max(box[0] - mask_box[0], box[1] - mask_box[1],
                      mask_box[2] - box[2], mask_box[3] - box[3])
        rows.append({
            "region_idx": i, "has_mask": True,
            "region_bbox": box, "mask_bbox": mask_box,
            "outside_px": int(outside), "match": outside <= tolerance_px,
        })
    return {
        "check": "mask_within_region_bbox",
        "ok": all(r["match"] for r in rows),
        "tolerance_px": tolerance_px,
        "per_idx": rows,
    }


def verify(store: IndexStore, frame_id: int, idxs: Sequence[int]) -> tuple[dict, bool]:
    idxs = _check_idxs(idxs)
    snap = store.load_snapshot(frame_id, idxs)
    if snap is None:
        raise VrsError(
            f"no answer snapshot for frame {frame_id} regions {idxs}",
            hint=f"run `vrs answer <index_dir> {frame_id} {' '.join(str(i) for i in idxs)}` "
            "first — verify checks a delivered answer against the live region table, "
            "so there must be an answer to check.",
            code=2,
        )

    current, detail = build_answer(store, frame_id, idxs)
    checks: list[dict] = []

    checks.append({
        "check": "regions_file_unchanged_since_answer",
        "ok": snap.get("regions_sha256") == detail["regions_sha256"],
        "snapshot_sha256": snap.get("regions_sha256"),
        "current_sha256": detail["regions_sha256"],
    })
    checks.append({
        "check": "answer_reproducible_from_current_regions",
        "ok": snap.get("bbox_2d_list") == current["bbox_2d_list"],
        "per_idx": [
            {
                "region_idx": i,
                "snapshot_bbox": a,
                "regions_bbox": b,
                "match": a == b,
            }
            for i, a, b in zip(idxs, snap.get("bbox_2d_list", []), current["bbox_2d_list"])
        ],
    })
    checks.append({
        "check": "keyframe_timestamp_consistent",
        "ok": snap.get("keyframe_timestamp") == current["keyframe_timestamp"],
        "snapshot": snap.get("keyframe_timestamp"),
        "current": current["keyframe_timestamp"],
    })
    checks.append(_mask_within_bbox(store, frame_id, idxs))

    ok = all(c["ok"] for c in checks)
    payload = {
        "ok": ok,
        "frame_id": frame_id,
        "region_idxs": list(idxs),
        "snapshot": for_agent(store.snapshot_path(frame_id, idxs)),
        "checks": checks,
    }
    if not ok:
        payload["error"] = "delivered answer disagrees with the current region table"
        payload["hint"] = (
            "the region table was changed after `answer` ran (re-index, or a `detect` "
            "that reassigned boxes); re-run `vrs answer` and re-submit."
        )
    return payload, ok
