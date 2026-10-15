"""On-disk index: the single source of truth for regions and answers.

Layout (all paths under one ``index_dir``)::

    index.json                 meta: video hash, probe, frame table, timings
    frames/f000123.png         extracted frame image
    regions/f000123.json       candidate region table for that frame
    masks/f000123.npz          bool masks keyed by region idx (optional)
    relations/f000123.json     last relation graph computed for that frame
    answers/f000123_r0-2.json  snapshot written by ``answer`` (see answers.py)
    outputs/                   show / crop / seg artifacts

Two invariants live here and nowhere else:

1. **A region index is permanent.**  Region ``idx`` is its position in
   ``regions[]``; the list is append-only, so ``regions``/``seg``/``answer``
   always mean the same box for the same idx.
2. **A bbox is written once.**  ``regions`` stores integer pixel boxes, and
   ``answer``/``verify``/``seg`` read *that* array rather than recomputing it,
   so the delivered coordinates cannot drift from the inspected ones.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .errors import VrsError
from .paths import for_agent, for_windows

SCHEMA = 1

#: Relation model name recorded on disk.  Kept here (not imported from
#: ``indexer``) so ``store`` stays the lower layer with no upward import.
RELATION_MODEL = "relsgg-vits16plus"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relation_vocabulary_signature(predicates: "Sequence[str] | None") -> str:
    """Stable id for a predicate vocabulary.

    A relation graph is only valid *for the vocabulary it was scored with*:
    RelateAnything's scores are produced by a softmax over the phrases it was
    given, so the same regions under ``["riding"]`` and under the 35-phrase
    default list are different facts.  Keying the cache on this signature (in
    addition to ``regions_sha256``) is what lets ``recall --relation`` and
    ``index --with-relations`` share one file without ever confusing the two.
    """
    phrases = [str(p).strip() for p in (predicates or []) if str(p).strip()]
    return sha256_json(phrases)[:16]


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class IndexStore:
    """Reader/writer for one index directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------

    @property
    def meta_path(self) -> Path:
        return self.root / "index.json"

    def frame_image_path(self, frame_id: int) -> Path:
        return self.root / "frames" / f"f{frame_id:06d}.png"

    def regions_path(self, frame_id: int) -> Path:
        return self.root / "regions" / f"f{frame_id:06d}.json"

    def masks_path(self, frame_id: int) -> Path:
        return self.root / "masks" / f"f{frame_id:06d}.npz"

    def relations_path(self, frame_id: int) -> Path:
        return self.root / "relations" / f"f{frame_id:06d}.json"

    def senses_path(self, frame_id: int) -> Path:
        return self.root / "senses" / f"f{frame_id:06d}.json"

    def snapshot_path(self, frame_id: int, idxs: Sequence[int]) -> Path:
        return self.root / "answers" / f"f{frame_id:06d}_r{'-'.join(str(i) for i in idxs)}.json"

    def output_path(self, name: str) -> Path:
        return self.root / "outputs" / name

    # -- meta ----------------------------------------------------------------

    def exists(self) -> bool:
        return self.meta_path.is_file()

    def require(self) -> dict:
        if not self.exists():
            raise VrsError(
                f"not a vrs index directory: {self.root}",
                hint="run `vrs index <video>` first; it prints the resolved index_dir to use.",
            )
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VrsError(
                f"index.json is not valid JSON ({exc})",
                hint="re-run `vrs index <video>` against a fresh --out to rebuild.",
            ) from exc

    def save_meta(self, meta: dict) -> None:
        atomic_write_json(self.meta_path, meta)

    def frame_ids(self) -> list[int]:
        return [f["frame_id"] for f in self.require().get("frames", [])]

    def require_frame(self, frame_id: int) -> dict:
        meta = self.require()
        for f in meta.get("frames", []):
            if f["frame_id"] == frame_id:
                return f
        known = [f["frame_id"] for f in meta.get("frames", [])]
        preview = known[:12] + (["..."] if len(known) > 12 else [])
        raise VrsError(
            f"frame_id {frame_id} is not in this index",
            hint=f"known frame_ids: {preview} (see `vrs frames <index_dir>`); "
            "pass one of those, or re-run `vrs index` with a larger --frames.",
        )

    # -- regions -------------------------------------------------------------

    def load_regions(self, frame_id: int) -> dict | None:
        p = self.regions_path(frame_id)
        if not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def require_regions(self, frame_id: int) -> dict:
        rec = self.load_regions(frame_id)
        if rec is None:
            self.require_frame(frame_id)
            raise VrsError(
                f"no region table stored for frame {frame_id}",
                hint="re-run `vrs index` (the frame is listed but its regions are missing).",
            )
        return rec

    def save_regions(self, frame_id: int, rec: dict) -> None:
        atomic_write_json(self.regions_path(frame_id), rec)

    def get_region(self, frame_id: int, idx: int) -> dict:
        rec = self.require_regions(frame_id)
        regions = rec.get("regions", [])
        if idx < 0 or idx >= len(regions) or regions[idx]["idx"] != idx:
            raise VrsError(
                f"region_idx {idx} does not exist on frame {frame_id} "
                f"(valid: 0..{len(regions) - 1})",
                hint=f"run `vrs regions <index_dir> {frame_id}` to list the candidates.",
            )
        return regions[idx]

    @staticmethod
    def merge_regions(existing: list[dict], incoming: Iterable[dict], *,
                      iou_threshold: float = 0.8) -> tuple[list[dict], list[dict]]:
        """Append *incoming* to *existing* without disturbing stable ids.

        A candidate is a duplicate of an existing region when the label matches
        (case-insensitive) **and** the boxes overlap by ``iou_threshold``; the
        existing idx is then reused, which makes repeated ``detect`` calls
        idempotent instead of growing the table forever.  A different label on
        the same pixels is a *new* hypothesis and gets a new idx on purpose.

        Returns ``(merged_regions, added_regions)``.
        """
        merged = list(existing)
        added: list[dict] = []

        def _iou(a: Sequence[float], b: Sequence[float]) -> float:
            ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
            ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / ua if ua > 0 else 0.0

        for cand in incoming:
            hit = None
            for r in merged:
                if r["label"].lower() != cand["label"].lower():
                    continue
                if _iou(r["bbox"], cand["bbox"]) >= iou_threshold:
                    hit = r
                    break
            if hit is not None:
                # keep the newer, higher score, but never touch idx or bbox:
                # the box is already published to the agent.
                if cand.get("score", 0) > hit.get("score", 0):
                    hit["score"] = cand["score"]
                    hit["source"] = cand.get("source", hit.get("source"))
                continue
            new = dict(cand)
            new["idx"] = len(merged)
            merged.append(new)
            added.append(new)
        return merged, added

    def append_regions(self, frame_id: int, incoming: Iterable[dict]) -> tuple[dict, list[dict]]:
        """Merge *incoming* into the stored table and persist the result."""
        rec = self.require_regions(frame_id)
        merged, added = self.merge_regions(rec["regions"], incoming)
        rec["regions"] = merged
        rec["next_idx"] = len(merged)
        self.save_regions(frame_id, rec)
        # keep the frame table's n_regions honest
        meta = self.require()
        for f in meta["frames"]:
            if f["frame_id"] == frame_id:
                f["n_regions"] = len(merged)
        self.save_meta(meta)
        return rec, added

    # -- masks ---------------------------------------------------------------

    def load_masks(self, frame_id: int) -> dict[int, np.ndarray]:
        p = self.masks_path(frame_id)
        if not p.is_file():
            return {}
        with np.load(p) as z:
            idxs = z["idx"].tolist()
            arr = z["masks"]
        return {int(i): arr[k] for k, i in enumerate(idxs)}

    def save_masks(self, frame_id: int, masks: dict[int, np.ndarray]) -> None:
        p = self.masks_path(frame_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not masks:
            if p.exists():
                p.unlink()
            return
        idxs = sorted(masks)
        arr = np.stack([np.asarray(masks[i], dtype=bool) for i in idxs])
        np.savez_compressed(p, idx=np.asarray(idxs, dtype=np.int32), masks=arr)

    # -- relations -----------------------------------------------------------

    def save_relations(self, frame_id: int, payload: dict) -> None:
        atomic_write_json(self.relations_path(frame_id), payload)

    def load_relations(self, frame_id: int) -> dict | None:
        p = self.relations_path(frame_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def append_relation_set(self, frame_id: int, regions_sha256: str,
                            predicates: Sequence[str], edges: list[dict],
                            info: dict | None = None, *, model: str = RELATION_MODEL) -> None:
        """Append one vocabulary's relation graph, keeping earlier ones.

        The frame's relation file is a *container*: ``sets[<vocab sig>]`` holds
        one graph per predicate vocabulary, so ``index --with-relations`` and
        later ``recall --relation`` calls can each add their own graph without
        destroying the other.  The top-level ``relation``/``edges`` fields keep
        the newest set, so a reader that predates this schema still sees a
        complete graph.

        A changed ``regions_sha256`` means every stored graph describes a
        region table that no longer exists, so the whole container is dropped
        before the new set is written (same rule as the old flat file).
        """
        p = self.relations_path(frame_id)
        payload: dict = {}
        if p.is_file():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        if payload.get("regions_sha256") not in (None, regions_sha256):
            payload = {}
        sets = dict(payload.get("sets") or {})
        # A legacy flat file (written by `relate` before this schema existed)
        # is one set too; adopt it so its graph is reusable instead of dropped.
        legacy_preds = (payload.get("relation") or {}).get("predicates")
        if legacy_preds and payload.get("edges") is not None and "sets" not in payload:
            sets.setdefault(
                relation_vocabulary_signature(legacy_preds),
                {"predicates": list(legacy_preds), "edges": payload.get("edges", []),
                 "info": payload.get("info"), "regions_sha256": payload.get("regions_sha256")},
            )
        phrases = [str(x).strip() for x in predicates if str(x).strip()]
        sets[relation_vocabulary_signature(phrases)] = {
            "predicates": phrases,
            "edges": list(edges),
            "info": dict(info or {}),
            "regions_sha256": regions_sha256,
            "computed_at": _now(),
        }
        payload.update({
            "frame_id": frame_id,
            "regions_sha256": regions_sha256,
            "relation": {"model": model, "predicates": phrases},
            "info": dict(info or {}),
            "edges": list(edges),
            "sets": sets,
        })
        atomic_write_json(p, payload)

    def find_relation_set(self, frame_id: int, regions_sha256: str,
                          predicates: Sequence[str]) -> dict | None:
        """The stored graph for *exactly* this region table and vocabulary."""
        data = self.load_relations(frame_id)
        if not data or data.get("regions_sha256") != regions_sha256:
            return None
        sig = relation_vocabulary_signature(predicates)
        got = (data.get("sets") or {}).get(sig)
        if isinstance(got, dict):
            return got
        # Back-compat: a flat file whose own vocabulary happens to match.
        flat_preds = (data.get("relation") or {}).get("predicates")
        if flat_preds and relation_vocabulary_signature(flat_preds) == sig:
            return {"predicates": list(flat_preds), "edges": data.get("edges", []),
                    "info": data.get("info"), "regions_sha256": regions_sha256}
        return None

    # -- senses --------------------------------------------------------------

    def save_senses(self, frame_id: int, payload: dict) -> None:
        atomic_write_json(self.senses_path(frame_id), payload)

    def load_senses(self, frame_id: int) -> dict | None:
        p = self.senses_path(frame_id)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    # -- answers -------------------------------------------------------------

    def save_snapshot(self, frame_id: int, idxs: Sequence[int], payload: dict) -> Path:
        p = self.snapshot_path(frame_id, idxs)
        atomic_write_json(p, payload)
        return p

    def load_snapshot(self, frame_id: int, idxs: Sequence[int]) -> dict | None:
        p = self.snapshot_path(frame_id, idxs)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    # -- helpers -------------------------------------------------------------

    def frame_image(self, frame_id: int) -> Path:
        p = self.frame_image_path(frame_id)
        if not p.is_file():
            raise VrsError(
                f"frame image missing on disk: {for_agent(p)}",
                hint="re-run `vrs index <video>`; the index directory looks damaged.",
            )
        return p

    def describe(self) -> dict:
        """Small self-description embedded in every success payload."""
        return {"index_dir": for_agent(self.root), "index_dir_win": for_windows(self.root)}
