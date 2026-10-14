"""``vrs index``: decode -> prompt-free YOLOE -> disk (relations on demand).

The index answers one question: *what regions are in this video?*  That is
detector work, and it is cheap (0.29 s/frame measured on the demo video).
Relations are a different question -- they depend on which predicates matter,
which is exactly what an index build cannot know -- and they cost ~0.93 s/frame.
So indexing computes regions only by default and leaves relation graphs to the
query layer (``vrs recall --relation``), which backfills them per frame and
appends one graph per vocabulary.  ``--with-relations`` restores the old
compute-everything behaviour for callers that want it.

Because the index directory is named after the video's sha256, every cached
frame in it belongs to exactly this video, and re-running ``index`` reuses
whatever matches the *current* detector/relation configuration instead of
recomputing it.  Only the frames that are missing (a larger ``--frames``) or
stale (a different ``conf``) are touched, so a re-index with 32 frames on top
of a 16-frame index costs 16 frames of work, not 32.

Cost is the point of the ``timings`` block: the agent decides how large a
``--frames`` it can afford from measured numbers, not from a guess.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from . import paths
from .errors import VrsError
from .paths import for_agent, for_windows
from .store import IndexStore, atomic_write_json, sha256_json
from .video import probe, read_frames, sample_frame_ids, video_fingerprint

DETECTOR_MODE = "prompt-free-lvis4585"
DETECTOR_MODEL = "yoloe-11s-seg-pf.pt"
RELATION_MODEL = "relsgg-vits16plus"
IMGSZ = 640
#: Bump when the *meaning* of a cached region/mask changes (post-processing fix,
#: new field semantics). It is part of the cache key, so a code change that
#: would alter stored data invalidates the old frames instead of silently
#: reusing them. v2: masks are cropped out of the letterbox before scaling.
PIPELINE_VERSION = 2


def _detector_id(conf: float) -> dict:
    return {"model": DETECTOR_MODEL, "mode": DETECTOR_MODE, "conf": conf, "imgsz": IMGSZ,
            "pipeline": PIPELINE_VERSION}


def index_dir_for(out_dir: Path, video: Path, digest: str) -> Path:
    return Path(out_dir) / f"{video.stem}-{digest[:8]}"


def _dir_size_mb(root: Path) -> float:
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return round(total / (1024 * 1024), 2)


def _frame_table(store: IndexStore, frame_ids: list[int], fps: float) -> list[dict]:
    table = []
    for fid in frame_ids:
        rec = store.load_regions(fid)
        img = store.frame_image_path(fid)
        table.append(
            {
                "frame_id": fid,
                "t": round(fid / fps, 3),
                "n_regions": len(rec["regions"]) if rec else 0,
                "image": for_agent(img),
                "image_win": for_windows(img),
                "regions_file": for_agent(store.regions_path(fid)),
            }
        )
    return table


def run_index(video_path: Path, out_dir: Path, n_frames: int = 16, conf: float = 0.25,
              topk: int = 20, with_relations: bool = False) -> dict:
    t_start = time.time()
    meta_probe = probe(video_path)
    digest = video_fingerprint(video_path)
    fps = meta_probe["fps"]

    root = index_dir_for(out_dir, video_path, digest)
    store = IndexStore(root)
    root.mkdir(parents=True, exist_ok=True)

    frame_ids = sample_frame_ids(n_frames, meta_probe["n_video_frames"])
    det_id = _detector_id(conf)

    # ---- plan -------------------------------------------------------------
    need_detect: list[int] = []
    for fid in frame_ids:
        rec = store.load_regions(fid)
        if rec is None or rec.get("detector") != det_id or not store.frame_image_path(fid).is_file():
            need_detect.append(fid)

    # ---- compute detectors (only if something is missing) -----------------
    timings: dict = {}
    if need_detect:
        from .detector import PromptFreeDetector

        t0 = time.time()
        det = PromptFreeDetector(conf=conf)
        t_load_det = time.time() - t0
        frames = read_frames(video_path, need_detect)

        t_det_all = 0.0
        n_det = 0
        for fid, frame in frames.items():
            t1 = time.time()
            cands, masks = det.detect(frame)
            t_det_all += time.time() - t1
            n_det += 1
            # frame image first: it is the source of truth for every later artifact
            import cv2

            store.frame_image_path(fid).parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(store.frame_image_path(fid)), frame):
                raise VrsError(
                    f"could not write frame {fid} to {store.frame_image_path(fid)}",
                    hint="check the --out drive has space and is writable.",
                )
            regions = []
            mask_map: dict[int, np.ndarray] = {}
            for cand in cands:
                cand = dict(cand)
                cand["idx"] = len(regions)
                regions.append(cand)
            for cand, mask in zip(regions, masks):
                if mask is not None:
                    mask_map[cand["idx"]] = mask
            rec = {
                "frame_id": fid,
                "t": round(fid / fps, 3),
                "width": meta_probe["width"],
                "height": meta_probe["height"],
                "image": for_agent(store.frame_image_path(fid)),
                "detector": det_id,
                "next_idx": len(regions),
                "regions": regions,
            }
            store.save_regions(fid, rec)
            # an empty candidate set must still leave a valid (empty) table
            store.save_masks(fid, mask_map)
            # detector params changed -> the old relation graph no longer applies
            if store.relations_path(fid).exists():
                store.relations_path(fid).unlink()

        timings["model_load_detector_s"] = round(t_load_det, 2)
        timings["detect_s_per_frame"] = round(t_det_all / max(n_det, 1), 3)
        timings["frames_detected"] = n_det

    # ---- relations (opt-in; the query layer backfills them by default) ----
    need_relate: list[int] = []
    if with_relations:
        try:
            from relsgg import DEFAULT_PREDICATES  # lightweight: a list, no model
        except ImportError:
            DEFAULT_PREDICATES = None  # let the cache miss and load the model
        for fid in frame_ids:
            rec = store.require_regions(fid)
            want = sha256_json(rec["regions"])
            if store.find_relation_set(fid, want, DEFAULT_PREDICATES) is None:
                need_relate.append(fid)

    if need_relate:
        from .relation import Relater

        t0 = time.time()
        relater = Relater()
        timings["model_load_relater_s"] = round(time.time() - t0, 2)
        frames = read_frames(video_path, need_relate)
        t_rel_all = 0.0
        for fid in need_relate:
            frame = frames.get(fid)
            rec = store.require_regions(fid)
            regions_sha = sha256_json(rec["regions"])
            if frame is None or len(rec["regions"]) < 2:
                store.append_relation_set(
                    fid, regions_sha, relater.predicates, [],
                    {"n_regions": len(rec["regions"]), "n_used": len(rec["regions"]),
                     "truncated": False,
                     "reason": "RelateAnything needs at least 2 regions to form a pair"},
                    model=RELATION_MODEL,
                )
                continue
            t1 = time.time()
            edges, info = relater.relate(frame, rec["regions"], store.load_masks(fid),
                                         topk=topk, min_score=0.0)
            t_rel_all += time.time() - t1
            store.append_relation_set(fid, regions_sha, info["vocabulary"], edges, info,
                                      model=RELATION_MODEL)
        timings["relate_s_per_frame"] = round(t_rel_all / max(len(need_relate), 1), 3)
    # `frames_related` is always present: 0 is the honest answer for the
    # default layered run and must not be inferred from a missing key.
    timings["frames_related"] = len(need_relate)

    # ---- meta -------------------------------------------------------------
    previous = store.require() if store.exists() else {}
    table = _frame_table(store, frame_ids, fps)
    # A one-frame top-up run is a bad benchmark; only a run that measured a
    # useful sample (or the first ever run) is allowed to replace the numbers.
    computed = len(need_detect) + len(need_relate)
    benchmark = timings if (computed >= 4 or not previous.get("timings")) else previous["timings"]
    meta = {
        "schema": 1,
        "video": for_agent(video_path),
        "video_win": for_windows(video_path),
        "video_sha256": digest,
        "video_bytes": video_path.stat().st_size,
        "fps": fps,
        "n_video_frames": meta_probe["n_video_frames"],
        "duration_s": meta_probe["duration_s"],
        "width": meta_probe["width"],
        "height": meta_probe["height"],
        "n_frames": len(table),
        "frame_sampling": "even, endpoints inclusive; frame_id == decoded video frame index; t = frame_id / fps",
        "detector": det_id,
        "relation": {
            "model": RELATION_MODEL,
            "mode": "index-time" if with_relations else "on-demand",
            "index_vocabulary": ("relsgg DEFAULT_PREDICATES (prompt-free at index time)"
                                 if with_relations else None),
        },
        "frames": table,
        "timings": benchmark,
        "last_run_s": round(time.time() - t_start, 2),
        "last_run_work": "all frames reused" if not (need_detect or need_relate) else {
            "frames_detected": len(need_detect), "frames_related": len(need_relate),
        },
    }
    store.save_meta(meta)

    return _summary(store, meta, repeated=not (need_detect or need_relate))


def _summary(store: IndexStore, meta: dict, repeated: bool) -> dict:
    payload = {
        "index_dir": for_agent(store.root),
        "index_dir_win": for_windows(store.root),
        "video": meta["video"],
        "video_sha256": meta["video_sha256"],
        "n_frames": meta["n_frames"],
        "n_video_frames": meta["n_video_frames"],
        "fps": meta["fps"],
        "duration_s": meta["duration_s"],
        "width": meta["width"],
        "height": meta["height"],
        "reused": repeated,
        "frames": meta["frames"],
        "detector": meta["detector"],
        "relation": meta["relation"],
        "timings": meta["timings"],
        "size_mb": _dir_size_mb(store.root),
    }
    if repeated:
        payload["note"] = ("nothing recomputed: every requested frame already has a region table "
                           "and relation graph for this video hash and these settings")
    return payload


def load_index_summary(store: IndexStore) -> dict:
    meta = store.require()
    return {**_summary(store, meta, repeated=True), "reused": True}


def refresh_frames(store: IndexStore) -> dict:
    meta = store.require()
    table = _frame_table(store, [f["frame_id"] for f in meta["frames"]], meta["fps"])
    meta["frames"] = table
    store.save_meta(meta)
    return {
        "index_dir": for_agent(store.root),
        "index_dir_win": for_windows(store.root),
        "video": meta["video"],
        "fps": meta["fps"],
        "duration_s": meta["duration_s"],
        "n_frames": len(table),
        "frames": table,
    }


def clean_index(store: IndexStore) -> dict:
    """Remove a whole index directory (used by tests; not exposed as a subcommand)."""
    if store.root.exists():
        shutil.rmtree(store.root)
    return {"removed": for_agent(store.root)}
