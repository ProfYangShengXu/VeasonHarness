"""``veason senses``: the round-3 sense axes, computed once and stored per frame.

Round 1 gave the CLI a detector (what *objects* are in a frame).  Round 2 gave it
retrieval over those objects (``recall``).  Both answer questions on the
**category / attribute** axis -- and a query like "the yellow cartoon character
*that is laughing*" has a part ("laughing") that neither can express, because no
stored field carries it.  ``recall`` then correctly ranks the frames it *can*
score, which is not the same as the frames that match.

This module adds the missing axes as **bounded, per-region fields** computed by
small local models, so a state like "laughing" becomes a number a route can
threshold and rank -- not something only an LLM can judge:

===============  =========================================================  ============
axis             field (per region)                                        model
===============  =========================================================  ============
``face``         ``jaw_open`` / ``mouth_smile_l/r`` / ``cheek_squint_l/r``   MediaPipe Face Landmarker
                 ``eye_squint_l/r`` / ``expression`` / ``expression_score`` + CLIP zero-shot expression head
``action``       ``label`` / ``score`` / ``top[]``                          VideoMAE-small (Kinetics-400)
``depth``        ``mean_rel`` / ``rank_in_frame``                          Depth Anything V2-Small (ONNX)
``text``         ``[{s, box, score}]``                                     RapidOCR (ONNX)
``pose``         ``track.track_id`` / ``track.dx_per_s`` / ``pose.kpt17``   yolo11n-pose + ByteTrack
===============  =========================================================  ============

Design rules, inherited from the rest of the project:

* **Layering.**  ``veason index`` does *not* run any of this.  Like relation
  graphs, senses are an opt-in pass (``veason senses``) and the index keeps only
  what a build can know for free.  ``veason recall`` reads the stored files and
  loads no model of its own.
* **A stale field is worse than a missing one.**  Each frame's file records the
  ``regions_sha256`` it was computed against; a ``detect`` that changes the
  table invalidates the file, exactly like a relation graph.
* **Every number must be traceable.**  Each block carries ``src``; the
  thresholds live in ``recall`` next to the measurement that set them.
"""
from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .paths import (DEPTH_ONNX, DEPTH_PREPROCESSOR, FACE_LANDMARKER_TASK,
                    POSE_WEIGHTS, VIDEOMAE_DIR)
from .store import IndexStore, sha256_json

SCHEMA = 2   # v2: face slot 的 src 改指真正产出该值的件（规格 §4.3 纪律③）；见下面复用判据

#: ``pose`` covers both the tracker and the keypoints: they come from one pass.
DEFAULT_AXES = ("face", "action", "depth", "text", "pose")

AXIS_MODELS = {
    "face": ["mediapipe-face-landmarker", "clip-vitb32-expression-zeroshot"],
    "action": ["videomae-small-kinetics"],
    "depth": ["depth-anything-v2-small"],
    "text": ["rapidocr-onnxruntime"],
    "pose": ["yolo11n-pose", "bytetrack"],
}

#: Blendshape keys kept from MediaPipe (the model outputs ~52; these are the
#: ones a query can name).  ``mouth_smile_*``/``jaw_open`` are the "laughing"
#: evidence, ``cheek_squint``/``eye_squint`` the "eyes squeezed shut" evidence.
FACE_BLENDSHAPES = {
    "jawOpen": "jaw_open",
    "mouthSmileLeft": "mouth_smile_l",
    "mouthSmileRight": "mouth_smile_r",
    "cheekSquintLeft": "cheek_squint_l",
    "cheekSquintRight": "cheek_squint_r",
    "eyeSquintLeft": "eye_squint_l",
    "eyeSquintRight": "eye_squint_r",
    "eyeWideLeft": "eye_wide_l",
    "eyeWideRight": "eye_wide_r",
}

#: Fixed, closed expression bank for the zero-shot head.  It is *fixed on
#: purpose*: a bounded field has to have a stable label set, unlike the free
#: ``--label`` phrases of the semantic route.  Measured separations on
#: ``dragon.mp4``: "laughing" 0.42-0.52 on the 13.8-14.9 s laugh, 0.04-0.17 on
#: the 6-8 s neutral frames (full table in REPORT.md, round 3).
EXPRESSION_BANK = (
    "a laughing face",
    "a smiling face",
    "a neutral face",
    "a sad face",
    "a crying face",
    "an angry face",
    "a surprised face",
    "a face with a closed mouth",
    "a face with an open mouth",
)
EXPRESSION_LABELS = {
    "a laughing face": "laughing",
    "a smiling face": "smiling",
    "a neutral face": "neutral",
    "a sad face": "sad",
    "a crying face": "crying",
    "an angry face": "angry",
    "a surprised face": "surprised",
    "a face with a closed mouth": "mouth_closed",
    "a face with an open mouth": "mouth_open",
}

#: Only the biggest few regions per frame get a CLIP expression pass; the
#: expression of a 2%-area region is not worth 0.09 s.
EXPRESSION_MAX_REGIONS = 4
EXPRESSION_MIN_AREA = 0.02

#: Depth Anything V2 wants the short side at 518 and both sides a multiple of 14.
DEPTH_SIZE = 518
DEPTH_MULTIPLE = 14

POSE_CONF = 0.10
#: Keypoint confidence below this is treated as absent (COCO-17).
KPT_CONF = 0.30


def _snake(name: str) -> str:
    """``browDownLeft`` -> ``brow_down_left`` (stable, queryable key names)."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


@contextlib.contextmanager
def _silence_native_stderr():
    """Silence C-level logging (TFLite/absl) that ``redirect_stderr`` misses.

    stderr is a JSON channel in this CLI; MediaPipe's graph warnings arrive
    from C++ and would otherwise be parsed as a malformed error record.
    """
    try:
        devnull = open(os.devnull, "w")
    except OSError:  # pragma: no cover - no /dev/null is not a supported box
        yield
        return
    with devnull:
        old = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(old, 2)
            os.close(old)


def _box_area(b) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _intersection(a, b) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _assign_region(box, regions: Sequence[dict]) -> Optional[int]:
    """The region idx whose box overlaps *box* the most, or None.

    IoU would punish a small face inside a big body region; raw intersection
    area picks the region that actually contains the detection.
    """
    best_idx, best = None, 0.0
    for r in regions:
        inter = _intersection(box, r["bbox"])
        if inter > best:
            best_idx, best = int(r["idx"]), inter
    return best_idx


def _to_box(pts: Sequence[Sequence[float]], w: int, h: int):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return [int(max(0, min(xs)) * w), int(max(0, min(ys)) * h),
            int(min(1, max(xs)) * w), int(min(1, max(ys)) * h)]


# --------------------------------------------------------------------------
# per-axis sensors
# --------------------------------------------------------------------------

class FaceSensor:
    """MediaPipe Face Landmarker (blendshapes) + a CLIP expression head.

    Two providers, because they answer different questions and are good on
    different material:

    * MediaPipe gives *landmark* blendshapes (``jaw_open``, ``mouth_smile_*``)
      which are the direct, calibrated output for a real face.  On the demo
      cartoon it detects nothing (see REPORT.md round 3, "坑 3"), but it is
      what makes ``--face-smile-min`` meaningful on real footage.
    * The expression head is a **fixed-bank zero-shot classifier** over the
      region crop.  It exists because that cartoon face is out of BlazeFace's
      domain: measured "laughing" is 0.42-0.52 at the laugh peak vs 0.04-0.17
      on the neutral frames, so it is a usable bounded field where the
      landmarker is blind.  It is recorded with its own ``expression_src`` so
      a caller can tell which provider produced a number.
    """

    def __init__(self) -> None:
        self._lm = None
        self._clip = None

    def _landmarker(self):
        if self._lm is None:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            with _silence_native_stderr():
                self._lm = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=mp_python.BaseOptions(
                            model_asset_path=str(FACE_LANDMARKER_TASK)),
                        running_mode=vision.RunningMode.IMAGE,
                        num_faces=5,
                        output_face_blendshapes=True,
                    ))
        return self._lm

    def landmarks(self, img: np.ndarray) -> list[dict]:
        """Per detected face: normalized landmarks + named blendshape scores."""
        import mediapipe as mp

        h, w = img.shape[:2]
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        with _silence_native_stderr():
            res = self._landmarker().detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for i, bs in enumerate(res.face_blendshapes or []):
            scores = {FACE_BLENDSHAPES.get(c.category_name, _snake(c.category_name)):
                      round(float(c.score), 4) for c in bs}
            pts = None
            if res.face_landmarks:
                pts = [[float(p.x), float(p.y)] for p in res.face_landmarks[i]]
            out.append({"blendshapes": scores, "bbox": _to_box(pts, w, h) if pts else None})
        return out

    def expression(self, crops: list) -> list[dict]:
        """Zero-shot expression distribution for mask-aware region crops."""
        if not crops:
            return []
        from PIL import Image

        from .recall import _ClipScorer  # shared loader: one CLIP per process

        scorer = _ClipScorer()
        img_feats = scorer.encode_crops(crops)
        txt_feats = scorer.encode_text(list(EXPRESSION_BANK))
        probs = _softmax(100.0 * (img_feats @ txt_feats.T), axis=1)
        rows = []
        for row in probs:
            named = {EXPRESSION_LABELS[p]: round(float(row[i]), 4)
                     for i, p in enumerate(EXPRESSION_BANK)}
            top = max(named.items(), key=lambda kv: kv[1])
            rows.append({"expression": top[0], "expression_score": top[1],
                         "expression_scores": named})
        return rows


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class ActionSensor:
    """VideoMAE-small fine-tuned on Kinetics-400, run on a 16-frame clip.

    Chosen over a per-frame CNN because "laughing" is a *motion* class and
    Kinetics-400 contains it literally (``id2label[180] == "laughing"``); the
    model is 88 MB and ~1.0 s/clip on this CPU.

    The label is clip-level, so it is attached to every region of the frame and
    the block says so in ``note`` -- inventing a per-region action score would
    be a made-up number.
    """

    def __init__(self, frame_ids: Sequence[int], store: IndexStore) -> None:
        self.frame_ids = list(frame_ids)
        self.store = store
        self._model = None
        self._proc = None
        self._labels: dict = {}

    def _load(self) -> None:
        if self._model is not None:
            return
        import warnings

        import torch  # noqa: F401
        import transformers
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        # stderr is a JSON channel here: a LOAD REPORT ("q_bias UNEXPECTED ...")
        # or a tqdm bar would look like a malformed error record to the caller.
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()

        path = str(VIDEOMAE_DIR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._proc = VideoMAEImageProcessor.from_pretrained(path)
            self._model = VideoMAEForVideoClassification.from_pretrained(path)
        # ---- checkpoint compatibility -------------------------------------
        # The MCG-NJU checkpoint stores one fused `qkv.weight` plus `q_bias`/
        # `v_bias`; transformers >=5.8 models attention as three separate
        # Linear layers.  `from_pretrained` succeeds but leaves query/key/value
        # biases *randomly initialised* (its LOAD REPORT calls them MISSING),
        # i.e. the model runs and returns noise.  Re-apply the checkpoint's
        # biases explicitly instead of trusting the load.
        state = _torch_load(VIDEOMAE_DIR / "pytorch_model.bin")
        patched = 0
        for i, layer in enumerate(self._model.videomae.encoder.layer):
            att = layer.attention.attention
            pre = f"videomae.encoder.layer.{i}.attention.attention."
            if pre + "q_bias" not in state:
                continue
            att.query.bias.data.copy_(state[pre + "q_bias"])
            att.value.bias.data.copy_(state[pre + "v_bias"])
            att.key.bias.data.zero_()
            patched += 1
        self._biases_patched = patched
        self._model.eval()
        self._labels = {int(k): v for k, v in self._model.config.id2label.items()}

    def _clip(self, i: int) -> list:
        import cv2
        from PIL import Image

        n = len(self.frame_ids)
        start = max(0, min(n - 16, i - 8))
        frames = []
        for fid in self.frame_ids[start:start + 16]:
            img = cv2.imread(str(self.store.frame_image(fid)))
            frames.append(Image.fromarray(np.ascontiguousarray(img[:, :, ::-1])))
        return frames

    def predict(self, i: int, topk: int = 3) -> list[dict]:
        import torch

        self._load()
        x = self._proc(self._clip(i), return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            probs = self._model(x).logits[0].softmax(-1)
        vals, idxs = probs.topk(topk)
        return [{"label": self._labels[int(j)], "score": round(float(v), 4),
                 "src": "videomae-small-kinetics"}
                for v, j in zip(vals, idxs)]


def _torch_load(path: Path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        return torch.load(str(path), map_location="cpu")


class DepthSensor:
    """Depth Anything V2-Small (ONNX, 99 MB) -> per-region relative depth.

    The model outputs *relative inverse depth*: larger means nearer.  It is
    min-max normalised **per frame** (so ``mean_rel`` is comparable only within
    a frame, never across frames) and the per-frame ranking is stored as
    ``rank_in_frame`` (1 = nearest) because that is the scale-free fact.
    """

    def __init__(self) -> None:
        self._sess = None

    def _session(self):
        if self._sess is None:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            self._sess = ort.InferenceSession(str(DEPTH_ONNX), so,
                                              providers=["CPUExecutionProvider"])
        return self._sess

    def map(self, img: np.ndarray) -> np.ndarray:
        import cv2

        sess = self._session()
        h, w = img.shape[:2]
        scale = DEPTH_SIZE / min(h, w)
        nh = max(DEPTH_MULTIPLE, int(round(h * scale / DEPTH_MULTIPLE)) * DEPTH_MULTIPLE)
        nw = max(DEPTH_MULTIPLE, int(round(w * scale / DEPTH_MULTIPLE)) * DEPTH_MULTIPLE)
        x = cv2.resize(img, (nw, nh)).astype(np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        out = sess.run(None, {"pixel_values": x.transpose(2, 0, 1)[None]})[0][0]
        d = cv2.resize(out, (w, h))
        lo, hi = float(d.min()), float(d.max())
        return (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)

    @staticmethod
    def region_stats(dmap: np.ndarray, region: dict, mask: Optional[np.ndarray]) -> dict:
        x1, y1, x2, y2 = [int(v) for v in region["bbox"]]
        patch = dmap[y1:y2, x1:x2]
        if mask is not None and getattr(mask, "shape", None) == dmap.shape:
            sub = mask[y1:y2, x1:x2]
            vals = patch[sub] if sub.any() else patch.ravel()
        else:
            vals = patch.ravel()
        if vals.size == 0:
            return {"mean_rel": 0.0, "p50_rel": 0.0, "src": "depth-anything-v2-small"}
        return {"mean_rel": round(float(vals.mean()), 4),
                "p50_rel": round(float(np.median(vals)), 4),
                "src": "depth-anything-v2-small"}


class TextSensor:
    """RapidOCR (ONNX) over the whole frame; boxes are attached to regions.

    ``--text`` matches the concatenated per-region strings; a frame-level list
    is kept too, so text that lies on no region is not silently dropped.
    """

    def __init__(self) -> None:
        self._engine = None

    def _get(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def read(self, img: np.ndarray) -> list[dict]:
        result, _ = self._get()(img)
        out = []
        for box, text, score in (result or []):
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            out.append({"s": str(text), "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                        "score": round(float(score), 4), "src": "rapidocr"})
        return out


class PoseSensor:
    """yolo11n-pose per frame + ByteTrack across the sampled frames.

    ByteTrack is fed ``result.boxes.cpu().numpy()`` -- in ultralytics 8.4 its
    ``update`` wants a boxes array, not a ``Results`` object (passing the
    latter raises ``AttributeError: no attribute 'conf'``).  Its return rows
    are ``[x1, y1, x2, y2, track_id, score, cls, det_index]``; ``det_index``
    maps back to the keypoints of the detection it came from.
    """

    def __init__(self) -> None:
        self._model = None
        self._tracker = None
        self._prev: dict[int, dict] = {}

    def _get(self):
        if self._model is None:
            from ultralytics import YOLO
            from ultralytics.trackers.byte_tracker import BYTETracker
            from types import SimpleNamespace

            self._model = YOLO(str(POSE_WEIGHTS))
            self._tracker = BYTETracker(SimpleNamespace(
                track_high_thresh=0.25, track_low_thresh=0.10, new_track_thresh=0.25,
                track_buffer=30, match_thresh=0.8, fuse_score=True))
        return self._model, self._tracker

    def detect(self, img: np.ndarray, t: float) -> tuple[list[dict], list[dict]]:
        import torch

        model, tracker = self._get()
        res = model.predict(img, verbose=False, conf=POSE_CONF, imgsz=640)[0]
        boxes = res.boxes
        tracks = np.asarray(tracker.update(boxes.cpu().numpy(), img))
        kpts = res.keypoints.data if res.keypoints is not None else None
        out = []
        for row in tracks:
            x1, y1, x2, y2, tid, score = (float(row[0]), float(row[1]), float(row[2]),
                                          float(row[3]), int(row[4]), float(row[5]))
            det_i = int(row[-1])
            kp = kpts[det_i].tolist() if kpts is not None and det_i < len(kpts) else []
            out.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "track_id": tid,
                        "score": round(score, 4), "kpt17": kp})
        return out, _velocity(out, self._prev, t)


def _velocity(tracks: list[dict], prev: dict[int, dict], t: float) -> list[dict]:
    """Per-track centre displacement per second since its previous sampled frame."""
    out = []
    for tr in tracks:
        cx = (tr["bbox"][0] + tr["bbox"][2]) / 2.0
        cy = (tr["bbox"][1] + tr["bbox"][3]) / 2.0
        p = prev.get(tr["track_id"])
        item = {"track_id": tr["track_id"], "src": "bytetrack"}
        if p is not None and t > p["t"]:
            dt = t - p["t"]
            item["dx_per_s"] = round((cx - p["cx"]) / dt, 3)
            item["dy_per_s"] = round((cy - p["cy"]) / dt, 3)
        item["bbox"] = tr["bbox"]
        out.append(item)
        prev[tr["track_id"]] = {"cx": cx, "cy": cy, "t": t}
    return out


def _arms_raised(kpt17: Sequence[Sequence[float]]) -> bool:
    """Both wrists clearly above both shoulders (image y grows downward)."""
    if len(kpt17) < 11:
        return False
    l_sh, r_sh = kpt17[5], kpt17[6]
    l_wr, r_wr = kpt17[9], kpt17[10]
    if min(l_sh[2], r_sh[2], l_wr[2], r_wr[2]) < KPT_CONF:
        return False
    return l_wr[1] < l_sh[1] and r_wr[1] < r_sh[1]


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

#: Which region/frame keys each axis owns, so a partial recompute knows what to
#: replace and what to carry over from the previous file.
_AXIS_REGION_KEYS = {"face": ("face",), "action": ("action",), "depth": ("depth",),
                     "text": ("text",), "pose": ("track", "pose")}
_AXIS_FRAME_KEYS = {"face": ("faces",), "action": ("action",), "depth": ("depth_stats",),
                    "text": ("text",), "pose": ("n_tracks",)}


def _merge_axes(prev: Optional[dict], region_axes: dict, frame_axes: dict,
                axes: Sequence[str]) -> tuple[list[str], dict, dict]:
    """Carry over stored axes that this run did not recompute."""
    if not prev:
        return list(axes), region_axes, frame_axes
    kept = list(prev.get("axes") or [])
    for a in axes:
        if a not in kept:
            kept.append(a)
    stale = [a for a in kept if a not in axes]
    for ridx, old in (prev.get("regions") or {}).items():
        slot = region_axes.setdefault(ridx, {})
        for a in stale:
            for k in _AXIS_REGION_KEYS.get(a, ()):
                if k in old and k not in slot:
                    slot[k] = old[k]
    for a in stale:
        for k in _AXIS_FRAME_KEYS.get(a, ()):
            if k in (prev.get("frame") or {}) and k not in frame_axes:
                frame_axes[k] = prev["frame"][k]
    return kept, region_axes, frame_axes


def run_senses(store: IndexStore, *, axes: Optional[Sequence[str]] = None,
               frames: Optional[Sequence[int]] = None, force: bool = False) -> dict:
    """Compute and store the requested axes for the requested frames."""
    import cv2

    axes = [a for a in (axes or DEFAULT_AXES)]
    unknown = [a for a in axes if a not in AXIS_MODELS]
    if unknown:
        raise ValueError(f"unknown sense axis(es): {unknown}; known: {list(AXIS_MODELS)}")
    meta = store.require()
    all_frames = [int(f["frame_id"]) for f in meta["frames"]]
    t_of = {int(f["frame_id"]): float(f.get("t") or 0.0) for f in meta["frames"]}
    wanted = list(frames) if frames else all_frames
    for fid in wanted:
        if fid not in t_of:
            store.require_frame(fid)

    face = FaceSensor() if "face" in axes else None
    depth = DepthSensor() if "depth" in axes else None
    text = TextSensor() if "text" in axes else None
    pose = PoseSensor() if "pose" in axes else None
    action = ActionSensor(all_frames, store) if "action" in axes else None

    computed, reused, timings = [], [], {}
    t_all = time.time()
    for i, fid in enumerate(all_frames):
        if fid not in wanted:
            continue
        rec = store.require_regions(fid)
        regions = rec["regions"]
        sha = sha256_json(regions)
        prev = store.load_senses(fid)
        if prev and prev.get("regions_sha256") != sha:
            prev = None
        # ★ schema 也是缓存键：v2 修了 face slot 的 src 标错（expression 产的数
        #   被写成 mediapipe）。只看 regions_sha256 的话，旧的错值会一直被复用，
        #   而 region 表一个字节都没变 —— 这正是「静默复用错值」的经典形态。
        if prev is not None and prev.get("schema") != SCHEMA:
            prev = None
        if not force and prev is not None and set(axes) <= set(prev.get("axes") or []):
            reused.append(fid)
            continue
        img = cv2.imread(str(store.frame_image(fid)))
        if img is None:
            continue
        masks = store.load_masks(fid)
        t0 = time.time()
        region_axes, frame_axes = _sense_frame(
            fid, img, regions, masks, axes, face, depth, text, pose, action, i, t_of[fid])
        timings["_frame_s"] = timings.get("_frame_s", 0.0) + (time.time() - t0)
        # An axis is expensive; a second `veason senses --axes depth` must not throw
        # away the face/action work already on disk.
        kept_axes, region_axes, frame_axes = _merge_axes(prev, region_axes, frame_axes, axes)
        store.save_senses(fid, {
            "schema": SCHEMA,
            "frame_id": fid,
            "t": rec.get("t", t_of[fid]),
            "regions_sha256": sha,
            "axes": kept_axes,
            "models": {a: AXIS_MODELS[a] for a in kept_axes},
            "regions": region_axes,
            "frame": frame_axes,
            "computed_at": _now(),
        })
        computed.append(fid)

    n = max(len(computed), 1)
    out_timings = {
        "senses_s_per_frame": round(timings.get("_frame_s", 0.0) / n, 3),
        "frames_sensed": len(computed),
        "frames_reused": len(reused),
        "wall_s": round(time.time() - t_all, 2),
    }
    if action is not None and getattr(action, "_biases_patched", None) is not None:
        out_timings["videomae_attention_biases_patched"] = action._biases_patched
    return {
        "index_dir": store.describe()["index_dir"],
        "axes": axes,
        "frames": {"computed": computed, "reused": reused},
        "n_frames_sensed": len(computed),
        "n_frames_reused": len(reused),
        "timings": out_timings,
        "models": {a: AXIS_MODELS[a] for a in axes},
    }


def _sense_frame(fid, img, regions, masks, axes, face, depth, text, pose, action, i, t):
    region_axes: dict[str, dict] = {str(int(r["idx"])): {} for r in regions}
    frame_axes: dict = {}

    if face is not None:
        found = face.landmarks(img)
        frame_axes["faces"] = [{"bbox": f["bbox"]} for f in found]
        for f in found:
            if not f["bbox"]:
                continue
            ridx = _assign_region(f["bbox"], regions)
            if ridx is None:
                continue
            slot = region_axes[str(ridx)].setdefault("face", {"n_faces": 0})
            slot["n_faces"] += 1
            slot["face_bbox"] = f["bbox"]
            slot.update(f["blendshapes"])
            slot["src"] = "mediapipe-face-landmarker"
        # zero-shot expression head (mask-aware crops, biggest regions first)
        order = sorted(regions, key=lambda r: -float(r.get("area_ratio", 0.0)))
        chosen = [r for r in order
                  if float(r.get("area_ratio", 0.0)) >= EXPRESSION_MIN_AREA][:EXPRESSION_MAX_REGIONS]
        crops, keys = [], []
        from .recall import _crop_bgr

        for r in chosen:
            patch = _crop_bgr(img, r["bbox"], mask=masks.get(int(r["idx"])))
            if patch is None:
                continue
            from PIL import Image

            crops.append(Image.fromarray(np.ascontiguousarray(patch[:, :, ::-1])))
            keys.append(int(r["idx"]))
        for key, row in zip(keys, face.expression(crops)):
            slot = region_axes[str(key)].setdefault("face", {"n_faces": 0})
            slot.update(row)
            slot["expression_src"] = "clip-vitb32-expression-zeroshot"
            if "src" not in slot:
                # ★ 这个 slot 是【表情头】建出来的（MediaPipe 没在这块找到脸），
                #   所以产出这些数的是 CLIP，不是 MediaPipe。写 mediapipe 会让
                #   「多 provider 并存时标错出处」—— 规格 §4.3 纪律③ 明确禁止。
                #   两个件都答过的 slot 保留 blendshape 的 src，CLIP 那部分由
                #   expression_src 指认（evidence.py 按「读哪个字段」定 src，不靠这个键）。
                slot["src"] = "clip-vitb32-expression-zeroshot"

    if action is not None:
        top = action.predict(i)
        frame_axes["action"] = top
        if top:
            for key in region_axes:
                region_axes[key]["action"] = {
                    "label": top[0]["label"], "score": top[0]["score"], "top": top,
                    "src": "videomae-small-kinetics",
                    "note": "clip-level label over the 16 sampled frames centred here; "
                            "copied to every region of the frame",
                }

    if depth is not None:
        dmap = depth.map(img)
        stats = [(int(r["idx"]), DepthSensor.region_stats(dmap, r, masks.get(int(r["idx"]))))
                 for r in regions]
        ranked = sorted(stats, key=lambda kv: (-kv[1]["mean_rel"], kv[0]))
        rank_of = {idx: k + 1 for k, (idx, _) in enumerate(ranked)}
        for idx, s in stats:
            s = dict(s)
            s["rank_in_frame"] = rank_of[idx]
            region_axes[str(idx)]["depth"] = s
        frame_axes["depth_stats"] = {
            "min": 0.0, "max": 1.0,
            "median": round(float(np.median(dmap)), 4),
            "n_regions_ranked": len(regions),
            "src": "depth-anything-v2-small",
            "note": "relative inverse depth, min-max normalised per frame; larger = nearer",
        }

    if text is not None:
        found = text.read(img)
        frame_axes["text"] = found
        for item in found:
            ridx = _assign_region(item["box"], regions)
            if ridx is None:
                continue
            region_axes[str(ridx)].setdefault("text", []).append(item)

    if pose is not None:
        tracks, vel = pose.detect(img, t)
        frame_axes["n_tracks"] = len(tracks)
        for tr in tracks:
            ridx = _assign_region(tr["bbox"], regions)
            if ridx is None:
                continue
            v = next((x for x in vel if x["track_id"] == tr["track_id"]), {"track_id": tr["track_id"]})
            region_axes[str(ridx)]["track"] = dict(v)
            region_axes[str(ridx)]["pose"] = {
                "kpt17": [[round(float(v2), 2) for v2 in k] for k in tr["kpt17"]],
                "arms_raised": _arms_raised(tr["kpt17"]),
                "n_kpts": len(tr["kpt17"]),
                "score": tr["score"],
                "src": "yolo11n-pose",
            }
    return region_axes, frame_axes


def load_valid_senses(store: IndexStore, frame_id: int, regions: list[dict]) -> Optional[dict]:
    """The stored senses for a frame, only if they describe the *current* table."""
    data = store.load_senses(frame_id)
    if not data:
        return None
    if data.get("regions_sha256") != sha256_json(regions):
        return None
    return data


def read_senses(store: IndexStore, frame_id: int) -> dict:
    store.require_frame(frame_id)
    data = store.load_senses(frame_id)
    if data is None:
        from .errors import VrsError

        raise VrsError(
            f"no senses stored for frame {frame_id}",
            hint="run `veason senses <index_dir>` first (it computes every axis once and caches it).",
            code=3,
        )
    return data
