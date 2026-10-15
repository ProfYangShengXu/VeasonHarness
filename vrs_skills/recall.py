"""``vrs recall``: multi-route structured retrieval over an index.

The caller (an LLM) does natural-language understanding *before* calling this:
it turns "the laughing yellow dragon" into the structured query
``--label yellow "cartoon character" --relation "smiling at"``.  This module is
pure computation over the index -- no NLU, no vocabulary invented here -- so a
query is reproducible and its evidence (``hits``) is auditable.

Five routes, each named after the column of the region table it reads:

===============  ==========================  ==========================================
route            field                       what it costs
===============  ==========================  ==========================================
``label``        ``region.label``            free text match (phrase-coverage scored)
``relation``     ``relations/f*.json``       RelateAnything, *backfilled on demand*
``semantic``     region crop                 CLIP image/text cosine (attribute holes)
``geometry``     ``bbox`` / ``area_ratio``   free filter + ranking
``cooc``         relation edges              free: neighbours of an already-hit region
===============  ==========================  ==========================================

Round 3 adds six **sense routes** over the per-frame files written by
``vrs senses`` (see ``senses.py``).  They cost nothing at query time because
their models ran once, at senses time; ``recall`` only reads the files:

===============  ==========================  ==========================================
route            sense field                 query flag
===============  ==========================  ==========================================
``face``         ``face.mouth_smile_l/r``    ``--face-smile-min`` / ``--jaw-open-min``
``expression``   ``face.expression_scores``  ``--expression WORD`` / ``--expression-min``
``action``       ``action.label``            ``--action WORD`` / ``--action-min``
``depth``        ``depth.mean_rel``          ``--depth-near`` / ``--depth-min``
``text``         ``text[].s``                ``--text WORD`` / ``--text-min``
``pose``         ``track.track_id`` / pose   ``--track ID`` / ``--arms-raised``
===============  ==========================  ==========================================

``--no-face`` switches the whole face axis off (landmarker *and* expression
route); the other sense routes are active only when their own flag is given.

Ranks are fused with Reciprocal Rank Fusion (Cormack et al., SIGIR 2009):
``score = sum_routes 1 / (RRF_K + rank)``.  RRF consumes *ranks*, not scores,
so a cosine in [-1, 1], a blendshape in [0, 1] and a detector confidence in
[0, 1] need no hand-tuned weight.  A region that no content route surfaced is
never returned: an empty result is a result, and "the closest thing" is not.
"""
from __future__ import annotations

import math
import re
import time
from typing import Iterable, Optional, Sequence

import numpy as np

from .errors import VrsError
from .paths import ULTRALYTICS_WEIGHTS_DIR
from .store import IndexStore, sha256_json

#: RRF constant.  60 is the value from the original paper and the default of
#: every off-the-shelf implementation (e.g. PyTerrier ``RRFusion(k=60)``).
RRF_K = 60
DEFAULT_TOP_K = 12
DEFAULT_AREA_MIN = 0.05
#: Calibrated on the demo material: true matches land at 0.25-0.29, absent
#: objects at 0.18-0.21.  0.24 keeps "purple elephant" empty while "yellow
#: dragon" / "laughing mouth" pass.
DEFAULT_SEM_MIN = 0.24
DEFAULT_RELATE_MAX_FRAMES = 16
#: CLIP on CPU costs ~0.09 s/crop plus crop preparation; 96 keeps a full
#: semantic recall near 15 s on the demo machine while still covering every
#: region big enough to matter.  Raise it for many small targets.
DEFAULT_SEM_MAX_CROPS = 96

# --------------------------------------------------------------------------
# sense-route thresholds (round 3).  Every constant here has a provenance; a
# number with no source does not belong in this file.
# --------------------------------------------------------------------------

#: MediaPipe blendshape scores are calibrated to [0, 1] by the model and its
#: own documentation treats > 0.5 as "the expression is present", so 0.5 is the
#: model's convention rather than a value fitted here.
DEFAULT_FACE_SMILE_MIN = 0.5
DEFAULT_JAW_OPEN_MIN = 0.5
#: Zero-shot "laughing" on dragon.mp4: 0.42 / 0.52 / 0.51 / 0.44 on the four
#: laugh frames (13.83-14.83 s), 0.04-0.17 on every 6-8 s frame, and <= 0.34 on
#: every other frame of the clip.  0.35 sits just above that whole-clip
#: negative maximum, with a 1.2x margin to the weakest true positive.  Like
#: --sem-min it is fitted to this footage, not universal.
DEFAULT_EXPRESSION_MIN = 0.35
#: Kinetics classifiers on real footage return the right class around 0.2-0.6
#: and VideoMAE-small's softmax over 400 classes is flat; 0.10 keeps the route
#: from firing on noise.  (On the cartoon demo the labels are low-confidence
#: and not trustworthy -- see REPORT.md.)
DEFAULT_ACTION_MIN = 0.10
#: ``mean_rel`` is min-max normalised per frame, so 0.5 is that frame's
#: mid-depth: "nearer than half the frame's depth range".  Measured on the
#: dragon frame 850: character 0.53, dunes 0.03-0.39, sky 0.12.
DEFAULT_DEPTH_MIN = 0.5
#: RapidOCR's own recogniser is trained with a 0.5 rec threshold; below it the
#: recogniser itself would have called the box "not text".
DEFAULT_TEXT_MIN = 0.5

#: A multi-word ``--label`` phrase must cover this fraction of its own words on
#: the label, otherwise it is not a hit.  Round 2 treated *any* shared word as
#: a full hit, so "cartoon character" matched "character sculpture" (1 of 2)
#: exactly like an exact label would have.
LABEL_MIN_COVERAGE = 0.5

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def _rank(scored: dict) -> dict:
    """``{(fid, idx): (raw, hit)}`` -> ``{key: {"rank", "raw", "hit"}}``.

    Deterministic tie-break on the key so two runs over one index agree
    byte-for-byte (tests assert stability).
    """
    order = sorted(scored.items(), key=lambda kv: (-float(kv[1][0]), kv[0]))
    return {key: {"rank": i + 1, "raw": float(raw), "hit": hit}
            for i, (key, (raw, hit)) in enumerate(order)}


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

def _label_route(eligible: list[dict], labels: Sequence[str]) -> dict:
    """Phrase-coverage label match.

    Three tiers, in decreasing strength:

    1. the query *is* the label (or a whole-phrase substring of it) -> full;
    2. the query's words are all covered by the label's words -> full;
    3. a partial word overlap -> kept only when it covers at least
       ``LABEL_MIN_COVERAGE`` of the query's own words, and ranked below the
       full tiers by that coverage (``raw`` grows with coverage).  ``hits``
       spells the fraction out (``label:character(1/2)``) so a caller can see
       it is a partial match rather than a complete one.
    """
    queries = [q.strip().lower() for q in labels if q and q.strip()]
    if not queries:
        return {}
    scored: dict = {}
    for item in eligible:
        region = item["region"]
        lab = str(region.get("label", "")).lower()
        lab_tokens = set(_tokens(lab))
        best = None  # (raw_without_score, quality, coverage, token)
        for q in queries:
            q_tokens = set(_tokens(q))
            if q == lab:
                cand = (3.0, 3, 1.0, lab)
            elif q in lab or lab in q:
                cand = (2.0, 2, 1.0, q)
            else:
                overlap = lab_tokens & q_tokens
                if not overlap or not q_tokens:
                    continue
                coverage = len(overlap) / len(q_tokens)
                if coverage < LABEL_MIN_COVERAGE:
                    continue
                token = sorted(overlap)[0]
                if coverage < 1.0:
                    token = f"{token}({len(overlap)}/{len(q_tokens)})"
                cand = (1.0 + coverage * 0.5, 1, coverage, token)
            if best is None or cand[0] > best[0]:
                best = cand
        if best is None:
            continue
        key = (item["frame_id"], int(region["idx"]))
        raw = best[0] + min(float(region.get("score", 0.0)), 0.999) / 1000.0
        if raw > scored.get(key, (-1.0, ""))[0]:
            scored[key] = (raw, f"label:{best[3]}")
    return _rank(scored)


class _ClipScorer:
    """CLIP ViT-B/32 image+text, loaded lazily and cached for the process.

    The YOLOE asset (``mobileclip_blt.ts``) is the *full* MobileCLIP, but the
    file was traced for its text tower only: ``image_encoder`` is present as
    dead parameters and exposes no scripted ``forward``, so it cannot embed a
    region crop.  OpenAI CLIP is already installed (``vrs detect`` needs it)
    and its weights are cached next to the other models, so the semantic route
    adds no framework dependency and no second text encoder choice.
    """

    _cache = None

    def __init__(self) -> None:
        if _ClipScorer._cache is None:
            import warnings

            import clip
            root = str(ULTRALYTICS_WEIGHTS_DIR / "clip")
            # `clip.load` goes through `torch.jit.load`, which prints a
            # deprecation FutureWarning on stderr.  stderr is a JSON channel
            # here (warnings + errors), so keep it parseable.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                model, pre = clip.load("ViT-B/32", device="cpu", download_root=root, jit=False)
            model.eval()
            _ClipScorer._cache = (model, pre, clip)
        self.model, self.pre, self._clip = _ClipScorer._cache

    def encode_text(self, phrases: Sequence[str]) -> np.ndarray:
        import torch
        tokens = self._clip.tokenize([p for p in phrases])
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.numpy()

    def encode_crops(self, pil_images: list, batch: int = 16) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(pil_images), batch):
                x = torch.stack([self.pre(im) for im in pil_images[i:i + batch]])
                feats = self.model.encode_image(x)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), np.float32)


def _crop_bgr(img: np.ndarray, bbox: Sequence[int], pad: int = 4,
              mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Region crop, mask-aware when a mask exists.

    A detector bbox on a cartoony video often contains a large slice of
    background, and CLIP reads that background: measured on the demo clips,
    the dragon's masked crop scores 0.26-0.28 for "a yellow cartoon dragon"
    while the same-size dune background scores ~0.20.  So when the index has a
    mask, crop to the mask's extent and paint the outside neutral gray; else
    fall back to the box.
    """
    h, w = img.shape[:2]
    use_mask = mask is not None
    if use_mask:
        mask = np.asarray(mask)
        if mask.shape[:2] != (h, w) or not mask.any():
            use_mask = False
        else:
            ys, xs = np.where(mask)
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    if not use_mask:
        box = [int(v) for v in bbox]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = img[y1:y2, x1:x2]
    if use_mask:
        patch = patch.copy()
        patch[~mask[y1:y2, x1:x2]] = 127  # neutral gray, not the background
    return patch


def _semantic_route(store: IndexStore, eligible: list[dict], phrases: Sequence[str],
                    sem_min: float, max_crops: int) -> tuple[dict, dict]:
    """Region crop vs query text; ``max_crops`` bounds the model cost."""
    phrases = [p for p in dict.fromkeys(p.strip() for p in phrases) if p]
    info = {"phrases": phrases, "n_crops": 0, "max_crops": max_crops, "threshold": sem_min}
    if not phrases or not eligible:
        return {}, info
    # Bigger regions carry more signal and are fewer; take them first.
    chosen = sorted(eligible, key=lambda it: (-float(it["region"].get("area_ratio", 0.0)),
                                              it["frame_id"], int(it["region"]["idx"])))[:max_crops]
    import cv2
    from PIL import Image

    crops: list = []
    keys: list = []
    cache_img: dict[int, np.ndarray] = {}
    cache_masks: dict[int, dict] = {}
    for item in chosen:
        fid = item["frame_id"]
        if fid not in cache_img:
            cache_img[fid] = cv2.imread(str(store.frame_image(fid)))
            cache_masks[fid] = store.load_masks(fid)
        img = cache_img[fid]
        if img is None:
            continue
        idx = int(item["region"]["idx"])
        patch = _crop_bgr(img, item["region"]["bbox"], mask=cache_masks[fid].get(idx))
        if patch is None:
            continue
        crops.append(Image.fromarray(patch[..., ::-1]))
        keys.append((fid, idx))
    if not crops:
        return {}, info
    scorer = _ClipScorer()
    t0 = time.time()
    img_feats = scorer.encode_crops(crops)
    txt_feats = scorer.encode_text(phrases)
    info["elapsed_s"] = round(time.time() - t0, 2)
    info["n_crops"] = len(crops)
    sims = img_feats @ txt_feats.T                       # [n_crops, n_phrases]
    scored: dict = {}
    for k, key in enumerate(keys):
        j = int(np.argmax(sims[k]))
        raw = float(sims[k, j])
        if raw < sem_min:
            continue
        scored[key] = (raw, f"sem:{raw:.2f}")
    info["n_above_threshold"] = len(scored)
    return _rank(scored), info


def _relation_matches(relation: str, phrases: Sequence[str]) -> bool:
    if not phrases:
        return True
    rt = set(_tokens(relation))
    for p in phrases:
        pt = set(_tokens(p))
        if pt and (rt & pt):
            return True
    return False


def _elect_candidate_frames(eligible: list[dict], cheap: dict,
                            relate_max_frames: int) -> list[int]:
    """Frames worth spending RelateAnything on.

    Cheap routes (label + semantic) already rank the frames; only their top
    ``relate_max_frames`` are related.  When those routes found nothing at all,
    fall back to the largest-area frames so the relation route still has a
    bounded, deterministic set to look at.
    """
    by_frame: dict[int, float] = {}
    if cheap:
        for key, entry in cheap.items():
            by_frame[key[0]] = max(by_frame.get(key[0], -math.inf), float(entry["raw"]))
    else:
        for item in eligible:
            by_frame[item["frame_id"]] = max(
                by_frame.get(item["frame_id"], -math.inf),
                float(item["region"].get("area_ratio", 0.0)))
    order = sorted(by_frame.items(), key=lambda kv: (-kv[1], kv[0]))
    return [fid for fid, _ in order[:max(relate_max_frames, 1)]]


def _ensure_relation_sets(store: IndexStore, frames: Sequence[int], relations: Sequence[str],
                          topk: int) -> tuple[dict, dict]:
    """Backfill the requested vocabulary, one frame at a time, and reuse cache."""
    from .relation import Relater

    edges_by_frame: dict[int, list] = {}
    computed: list[int] = []
    reused: list[int] = []
    skipped: list[int] = []
    relater = None
    model_load_s = 0.0
    for fid in frames:
        rec = store.require_regions(fid)
        regions_sha = sha256_json(rec["regions"])
        cached = store.find_relation_set(fid, regions_sha, relations)
        if cached is not None:
            edges_by_frame[fid] = cached.get("edges", [])
            reused.append(fid)
            continue
        if len(rec["regions"]) < 2:
            store.append_relation_set(fid, regions_sha, relations, [],
                                      {"n_regions": len(rec["regions"]),
                                       "n_used": len(rec["regions"]), "truncated": False,
                                       "reason": "RelateAnything needs at least 2 regions"})
            edges_by_frame[fid] = []
            computed.append(fid)
            continue
        import cv2
        img = cv2.imread(str(store.frame_image(fid)))
        if img is None:
            skipped.append(fid)
            continue
        if relater is None:
            t0 = time.time()
            relater = Relater(predicates=relations)
            model_load_s = round(time.time() - t0, 2)
        edges, info = relater.relate(img, rec["regions"], store.load_masks(fid),
                                     topk=topk, min_score=0.0)
        store.append_relation_set(fid, regions_sha, info["vocabulary"], edges, info)
        edges_by_frame[fid] = edges
        computed.append(fid)
    return edges_by_frame, {
        "frames_related": computed, "n_frames_related": len(computed),
        "frames_reused": reused, "n_frames_reused": len(reused),
        "frames_skipped": skipped, "n_frames_considered": len(frames),
        "model_load_s": model_load_s,
    }


def _relation_route(edges_by_frame: dict, relations: Sequence[str], relation_min: float) -> dict:
    scored: dict = {}
    for fid, edges in edges_by_frame.items():
        for e in edges:
            raw = float(e.get("score", 0.0))
            if raw < relation_min:
                continue
            if not _relation_matches(str(e.get("relation", "")), relations):
                continue
            for idx in (e.get("subject"), e.get("object")):
                key = (fid, int(idx))
                if raw > scored.get(key, (-1.0, ""))[0]:
                    scored[key] = (raw, f"rel:{e.get('relation')}:{raw:.2f}")
    return _rank(scored)


def _cooccurrence_route(store: IndexStore, hits: set, relation_min: float) -> dict:
    """Neighbours of an already-hit region, read off the frame's relation graph."""
    scored: dict = {}
    for fid in sorted({k[0] for k in hits}):
        rec = store.require_regions(fid)
        sha = sha256_json(rec["regions"])
        data = store.load_relations(fid)
        if not data or data.get("regions_sha256") != sha:
            continue
        for e in data.get("edges", []):
            raw = float(e.get("score", 0.0))
            if raw < relation_min:
                continue
            sub, obj = int(e["subject"]), int(e["object"])
            for a, b in ((sub, obj), (obj, sub)):
                if (fid, a) in hits and (fid, b) not in hits:
                    key = (fid, b)
                    hit = f"cooc:{e.get('relation')}"
                    if raw > scored.get(key, (-1.0, ""))[0]:
                        scored[key] = (raw, hit)
    return _rank(scored)


def _geometry_route(eligible: list[dict], content_keys: set) -> dict:
    """Area ranking, restricted to regions a content route already surfaced."""
    scored: dict = {}
    for item in eligible:
        key = (item["frame_id"], int(item["region"]["idx"]))
        if key not in content_keys:
            continue
        raw = float(item["region"].get("area_ratio", 0.0))
        scored[key] = (raw, f"area:{raw:.2f}")
    return _rank(scored)


# --------------------------------------------------------------------------
# sense routes (round 3): pure reads of ``senses/f*.json``
# --------------------------------------------------------------------------

#: Canonical fusion order.  Every route must appear here or be discovered from
#: the ``routes`` dict -- a hardcoded tuple that silently drops a new route was
#: exactly the trap this list replaces.
SENSE_ROUTE_ORDER = ("label", "semantic", "relation", "cooccurrence", "geometry",
                     "face", "expression", "action", "depth", "text", "pose")


def _sense_table(store: IndexStore, eligible: list[dict]) -> tuple[dict, dict]:
    """``{(fid, idx): region_senses}`` for frames whose senses still match.

    Staleness rule (same as relations): the file is only used when its
    ``regions_sha256`` equals the hash of the *current* region table, so a
    ``detect`` that appended a region cannot leak a field computed for the
    old table.
    """
    table: dict = {}
    with_senses = set()
    known = {}
    for item in eligible:
        fid = item["frame_id"]
        if fid not in known:
            rec = store.load_regions(fid) or {"regions": []}
            data = store.load_senses(fid)
            if data and data.get("regions_sha256") == sha256_json(rec.get("regions", [])):
                known[fid] = data
                with_senses.add(fid)
            else:
                known[fid] = None
        data = known[fid]
        if not data:
            continue
        ridx = str(int(item["region"]["idx"]))
        table[(fid, int(item["region"]["idx"]))] = (data.get("regions") or {}).get(ridx, {})
    return table, {"frames_with_senses": len(with_senses), "frames_considered": len(known)}


def _sym(v) -> float:
    try:
        return max(0.0, float(v))
    except (TypeError, ValueError):
        return 0.0


def _face_route(sense: dict, smile_min: Optional[float], jaw_min: Optional[float]) -> dict:
    if smile_min is None and jaw_min is None:
        return {}
    scored: dict = {}
    for key, s in sense.items():
        f = s.get("face") or {}
        cands = []
        if smile_min is not None:
            v = max(_sym(f.get("mouth_smile_l")), _sym(f.get("mouth_smile_r")))
            if v >= smile_min:
                cands.append((v, f"smile:{v:.2f}"))
        if jaw_min is not None:
            v = _sym(f.get("jaw_open"))
            if v >= jaw_min:
                cands.append((v, f"jaw:{v:.2f}"))
        if cands:
            scored[key] = max(cands)
    return _rank(scored)


def _expression_route(sense: dict, word: Optional[str], expr_min: float) -> dict:
    if not word:
        return {}
    want = word.strip().lower().replace(" ", "_")
    scored: dict = {}
    for key, s in sense.items():
        f = s.get("face") or {}
        scores = {str(k).lower(): _sym(v)
                  for k, v in (f.get("expression_scores") or {}).items()}
        if f.get("expression") and want not in scores:
            scores[str(f["expression"]).lower()] = _sym(f.get("expression_score"))
        v = scores.get(want)
        if v is None or v < expr_min:
            continue
        scored[key] = (v, f"expr:{want}:{v:.2f}")
    return _rank(scored)


def _action_route(sense: dict, words: Sequence[str], action_min: float) -> dict:
    want = [w.strip().lower() for w in words if w and w.strip()]
    if not want:
        return {}
    scored: dict = {}
    for key, s in sense.items():
        a = s.get("action") or {}
        lab = str(a.get("label", "")).lower()
        v = _sym(a.get("score"))
        if v < action_min:
            continue
        if not any(w == lab or w in lab for w in want):
            continue
        scored[key] = (v, f"act:{lab}:{v:.2f}")
    return _rank(scored)


def _depth_route(sense: dict, enabled: bool, depth_min: float) -> dict:
    if not enabled:
        return {}
    scored: dict = {}
    for key, s in sense.items():
        d = s.get("depth") or {}
        v = _sym(d.get("mean_rel"))
        if v < depth_min:
            continue
        scored[key] = (v, f"depth:{v:.2f}#{d.get('rank_in_frame', '?')}")
    return _rank(scored)


def _text_route(sense: dict, words: Sequence[str], text_min: float) -> dict:
    want = [w.strip().lower() for w in words if w and w.strip()]
    if not want:
        return {}
    scored: dict = {}
    for key, s in sense.items():
        best = None
        for it in (s.get("text") or []):
            t = str(it.get("s", "")).lower()
            sc = _sym(it.get("score"))
            if sc < text_min or not any(w in t for w in want):
                continue
            if best is None or sc > best[0]:
                best = (sc, t)
        if best:
            scored[key] = (best[0], f"text:{best[1]}:{best[0]:.2f}")
    return _rank(scored)


def _pose_route(sense: dict, track: Optional[int], arms_raised: bool) -> dict:
    if track is None and not arms_raised:
        return {}
    scored: dict = {}
    for key, s in sense.items():
        pose = s.get("pose") or {}
        tr = s.get("track") or {}
        if track is not None and int(tr.get("track_id", -1)) == track:
            speed = abs(_sym(tr.get("dx_per_s"))) + abs(_sym(tr.get("dy_per_s")))
            scored[key] = (speed, f"track:{track}:v={speed:.0f}px/s")
        if arms_raised and pose.get("arms_raised"):
            scored.setdefault(key, (1.0, "pose:arms_raised"))
    return _rank(scored)


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def fuse(routes: dict, rrf_k: int = RRF_K) -> dict:
    """RRF over ``{route: {key: {"rank", "raw", "hit"}}}``.

    Iterates the union of the canonical order and whatever the caller put in
    ``routes``: a route can never be computed and then silently dropped.
    """
    names = [n for n in SENSE_ROUTE_ORDER if n in routes]
    names += [n for n in routes if n not in SENSE_ROUTE_ORDER]
    fused: dict = {}
    for name in names:
        table = routes.get(name) or {}
        for key, entry in table.items():
            slot = fused.setdefault(key, {"score": 0.0, "hits": [], "routes": {}})
            slot["score"] += 1.0 / (rrf_k + int(entry["rank"]))
            slot["hits"].append(entry["hit"])
            slot["routes"][name] = {"rank": int(entry["rank"]), "raw": round(float(entry["raw"]), 4)}
    return fused


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def run_recall(store: IndexStore, *, labels: Sequence[str] = (), relations: Sequence[str] = (),
               area_min: float = DEFAULT_AREA_MIN,
               frame_min: Optional[int] = None, frame_max: Optional[int] = None,
               top_k: int = DEFAULT_TOP_K, use_clip: bool = True, explain: bool = False,
               relate_max_frames: int = DEFAULT_RELATE_MAX_FRAMES,
               sem_min: float = DEFAULT_SEM_MIN, relation_min: float = 0.0,
               sem_max_crops: int = DEFAULT_SEM_MAX_CROPS,
               relate_topk: int = 20,
               no_face: bool = False,
               face_smile_min: Optional[float] = None,
               jaw_open_min: Optional[float] = None,
               expression: Optional[str] = None,
               expression_min: float = DEFAULT_EXPRESSION_MIN,
               actions: Sequence[str] = (),
               action_min: float = DEFAULT_ACTION_MIN,
               depth_near: bool = False,
               depth_min: float = DEFAULT_DEPTH_MIN,
               texts: Sequence[str] = (),
               text_min: float = DEFAULT_TEXT_MIN,
               track: Optional[int] = None,
               arms_raised: bool = False) -> tuple[list[dict], dict]:
    labels = [x for x in labels if x and x.strip()]
    relations = [x for x in relations if x and x.strip()]
    actions = [x for x in actions if x and x.strip()]
    texts = [x for x in texts if x and x.strip()]
    # A face query with --no-face is still a query: the caller asked for the
    # face axis and deliberately switched it off, so the honest answer is an
    # empty result, not a usage error.
    asked_face = (face_smile_min is not None or jaw_open_min is not None or bool(expression))
    if no_face:
        face_smile_min = jaw_open_min = None
        expression = None
    sense_query = (asked_face or bool(actions) or depth_near or bool(texts)
                   or track is not None or arms_raised)
    if not labels and not relations and not sense_query and frame_min is None and frame_max is None:
        raise VrsError(
            "recall needs a query",
            hint='usage: vrs recall <index_dir> --label "yellow" "dragon" '
                 '[--relation "smiling at"] [--expression laughing] [--area-min 0.05]. '
                 "The LLM turns the question into these structured fields; the CLI never parses NL.",
            code=2,
        )

    meta = store.require()
    eligible: list[dict] = []
    for f in meta.get("frames", []):
        fid = int(f["frame_id"])
        if frame_min is not None and fid < frame_min:
            continue
        if frame_max is not None and fid > frame_max:
            continue
        rec = store.load_regions(fid) or {"regions": []}
        for r in rec.get("regions", []):
            if float(r.get("area_ratio", 0.0)) >= area_min:
                eligible.append({"frame_id": fid, "t": f.get("t"),
                                 "region": r})

    label_table = _label_route(eligible, labels)
    semantic_table: dict = {}
    semantic_info: dict = {}
    if use_clip:
        try:
            semantic_table, semantic_info = _semantic_route(
                store, eligible, list(labels) + list(relations), sem_min, sem_max_crops)
        except Exception as exc:  # noqa: BLE001 - degrade, but say so loudly
            semantic_info = {"error": f"{type(exc).__name__}: {exc}", "phrases": []}
    else:
        semantic_info = {"disabled": True, "phrases": []}

    relation_table: dict = {}
    cooc_table: dict = {}
    backfill: dict | None = None
    if relations:
        cheap = {**label_table, **semantic_table}
        candidate_frames = _elect_candidate_frames(eligible, cheap, relate_max_frames)
        edges_by_frame, backfill = _ensure_relation_sets(store, candidate_frames, relations, relate_topk)
        relation_table = _relation_route(edges_by_frame, relations, relation_min)
        cooc_table = _cooccurrence_route(store, set(label_table) | set(semantic_table), relation_min)

    # ---- sense routes: disk reads only, no model in this process ----------
    sense, senses_info = _sense_table(store, eligible)
    face_table = {} if no_face else _face_route(sense, face_smile_min, jaw_open_min)
    expression_table = {} if no_face else _expression_route(sense, expression, expression_min)
    action_table = _action_route(sense, actions, action_min)
    depth_table = _depth_route(sense, depth_near, depth_min)
    text_table = _text_route(sense, texts, text_min)
    pose_table = _pose_route(sense, track, arms_raised)

    content_keys = (set(label_table) | set(semantic_table) | set(relation_table) | set(cooc_table)
                    | set(face_table) | set(expression_table) | set(action_table)
                    | set(depth_table) | set(text_table) | set(pose_table))
    geometry_table = _geometry_route(eligible, content_keys)
    fused = fuse({"label": label_table, "semantic": semantic_table, "relation": relation_table,
                  "cooccurrence": cooc_table, "geometry": geometry_table,
                  "face": face_table, "expression": expression_table, "action": action_table,
                  "depth": depth_table, "text": text_table, "pose": pose_table})

    by_key = {(it["frame_id"], int(it["region"]["idx"])): it for it in eligible}
    ordered = sorted(fused.items(), key=lambda kv: (-kv[1]["score"], kv[0][0], kv[0][1]))
    results: list[dict] = []
    for key, slot in ordered[:max(top_k, 0)]:
        item = by_key.get(key)
        if item is None:
            continue
        region = item["region"]
        row = {
            "frame_id": key[0],
            "region_idx": key[1],
            "score": round(slot["score"], 4),
            "hits": slot["hits"],
            "label": region.get("label"),
            "bbox": list(region.get("bbox", [])),
            "t": item.get("t"),
            "area_ratio": round(float(region.get("area_ratio", 0.0)), 6),
        }
        if explain:
            row["routes"] = slot["routes"]
            senses = (sense.get(key) or {})
            if senses:
                row["senses"] = senses
        results.append(row)

    all_tables = {
        "label": label_table, "semantic": semantic_table, "relation": relation_table,
        "cooccurrence": cooc_table, "geometry": geometry_table,
        "face": face_table, "expression": expression_table, "action": action_table,
        "depth": depth_table, "text": text_table, "pose": pose_table,
    }
    senses_missing = []
    if sense_query and senses_info["frames_with_senses"] == 0 and eligible:
        senses_missing = ["no frame has senses matching its current region table"]
    info = {
        "video": meta.get("video"),
        "n_regions_considered": len(eligible),
        "routes": {name: len(t) for name, t in all_tables.items()},
        "semantic": semantic_info,
        "relation_backfill": backfill,
        "senses": senses_info,
        "senses_unavailable": senses_missing or None,
        "empty": not results,
        "empty_reason": None if results else (
            "no route surfaced a candidate above its threshold; an empty recall is a result, "
            "not an error (lower --sem-min / --relation-min / --expression-min only if you mean to)"),
    }
    return results, info
