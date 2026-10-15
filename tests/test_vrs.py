"""Self-checks for vrs-skills.  No pytest needed:

    python tests/test_vrs.py            # fast: synthetic index, no models
    python tests/test_vrs.py --slow     # also runs the model-backed reject path

The point of these tests is not that the happy path works (the acceptance run
shows that).  It is that every assertion here has been *observed to fail* when
the thing it guards is broken -- tampered regions, a mask that escapes its box,
a naive ``round()`` -- because a check that cannot fail is not a check.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vrs_skills.answers import keyframe_timestamp  # noqa: E402
from vrs_skills.cli import build_parser  # noqa: E402
from vrs_skills.recall import RRF_K, fuse  # noqa: E402
from vrs_skills.store import IndexStore, sha256_json  # noqa: E402

PY = sys.executable
VRS = str(ROOT / "vrs.py")
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def run(*args: str) -> tuple[int, str, str]:
    p = subprocess.run([PY, VRS, *args], capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout, p.stderr


def make_index(root: Path, t: float = 2.5, bbox=(10, 20, 110, 120)) -> IndexStore:
    """A 2-region, 1-frame index with real files, built without any model."""
    store = IndexStore(root)
    img = np.full((160, 200, 3), 60, np.uint8)
    img[20:120, 10:110] = 200
    store.frame_image_path(7).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(store.frame_image_path(7)), img)
    mask = np.zeros((160, 200), bool)
    mask[20:120, 10:110] = True
    store.save_masks(7, {0: mask})
    regions = [
        {"idx": 0, "label": "person", "score": 0.9, "bbox": list(bbox),
         "area_ratio": 0.3, "has_mask": True, "source": "test"},
        {"idx": 1, "label": "horse", "score": 0.8, "bbox": [120, 30, 190, 150],
         "area_ratio": 0.2, "has_mask": False, "source": "test"},
    ]
    store.save_regions(7, {"frame_id": 7, "t": t, "width": 200, "height": 160,
                           "detector": {}, "next_idx": 2, "regions": regions})
    store.save_meta({
        "schema": 1, "video": "synthetic.mp4", "video_sha256": "deadbeef", "fps": 12.0,
        "n_video_frames": 100, "duration_s": 8.0, "width": 200, "height": 160, "n_frames": 1,
        "detector": {}, "relation": {},
        "frames": [{"frame_id": 7, "t": t, "n_regions": 2,
                    "image": str(store.frame_image_path(7)), "regions_file": ""}],
        "timings": {},
    })
    return store


def test_timestamp_is_half_up() -> None:
    # 2.5 -> 3 is the discriminating case: Python's round() would give 2.
    check("timestamp 2.5 -> 3 (round() would give 2)", keyframe_timestamp(2.5) == 3)
    check("timestamp 0.5 -> 1", keyframe_timestamp(0.5) == 1)
    check("timestamp 1.5 -> 2", keyframe_timestamp(1.5) == 2)
    check("timestamp 29.75 -> 30", keyframe_timestamp(29.75) == 30)
    check("timestamp is int", isinstance(keyframe_timestamp(3.2), int))


def test_merge_is_idempotent(tmp: Path) -> None:
    store = make_index(tmp / "merge")
    existing = store.require_regions(7)["regions"]
    same = {"label": "person", "score": 0.95, "bbox": [11, 21, 111, 121],
            "area_ratio": 0.3, "has_mask": False, "source": "detect"}
    merged, added = store.merge_regions(existing, [same])
    check("duplicate detect adds nothing", added == [])
    check("duplicate detect keeps idxs", [r["idx"] for r in merged] == [0, 1])
    check("duplicate detect keeps the old bbox", merged[0]["bbox"] == [10, 20, 110, 120])
    new = {"label": "hat", "score": 0.5, "bbox": [40, 40, 60, 60], "area_ratio": 0.01,
           "has_mask": False, "source": "detect"}
    merged2, added2 = store.merge_regions(merged, [new])
    check("new label gets the next idx", [r["idx"] for r in added2] == [2])
    check("existing idxs untouched", [r["idx"] for r in merged2[:2]] == [0, 1])


def test_regions_are_stable(tmp: Path) -> None:
    store = make_index(tmp / "stable")
    rc1, out1, _ = run("regions", str(store.root), "7")
    rc2, out2, _ = run("regions", str(store.root), "7")
    check("regions exits 0", rc1 == 0 and rc2 == 0)
    check("regions output byte-identical across calls", out1 == out2)


def test_answer_shape(tmp: Path) -> None:
    store = make_index(tmp / "answer")
    rc, out, err = run("answer", str(store.root), "7", "0", "1")
    check("answer exits 0", rc == 0, err)
    payload = json.loads(out)
    check("answer has exactly the official keys",
          set(payload) == {"keyframe_timestamp", "bbox_2d_list"}, str(list(payload)))
    check("answer timestamp is int seconds", payload["keyframe_timestamp"] == 3)
    check("answer bbox matches the region table",
          payload["bbox_2d_list"] == [[10, 20, 110, 120], [120, 30, 190, 150]])


def test_verify_detects_tampering(tmp: Path) -> None:
    store = make_index(tmp / "tamper")
    rc, out, err = run("answer", str(store.root), "7", "0")
    check("answer before tamper exits 0", rc == 0, err)
    rc, out, err = run("verify", str(store.root), "7", "0")
    check("verify passes on untouched data", rc == 0 and json.loads(out)["ok"] is True, err)

    rec = store.require_regions(7)
    rec["regions"][0]["bbox"] = [12, 20, 110, 120]      # 2 px shift, same shape
    store.save_regions(7, rec)

    rc, out, err = run("verify", str(store.root), "7", "0")
    payload = json.loads(out)
    check("verify FAILS after a 2px bbox tamper", rc == 4 and payload["ok"] is False, f"rc={rc}")
    check("verify reports the offending idx",
          any(not c["ok"] and c["check"] == "answer_reproducible_from_current_regions"
              for c in payload["checks"]))
    check("verify failure carries a hint", bool(payload.get("hint")))


def test_verify_detects_mask_escaping_bbox(tmp: Path) -> None:
    store = make_index(tmp / "maskescape")
    run("answer", str(store.root), "7", "0")
    store.save_masks(7, {0: np.ones((160, 200), bool)})   # mask >> bbox
    rc, out, _ = run("verify", str(store.root), "7", "0")
    payload = json.loads(out)
    mask_check = [c for c in payload["checks"] if c["check"] == "mask_within_region_bbox"][0]
    check("verify FAILS when the mask escapes its bbox", rc == 4 and mask_check["ok"] is False,
          f"rc={rc} {mask_check}")
    check("mask check reports the outside distance",
          mask_check["per_idx"][0]["outside_px"] >= 40, str(mask_check["per_idx"][0]))


def test_guided_failures(tmp: Path) -> None:
    store = make_index(tmp / "guided")
    rc, out, err = run("regions", str(tmp / "does-not-exist"), "0")
    body = json.loads(err)
    check("missing index dir -> exit 3", rc == 3, f"rc={rc}")
    check("missing index dir -> stdout empty", out == "")
    check("missing index dir -> hint present", "vrs index" in body.get("hint", ""))

    rc, out, err = run("regions", str(store.root), "999")
    body = json.loads(err)
    check("unknown frame_id -> exit 3 + hint", rc == 3 and "frame_ids" in body.get("hint", ""))

    rc, out, err = run("seg", str(store.root), "7", "99")
    body = json.loads(err)
    check("unknown region_idx -> exit 3 + hint", rc == 3 and "valid: 0..1" in body.get("error", ""))

    rc, out, err = run("verify", str(store.root), "7", "1")
    body = json.loads(err)
    check("verify without an answer -> exit 2 + hint", rc == 2 and "vrs answer" in body.get("hint", ""))

    rc, out, err = run("answer", str(store.root), "7")
    check("answer without idxs -> exit 2", rc == 2)


def test_no_traceback_leaks(tmp: Path) -> None:
    """Every failure must be a JSON object, never a Python traceback."""
    store = make_index(tmp / "leak")
    for args in (("regions", str(tmp / "nope"), "0"),
                 ("crop", str(store.root), "7", "99"),
                 ("index", str(tmp / "no-such-video.mp4"))):
        rc, out, err = run(*args)
        check(f"no traceback for: {' '.join(args)}", "Traceback" not in err, err[:200])


def test_seg_and_crop(tmp: Path) -> None:
    store = make_index(tmp / "render")
    for cmd in ("show", "crop", "seg"):
        args = [cmd, str(store.root), "7"] + ([] if cmd == "show" else ["0"])
        rc, out, err = run(*args)
        payload = json.loads(out)
        check(f"{cmd} exits 0", rc == 0, err)
        check(f"{cmd} wrote a file", Path(payload["out_win"]).is_file(), payload.get("out", ""))
    rc, out, _ = run("seg", str(store.root), "7", "1")     # region 1 has no mask
    check("seg without a mask reports box-only", json.loads(out)["mask_source"] == "box-only")


def test_slow_reject_path() -> None:
    """Model-backed: a phrase that cannot be in the video must yield [] , exit 0."""
    idx = ROOT / "index"
    if not idx.exists():
        print("SKIP  slow reject path (no index/ built)")
        return
    candidates = [p for p in sorted(idx.iterdir()) if (p / "index.json").is_file()]
    if not candidates:
        print("SKIP  slow reject path (no index.json under index/)")
        return
    index_dir = candidates[0]
    meta = json.loads((index_dir / "index.json").read_text())
    fid = meta["frames"][0]["frame_id"]
    rc, out, err = run("detect", str(index_dir), str(fid), "a purple elephant")
    check("detect on an absent target exits 0", rc == 0, err)
    check("detect on an absent target returns []", json.loads(out) == [], out)


# --------------------------------------------------------------------------
# recall (round 2)
# --------------------------------------------------------------------------

def test_recall_label_and_empty(tmp: Path) -> None:
    store = make_index(tmp / "recall")
    rc, out, err = run("recall", str(store.root), "--label", "person", "--no-clip", "--area-min", "0", "--explain")
    check("recall exits 0", rc == 0, err)
    rows = json.loads(out)
    hit = [r for r in rows if r["region_idx"] == 0]
    check("label route surfaced the person region", bool(hit), out)
    check("hits names the label field", bool(hit) and any(h.startswith("label:person") for h in hit[0]["hits"]),
          str(hit[:1]))
    check("explain adds per-route ranks", bool(hit) and "routes" in hit[0])

    rc, out, err = run("recall", str(store.root), "--label", "purple elephant", "--no-clip", "--area-min", "0")
    check("absent target exits 0", rc == 0, err)
    check("absent target returns []", json.loads(out) == [], out)
    rc, out, err = run("recall", str(store.root), "--label", "purple elephant", "--no-clip", "--area-min", "0")
    check("absent target writes a machine-readable empty reason", "recall_empty" in err, err)


def test_recall_geometry_filters(tmp: Path) -> None:
    store = make_index(tmp / "recallgeo")
    rc, out, _ = run("recall", str(store.root), "--label", "person", "--no-clip", "--area-min", "0", "--frame-min", "8")
    check("--frame-min past the only frame -> []", rc == 0 and json.loads(out) == [], out)
    rc, out, _ = run("recall", str(store.root), "--label", "person", "--no-clip", "--area-min", "0", "--frame-min", "7")
    check("--frame-min including the frame keeps it", rc == 0 and len(json.loads(out)) == 1, out)
    rc, out, _ = run("recall", str(store.root), "--label", "person", "--no-clip", "--area-min", "0.5")
    check("--area-min 0.5 drops a 0.3 region", rc == 0 and json.loads(out) == [], out)


def test_recall_needs_a_query(tmp: Path) -> None:
    store = make_index(tmp / "recallnoquery")
    rc, out, err = run("recall", str(store.root), "--no-clip")
    check("recall with no query is a usage error", rc == 2 and out == "", f"rc={rc} {err}")


def test_rrf_is_not_any_single_route() -> None:
    # A is never ranked first by any route, but three routes agree on it; a
    # rank-blind raw-score sum would crown B (raw 1.0) instead.
    routes = {
        "label": {"B": {"rank": 1, "raw": 1.0, "hit": "label:b"},
                  "A": {"rank": 2, "raw": 0.1, "hit": "label:a"}},
        "semantic": {"D": {"rank": 1, "raw": 0.9, "hit": "sem:d"},
                     "A": {"rank": 3, "raw": 0.1, "hit": "sem:a"}},
        "geometry": {"C": {"rank": 1, "raw": 0.8, "hit": "area:c"},
                     "A": {"rank": 2, "raw": 0.1, "hit": "area:a"}},
    }
    fused = fuse(routes)
    order = sorted(fused.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
    check("RRF top is the 3-route region, not any route's rank-1", order[0][0] == "A",
          str([(k, round(v['score'], 4)) for k, v in order]))
    check("A's per-route ranks are 2/3/2 (never 1)",
          [fused["A"]["routes"][r]["rank"] for r in ("label", "semantic", "geometry")] == [2, 3, 2])
    expected = 1 / (RRF_K + 2) + 1 / (RRF_K + 3) + 1 / (RRF_K + 2)
    check("score is sum of 1/(k+rank)", abs(fused["A"]["score"] - expected) < 1e-12,
          f"{fused['A']['score']} vs {expected}")


def test_relation_cache_is_vocabulary_keyed(tmp: Path) -> None:
    """The round-2 fix: a graph is only reusable for the vocabulary it used."""
    store = make_index(tmp / "relcache")
    rec = store.require_regions(7)
    sha = sha256_json(rec["regions"])
    store.append_relation_set(7, sha, ["riding"],
                              [{"subject": 0, "relation": "riding", "object": 1, "score": 0.5}],
                              {"vocabulary": ["riding"]})
    store.append_relation_set(7, sha, ["smiling at"],
                              [{"subject": 1, "relation": "smiling at", "object": 0, "score": 0.4}],
                              {"vocabulary": ["smiling at"]})
    riding = store.find_relation_set(7, sha, ["riding"])
    smiling = store.find_relation_set(7, sha, ["smiling at"])
    check("both vocabularies survive in one file",
          riding and smiling and riding["edges"][0]["relation"] == "riding"
          and smiling["edges"][0]["relation"] == "smiling at")
    check("a different vocabulary is a cache miss", store.find_relation_set(7, sha, ["holding"]) is None)
    check("a different region table invalidates every set",
          store.find_relation_set(7, "0" * 64, ["riding"]) is None)
    # A flat file (legacy `relate`) is still readable when the vocabulary matches.
    store.save_relations(7, {"frame_id": 7, "regions_sha256": sha,
                             "relation": {"model": "x", "predicates": ["carrying"]},
                             "edges": [{"subject": 0, "relation": "carrying", "object": 1, "score": 0.3}]})
    legacy = store.find_relation_set(7, sha, ["carrying"])
    check("legacy flat graph still matches its own vocabulary",
          legacy is not None and legacy["edges"][0]["relation"] == "carrying")


def test_index_layer_switch() -> None:
    parser = build_parser()
    default = parser.parse_args(["index", "v.mp4"])
    check("index defaults to detection only", default.with_relations is False)
    flagged = parser.parse_args(["index", "v.mp4", "--with-relations"])
    check("--with-relations opts back in", flagged.with_relations is True)
    recall_args = parser.parse_args(["recall", "d", "--label", "yellow", "dragon", "--relation", "riding"])
    check("--label/--relation accept several words",
          recall_args.label == ["yellow", "dragon"] and recall_args.relation == ["riding"])


def test_slow_recall_reject() -> None:
    """Model-free but index-backed: an absent label must be [] on the real index."""
    idx = ROOT / "index"
    if not idx.exists():
        print("SKIP  slow recall reject (no index/ built)")
        return
    candidates = [p for p in sorted(idx.iterdir()) if (p / "index.json").is_file()]
    if not candidates:
        print("SKIP  slow recall reject (no index.json under index/)")
        return
    rc, out, err = run("recall", str(candidates[0]), "--label", "purple elephant",
                       "--area-min", "0.2", "--no-clip")
    check("recall on an absent target exits 0", rc == 0, err)
    check("recall on an absent target returns []", json.loads(out) == [], out)


# --------------------------------------------------------------------------
# round 3: sense routes + label phrase coverage
# --------------------------------------------------------------------------

def make_sense_index(root: Path) -> IndexStore:
    """Two frames with real files and hand-written senses (no model needed)."""
    from vrs_skills.senses import SCHEMA as SENSE_SCHEMA

    store = IndexStore(root)
    frames = [
        (7, 2.5, [{"idx": 0, "label": "character sculpture", "score": 0.9,
                   "bbox": [10, 20, 110, 120], "area_ratio": 0.3, "has_mask": True,
                   "source": "test"}]),
        (8, 3.0, [{"idx": 0, "label": "purple elephant", "score": 0.7,
                   "bbox": [12, 22, 100, 110], "area_ratio": 0.3, "has_mask": True,
                   "source": "test"}]),
    ]
    for fid, t, regions in frames:
        img = np.full((160, 200, 3), 60, np.uint8)
        img[20:120, 10:110] = 200
        store.frame_image_path(fid).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(store.frame_image_path(fid)), img)
        mask = np.zeros((160, 200), bool)
        mask[20:120, 10:110] = True
        store.save_masks(fid, {0: mask})
        store.save_regions(fid, {"frame_id": fid, "t": t, "width": 200, "height": 160,
                                 "detector": {}, "next_idx": len(regions), "regions": regions})
        sha = sha256_json(regions)
        store.save_senses(fid, {
            "schema": SENSE_SCHEMA, "frame_id": fid, "t": t, "regions_sha256": sha,
            "axes": ["face", "action", "depth", "text", "pose"], "models": {},
            "regions": {"0": {
                "face": {"n_faces": 1, "mouth_smile_l": 0.88 if fid == 7 else 0.02,
                         "mouth_smile_r": 0.86 if fid == 7 else 0.01,
                         "jaw_open": 0.41 if fid == 7 else 0.01,
                         "expression": "laughing" if fid == 7 else "neutral",
                         "expression_score": 0.51 if fid == 7 else 0.05,
                         "expression_scores": {"laughing": 0.51 if fid == 7 else 0.05,
                                               "smiling": 0.10, "neutral": 0.05 if fid == 7 else 0.90},
                         "src": "mediapipe-face-landmarker",
                         "expression_src": "clip-vitb32-expression-zeroshot"},
                "action": {"label": "waving hand" if fid == 7 else "sitting",
                           "score": 0.42 if fid == 7 else 0.03, "top": [],
                           "src": "videomae-small-kinetics"},
                "depth": {"mean_rel": 0.84 if fid == 7 else 0.21, "p50_rel": 0.8,
                          "rank_in_frame": 1, "src": "depth-anything-v2-small"},
                "text": ([{"s": "EXIT", "box": [12, 22, 60, 40], "score": 0.94, "src": "rapidocr"}]
                         if fid == 7 else []),
                "track": {"track_id": 7 if fid == 7 else 3, "dx_per_s": 12.5, "src": "bytetrack"},
                "pose": {"kpt17": [[1.0, 1.0, 0.9]] * 17, "arms_raised": fid == 7,
                         "n_kpts": 17, "src": "yolo11n-pose"},
            }},
            "frame": {"faces": [{"bbox": [10, 20, 110, 120]}], "action": [],
                      "text": [], "depth_stats": {"median": 0.5}, "n_tracks": 1},
            "computed_at": "test",
        })
    store.save_meta({
        "schema": 1, "video": "synthetic.mp4", "video_sha256": "deadbeef", "fps": 12.0,
        "n_video_frames": 100, "duration_s": 8.0, "width": 200, "height": 160, "n_frames": 2,
        "detector": {}, "relation": {},
        "frames": [{"frame_id": fid, "t": t, "n_regions": len(regions),
                    "image": str(store.frame_image_path(fid)), "regions_file": ""}
                   for fid, t, regions in frames],
        "timings": {},
    })
    return store


def test_label_phrase_coverage(tmp: Path) -> None:
    """A 1-of-2 word overlap is a *partial* match, not a complete one."""
    store = make_sense_index(tmp / "coverage")
    rc, out, err = run("recall", str(store.root), "--label", "cartoon character",
                       "--no-clip", "--area-min", "0", "--explain")
    rows = json.loads(out)
    hits = [h for r in rows for h in r["hits"] if h.startswith("label:")]
    check("partial label overlap is marked with its coverage",
          any("(1/2)" in h for h in hits), str(hits))
    check("partial label overlap is not reported as an exact match",
          not any(h == "label:character sculpture" for h in hits), str(hits))

    rc, out, _ = run("recall", str(store.root), "--label", "cartoon character",
                     "--no-clip", "--area-min", "0")
    check("a partially covering phrase still surfaces the region (discount, not drop)",
          rc == 0 and json.loads(out), out)

    rc, out, _ = run("recall", str(store.root), "--label", "cartoon bicycle",
                     "--no-clip", "--area-min", "0")
    check("a phrase with no covered word returns []", json.loads(out) == [], out)

    rc, out, _ = run("recall", str(store.root), "--label", "character sculpture",
                     "--no-clip", "--area-min", "0", "--explain")
    exact = [r for r in json.loads(out) if r["region_idx"] == 0 and r["frame_id"] == 7]
    check("an exact label match still ranks first in the label route",
          bool(exact) and exact[0]["routes"]["label"]["rank"] == 1, out)


def test_sense_routes_read_from_disk(tmp: Path) -> None:
    store = make_sense_index(tmp / "sense")
    idx = str(store.root)

    rc, out, err = run("recall", idx, "--expression", "laughing", "--area-min", "0", "--explain")
    rows = json.loads(out)
    check("expression route exits 0", rc == 0, err)
    check("expression route surfaces only the laughing frame",
          [r["frame_id"] for r in rows] == [7], out)
    check("hits names the expression route",
          any(h.startswith("expr:laughing") for h in rows[0]["hits"]), str(rows[0]["hits"]))
    check("explain shows the expression route's rank",
          rows[0]["routes"]["expression"]["rank"] == 1, str(rows[0].get("routes")))

    rc, out, err = run("recall", idx, "--label", "character sculpture",
                       "--expression", "laughing", "--area-min", "0", "--no-face")
    rows = json.loads(out)
    check("--no-face switches the expression route off",
          all(not h.startswith("expr:") for r in rows for h in r["hits"]), out)
    rc, out, err = run("recall", idx, "--expression", "laughing", "--area-min", "0", "--no-face")
    check("--no-face on a face-only query is an empty result, not a usage error",
          rc == 0 and json.loads(out) == [], f"rc={rc} {out} {err}")

    rc, out, _ = run("recall", idx, "--face-smile-min", "0.7", "--area-min", "0", "--explain")
    rows = json.loads(out)
    check("face-smile route reads MediaPipe blendshapes",
          rc == 0 and [r["frame_id"] for r in rows] == [7]
          and any(h.startswith("smile:") for h in rows[0]["hits"]), out)
    rc, out, err = run("recall", idx, "--label", "character sculpture",
                       "--face-smile-min", "0.7", "--area-min", "0", "--no-face")
    rows = json.loads(out)
    check("--no-face switches the landmarker route off",
          all(not h.startswith("smile:") for r in rows for h in r["hits"]), out)

    rc, out, _ = run("recall", idx, "--depth-near", "--area-min", "0", "--explain")
    rows = json.loads(out)
    check("depth route ranks the nearer region first",
          rc == 0 and rows[0]["frame_id"] == 7
          and any(h.startswith("depth:") for h in rows[0]["hits"]), out)

    rc, out, _ = run("recall", idx, "--text", "EXIT", "--area-min", "0", "--explain")
    rows = json.loads(out)
    check("text route matches the OCR string",
          rc == 0 and [r["frame_id"] for r in rows] == [7]
          and any(h.startswith("text:exit") for h in rows[0]["hits"]), out)
    rc, out, _ = run("recall", idx, "--text", "NOPE", "--area-min", "0")
    check("text route returns [] when the string is absent", json.loads(out) == [], out)

    rc, out, _ = run("recall", idx, "--track", "7", "--area-min", "0", "--explain")
    rows = json.loads(out)
    check("track route matches the ByteTrack id",
          rc == 0 and [r["frame_id"] for r in rows] == [7]
          and any(h.startswith("track:7") for h in rows[0]["hits"]), out)
    rc, out, _ = run("recall", idx, "--track", "99", "--area-min", "0")
    check("an unknown track id returns []", json.loads(out) == [], out)

    rc, out, _ = run("recall", idx, "--arms-raised", "--area-min", "0", "--explain")
    rows = json.loads(out)
    check("arms-raised filter uses the pose keypoints",
          rc == 0 and [r["frame_id"] for r in rows] == [7]
          and "pose:arms_raised" in rows[0]["hits"], out)

    rc, out, _ = run("recall", idx, "--action", "waving", "--area-min", "0", "--explain")
    rows = json.loads(out)
    check("action route matches the VideoMAE label",
          rc == 0 and [r["frame_id"] for r in rows] == [7]
          and any(h.startswith("act:waving hand") for h in rows[0]["hits"]), out)
    rc, out, _ = run("recall", idx, "--action", "waving", "--action-min", "0.9", "--area-min", "0")
    check("action-min drops a low-confidence label", json.loads(out) == [], out)

    rc, out, _ = run("recall", idx, "--expression", "laughing", "--area-min", "0")
    check("a sense-only query is a valid query", rc == 0, out)


def test_stale_senses_are_ignored(tmp: Path) -> None:
    """A senses file written for an older region table must not be used."""
    store = make_sense_index(tmp / "stale")
    data = store.load_senses(7)
    data["regions_sha256"] = "0" * 64
    store.save_senses(7, data)
    rc, out, _ = run("recall", str(store.root), "--expression", "laughing", "--area-min", "0")
    check("stale senses (region table changed) are not used", json.loads(out) == [], out)
    # frame 8 is untouched, so the route still works there when asked for neutral
    rc, out, _ = run("recall", str(store.root), "--expression", "neutral",
                     "--expression-min", "0.5", "--area-min", "0")
    check("a frame with matching senses still matches", [r["frame_id"] for r in json.loads(out)] == [8], out)


def test_fuse_never_drops_a_route() -> None:
    routes = {
        "label": {"A": {"rank": 1, "raw": 1.0, "hit": "label:a"}},
        "brand_new_route": {"B": {"rank": 1, "raw": 0.5, "hit": "new:b"}},
    }
    fused = fuse(routes)
    check("a route outside the canonical order still fuses", "B" in fused,
          str(list(fused)))
    check("the canonical RRF formula is unchanged for known routes",
          abs(fused["A"]["score"] - 1 / (RRF_K + 1)) < 1e-12)


# --------------------------------------------------------------------------
# round-3: the senses command itself
# --------------------------------------------------------------------------

def test_senses_axes_are_pluggable() -> None:
    parser = build_parser()
    args = parser.parse_args(["senses", "d", "--axes", "depth", "text"])
    check("senses --axes accepts a subset", args.axe == ["depth", "text"])
    args = parser.parse_args(["senses", "d"])
    check("senses defaults to every axis (None => all)", args.axe is None)
    args = parser.parse_args(["recall", "d", "--expression", "laughing", "--no-face"])
    check("recall exposes the face switch", args.no_face is True and args.expression == "laughing")
    args = parser.parse_args(["recall", "d", "--track", "7", "--depth-near"])
    check("recall exposes track/depth switches", args.track == 7 and args.depth_near is True)


def main() -> int:
    slow = "--slow" in sys.argv
    test_timestamp_is_half_up()
    tmp = Path(tempfile.mkdtemp(prefix="vrs_test_"))
    try:
        test_merge_is_idempotent(tmp)
        test_regions_are_stable(tmp)
        test_answer_shape(tmp)
        test_verify_detects_tampering(tmp)
        test_verify_detects_mask_escaping_bbox(tmp)
        test_guided_failures(tmp)
        test_no_traceback_leaks(tmp)
        test_seg_and_crop(tmp)
        test_recall_label_and_empty(tmp)
        test_recall_geometry_filters(tmp)
        test_recall_needs_a_query(tmp)
        test_rrf_is_not_any_single_route()
        test_relation_cache_is_vocabulary_keyed(tmp)
        test_index_layer_switch()
        test_label_phrase_coverage(tmp)
        test_sense_routes_read_from_disk(tmp)
        test_stale_senses_are_ignored(tmp)
        test_fuse_never_drops_a_route()
        test_senses_axes_are_pluggable()
        if slow:
            test_slow_reject_path()
            test_slow_recall_reject()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
