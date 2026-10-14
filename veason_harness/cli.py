"""``veason`` entry point: one subcommand per perception skill.

Every command is non-interactive, prints one JSON document to stdout, and
writes its failures as a single JSON object to stderr with a non-zero exit
code.  The handler functions read stdout and decide the next command; nothing
here ever prompts, and nothing prints a bare traceback.

This module is the composition root: it is the only place that turns strings
the agent typed into objects (index dir, detector, relater, renderer).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep the run offline and quiet: every model here is already on disk, and a
# hanging hub lookup would look like a crash to the calling agent.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from . import paths  # noqa: E402
from .answers import EXIT_VERIFY_FAILED, answer, verify  # noqa: E402
from .errors import VrsError, emit_error  # noqa: E402
from .store import IndexStore, sha256_json  # noqa: E402

PROG = "veason"


def _print(payload) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    sys.stdout.flush()


def _emit_events(info) -> None:
    """降级/成本台账走 stderr，一行一个 JSON。

    ★ 四环一律沿用 `cmd_recall` 立下的模式：**stdout 是契约，stderr 是事实**
      （哪条路降级了、Jev 被调了几次）。把降级塞进 stdout 会污染契约，
      把降级丢掉则会让「读数为 0」的原因永远查不出来。
    """
    for event in (info or {}).get("events", []):
        sys.stderr.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stderr.flush()


def _store(index_dir: str) -> IndexStore:
    store = IndexStore(paths.resolve(index_dir))
    store.require()
    return store


def senses_axes() -> tuple:
    """The round-3 sense axes, imported lazily so the CLI stays cheap to build."""
    from .senses import AXIS_MODELS

    return tuple(AXIS_MODELS)


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

def cmd_index(args) -> int:
    from .indexer import run_index

    payload = run_index(
        video_path=paths.resolve(args.video),
        out_dir=paths.resolve(args.out),
        n_frames=args.frames,
        conf=args.conf,
        topk=args.topk,
        with_relations=args.with_relations,
    )
    _print(payload)
    return 0


def cmd_frames(args) -> int:
    from .indexer import refresh_frames

    _print(refresh_frames(_store(args.index_dir)))
    return 0


def cmd_regions(args) -> int:
    store = _store(args.index_dir)
    store.require_frame(args.frame_id)
    _print(store.require_regions(args.frame_id)["regions"])
    return 0


def cmd_detect(args) -> int:
    import cv2

    from .detector import PromptedDetector

    store = _store(args.index_dir)
    store.require_frame(args.frame_id)
    rec = store.require_regions(args.frame_id)
    before_sha = sha256_json(rec["regions"])
    frame = cv2.imread(str(store.frame_image(args.frame_id)))
    if frame is None:
        raise VrsError(
            f"frame image unreadable: {store.frame_image(args.frame_id)}",
            hint="re-run `veason index`; the stored PNG is corrupt.",
        )
    cands, masks = PromptedDetector(conf=args.conf).detect(frame, args.phrase)
    # each candidate gets a provisional idx; merge_regions overwrites it with the
    # stable idx (existing region) or the next free one (new region).
    for c in cands:
        c["idx"] = -1
    rec, added = store.append_regions(args.frame_id, cands)
    # A `detect` can change the table without adding a region: a duplicate hit
    # with a higher score updates the stored score in place (added == []), which
    # changes the region hash.  Any relation/co-occurrence graph keyed on the old
    # table is now stale, so invalidate it on *any* table change, not just adds.
    if sha256_json(rec["regions"]) != before_sha:
        if added:
            mask_map = store.load_masks(args.frame_id)
            by_bbox = {tuple(c["bbox"]): m for c, m in zip(cands, masks) if m is not None}
            for r in added:
                m = by_bbox.get(tuple(r["bbox"]))
                if m is not None:
                    mask_map[r["idx"]] = m
            store.save_masks(args.frame_id, mask_map)
        if store.relations_path(args.frame_id).exists():
            store.relations_path(args.frame_id).unlink()  # region table changed
    _print(added)
    return 0


def cmd_relate(args) -> int:
    import cv2

    from .relation import Relater

    store = _store(args.index_dir)
    store.require_frame(args.frame_id)
    rec = store.require_regions(args.frame_id)
    frame = cv2.imread(str(store.frame_image(args.frame_id)))
    if frame is None:
        raise VrsError(
            f"frame image unreadable: {store.frame_image(args.frame_id)}",
            hint="re-run `veason index`; the stored PNG is corrupt.",
        )
    relater = Relater(predicates=args.relation)
    edges, info = relater.relate(frame, rec["regions"], store.load_masks(args.frame_id),
                                 topk=args.topk, min_score=args.min_score)
    store.save_relations(
        args.frame_id,
        {
            "frame_id": args.frame_id,
            "regions_sha256": sha256_json(rec["regions"]),
            "relation": {"model": "relsgg-vits16plus", "predicates": info["vocabulary"]},
            "info": info,
            "edges": edges,
        },
    )
    # stdout is the contract (a list of triplets); truncation is a warning, not a silent fact.
    if info.get("truncated"):
        sys.stderr.write(json.dumps({
            "warning": "region table larger than the relation model's cap",
            "n_regions": info["n_regions"], "n_used": info["n_used"],
            "hint": f"only the {info['n_used']} highest-scoring regions were used; "
                    "lower --conf or use `detect` to narrow the table.",
        }, ensure_ascii=False) + "\n")
    _print(edges)
    return 0


def cmd_show(args) -> int:
    from . import render

    _print(render.show(_store(args.index_dir), args.frame_id,
                       paths.resolve(args.out) if args.out else None))
    return 0


def cmd_crop(args) -> int:
    from . import render

    _print(render.crop(_store(args.index_dir), args.frame_id, args.region_idx,
                       out=paths.resolve(args.out) if args.out else None))
    return 0


def cmd_seg(args) -> int:
    from . import render

    _print(render.seg(_store(args.index_dir), args.frame_id, args.region_idx,
                      out=paths.resolve(args.out) if args.out else None))
    return 0


def cmd_answer(args) -> int:
    _print(answer(_store(args.index_dir), args.frame_id, args.region_idx))
    return 0


def cmd_verify(args) -> int:
    payload, ok = verify(_store(args.index_dir), args.frame_id, args.region_idx)
    _print(payload)
    if not ok:
        return EXIT_VERIFY_FAILED
    return 0


def cmd_recall(args) -> int:
    from .recall import run_recall

    results, info = run_recall(
        _store(args.index_dir),
        labels=args.label,
        relations=args.relation,
        area_min=args.area_min,
        frame_min=args.frame_min,
        frame_max=args.frame_max,
        top_k=args.top_k,
        use_clip=not args.no_clip,
        explain=args.explain,
        relate_max_frames=args.relate_max_frames,
        sem_min=args.sem_min,
        relation_min=args.relation_min,
        sem_max_crops=args.sem_max_crops,
        no_face=args.no_face,
        face_smile_min=args.face_smile_min,
        jaw_open_min=args.jaw_open_min,
        expression=args.expression,
        expression_min=args.expression_min,
        actions=args.action or [],
        action_min=args.action_min,
        depth_near=args.depth_near,
        depth_min=args.depth_min,
        texts=args.text or [],
        text_min=args.text_min,
        track=args.track,
        arms_raised=args.arms_raised,
    )
    backfill = info.get("relation_backfill")
    if backfill:
        # stdout is the contract (a JSON list); the backfill ledger is a fact
        # about cost, reported on the warning channel like `relate`'s truncation.
        sys.stderr.write(json.dumps({"event": "recall_relation_backfill", **backfill},
                                    ensure_ascii=False) + "\n")
    semantic = info.get("semantic") or {}
    if semantic.get("error"):
        sys.stderr.write(json.dumps({"event": "recall_semantic_unavailable",
                                     "error": semantic["error"],
                                     "hint": "the remaining routes still ran; use --no-clip "
                                             "to skip the load next time."},
                                    ensure_ascii=False) + "\n")
    if info.get("senses_unavailable"):
        sys.stderr.write(json.dumps({
            "event": "recall_senses_unavailable",
            "senses": info.get("senses"),
            "hint": "run `veason senses <index_dir>` once to compute the sense axes "
                    "(face/action/depth/text/pose); recall reads them from disk and loads no model.",
        }, ensure_ascii=False) + "\n")
    if info.get("empty"):
        sys.stderr.write(json.dumps({"event": "recall_empty", "reason": info["empty_reason"],
                                     "routes": info["routes"]}, ensure_ascii=False) + "\n")
    _print(results)
    return 0


def cmd_senses(args) -> int:
    from . import senses

    store = _store(args.index_dir)
    if args.frame_id is not None:
        _print(senses.read_senses(store, args.frame_id))
        return 0
    payload = senses.run_senses(store, axes=args.axe or None, force=args.force)
    _print(payload)
    return 0


# ---- 五环链的四条新命令（①②③⑤；④ 由 agent 自己做，见 veason_pipeline.py）----

def cmd_screen(args) -> int:
    from .screen import parse_synonyms, run_screen

    payload, info = run_screen(
        _store(args.index_dir), args.query, args.top_n, args.oov_rule,
        oov_threshold=args.oov_threshold,
        jev_backend=args.jev_backend,
        synonyms=parse_synonyms(args.synonym),
        frames=args.frames,
    )
    _emit_events(info)
    _print(payload)
    return 0


def cmd_locate(args) -> int:
    from .locate import run_locate

    payload, info = run_locate(
        _store(args.index_dir), args.frames, args.query,
        pad_to=args.pad_to, veason_backend=args.veason_backend,
    )
    _emit_events(info)
    _print(payload)
    return 0


def cmd_evidence(args) -> int:
    from .evidence import run_evidence

    payload, info = run_evidence(
        _store(args.index_dir), args.frames, args.query, args.axes,
        axis_backend=args.axis_backend,
    )
    _emit_events(info)
    _print(payload)
    return 0


def cmd_reflect(args) -> int:
    from .reflect import parse_state, run_reflect

    payload, info = run_reflect(
        _store(args.index_dir), args.round, parse_state(args.state),
        fuse_rounds=args.fuse_rounds, reflect_first=args.reflect_first,
        crop_source=args.crop_source, min_area=args.min_area,
        jev_backend=args.jev_backend, axis_backend=args.axis_backend,
    )
    _emit_events(info)
    _print(payload)
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="VRS perception skills for an LLM agent: index a video, then "
                    "hand back region idxs instead of coordinates.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every command prints JSON on stdout. Failures print JSON on stderr "
               "with a `hint` and exit non-zero. Paths may be WSL (/mnt/c/...) or Windows.",
    )
    p.add_argument("--debug", action="store_true",
                   help="let unexpected exceptions raise, for debugging the CLI itself")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_):
        return sub.add_parser(name, help=help_)

    sp = add("index", "decode a video and build/extend an index directory")
    sp.add_argument("video")
    sp.add_argument("--frames", type=int, default=16,
                    help="frames to sample, evenly spaced, endpoints inclusive (default 16; cost is linear)")
    sp.add_argument("--out", default=str(paths.DEFAULT_INDEX_DIR), help="parent directory for the index")
    sp.add_argument("--conf", type=float, default=0.25, help="prompt-free detector confidence (default 0.25)")
    sp.add_argument("--topk", type=int, default=20, help="relation triplets kept per frame (default 20)")
    sp.add_argument("--with-relations", action="store_true",
                    help="also compute RelateAnything for every frame (legacy behaviour; "
                         "costs ~3x the detector and is not needed by `recall`, which backfills on demand)")
    sp.set_defaults(func=cmd_index)

    sp = add("frames", "list the frames of an index")
    sp.add_argument("index_dir")
    sp.set_defaults(func=cmd_frames)

    sp = add("regions", "candidate regions of one frame, with stable idxs")
    sp.add_argument("index_dir")
    sp.add_argument("frame_id", type=int)
    sp.set_defaults(func=cmd_regions)

    sp = add("detect", "re-run YOLOE with your own phrases and add the hits to the table")
    sp.add_argument("index_dir")
    sp.add_argument("frame_id", type=int)
    sp.add_argument("phrase", nargs="+", help='open-vocabulary phrases, e.g. "red dress" "a person"')
    sp.add_argument("--conf", type=float, default=0.10, help="prompted detector confidence (default 0.10)")
    sp.set_defaults(func=cmd_detect)

    sp = add("relate", "re-run RelateAnything with your own relation phrases")
    sp.add_argument("index_dir")
    sp.add_argument("frame_id", type=int)
    sp.add_argument("relation", nargs="+", help='relation phrases, e.g. "riding" "holding"')
    sp.add_argument("--topk", type=int, default=20)
    sp.add_argument("--min-score", type=float, default=0.0, help="drop triplets below this score")
    sp.set_defaults(func=cmd_relate)

    sp = add("show", "export the whole frame as a PNG")
    sp.add_argument("index_dir")
    sp.add_argument("frame_id", type=int)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_show)

    sp = add("crop", "export one region as a PNG")
    sp.add_argument("index_dir")
    sp.add_argument("frame_id", type=int)
    sp.add_argument("region_idx", type=int)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_crop)

    sp = add("seg", "segmentation overlay for one region (the deliverable picture)")
    sp.add_argument("index_dir")
    sp.add_argument("frame_id", type=int)
    sp.add_argument("region_idx", type=int)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_seg)

    sp = add("answer", "map region idxs to the official VRS answer format")
    sp.add_argument("index_dir")
    sp.add_argument("frame_id", type=int)
    sp.add_argument("region_idx", type=int, nargs="+")
    sp.set_defaults(func=cmd_answer)

    sp = add("verify", "check a delivered answer against the current region table")
    sp.add_argument("index_dir")
    sp.add_argument("frame_id", type=int)
    sp.add_argument("region_idx", type=int, nargs="+")
    sp.set_defaults(func=cmd_verify)

    sp = add("recall", "structured multi-route retrieval over an index (label/relation/semantic/geometry/co-occurrence, fused with RRF)")
    sp.add_argument("index_dir")
    sp.add_argument("--label", nargs="+", action="extend", default=[], metavar="WORD",
                    help="category/attribute words matched against region.label (repeatable)")
    sp.add_argument("--relation", nargs="+", action="extend", default=[], metavar="PHRASE",
                    help="relation phrases; RelateAnything graphs are backfilled for exactly this vocabulary")
    sp.add_argument("--area-min", type=float, default=0.05,
                    help="ignore regions smaller than this fraction of the frame (default 0.05)")
    sp.add_argument("--frame-min", type=int, default=None, help="lowest frame_id to consider")
    sp.add_argument("--frame-max", type=int, default=None, help="highest frame_id to consider")
    sp.add_argument("--top-k", type=int, default=12, help="candidates to return (default 12)")
    sp.add_argument("--no-clip", action="store_true",
                    help="skip the CLIP semantic route entirely (no model load, no image-text scores)")
    sp.add_argument("--explain", action="store_true", help="add each route's rank and raw score to every result")
    sp.add_argument("--sem-min", type=float, default=0.24,
                    help="minimum CLIP cosine for the semantic route (default 0.24, calibrated on the demo clips)")
    sp.add_argument("--relation-min", type=float, default=0.0,
                    help="minimum RelateAnything score for the relation/co-occurrence routes")
    sp.add_argument("--relate-max-frames", type=int, default=16,
                    help="max frames to backfill relations for per call (default 16)")
    sp.add_argument("--sem-max-crops", type=int, default=96,
                    help="max region crops to embed with CLIP per call (default 96; largest regions first)")
    # ---- round-3 sense routes (read from `veason senses` output, no model load) ----
    sp.add_argument("--no-face", action="store_true",
                    help="switch the whole face axis off (landmarker blendshapes and the "
                         "zero-shot expression head)")
    sp.add_argument("--face-smile-min", type=float, default=None, metavar="F",
                    help="face route: keep regions whose MediaPipe mouth_smile (L or R) >= F "
                         "(model convention: 0.5 = expression present)")
    sp.add_argument("--jaw-open-min", type=float, default=None, metavar="F",
                    help="face route: keep regions whose MediaPipe jaw_open >= F (0.5 = present)")
    sp.add_argument("--expression", default=None, metavar="WORD",
                    help='expression route: keep regions whose zero-shot expression == WORD '
                         '(e.g. laughing / smiling / mouth_open)')
    sp.add_argument("--expression-min", type=float, default=0.35,
                    help="minimum zero-shot expression score (default 0.35, calibrated on the demo clips)")
    sp.add_argument("--action", nargs="+", action="extend", default=[], metavar="WORD",
                    help="action route: keep regions whose VideoMAE clip label matches (substring ok)")
    sp.add_argument("--action-min", type=float, default=0.10,
                    help="minimum VideoMAE label score (default 0.10)")
    sp.add_argument("--depth-near", action="store_true",
                    help="depth route: rank regions by mean relative depth (larger = nearer)")
    sp.add_argument("--depth-min", type=float, default=0.5,
                    help="minimum per-frame normalised mean depth (default 0.5 = nearer half)")
    sp.add_argument("--text", nargs="+", action="extend", default=[], metavar="WORD",
                    help="text route: keep regions whose OCR string contains any of these words")
    sp.add_argument("--text-min", type=float, default=0.5,
                    help="minimum RapidOCR recognition score (default 0.5)")
    sp.add_argument("--track", type=int, default=None, metavar="ID",
                    help="pose route: keep regions whose ByteTrack track_id == ID")
    sp.add_argument("--arms-raised", action="store_true",
                    help="pose route: keep regions whose yolo11n-pose keypoints have both wrists "
                         "above both shoulders")
    sp.set_defaults(func=cmd_recall)

    sp = add("senses", "compute (or read) the face/action/depth/text/pose sense axes of an index")
    sp.add_argument("index_dir")
    sp.add_argument("--frame-id", type=int, default=None,
                    help="read the stored senses of one frame instead of computing")
    sp.add_argument("--axes", dest="axe", nargs="+", choices=list(senses_axes()), default=None,
                    metavar="AXIS", help="axes to compute (default: all five)")
    sp.add_argument("--force", action="store_true",
                    help="recompute even when a matching senses file already exists")
    sp.set_defaults(func=cmd_senses)

    # ---- 五环链 ①②③⑤（④ 是 agent 自己的活，没有对应命令）----
    sp = add("screen", "① 初筛：YOLO 主力，OOV 才上 Jev（高召回，输出带 route 与成本）")
    sp.add_argument("index_dir")
    sp.add_argument("--query", required=True, help="指代表达式（自然语言，中英文都行）")
    sp.add_argument("--top-n", type=int, default=6,
                    help="返回几个候选（默认 6；★ 实测 >3 个进打分器就崩，>6 更要小心）")
    sp.add_argument("--oov-rule", default="query_miss", choices=("query_miss", "score_threshold"),
                    help="OOV 判据：query_miss = query 词零命中（可解释，默认）；"
                         "score_threshold = YOLO 最高分低于 --oov-threshold（更连续）")
    sp.add_argument("--oov-threshold", type=float, default=0.30,
                    help="score_threshold 规则的门限（默认 0.30；★ 推断值，未标定）")
    sp.add_argument("--frames", default="",
                    help="只筛这些帧（逗号分隔）；不给 = 索引里的全部帧")
    sp.add_argument("--synonym", action="extend", nargs="+", default=[], metavar="KEY=WORD[,WORD]",
                    help="中文 query ↔ 英文 label 的显式映射，可重复。"
                         '例：--synonym "奶龙=parachute,stuffed toy"')
    sp.add_argument("--jev-backend", default=None,
                    help="jev 后端：auto(默认→local，同进程真件) | local | fake | unavailable（env VEASON_JEV_BACKEND）")
    sp.set_defaults(func=cmd_screen)

    sp = add("locate", "② 切分：Veason-R1 定位（只定位，不选帧；一次 8 帧）")
    sp.add_argument("index_dir")
    sp.add_argument("--frames", required=True, help="要定位的帧号，逗号分隔，例：850,384")
    sp.add_argument("--query", required=True, help="指代表达式")
    sp.add_argument("--pad-to", type=int, default=8,
                    help="一次前向的帧数（Veason-R1 官方口径 = 8）；不足时用【没看过的】帧补齐")
    sp.add_argument("--veason-backend", default=None,
                    help="veason 后端：auto(默认→local，同进程真件) | local | fake | unavailable（env VEASON_VEASON_BACKEND）")
    sp.set_defaults(func=cmd_locate)

    sp = add("evidence", "③ 打标签 + 按轴路由小模型（面积/数量纯计算，零模型）")
    sp.add_argument("index_dir")
    sp.add_argument("--frames", required=True, help="要取证据的帧号，逗号分隔")
    sp.add_argument("--query", required=True, help="指代表达式")
    sp.add_argument("--axes", default="area,action,occlusion", metavar="AXIS[,AXIS...]",
                    help="要哪些轴：area,count,attribute,relation,action,occlusion,text"
                         "（默认 area,action,occlusion）")
    sp.add_argument("--axis-backend", default=None,
                    help="轴后端：auto(默认→senses-cache) | fake | senses-cache | local | unavailable")
    sp.set_defaults(func=cmd_evidence)

    sp = add("reflect", "④ 分路反思：换帧/放大/换轴，每轮必须带【新测量】；"
                        "★ 不设次数上限，唯一正常终止是 no_new_information（没有新测量可拿）")
    sp.add_argument("index_dir")
    sp.add_argument("--round", type=int, required=True, help="这是第几轮（从 1 开始）")
    sp.add_argument("--state", required=True,
                    help="上一轮的状态：内联 JSON，或一个 .json 文件路径（两种都吃）")
    sp.add_argument("--fuse-rounds", "--max-rounds", dest="fuse_rounds", type=int, default=1000,
                    help="★ 保险丝，不是轮次上限（默认 1000；0 = 完全不设）。正常终止只有一个："
                         "reflect 报 no_new_information（没有新测量可拿）")
    sp.add_argument("--reflect-first", default="low_visibility",
                    choices=("low_visibility", "by_measurement"),
                    help="第一轮先跑哪条：low_visibility = 确定性、零成本（默认）| by_measurement")
    sp.add_argument("--crop-source", default="yolo", choices=("yolo", "veason"),
                    help="「放大」剪哪个框：yolo = ① 的 region 框（默认，现成）| "
                         "veason = ② 的框（更准，但要等 ② 跑完）")
    sp.add_argument("--min-area", type=float, default=0.02,
                    help="「低可见」判据的面积门限（默认 0.02；★ 推断值，未标定 —— "
                         "规格只写「< 阈值」，切点要按素材标）")
    sp.add_argument("--jev-backend", default=None, help="放大后重问用的后端（auto→local | local | fake | unavailable）")
    sp.add_argument("--axis-backend", default=None, help="换轴用的后端")
    sp.set_defaults(func=cmd_reflect)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.cmd
    try:
        return int(args.func(args))
    except VrsError as exc:
        emit_error(cmd, exc.error, exc.hint, exc.code)
    except KeyboardInterrupt:  # pragma: no cover
        emit_error(cmd, "interrupted", "nothing was left half-written: index files are replaced atomically.")
    except FileNotFoundError as exc:
        emit_error(cmd, f"file not found: {exc.filename}",
                   "check the path; WSL (/mnt/c/...) and Windows (C:\\...) forms are both accepted.")
    except Exception as exc:  # noqa: BLE001 - the agent must never see a traceback
        if getattr(args, "debug", False):
            raise
        emit_error(
            cmd,
            f"{type(exc).__name__}: {exc}",
            "unexpected failure. Re-run the same command with `--debug` (before the subcommand) "
            "to see the traceback, and report the command line + `veason --help` output.",
        )
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
