"""④ 分路反思：按问题类型路由，每轮必须拿【新信息】。

## 铁律（规格 §6.1，用户 2026-10-13 拍板）

> **「重排旧信息」不算反思。**
> 判断一轮值不值得开，就看它有没有引入【新的测量】。

这条铁律在代码里只有一个落点：``new_info`` 里每一条都必须写清
``measurement`` —— 「这一轮多出来的那个测量是什么」。写不出 ``measurement``
的条目根本进不了 ``new_info``，而 pipeline 的判据是
``if not reflected.get("new_info"): break``（``veason_pipeline.py:158``）。
⇒ 重排旧信息在这里**结构上产不出** ``new_info``。

## 三种新信息，权重不同（规格 §6.1，按 ReVOS 视频长度实测加权）

===== ================================ ========
新信息                                 权重
===== ================================ ========
**放大** 候选区域裁出来单独看             ★★ 主力
**换轴** 换个问法问原来看不到的属性        中
**换帧** 取视频里原本没看过的帧            低
===== ================================ ========

★★ 为什么放大是主力（实测依据，不是偏好）：ReVOS 269 个视频中位 18 帧（96% 在
17–32 帧），8 帧采样已经抽掉一半多，「换帧」剩下的帧与已看的几乎相邻；
而模型每问只有 1024 个图像 token 预算，摊在整帧上目标那块可能只分到几十个。
裁出来单独喂 = 把全部预算给这一块，是**真·分辨率提升**，不是换摆法。

## 判据必须【测量出来】，不是【问出来】（规格 §6.2）

本模块只做确定性测量：面积、IoU、分数距离、掩码一致。一次模型也没问
「你觉得这题是不是指代不清」—— 那是把路由键变成一个不可复算的判断。
``measurements`` 每个数都写在输出里，谁都能拿同一份素材复算一遍。

## ★ 两个必须钉死的边界

1. **「换帧」必须真的换成【没看过的】帧**。``state["seen_frames"]`` 是唯一
   依据；本模块返回的 ``frames`` 与 ``seen_frames`` 的交集必须为空
   （``_assert_unseen`` 在出口处强制，不靠调用方自觉）。
2. ★ **不设次数上限**。正常终止只有一个出口：``reason="no_new_information"``
   —— 没有新测量可拿。**判据在 agent 手上，不在一个常数上。**
   ``FUSE_ROUNDS`` 是**保险丝**（默认 1000），它触发时输出里必须带
   ``fuse: true`` 与明确措辞 —— 那是程序出错的信号，不是「想完了」。
   把保险丝当成正常的轮次上限，就等于用「重试 N 次」代替「想清楚」。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from .backends import BackendUnavailable, get_backend
from .errors import VrsError
from .paths import for_agent, for_windows

#: ★ 保险丝，**不是轮次上限**。正常终止是 ``no_new_information``
#: （没有新测量可拿）；这个数只用来兜住「循环里有 bug、每轮都假装有新信息」
#: 那种程序错误。所以它默认很大，且触发时输出必须带 ``fuse: true``。
#: ``None`` / ``0`` = 完全不设保险丝。
FUSE_ROUNDS = 1000

#: 低可见判据的面积门限。★ **推断值，未标定** —— 规格只说「面积占比 < 阈值」。
MIN_AREA = 0.02

#: 「候选间相似度高、分数接近」的两条门限。★ 也是推断值。
AMBIGUITY_AREA_REL = 0.20      # 面积相对差 ≤20% 算相似
AMBIGUITY_SCORE_GAP = 0.10     # 分数绝对差 ≤0.10 算接近

#: 「选中帧与相邻帧掩码不一致」的 IoU 门限。★ 推断值。
TEMPORAL_MASK_IOU = 0.30

#: 可按需补问的轴，顺序 = 补问的优先级（按规格 §4.2 的成本从低到高）。
CANDIDATE_AXES = ("occlusion", "text", "relation", "action")


def parse_state(raw: str) -> dict:
    """``--state`` 两吃：文件路径（规格 §6.3）或内联 JSON（pipeline 传法）。

    两个调用方在这一点上不一致，而两边都合理（文件更适合 agent 手写，
    内联更适合被程序调用）。先按路径试，不是文件就当 JSON —— 顺序不能反：
    一个合法的 JSON 字符串也可能是路径不存在的文件名，但反过来不会。
    """
    if raw is None or not str(raw).strip():
        raise VrsError("no --state given",
                       hint='--state <json> or --state <file.json>; the state carries '
                            'query/candidates/evidence/decision/seen_frames', code=2)
    s = str(raw).strip()
    p = Path(s)
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise VrsError(f"--state file is not valid JSON: {s} ({exc})",
                           hint="fix the file, or pass the state inline as JSON.", code=2) from exc
    try:
        return json.loads(s)
    except json.JSONDecodeError as exc:
        raise VrsError(
            f"--state is neither a readable file nor JSON ({exc})",
            hint="pass either an existing .json path or an inline JSON object.",
            code=2,
        ) from exc


def _int_list(v: Any) -> list[int]:
    out: list[int] = []
    for x in (v or []):
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            raise VrsError(f"seen_frames contains a non-integer: {x!r}",
                           hint="seen_frames is a list of frame ids (ints).", code=2) from None
    return out


def _dig(v: Any) -> dict:
    d = v.get("value") if isinstance(v, dict) else None
    return d if isinstance(d, dict) else {}


def _area(v: Any) -> float:
    if isinstance(v, dict):
        a = v.get("area_ratio")
        if a is None:
            a = _dig(v).get("area_ratio")
        try:
            return float(a)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _mask_iou(a: Any, b: Any) -> Optional[float]:
    """两个整帧 bool 掩码的 IoU。纯计算（纪律①），不问模型。"""
    if a is None or b is None:
        return None
    import numpy as np

    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    if a.shape != b.shape:
        return None
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else None


# --------------------------------------------------------------------------
# 测量：每个判据都要能被另一台机器按同一份素材复算
# --------------------------------------------------------------------------

def measure(store: Any, state: dict, min_area: float = MIN_AREA) -> dict:
    cands = list(state.get("candidates") or [])
    dec = state.get("decision") or {}
    seen = set(_int_list(state.get("seen_frames")))
    fid = dec.get("frame_id")
    ridx = dec.get("region_idx")

    areas = sorted((_area(c) for c in cands), reverse=True)
    gap = (areas[0] - areas[1]) if len(areas) > 1 else 1.0
    ambiguity_area = bool(len(areas) > 1 and areas[0] > 0
                          and (areas[0] - areas[1]) / areas[0] <= AMBIGUITY_AREA_REL)
    scores = [float(c.get("score", c.get("jev_yes", 0.0)) or 0.0) for c in cands]
    ambiguity_score = bool(len(scores) > 1 and abs(scores[0] - scores[1]) <= AMBIGUITY_SCORE_GAP)
    # ★ 判据 ②：不许用「谁更自信」解冲突。所以这里只把分数距离当【一个测量】记下来，
    #   它单独不足以触发任何动作（见 _plan 里 ambiguity 要求面积也接近）。
    ambiguity = bool(ambiguity_area and ambiguity_score)

    my_area = _candidate_area(cands, fid, ridx)
    regions: list[dict] = []
    if fid is not None and int(fid) in set(store.frame_ids()):
        regions = store.require_regions(int(fid)).get("regions", [])

    # ★ 面积必须能从【索引】里量出来，不能只依赖候选里带没带。
    #   候选没带时退回 0.0 是危险的：0.0 会被读成「低可见」，而低可见是
    #   一条会触发换帧的路由。索引里的 region 表才是面积的权威来源。
    if my_area <= 0.0 and fid is not None and ridx is not None and int(ridx) >= 0:
        for r in regions:
            if int(r["idx"]) == int(ridx):
                my_area = float(r.get("area_ratio") or 0.0)
                break

    # 干扰源：同一帧里还有跟目标面积相近、且被选中的不是最大的那个
    distractor = False
    if len(regions) > 1 and ridx is not None:
        by_area = sorted(regions, key=lambda r: -float(r.get("area_ratio", 0.0)))
        top = float(by_area[0].get("area_ratio", 0.0))
        distractor = bool(top > 0 and int(by_area[0]["idx"]) != int(ridx)
                          and (top - my_area) / top <= AMBIGUITY_AREA_REL)

    # 时序错误：选中帧的掩码与【已看过的】最近邻帧里同名 region 的掩码不一致
    temporal, temporal_iou = False, None
    if fid is not None and ridx is not None and regions:
        label = next((r.get("label") for r in regions if int(r["idx"]) == int(ridx)), None)
        masks_here = store.load_masks(int(fid))
        m_here = masks_here.get(int(ridx))
        avail = sorted(set(store.frame_ids()) & seen)
        if label and m_here is not None and avail:
            avail.sort(key=lambda f: (abs(f - int(fid)), f))
            for other in avail[:3]:
                if other == int(fid):
                    continue
                for r2 in store.require_regions(other).get("regions", []):
                    if r2.get("label") != label:
                        continue
                    iou = _mask_iou(m_here, store.load_masks(other).get(int(r2["idx"])))
                    if iou is None:
                        continue
                    temporal_iou = iou
                    temporal = iou < TEMPORAL_MASK_IOU
                    break
                if temporal_iou is not None:
                    break

    return {
        "decision_frame": None if fid is None else int(fid),
        "decision_region": None if ridx is None else int(ridx),
        "decision_area": round(my_area, 6),
        "candidate_areas_top2": [round(a, 6) for a in areas[:2]],
        "candidate_area_gap": round(gap, 6),
        "ambiguity_area_close": ambiguity_area,
        "ambiguity_score_close": ambiguity_score,
        "ambiguity": ambiguity,
        "low_visibility": bool(fid is not None and my_area < min_area),
        "min_area_threshold": min_area,
        "distractor": distractor,
        "temporal": temporal,
        "temporal_mask_iou": None if temporal_iou is None else round(float(temporal_iou), 4),
        "temporal_iou_threshold": TEMPORAL_MASK_IOU,
        "seen_frames": sorted(seen),
        "axes_already_measured": sorted({str(e.get("axis")) for e in (state.get("evidence") or [])}),
    }


def _candidate_area(cands: Sequence[dict], fid: Any, ridx: Any) -> float:
    for c in cands:
        if fid is not None and int(c.get("frame_id", -1)) == int(fid) \
                and ridx is not None and int(c.get("region_idx", -1)) == int(ridx):
            return _area(c)
    best = 0.0
    for c in cands:
        if fid is not None and int(c.get("frame_id", -1)) == int(fid):
            best = max(best, _area(c))
    return best


def _premise_false(state: dict) -> Optional[dict]:
    """前提为假：Jev 的 ``none`` 判定 ⇒ 终止，原样交付（不猜）。"""
    for e in (state.get("evidence") or []):
        v = _dig(e)
        for key in ("exists", "none", "jev_none"):
            if v.get(key) is False and key in ("exists", "jev_none"):
                return {"source": "evidence", "axis": e.get("axis"), "signal": {key: False}}
        if v.get("none") is True or v.get("jev_none") is True:
            return {"source": "evidence", "axis": e.get("axis"), "signal": v}
    for c in (state.get("candidates") or []):
        if c.get("route") == "jev_none" or _dig(c).get("none") is True:
            return {"source": "candidate", "frame_id": c.get("frame_id")}
    return None


# --------------------------------------------------------------------------
# 三个动作，每个都必须产出一条带 measurement 的新信息
# --------------------------------------------------------------------------

def _crop_box(store: Any, state: dict, fid: int, ridx: int, crop_source: str) -> tuple[list[int], str]:
    """放大用哪个框（待拍板③）。

    ``yolo``   —— region 表里那个框。**默认**，因为它在 ① 之后就有，不用多跑一轮。
    ``veason`` —— ② 的框（更准）。要等 ② 跑完，所以它只能从 ``state["locate"]``
                  里取；取不到就退回 yolo 并**把退回这件事记在返回值里**，
                  不许静默换源（换了源就等于说不清这块像素是哪来的）。
    """
    if crop_source == "veason":
        for item in (state.get("locate") or []):
            if int(item.get("frame_id", -1)) != int(fid):
                continue
            boxes = item.get("bbox_2d_list") or []
            if boxes:
                return [int(v) for v in boxes[0]], "veason-r1"
        return _region_box(store, fid, ridx), "yolo-region(fallback: state has no veason box)"
    return _region_box(store, fid, ridx), "yolo-region"


def _region_box(store: Any, fid: int, ridx: int) -> list[int]:
    region = store.get_region(fid, ridx)
    return [int(v) for v in region["bbox"]]


def _write_crop(store: Any, fid: int, box: list[int], out_path: Path) -> dict:
    """把 box（带 8px 余量）裁出来写成 PNG。**这一步就是新测量的物证。**"""
    import cv2

    img = cv2.imread(str(store.frame_image(fid)))
    if img is None:
        raise VrsError(f"frame image unreadable: {store.frame_image(fid)}",
                       hint="re-run `veason index`; the stored PNG is corrupt.")
    h, w = img.shape[:2]
    pad = 8
    x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad)
    x2, y2 = min(w, box[2] + pad), min(h, box[3] + pad)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        raise VrsError(f"region crops to nothing on frame {fid}",
                       hint="the stored bbox is degenerate; re-run `veason index`.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), patch):
        raise VrsError(f"could not write the zoom crop to {out_path}",
                       hint="check the index directory is writable and the drive has space.")
    return {"out": for_agent(out_path), "out_win": for_windows(out_path)}


def _zoom(store: Any, state: dict, m: dict, jev: Any,
          crop_source: str = "yolo") -> list[dict]:
    """放大：把候选区域裁出来单独看。

    ★ 主力项。这里的「新测量」是**这块像素第一次以原生分辨率被看到** ——
      上一轮模型看到的只是整帧经过缩放后的版本，目标那块可能只分到几十个
      image token。所以哪怕模型没接线，crop 本身也是一次新测量；
      但 ``answer`` 会诚实地写成 null，不会假装问过。
    """
    dec = state.get("decision") or {}
    targets: list[tuple[int, int]] = []
    if dec.get("frame_id") is not None and int(dec.get("region_idx", -1)) >= 0:
        targets.append((int(dec["frame_id"]), int(dec["region_idx"])))
    if m["ambiguity"] or m["distractor"]:
        # 指代不清 / 干扰源：两个候选都要裁 —— 提问不能只给一方。
        for c in (state.get("candidates") or [])[:2]:
            if c.get("frame_id") is None or int(c.get("region_idx", -1)) < 0:
                continue
            pair = (int(c["frame_id"]), int(c["region_idx"]))
            if pair not in targets:
                targets.append(pair)

    out: list[dict] = []
    for fid, ridx in targets:
        out_path = store.output_path(f"zoom/f{fid:06d}_r{ridx}.png")
        if out_path.is_file():
            # 已经裁过 = 这一轮没有新测量。★ 正是「重排旧信息不算反思」的落点。
            continue
        box, box_src = _crop_box(store, state, fid, ridx, crop_source)
        payload = _write_crop(store, fid, box, out_path)
        entry = {
            "kind": "zoom", "frame_id": fid, "region_idx": ridx,
            "crop": payload["out"], "crop_win": payload["out_win"],
            "bbox": box, "bbox_source": box_src,
            "measurement": "把该 region 的像素按原生分辨率单独取出（上一轮只有整帧缩放版）",
            "answer": None, "answer_src": None,
        }
        if jev is not None:
            import cv2

            patch = cv2.imread(str(out_path))
            # ★ 放大之后重问用的是【存在性（二选一）】，不是 N 选一：
            #   这里只有一个候选，没有比较压力，N 选一退化成「问它是不是」——
            #   而那正是实测 74% 都答「是」的那个形态。二选一至少把
            #   「是/不是」摆在同一道题的选项里。
            ans = jev.exists(patch, (state.get("query") or "")) if patch is not None else None
            if ans is not None:
                entry["answer"] = round(float(ans), 4)
                entry["answer_src"] = getattr(jev, "name", "jev")
                entry["answer_form"] = "existence-2way-on-crop"
                entry["measurement"] = ("在该 region 的原生分辨率裁剪上重问存在性："
                                        f"{getattr(jev, 'name', 'jev')} -> {entry['answer']}")
        out.append(entry)
    return out


def _new_frame(store: Any, state: dict, m: dict, reason: str) -> tuple[list[dict], list[int]]:
    """换帧：取视频里**原本没看过的**帧。

    ★ 出口处由 ``_assert_unseen`` 强制：与 seen_frames 的交集必须为空。
      这条不是风格 —— 规格 §6.1 的实测依据就是「剩下的帧与已看的几乎相邻」，
      一旦允许重复看已看过的帧，「换帧」这一整类新信息就退化成重试。
    """
    seen = set(_int_list(state.get("seen_frames")))
    dec = state.get("decision") or {}
    fid = dec.get("frame_id")
    ridx = dec.get("region_idx")
    cur_area = m["decision_area"]

    label = None
    if fid is not None and ridx is not None and int(ridx) >= 0:
        regions = store.require_regions(int(fid)).get("regions", [])
        label = next((r.get("label") for r in regions if int(r["idx"]) == int(ridx)), None)

    info: list[dict] = []
    frames: list[int] = []
    if label is None:
        # 连目标叫什么都不知道（决策落在零检出帧/无 region 上）⇒ 没法在别的帧上
        # 认出「同一个目标」。宁可这一轮拿不出新信息，也不拿别的物体充数。
        return info, frames
    for other in store.frame_ids():
        if other in seen or (fid is not None and int(other) == int(fid)):
            continue
        regions = store.require_regions(other).get("regions", [])
        # ★ 只认**同名**的 region。这里曾经写 `... or regions` 兜底，
        #   结果是把「同名目标更大」偷换成「这一帧里最大的那个东西更大」——
        #   而 measurement 文案还写着「同名目标」，于是读数看起来正常、结论是错的。
        #   认不出同名目标就该如实说「换帧拿不出新信息」，不能换个物体顶上。
        pool = [r for r in regions if r.get("label") == label]
        if not pool:
            continue
        best = max(pool, key=lambda r: (float(r.get("area_ratio", 0.0)), -int(r["idx"])))
        area = float(best.get("area_ratio", 0.0))
        if area <= cur_area:
            continue
        frames.append(int(other))
        info.append({
            "kind": "new_frame", "frame_id": int(other), "region_idx": int(best["idx"]),
            "area_ratio": round(area, 6), "previous_area_ratio": round(cur_area, 6),
            "label": best.get("label"), "tracked_by": f"label:{label}",
            "reason": reason,
            "measurement": (f"帧 {int(other)} 上同名目标（label={label!r}）面积 {area:.4f} > "
                            f"当前 {cur_area:.4f}：这是本帧之外的一次新测量，"
                            "且该帧从未进过 seen_frames"),
        })
        if len(info) >= 3:
            break
    return info, frames


def _new_axis(store: Any, state: dict, m: dict, backend: Any,
              backend_error: Optional[BackendUnavailable]) -> list[dict]:
    """换轴：换个问法问**原来看不到的**属性。"""
    already = set(m["axes_already_measured"])
    dec = state.get("decision") or {}
    fid = dec.get("frame_id")
    if fid is None:
        return []
    for axis in CANDIDATE_AXES:
        if axis in already:
            continue                     # 已经问过的轴 = 旧信息，不是新信息
        if backend is None or not backend.supports(axis):
            continue
        from .evidence import _read_frame      # 同一个取帧口径，只有一处

        regions = store.require_regions(int(fid)).get("regions", [])
        rows = backend.measure(axis, _read_frame(store, int(fid)), regions,
                               store.load_masks(int(fid)), state.get("query") or "")
        if not rows:
            continue
        return [{
            "kind": "new_axis", "frame_id": int(fid), "axis": axis,
            "value": rows,
            "src": (rows[0].get("src") if rows and isinstance(rows[0], dict) else None)
                   or getattr(backend, "name", None),
            "measurement": f"轴 {axis} 此前未被测量过（已测：{sorted(already) or '无'}）",
        }]
    return []


def _assert_unseen(new_frames: Sequence[int], seen: Sequence[int]) -> None:
    overlap = sorted(set(int(f) for f in new_frames) & set(int(f) for f in seen))
    if overlap:
        raise VrsError(
            f"reflect would re-show already-seen frames {overlap} as 'new information'",
            hint="this is a bug in reflect's frame selection, not a data problem: "
                 "re-asking about a frame the model already saw is a retry, not a reflection.",
        )


def run_reflect(store: Any, round_no: int, state: dict, *,
                fuse_rounds: Optional[int] = FUSE_ROUNDS,
                reflect_first: str = "low_visibility",
                crop_source: str = "yolo",
                min_area: float = MIN_AREA,
                jev_backend: Optional[str] = None,
                axis_backend: Optional[str] = None) -> tuple[dict, dict]:
    if reflect_first not in ("low_visibility", "by_measurement"):
        raise VrsError(f"unknown --reflect-first {reflect_first!r}",
                       hint="known: low_visibility (先跑确定性、零成本的那条) | by_measurement",
                       code=2)
    if crop_source not in ("yolo", "veason"):
        raise VrsError(f"unknown --crop-source {crop_source!r}",
                       hint="known: yolo (默认，① 的框) | veason (② 的框，更准但要等 ② 跑完)",
                       code=2)
    if int(round_no) < 1:
        raise VrsError(f"--round must be >= 1 (got {round_no})",
                       hint="轮次从 1 开始；上限由 --max-rounds 管（默认 2）。", code=2)
    store.require()

    events: list[dict] = []
    seen = _int_list(state.get("seen_frames"))
    m = measure(store, state, min_area)

    base = {"round": int(round_no), "measurements": m, "frames": [], "new_info": [],
            "fuse_rounds": None if not fuse_rounds else int(fuse_rounds)}

    false_premise = _premise_false(state)
    if false_premise is not None:
        # ★ 规格 §6.2：前提为假 ⇒ 终止，原样交付（不猜）。这里**不许**再产新信息，
        #   因为「目标不存在」这件事不是靠再测一次能翻案的。
        return {**base, "terminated": True, "reason": "premise_false",
                "premise_false": false_premise}, {"events": events}

    if fuse_rounds and int(round_no) > int(fuse_rounds):
        # ★ 这是保险丝，不是正常终止。措辞必须让人一眼看出「程序有问题」，
        #   而不是「想完了」—— 否则它会被当成轮次上限来规划预算。
        return {**base, "terminated": True, "reason": "fuse_round_limit", "fuse": True,
                "fuse_rounds": int(fuse_rounds),
                "note": ("保险丝触发：这不是正常终止。正常终止是 reason=no_new_information"
                         "（没有新测量可拿）。跑到这里说明每轮都在报新信息却停不下来，"
                         "先查 reflect 的新信息判据，别把它当成轮次上限。")}, {"events": events}

    jev = None
    jev_error = None
    try:
        jev = get_backend("jev", jev_backend)
    except BackendUnavailable as exc:
        jev_error = exc
    axis = None
    axis_error = None
    try:
        axis = get_backend("axis", axis_backend, store=store)
    except BackendUnavailable as exc:
        axis_error = exc

    plan: list[tuple[str, str]] = []
    if reflect_first == "low_visibility" and m["low_visibility"]:
        plan.append(("new_frame", "low_visibility"))
    if m["ambiguity"] or m["distractor"]:
        plan.append(("zoom", "ambiguity" if m["ambiguity"] else "distractor"))
    if m["temporal"]:
        plan.append(("new_frame", "temporal_mismatch"))
    if m["low_visibility"] and reflect_first != "low_visibility":
        plan.append(("new_frame", "low_visibility"))
    # 放大是主力（规格 §6.1 实测加权），所以它是兜底的第一顺位。
    plan.append(("zoom", "zoom_is_the_main_new_info"))
    plan.append(("new_axis", "axis_never_measured"))

    tried: list[str] = []
    for kind, reason in plan:
        if kind in tried:
            continue
        tried.append(kind)
        if kind == "zoom":
            info = _zoom(store, state, m, jev, crop_source)
            frames: list[int] = []
        elif kind == "new_frame":
            info, frames = _new_frame(store, state, m, reason)
        else:
            info = _new_axis(store, state, m, axis, axis_error)
            frames = []
        if info:
            _assert_unseen(frames, seen)
            if jev_error is not None and kind == "zoom":
                events.append({"event": "reflect_zoom_unread", "error": jev_error.error,
                               "hint": jev_error.hint,
                               "consequence": "crop 已产出（新测量），但没有模型读它。"})
            if kind == "new_axis" and axis_error is not None:
                events.append({"event": "reflect_axis_unavailable", "error": axis_error.error,
                               "hint": axis_error.hint})
            return {**base, "frames": frames, "new_info": info, "terminated": False,
                    "action": kind, "action_reason": reason,
                    "rounds_tried": tried}, {"events": events}

    # 三种都拿不出新测量 ⇒ 停。★ 这就是「重排旧信息不算反思」的出口，
    # 也是本模块【唯一】的正常终止：判据在 agent 手上（测量没了就停），
    # 不在一个数上。
    return {**base, "terminated": True, "normal_termination": True,
            "reason": "no_new_information", "rounds_tried": tried,
            "detail": ("放大已做过（crop 已存在）、没有没看过的帧、已测轴 "
                       f"{m['axes_already_measured']} 之外没有可用的件")
                      }, {"events": events}
