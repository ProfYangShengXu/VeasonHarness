"""④ 分路反思：按问题类型路由，每轮必须拿【新信息】。

## 铁律（用户 2026-10-13 拍板）

> **「重排旧信息」不算反思。判断一轮值不值得开，就看它有没有引入【新的测量】。**

落点只有一个：``new_info`` 里每一条都必须写清 ``measurement`` —— 「这一轮多出来的
那个测量是什么」。写不出 ``measurement`` 的条目根本进不了 ``new_info``；而终止条件
就是三种动作都产不出新测量（``reason="no_new_information"``）。

## ★ 没有轮次上限（新契约⑤）

旧版有 ``MAX_ROUNDS=2``，第 3 轮直接 ``round_limit`` 终止。**已去掉。** 终止权交给
agent：它按题目类型决定还要不要反思，本模块只负责回答「这一轮还拿不拿得到新测量」。
超限这种「防死循环」的裁决不该由 CLI 替 agent 下 —— 但铁律不变：拿不出新测量就停。

## 三种新信息，权重不同（实测依据）

===== ================================ ========
新信息                                 权重
===== ================================ ========
**放大** 候选区域裁出来单独看             ★★ 主力
**换轴** 换个问法问原来看不到的属性        中
**换帧** 取视频里原本没看过的帧            低
===== ================================ ========

★★ 为什么放大是主力：ReVOS 269 个视频中位 18 帧（96% 在 17–32 帧），8 帧采样已经
抽掉一半多，「换帧」剩下的帧与已看的几乎相邻；而模型每问只有 1024 个图像 token 预算，
摊在整帧上目标那块可能只分到几十个。裁出来单独喂 = 把全部预算给这一块。

## ★ 裁剪框用 Veason 的框（新契约④）

放大剪的是 ② 的框，不是 ① 的 YOLO 框。代价是 ①② 不能并行（② 跑完才有框），
接受。取不到 Veason 框就**报错**，不静默退回 YOLO —— 退了源就等于说不清这块像素
是哪来的，而那正是「放大」这条新信息的全部内容。

## ★ 谁来决定调哪个动作：agent，不是本模块（新契约⑥）

三种动作（放大 / 换帧 / 换轴）由本模块**实现**，但**选哪个不归它管**。
``run_reflect`` 有一个 ``chooser`` 缝：

* **不传** ``chooser`` —— 走下面那条确定性兜底 ``_default_plan``，行为与旧版逐字相同
  （现有测试因此全绿，向后兼容）。
* **传了** ``chooser`` —— 由它决定。真正的实现是 ``reflect_agent.MllmChooser``：
  它把**当前帧的图 + 元数据 + 动作菜单**交给 MLLM，让模型按规则做 agentic search。

★ 为什么必须留这条缝而不是直接换掉：确定性那条兜底有实测依据（放大是主力、
换帧权重低），它是**资产**不是包袱；而 MLLM 决策要能跟它**对照**，才谈得上谁更好。

## 判据必须【测量出来】，不是【问出来】

本模块只做确定性测量：面积、IoU、分数距离、掩码一致。一次模型也没问「你觉得这题
是不是指代不清」—— 那是把路由键变成不可复算的判断。``measurements`` 每个数都写在
输出里，谁都能拿同一份素材复算一遍。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .errors import VrsError
from .paths import for_agent, for_windows
from .video import load_bgr, region_area_ratio
from . import evidence as _evidence

#: 低可见判据的面积门限。★ **推断值，未标定**。
MIN_AREA = 0.02

#: 「候选间相似度高、分数接近」的两条门限。★ 也是推断值。
AMBIGUITY_AREA_REL = 0.20
AMBIGUITY_SCORE_GAP = 0.10

#: 「选中帧与相邻帧掩码不一致」的 IoU 门限。★ 推断值。
TEMPORAL_MASK_IOU = 0.30

#: 可按需补问的轴，顺序 = 补问的优先级（成本从低到高）。
CANDIDATE_AXES = ("occlusion", "text", "action", "relation")

#: 哪个证据轴靠哪些 senses 轴算出来（"relation" 不在其中：它由 RelateAnything 产）。
SENSES_AXES_FOR = {"occlusion": ("depth",), "text": ("text",), "action": ("action", "face")}


def _query_key(query: str) -> str:
    return hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:8]


def parse_state(raw: str) -> dict:
    """``--state`` 两吃：文件路径或内联 JSON。

    先按路径试，不是文件就当 JSON —— 顺序不能反：一个合法的 JSON 字符串也可能是
    路径不存在的文件名，但反过来不会。
    """
    if raw is None or not str(raw).strip():
        raise VrsError("no --state given",
                       hint='--state <json> or --state <file.json>; the state carries '
                            'query/candidates/evidence/decision/seen_frames/locate', code=2)
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


# --------------------------------------------------------------------------
# 类型边界（#514：reflect 是全 CLI 唯一吃 agent 手写 JSON 的命令）
# --------------------------------------------------------------------------

def _as_int(v: Any, what: str) -> int:
    if isinstance(v, bool) or v is None:
        raise VrsError(f"{what} must be an integer, got {v!r}",
                       hint="frame_id and region_idx are integers chosen from the tools' output.",
                       code=2)
    try:
        return int(v)
    except (TypeError, ValueError):
        raise VrsError(f"{what} must be an integer, got {v!r}",
                       hint="frame_id and region_idx are integers chosen from the tools' output.",
                       code=2) from None


def _int_list(v: Any, what: str) -> list[int]:
    out: list[int] = []
    for x in (v or []):
        out.append(_as_int(x, f"{what} entry"))
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
    """两个整帧 bool 掩码的 IoU。纯计算，不问模型。"""
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
# 测量
# --------------------------------------------------------------------------

def measure(store: Any, state: dict, min_area: float = MIN_AREA) -> dict:
    cands = list(state.get("candidates") or [])
    dec = state.get("decision") or {}
    if not isinstance(dec, dict):
        raise VrsError("decision must be a JSON object {frame_id, region_idx}",
                       hint="it is the agent's reply, not a bare number.", code=2)
    seen = set(_int_list(state.get("seen_frames"), "seen_frames"))
    fid = None if dec.get("frame_id") is None else _as_int(dec.get("frame_id"), "decision.frame_id")
    ridx = None if dec.get("region_idx") is None else _as_int(dec.get("region_idx"), "decision.region_idx")
    for c in cands:
        if not isinstance(c, dict):
            raise VrsError(f"candidate must be an object, got {type(c).__name__}",
                           hint="pass screen's `candidates` list unchanged.", code=2)
        if c.get("frame_id") is not None:
            _as_int(c.get("frame_id"), "candidate.frame_id")

    areas = sorted((_area(c) for c in cands), reverse=True)
    gap = (areas[0] - areas[1]) if len(areas) > 1 else 1.0
    ambiguity_area = bool(len(areas) > 1 and areas[0] > 0
                          and (areas[0] - areas[1]) / areas[0] <= AMBIGUITY_AREA_REL)
    scores = [float(c.get("score", c.get("jev_yes", 0.0)) or 0.0) for c in cands]
    ambiguity_score = bool(len(scores) > 1 and abs(scores[0] - scores[1]) <= AMBIGUITY_SCORE_GAP)
    # 分数距离只是一个【测量】记下来，单独不足以触发任何动作：不许用「谁更自信」解冲突。
    ambiguity = bool(ambiguity_area and ambiguity_score)

    my_area = _candidate_area(cands, fid, ridx)
    regions: list[dict] = []
    if fid is not None and int(fid) in set(store.frame_ids()):
        regions = store.require_regions(int(fid)).get("regions", [])

    if my_area <= 0.0 and fid is not None and ridx is not None and int(ridx) >= 0:
        for r in regions:
            if int(r["idx"]) == int(ridx):
                my_area = float(r.get("area_ratio") or 0.0)
                break

    distractor = False
    if len(regions) > 1 and ridx is not None:
        by_area = sorted(regions, key=lambda r: -float(r.get("area_ratio", 0.0)))
        top = float(by_area[0].get("area_ratio", 0.0))
        distractor = bool(top > 0 and int(by_area[0]["idx"]) != int(ridx)
                          and (top - my_area) / top <= AMBIGUITY_AREA_REL)

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


# --------------------------------------------------------------------------
# 裁剪框：★ 只用 Veason 的框（新契约④）
# --------------------------------------------------------------------------

def _veason_box(state: dict, fid: int) -> list[int]:
    """从 ``state["locate"]``（② 的输出）取这一帧的框。取不到就报错，不退回 YOLO。"""
    located = state.get("locate")
    if not located:
        raise VrsError(
            "zoom needs Veason's box, but the state carries no `locate` (② output)",
            hint="run `vrs locate` for the decision frame first and pass its `locate` array in "
                 "the state; the crop source is Veason by contract, not YOLO.",
        )
    for item in located:
        if int(item.get("frame_id", -1)) != int(fid):
            continue
        boxes = item.get("bbox_2d_list") or []
        if boxes:
            return [int(v) for v in boxes[0]]
    raise VrsError(
        f"Veason returned no box on frame {fid}, so there is nothing to zoom into",
        hint="pick a frame where ② produced a box (see the `locate` array), or fix ② first.",
    )


def _write_crop(store: Any, fid: int, box: list[int], out_path: Path) -> dict:
    """把 box（带 8px 余量）裁出来写成 PNG。**这一步就是新测量的物证。**"""
    import cv2

    img = load_bgr(store, fid)
    h, w = img.shape[:2]
    pad = 8
    x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad)
    x2, y2 = min(w, box[2] + pad), min(h, box[3] + pad)
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        raise VrsError(f"region crops to nothing on frame {fid}",
                       hint="the stored bbox is degenerate; re-run `vrs index`.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), patch):
        raise VrsError(f"could not write the zoom crop to {out_path}",
                       hint="check the index directory is writable and the drive has space.")
    return {"out": for_agent(out_path), "out_win": for_windows(out_path)}


def _crop_path(store: Any, query: str, fid: int, ridx: int, box: Sequence[int]) -> Path:
    """物证路径含 query + 框指纹，所以换 query 或换了框 = 一次新测量（#503）。"""
    boxh = hashlib.sha256(",".join(str(int(v)) for v in box).encode()).hexdigest()[:8]
    return store.output_path(f"zoom/q{_query_key(query)}/f{fid:06d}_r{ridx}_b{boxh}.png")


def _zoom(store: Any, state: dict, m: dict, jev: Any, events: Optional[list] = None) -> list[dict]:
    """放大：把候选区域按 **Veason 的框**裁出来单独看。

    ★ 逐个目标处理：某个候选没有 Veason 框（② 只对关键帧出框）时跳过它并记事件，
      不能让一个坏目标把整次放大动作打掉 —— 真实跑时正是这样丢掉了决策帧的 crop。
    """
    dec = state.get("decision") or {}
    targets: list[tuple[int, int]] = []
    if dec.get("frame_id") is not None and int(dec.get("region_idx", -1)) >= 0:
        targets.append((int(dec["frame_id"]), int(dec["region_idx"])))
    if m["ambiguity"] or m["distractor"]:
        for c in (state.get("candidates") or [])[:2]:
            if c.get("frame_id") is None or int(c.get("region_idx", -1)) < 0:
                continue
            pair = (int(c["frame_id"]), int(c["region_idx"]))
            if pair not in targets:
                targets.append(pair)

    query = state.get("query") or ""
    out: list[dict] = []
    for fid, ridx in targets:
        try:
            box = _veason_box(state, fid)
        except VrsError as exc:
            if events is not None:
                events.append({"event": "reflect_zoom_target_skipped", "frame_id": fid,
                               "error": exc.error, "hint": exc.hint})
            continue
        out_path = _crop_path(store, query, fid, ridx, box)
        if out_path.is_file():
            # 已经裁过这个 (query, 框) = 这一轮没有新测量。★「重排旧信息不算反思」。
            continue
        payload = _write_crop(store, fid, box, out_path)
        entry = {
            "kind": "zoom", "frame_id": fid, "region_idx": ridx,
            "crop": payload["out"], "crop_win": payload["out_win"],
            "bbox": box, "bbox_source": "veason-r1",
            "measurement": "把该 region 的像素按原生分辨率单独取出（上一轮只有整帧缩放版）",
            "answer": None, "answer_src": None,
        }
        if jev is not None:
            import cv2

            patch = cv2.imread(str(out_path))
            ans = None
            if patch is not None:
                ans = jev.confirm(patch, [0, 0, patch.shape[1], patch.shape[0]], query)
            if ans is not None:
                entry["answer"] = round(float(ans), 4)
                entry["answer_src"] = getattr(jev, "name", "jev")
                entry["measurement"] = ("在该 region 的原生分辨率裁剪上重问 query："
                                        f"{getattr(jev, 'name', 'jev')} -> {entry['answer']}")
        out.append(entry)
    return out


def _new_frame(store: Any, state: dict, m: dict, reason: str) -> tuple[list[dict], list[int]]:
    """换帧：取视频里**原本没看过的**、目标更大的一帧。"""
    seen = set(_int_list(state.get("seen_frames"), "seen_frames"))
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
        # 连目标叫什么都不知道 ⇒ 没法在别的帧上认出「同一个目标」。宁可这一轮拿不出
        # 新信息，也不拿别的物体充数。
        return info, frames
    for other in store.frame_ids():
        if other in seen or (fid is not None and int(other) == int(fid)):
            continue
        regions = store.require_regions(other).get("regions", [])
        # ★ 只认**同名**的 region。认不出同名目标就如实说「换帧拿不出新信息」，不能
        #   换个物体顶上 —— 那会让读数看起来正常、结论是错的。
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
                            f"当前 {cur_area:.4f}：这是本帧之外的一次新测量，且该帧从未进过 seen_frames"),
        })
        if len(info) >= 3:
            break
    return info, frames


def _run_real_senses(store: Any, axis_names: Sequence[str], fid: int) -> None:
    from .senses import run_senses

    run_senses(store, axes=list(axis_names), frames=[int(fid)])


def _new_axis(store: Any, state: dict, m: dict, *,
              senses_runner: Optional[Callable[..., Any]] = None,
              events: Optional[list] = None) -> list[dict]:
    """换轴：换个问法问**原来看不到的**属性。

    缓存里有就直接读（那也是一次真测量，只是早算过）；缓存里没有且该轴是 senses 能
    算的，就跑真 senses 补算。两条路都产出一条带 ``measurement`` 的新信息。

    ★ 某个轴的小模型没装（如 ``rapidocr_onnxruntime``）时**跳过它并记事件**，不把整条
      链打死：基座的 senses 本来就把每个轴当可选件（缺件该轴降级）。事件里带真实
      错误，所以这不是静默吞错。
    """
    already = set(m["axes_already_measured"])
    dec = state.get("decision") or {}
    fid = dec.get("frame_id")
    if fid is None:
        return []
    fid = int(fid)
    meta = store.require()
    regions = store.require_regions(fid).get("regions", [])
    senses, relations = _evidence.cached_frame_measurements(store, fid, regions)
    for axis in CANDIDATE_AXES:
        if axis in already:
            continue                     # 已经问过的轴 = 旧信息，不是新信息
        values, src = _evidence.read_axis(axis, regions, meta, senses, relations,
                                          state.get("query") or "")
        if not values and axis in SENSES_AXES_FOR:
            runner = senses_runner or _run_real_senses
            try:
                runner(store, SENSES_AXES_FOR[axis], fid)
            except Exception as exc:
                if events is not None:
                    events.append({
                        "event": "reflect_axis_unavailable", "frame_id": fid, "axis": axis,
                        "senses_axes": list(SENSES_AXES_FOR[axis]), "error": str(exc)[:300],
                        "hint": "install this axis model on the GPU box (or skip the axis); "
                                "the other axes are still tried.",
                    })
                continue
            senses, relations = _evidence.cached_frame_measurements(store, fid, regions)
            values, src = _evidence.read_axis(axis, regions, meta, senses, relations,
                                              state.get("query") or "")
        if not values:
            continue
        return [{
            "kind": "new_axis", "frame_id": fid, "axis": axis,
            "value": values, "src": src,
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


#: ★ 动作菜单（新契约⑥）：把三种能力**如实**摆给 agent 看。
#: 每条都写清「产出什么新测量」和「代价」—— 模型据此自己选，而不是被代码安排。
#: ``produces`` 必须写得出**测量**（铁律：写不出 measurement 的不算新信息）。
#: agent 能【执行】的动作（与菜单同源，防自创）
AGENT_KINDS = ("zoom", "new_frame", "new_axis")

#: ★ agent 的终止答复。提示词里明确允许它答 ``stop``（「信息够了」），
#: 所以校验表必须同样放行 —— 否则模型按提示词答 stop 会被判成坏产物，
#: 退回确定性兜底，把「agent 判断该停」这个信号整个丢掉（实测踩过）。
AGENT_STOP = "stop"

ACTION_MENU: tuple[dict[str, str], ...] = (
    {
        "kind": "zoom",
        "what": "把候选区域裁出来单独看（裁的是 ② 的框）",
        "produces": "一个更清晰的 crop + 该框内目标的掩码元数据"
                    "（面积 / 框内填充率 / 掩码外接框）",
        "cost": "便宜：一次裁剪 + 一次局部测量，不问模型",
        "when_useful": "候选之间分不开、或目标只占画面很小一块时",
    },
    {
        "kind": "new_frame",
        "what": "取视频里原本没看过的帧",
        "produces": "新帧的候选 + 该帧的面积/位置测量",
        "cost": "中等：要重新跑 ② 定位",
        "when_useful": "当前帧目标看不见、或时间对不上时",
    },
    {
        "kind": "new_axis",
        "what": "换个问法问此前没测过的属性轴",
        "produces": "一条此前没测过的轴上的值（如遮挡/关系/动作）",
        "cost": "较贵：可能要跑该轴的模型",
        "when_useful": "候选在几何量上分不开，但描述里提了属性/关系时",
    },
)


def action_menu() -> list[dict[str, str]]:
    """给 agent 看的动作菜单（深拷贝，调用方改不动常量）。"""
    return [dict(a) for a in ACTION_MENU]


def _default_plan(reflect_first: str, m: dict) -> list[tuple[str, str]]:
    """确定性兜底 plan（旧版逐字逻辑，抽成函数是为了能和 chooser 对照）。"""
    plan: list[tuple[str, str]] = []
    if reflect_first == "low_visibility" and m["low_visibility"]:
        plan.append(("new_frame", "low_visibility"))
    if m["ambiguity"] or m["distractor"]:
        plan.append(("zoom", "ambiguity" if m["ambiguity"] else "distractor"))
    if m["temporal"]:
        plan.append(("new_frame", "temporal_mismatch"))
    if m["low_visibility"] and reflect_first != "low_visibility":
        plan.append(("new_frame", "low_visibility"))
    plan.append(("zoom", "zoom_is_the_main_new_info"))     # 放大是主力，兜底第一顺位
    plan.append(("new_axis", "axis_never_measured"))
    return plan



def run_reflect(store: Any, round_no: int, state: dict, *,
                reflect_first: str = "low_visibility",
                min_area: float = MIN_AREA,
                jev: Optional[Any] = None,
                senses_runner: Optional[Callable[..., Any]] = None,
                chooser: Optional[Callable[[dict, dict, list[dict]], Any]] = None
                ) -> tuple[dict, dict]:
    """一次反思。★ 没有 ``max_rounds``：能不能停由 agent 按新信息判断（契约⑤）。"""
    if reflect_first not in ("low_visibility", "by_measurement"):
        raise VrsError(f"unknown --reflect-first {reflect_first!r}",
                       hint="known: low_visibility (先跑确定性、零成本的那条) | by_measurement",
                       code=2)
    if int(round_no) < 1:
        raise VrsError(f"--round must be >= 1 (got {round_no})",
                       hint="轮次从 1 开始，只是记账；没有上限。", code=2)
    store.require()

    events: list[dict] = []
    seen = _int_list(state.get("seen_frames"), "seen_frames")
    m = measure(store, state, min_area)

    base = {"round": int(round_no), "measurements": m, "frames": [], "new_info": []}

    if state.get("absent") is True:
        # ★ 前提为假 ⇒ 终止，原样交付（不猜）。这是 ① 的 ``absent`` 字段，不是靠
        #   模型再答一次翻案。
        return {**base, "terminated": True, "reason": "premise_false"}, {"events": events}

    if jev is None:
        try:
            from .screen import default_jev

            jev = default_jev()
        except VrsError:
            jev = None               # 放大仍可产出 crop 这次新测量；answer 诚实写 null

    # ★ 新契约⑥：谁来决定调哪个动作。给了 chooser 就交给它（MLLM 看着图 + 元数据选），
    #   不给就走确定性兜底 —— 两条路都在同一条铁律（new_info 必须有 measurement）之下。
    if chooser is not None:
        picked = chooser(state, m, action_menu())
        kind = str(picked[0] if picked else "")
        # ★ agent 说「停」是**正常终止**，不是坏产物：它判断没有新测量可拿了。
        #   这与本模块自己走到 ``no_new_information`` 是同一结论的两个来源，
        #   ``reason`` 上分开记，别混。
        if kind == AGENT_STOP:
            return {**base, "terminated": True, "reason": "agent_says_stop",
                    "action": None, "action_reason": str(picked[1])[:300],
                    "rounds_tried": []}, {"events": events + [{
                        "event": "reflect_terminated_by_agent",
                        "reason": str(picked[1])[:300],
                        "note": "agent 判断已无新测量可拿 —— 正常出口，不是失败"}]}
        # ★ 模型没给出合法动作（拒答/格式错/超时/自创动作）⇒ 退回确定性兜底，
        #   绝不把空动作当默认动作跑（那等于静默换了模型没选的那个动作）。
        if kind in AGENT_KINDS:
            plan = [(kind, str(picked[1]))]
            events.append({"event": "reflect_action_chosen_by_agent",
                           "kind": kind, "reason": str(picked[1])[:300]})
        else:
            plan = _default_plan(reflect_first, m)
            events.append({"event": "reflect_action_chooser_fallback",
                           "got": kind[:60],
                           "reason": str(picked[1])[:200] if picked else "",
                           "note": "chooser 没给出合法动作，这一轮按确定性兜底走"})
    else:
        plan = _default_plan(reflect_first, m)

    tried: list[str] = []
    for kind, reason in plan:
        if kind in tried:
            continue
        tried.append(kind)
        if kind == "zoom":
            try:
                info = _zoom(store, state, m, jev, events)
            except VrsError as exc:
                # 兜底：单个目标的缺框已经在 _zoom 里跳过了，这里只接意料之外的失败。
                events.append({"event": "reflect_zoom_unavailable", "error": exc.error,
                               "hint": exc.hint})
                info = []
            frames: list[int] = []
        elif kind == "new_frame":
            info, frames = _new_frame(store, state, m, reason)
        else:
            info = _new_axis(store, state, m, senses_runner=senses_runner, events=events)
            frames = []
        if info:
            _assert_unseen(frames, seen)
            return {**base, "frames": frames, "new_info": info, "terminated": False,
                    "action": kind, "action_reason": reason,
                    "rounds_tried": tried}, {"events": events}

    # 三种都拿不出新测量 ⇒ 停。★ 这就是「重排旧信息不算反思」的出口。
    return {**base, "terminated": True, "reason": "no_new_information",
            "rounds_tried": tried,
            "detail": ("放大已做过（该 query 的 crop 已存在）、没有没看过的更大同名帧、"
                       f"已测轴 {m['axes_already_measured']} 之外没有可用的件")}, {"events": events}
