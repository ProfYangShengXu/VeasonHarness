"""③ 打标签 + 按轴路由 —— 证据包，每条带 src。

## 三条纪律，每条都在代码里有个落点

① **能用确定性计算的一律不配模型**（面积 / 数量）
   → ``_read_area`` / ``_read_count`` 直接算，代码路径上碰不到任何模型。
② **每个件只对它负责的那枚标签出数**
   → ``_READERS`` 一个轴一个 reader，reader 返回什么决定 value 里有什么。
③ ★ **``src`` 必须指向真正产出该值的那个模型**
   → ``FIELD_SRC``：src 由「读的是哪个字段」决定，不由「谁被调过」决定。多 provider
     并存时标错 = 说不清出处。已经踩过：``senses.py`` 里 expression（CLIP 产出）被
     写成 ``mediapipe-face-landmarker`` —— 本模块不复制那个错，并且顺手修了它。

## 为什么读缓存，而不是每次都调模型

``vrs senses`` / ``vrs relate`` 已经把每帧每个 region 的轴算过一次并落在盘上，
且它们跑的就是真模型（CLIP / Depth-Anything / VideoMAE / OCR / RelateAnything）。
缓存里存的就是那次真测量的结果，``src`` 照样指得清，只是不重算。缓存失效判据用
``regions_sha256``（region 表变了，旧测量就不再描述这张表）。

新契约①取消了「可注入后端」注册表：没有 ``--axis-backend``，也没有
``auto -> unavailable``。某个轴缓存里没有就是一条显式
``evidence_axis_unavailable`` 事件，提示先跑 ``vrs senses`` / ``vrs relate``，
绝不静默编一个数出来。
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .errors import VrsError
from .store import sha256_json
from .video import load_bgr, parse_frame_ids, region_area_ratio

#: 轴 → 该轴该由哪个件答（路由表）。
AXIS_SRC = {
    "area": "computed",
    "count": "computed",
    "attribute": "yoloe-pf-lvis4585",
    "relation": "relsgg-vits16plus",
    "action": "clip-vitb32-expression",
    "occlusion": "depth-anything-v2-small",
    "text": "rapidocr-onnxruntime",
    # visibility 不新模型：它读的是 SAM2 掩码几何 + 同索引面积相对量，
    # 由 visibility.py 现算（见该模块顶部「为什么不合成单一分数」）。
    "visibility": "sam2+computed",
}
COMPUTED = ("area", "count")

#: ★ 纪律③的落点：读哪个字段 ⇒ 写哪个 src。看不清出处时不许猜。
FIELD_SRC = {
    "expression": "clip-vitb32-expression",
    "depth": "depth-anything-v2-small",
    "text": "rapidocr-onnxruntime",
    "action": "videomae-small-kinetics",
    "pose": "yolo11n-pose",
    "track": "bytetrack",
    "blendshape": "mediapipe-face-landmarker",
    "edges": "relsgg-vits16plus",
    "region_label": "yoloe-pf-lvis4585",
    "computed": "computed",
}

#: 可见性三档的门限。★ **推断值，未标定** —— 切点要按素材标。
VISIBILITY_HIGH = 0.10
VISIBILITY_MID = 0.02


def parse_axes(raw: str) -> list[str]:
    if raw is None or not str(raw).strip():
        raise VrsError("no --axes given",
                       hint=f"usage: vrs evidence <index_dir> --frames 1,2 --query \"...\" "
                            f"--axes {','.join(AXIS_SRC)}",
                       code=2)
    out: list[str] = []
    for part in str(raw).split(","):
        a = part.strip().lower()
        if not a:
            continue
        if a not in AXIS_SRC:
            raise VrsError(
                f"unknown axis {a!r}",
                hint=f"known axes: {', '.join(AXIS_SRC)}. The axis is the routing key: "
                     "one axis, one model (or zero, for area/count).",
                code=2,
            )
        if a not in out:
            out.append(a)
    if not out:
        raise VrsError(f"--axes {raw!r} contains no axis", hint="pass at least one.", code=2)
    return out


def visibility(area_ratio: float) -> str:
    """面积 → 三档。确定性，零模型（纪律①）。"""
    if area_ratio >= VISIBILITY_HIGH:
        return "high"
    if area_ratio >= VISIBILITY_MID:
        return "mid"
    return "low"


def _biggest(regions: Sequence[dict], meta: dict) -> Optional[dict]:
    if not regions:
        return None
    return max(regions, key=lambda r: (region_area_ratio(r, meta), -int(r["idx"])))


# --------------------------------------------------------------------------
# 每个轴的 reader：返回 {region_idx: value} + 该轴的 src
# --------------------------------------------------------------------------

def _read_area(regions, meta, senses, relations, query) -> tuple[dict, str]:
    return ({int(r["idx"]): {"area_ratio": round(region_area_ratio(r, meta), 6)}
             for r in regions}, FIELD_SRC["computed"])


def _read_count(regions, meta, senses, relations, query) -> tuple[dict, str]:
    # n_suspect = 检测器多框数一下（检测副产品），不是模型再答一次。
    return ({int(r["idx"]): {"n_suspect": len(regions)} for r in regions},
            FIELD_SRC["computed"])


def _read_attribute(regions, meta, senses, relations, query) -> tuple[dict, str]:
    """属性：用**索引里那次** prompt-free YOLOE 标签。

    ★ src 因此是 ``yoloe-pf-lvis4585``（索引时跑的），不是 ``set_classes`` 的在线
      结果：``yoloe-11s-seg-pf.pt`` 根本不收文本 prompt。把出处写成 prompted 会是
      「多 provider 并存时标错」那类错误 —— 宁可标出真正产出它的那个。
    """
    return ({int(r["idx"]): {"label": str(r.get("label", "")),
                             "score": round(float(r.get("score", 0.0)), 4)}
             for r in regions}, FIELD_SRC["region_label"])


def _read_relation(regions, meta, senses, relations, query) -> tuple[dict, str]:
    if not relations:
        return {}, FIELD_SRC["edges"]
    out: dict[int, dict] = {}
    for e in relations.get("edges", []):
        for side, other in (("subject", "object"), ("object", "subject")):
            idx = int(e[side])
            out.setdefault(idx, {"edges": []})["edges"].append({
                "relation": e.get("relation"), side: int(e[side]), other: int(e[other]),
                "score": e.get("score"),
            })
    return out, FIELD_SRC["edges"]


def _read_action(regions, meta, senses, relations, query) -> tuple[dict, str]:
    """动作/状态。★ 优先表情头，不优先 VideoMAE。

    实测：卡通域里 VideoMAE 是噪声，CLIP 零样本表情头可用。所以同一帧两个来源都在
    时选表情头 —— 并且 src 跟着换，因为它确实换了。
    """
    if not senses:
        return {}, FIELD_SRC["expression"]
    out: dict[int, dict] = {}
    src = FIELD_SRC["expression"]
    saw_expression = False
    for idx, slot in (senses.get("regions") or {}).items():
        face = slot.get("face") or {}
        if face.get("expression"):
            saw_expression = True
            out[int(idx)] = {"expression": face["expression"],
                             "expression_score": face.get("expression_score")}
    if not saw_expression:
        for idx, slot in (senses.get("regions") or {}).items():
            a = (slot.get("action") or {})
            if a.get("label"):
                out[int(idx)] = {"label": a["label"], "score": a.get("score")}
        src = FIELD_SRC["action"]
    return out, src


def _read_occlusion(regions, meta, senses, relations, query) -> tuple[dict, str]:
    out: dict[int, dict] = {}
    for idx, slot in ((senses or {}).get("regions") or {}).items():
        d = slot.get("depth")
        if d:
            out[int(idx)] = {"depth_rank": d.get("rank_in_frame"),
                             "mean_rel": d.get("mean_rel")}
    return out, FIELD_SRC["depth"]


def _read_text(regions, meta, senses, relations, query) -> tuple[dict, str]:
    out: dict[int, dict] = {}
    for idx, slot in ((senses or {}).get("regions") or {}).items():
        items = slot.get("text") or []
        if items:
            out[int(idx)] = {"texts": [i.get("text") for i in items],
                             "scores": [i.get("score") for i in items]}
    return out, FIELD_SRC["text"]


_READERS: dict[str, Callable[..., tuple[dict, str]]] = {
    "area": _read_area,
    "count": _read_count,
    "attribute": _read_attribute,
    "relation": _read_relation,
    "action": _read_action,
    "occlusion": _read_occlusion,
    # visibility 要读掩码原图 ⇒ 签名比其他 reader 多 store/frame_id 两个入参，
    # 由 read_axis 按需透传（见下面的分派）。
    "visibility": None,
    "text": _read_text,
}


def read_axis(axis: str, regions: list[dict], meta: dict, senses: Optional[dict],
              relations: Optional[dict], query: str,
              store: Any = None, frame_id: Optional[int] = None) -> tuple[dict, str]:
    """给 reflect 的「换轴」用：从同一批缓存里读一个轴。"""
    if axis not in _READERS:
        raise VrsError(f"unknown axis {axis!r}", hint=f"known: {', '.join(_READERS)}", code=2)
    if axis == "visibility":
        from . import visibility
        return visibility.read(regions, meta, senses, relations, query,
                               store=store, frame_id=frame_id)
    return _READERS[axis](regions, meta, senses, relations, query)


def cached_frame_measurements(store: Any, fid: int, regions: list[dict]
                              ) -> tuple[Optional[dict], Optional[dict]]:
    """读一帧的 senses / relations，并把「表变了」的旧测量当没有（带事件由调用方发）。"""
    senses = store.load_senses(fid)
    if senses and senses.get("regions_sha256") != sha256_json(regions):
        senses = None
    relations = store.load_relations(fid)
    if relations and relations.get("regions_sha256") != sha256_json(regions):
        relations = None
    return senses, relations


def run_evidence(store: Any, frames: str, query: str, axes: str) -> tuple[dict, dict]:
    if not (query or "").strip():
        raise VrsError("no query given",
                       hint='usage: vrs evidence <index_dir> --frames 1,2 --query "..."', code=2)
    wanted_axes = parse_axes(axes)
    wanted_frames = parse_frame_ids(frames)
    meta = store.require()
    events: list[dict] = []

    evidence: list[dict] = []
    for fid in wanted_frames:
        store.require_frame(fid)
        regions = store.require_regions(fid).get("regions", [])
        senses, relations = cached_frame_measurements(store, int(fid), regions)
        if senses is None and store.load_senses(int(fid)) is not None:
            events.append({"event": "evidence_stale_senses_dropped", "frame_id": int(fid),
                           "note": "senses 记录的是另一张 region 表的测量，已忽略。"})

        for axis in wanted_axes:
            values, src = read_axis(axis, regions, meta, senses, relations, query,
                                    store=store, frame_id=fid)
            if not values and axis not in COMPUTED:
                events.append({
                    "event": "evidence_axis_unavailable", "frame_id": int(fid), "axis": axis,
                    "src_that_would_have_produced_it": AXIS_SRC[axis],
                    "reason": "no cached measurement for this frame",
                    "hint": ("run `vrs senses <index_dir>` (occlusion/action/text) or "
                             "`vrs relate <index_dir> <frame_id> ...` (relation) once; "
                             "their output is reused, not recomputed."),
                })
                continue
            rep = _biggest(regions, meta)
            rep_idx = int(rep["idx"]) if rep is not None else None
            value = dict(values.get(rep_idx, {})) if rep_idx is not None else {}
            if not value:
                rep_idx = sorted(values)[0]
                value = dict(values[rep_idx])
            value["region_idx"] = rep_idx
            value["per_region"] = [
                {"region_idx": i, **values[i]} for i in sorted(values)
            ][:32]     # 一帧 region 数在个位数；32 只是防御性上限
            evidence.append({"frame_id": int(fid), "axis": axis, "value": value, "src": src})

    return {"evidence": evidence}, {"events": events}
