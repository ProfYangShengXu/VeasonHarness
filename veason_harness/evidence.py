"""③ 打标签 + 按轴路由小模型 —— 证据包，每条带 src。

## 三条纪律（规格 §4.2/§4.3），每条都在代码里有个落点

① **能用确定性计算的一律不配模型**（面积 / 数量 / IoU）
   → ``_COMPUTED`` 里的轴直接算，代码路径上碰不到任何 backend。
② **每个件只对它负责的那枚标签出数**
   → ``_READERS`` 一个轴一个 reader，reader 返回什么决定 value 里有什么；
     不存在「顺手把 depth 也塞进 action 的 value」这种可能。
③ ★ **``src`` 必须指向真正产出该值的那个模型**
   → ``FIELD_SRC``：src 由「读的是哪个字段」决定，不由「谁被调过」决定。
     多 provider 并存时标错 = 说不清出处 —— 这是已经踩过的坑：
     ``senses.py`` 里 expression（CLIP 产出）被写成
     ``src: mediapipe-face-landmarker``。本模块不复制那个错，并且顺手修了它。

## 为什么先读缓存，而不是每次都调模型

`veason senses` / `veason relate` 已经把每帧每个 region 的轴算过一次并落在盘上。
再调一次同一个模型只会得到同一个数（同样的随机性下），却要花 1.3–2.3 s/帧。
所以本命令的顺序是：**缓存（真测量，真 src）→ 注入的 backend（活算）→ 明确说不可用**。
「读缓存」不是降级 —— 缓存里存的就是那次真测量的结果，src 照样指得清。
缓存失效判据用 ``regions_sha256``（region 表变了，旧测量就不再描述这张表）。
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .backends import BackendUnavailable, get_backend
from .errors import VrsError
from .store import sha256_json

#: 轴 → 该轴该由哪个件答（规格 §4.2 的路由表）。
AXIS_SRC = {
    "area": "computed",
    "count": "computed",
    "attribute": "yoloe-prompted-set-classes",
    "relation": "relsgg-vits16plus",
    "action": "clip-vitb32-expression",
    "occlusion": "depth-anything-v2-small",
    "text": "rapidocr-onnxruntime",
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

#: 可见性三档的门限。★ **推断值，未标定** —— 规格只说「面积占比公式」「三档」，
#: 没给切点。0.10 / 0.02 是按 720p 帧上一个中等目标占 10% 面积取的。
VISIBILITY_HIGH = 0.10
VISIBILITY_MID = 0.02


def parse_axes(raw: str) -> list[str]:
    if raw is None or not str(raw).strip():
        raise VrsError("no --axes given",
                       hint=f"usage: veason evidence <index_dir> --frames 1,2 --query \"...\" "
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


def parse_frames(raw: str) -> list[int]:
    if raw is None or not str(raw).strip():
        raise VrsError("no --frames given",
                       hint='usage: veason evidence <index_dir> --frames 850,384 --query "..."', code=2)
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise VrsError(f"--frames {raw!r} is not a comma-separated list of ints",
                           hint="e.g. --frames 850,384", code=2) from None
    if not out:
        raise VrsError(f"--frames {raw!r} contains no frame ids", hint="pass at least one.", code=2)
    return out


def visibility(area_ratio: float) -> str:
    """面积 → 三档。确定性，零模型（纪律①）。"""
    if area_ratio >= VISIBILITY_HIGH:
        return "high"
    if area_ratio >= VISIBILITY_MID:
        return "mid"
    return "low"


def _area_ratio(region: dict, meta: dict) -> float:
    if "area_ratio" in region:
        return float(region["area_ratio"])
    w = float(meta.get("width") or 1) or 1.0
    h = float(meta.get("height") or 1) or 1.0
    x1, y1, x2, y2 = (float(v) for v in region["bbox"])
    return max(0.0, (x2 - x1) * (y2 - y1)) / (w * h)


def _biggest(regions: Sequence[dict], meta: dict) -> Optional[dict]:
    if not regions:
        return None
    return max(regions, key=lambda r: (_area_ratio(r, meta), -int(r["idx"])))


# --------------------------------------------------------------------------
# 每个轴的 reader：返回 {region_idx: value} + 该轴的 src
# --------------------------------------------------------------------------

def _read_area(regions: Sequence[dict], meta: dict, senses: Optional[dict],
               relations: Optional[dict], query: str) -> tuple[dict, str]:
    return ({int(r["idx"]): {"area_ratio": round(_area_ratio(r, meta), 6)}
             for r in regions}, FIELD_SRC["computed"])


def _read_count(regions: Sequence[dict], meta: dict, senses: Optional[dict],
                relations: Optional[dict], query: str) -> tuple[dict, str]:
    # n_suspect = 检测器多框数一下（检测副产品），不是模型再答一次。
    return ({int(r["idx"]): {"n_suspect": len(regions)} for r in regions},
            FIELD_SRC["computed"])


def _read_attribute(regions: Sequence[dict], meta: dict, senses: Optional[dict],
                    relations: Optional[dict], query: str) -> tuple[dict, str]:
    """属性：有 senses 也没有的属性槽 —— 这里用**索引里那次的** YOLOE 标签。

    ★ src 因此是 ``yoloe-pf-lvis4585``（prompt-free，索引时跑的），
      **不是** ``yoloe-prompted-set-classes``。规格 §4.2 把属性路由到
      ``set_classes``，那需要 query 词表在线跑一次；本机没接线时，盘上的
      标签就是唯一真实存在的属性测量。把它的出处写成 prompted 会是
      「多 provider 并存时标错」那类错误 —— 宁可标出真正产出它的那个。
    """
    return ({int(r["idx"]): {"label": str(r.get("label", "")),
                             "score": round(float(r.get("score", 0.0)), 4)}
             for r in regions}, FIELD_SRC["region_label"])


def _read_relation(regions: Sequence[dict], meta: dict, senses: Optional[dict],
                   relations: Optional[dict], query: str) -> tuple[dict, str]:
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


def _read_action(regions: Sequence[dict], meta: dict, senses: Optional[dict],
                 relations: Optional[dict], query: str) -> tuple[dict, str]:
    """动作/状态。★ 优先表情头，不优先 VideoMAE。

    规格 §4.2 实测：**卡通域里 VideoMAE 是噪声，CLIP 零样本表情头可用**。
    所以同一帧两个来源都在时，选表情头 —— 并且 src 跟着换，因为它确实换了。
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


def _read_occlusion(regions: Sequence[dict], meta: dict, senses: Optional[dict],
                    relations: Optional[dict], query: str) -> tuple[dict, str]:
    out: dict[int, dict] = {}
    for idx, slot in ((senses or {}).get("regions") or {}).items():
        d = slot.get("depth")
        if d:
            out[int(idx)] = {"depth_rank": d.get("rank_in_frame"),
                             "mean_rel": d.get("mean_rel")}
    return out, FIELD_SRC["depth"]


def _read_text(regions: Sequence[dict], meta: dict, senses: Optional[dict],
               relations: Optional[dict], query: str) -> tuple[dict, str]:
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
    "text": _read_text,
}


def run_evidence(store: Any, frames: str, query: str, axes: str, *,
                 axis_backend: Optional[str] = None) -> tuple[dict, dict]:
    if not (query or "").strip():
        raise VrsError("no query given",
                       hint='usage: veason evidence <index_dir> --frames 1,2 --query "..."', code=2)
    wanted_axes = parse_axes(axes)
    wanted_frames = parse_frames(frames)
    meta = store.require()
    events: list[dict] = []

    backend = None
    backend_error: Optional[BackendUnavailable] = None
    if any(a not in COMPUTED for a in wanted_axes):
        try:
            backend = get_backend("axis", axis_backend, store=store)
        except BackendUnavailable as exc:
            backend_error = exc

    evidence: list[dict] = []
    for fid in wanted_frames:
        store.require_frame(fid)
        regions = store.require_regions(fid).get("regions", [])
        # 缓存有效性判据：senses 里记的 regions_sha256 必须就是当前这张表。
        # 表变了（detect 追加过 region）旧测量就不再描述它，必须当没有。
        senses = store.load_senses(fid)
        if senses and senses.get("regions_sha256") != sha256_json(regions):
            events.append({"event": "evidence_stale_senses_dropped", "frame_id": int(fid),
                           "note": "senses 记录的是另一张 region 表的测量，已忽略。"})
            senses = None
        relations = store.load_relations(fid)
        if relations and relations.get("regions_sha256") != sha256_json(regions):
            relations = None

        for axis in wanted_axes:
            reader = _READERS[axis]
            values, src = reader(regions, meta, senses, relations, query)
            if not values and axis not in COMPUTED:
                values, src = _try_backend(backend, backend_error, axis, store, fid,
                                           regions, query, events)
            if not values:
                events.append({
                    "event": "evidence_axis_unavailable", "frame_id": int(fid), "axis": axis,
                    "src_that_would_have_produced_it": AXIS_SRC[axis],
                    "reason": ("no cached measurement for this frame and no live backend"
                               if backend_error else "the routed model returned nothing"),
                    "hint": (backend_error.hint if backend_error else
                             "run `veason senses <index_dir>` / `veason relate <index_dir> <fid> ...` "
                             "once; their output is reused, not recomputed."),
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
            ][:32]     # 一帧 region 数在个位数；32 只是防御性上限，不改变正常读数
            evidence.append({"frame_id": int(fid), "axis": axis, "value": value, "src": src})

    return {"evidence": evidence}, {"events": events}


def _read_frame(store: Any, fid: int) -> Any:
    import cv2

    img = cv2.imread(str(store.frame_image(fid)))
    if img is None:
        raise VrsError(f"frame image unreadable: {store.frame_image(fid)}",
                       hint="re-run `veason index`; the stored PNG is corrupt.")
    return img


def _try_backend(backend: Any, backend_error: Optional[BackendUnavailable], axis: str,
                 store: Any, fid: int, regions: list[dict], query: str,
                 events: list[dict]) -> tuple[dict, str]:
    """缓存里没有这个轴时才动模型。返回 ``(values, src)``。

    ★ 纪律②在这里被执行成一次显式检查：backend 答的轴必须就是被问的轴，
      不是就丢掉并记账 —— 否则一个「顺手多答一个轴」的件会污染归因。
    """
    if backend is None or not backend.supports(axis):
        return {}, AXIS_SRC[axis]
    try:
        rows = backend.measure(axis, _read_frame(store, fid), regions,
                               store.load_masks(fid), query)
    except BackendUnavailable:
        return {}, AXIS_SRC[axis]
    if not rows:
        return {}, AXIS_SRC[axis]
    values: dict[int, dict] = {}
    src = AXIS_SRC[axis]
    for row in rows:
        if row.get("axis") not in (None, axis):
            events.append({"event": "evidence_backend_answered_the_wrong_axis",
                           "frame_id": int(fid), "asked": axis, "answered": row.get("axis"),
                           "note": "已丢弃：一个件只许对它负责的那枚标签出数。"})
            continue
        values[int(row["region_idx"])] = dict(row.get("value") or {})
        src = row.get("src") or src
    return values, src
