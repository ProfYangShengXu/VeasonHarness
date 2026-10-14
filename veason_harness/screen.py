"""① 初筛：YOLO 主力，OOV 才上 Jev。

## OOV 定义成可测量的三档（规格 §2.1）

===== ============================ ==========================================
档    条件                          动作
===== ============================ ==========================================
A     YOLO 检出 && 命中 query 词     直接用，不上 Jev
B     YOLO 检出 && 未命中 query 词   ★ 上 Jev：问「这个区域是不是 query 说的那个」
C     YOLO 零检出                   ★ 上 Jev：问存在性（none 档）
===== ============================ ==========================================

★ 档 B 不是补丁，是实测逼出来的：开放词表检测器在卡通/非自然物体上
  **标签错、框对** —— 本仓 dragon 索引里那只黄色卡通奶龙被贴成
  ``parachute`` / ``character sculpture`` / ``stuffed toy``，框完全正确。
  所以「标签没命中」≠「目标不在」，这一档必须有人兜。

★ 档 B 问 Jev 的形态是【N 选一】，不是【逐个问是不是】：
  后者实测 74% 都答「是」（`资料/Jev用法-20261013.md` §6 纪律②），
  等于把这一档变成一个常数。**一帧一次前向**，候选摆进同一道题里比。

★ 高召回优先：**剔帧判据 = YOLO 与 Jev 都说不在**才剔；任一说在就留。
  Jev 不可用时不许当「说不在」——那条路径叫 ``jev_unavailable_kept``，
  因为它离「说不在」差了一整个模型。

★ 这里**不加载任何 YOLO 模型**。「YOLO 检出」= 索引时那次 prompt-free 检测
  已经落在盘上的 region 表（`veason index` 的产物）。理由有三：
  1. 索引本就不知道 query（prompt-free, LVIS 4,585 类），它产出的就是
     「YOLO 在 query 未知时说看到了什么」，正是档 B 需要的那个事实；
  2. 重跑一次 prompt-free 只会得到同一批框，白白花 0.29 s/帧；
  3. 让本命令在纯 CPU 上是零模型、毫秒级 —— 初筛的成本应该全花在 Jev 上。
  真需要用 query 词重跑开放词表检测时走 `veason detect`（那是它的活）。
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from .backends import BackendUnavailable, get_backend
from .errors import VrsError

#: 档 B 候选池的上限（= 一次 N 选一的选项数，还要给「都不是」留一格）。
#: 区域多到几十个时选项数会超 Jev 的 K_MAX=16（超了它直接 raise，不截断），
#: 所以按面积取前 N 个；截断本身记在 screened 里，不静默。
MAX_JEV_REGIONS = 8

#: 判 Jev「说是」的门限 —— **只用在档 C 的二选一存在性**上。
#: ★ 这是**推断值，未标定**：实测概率几乎分不开对错（判对 0.746 / 判错 0.708），
#:   所以档 B 的 N 选一刻意**不用**它；接线后要拿真读数重标这一条。
JEV_YES = 0.5

#: 中文查询切词时要丢的虚词。不丢的话「的」「一个」会去匹配 label，
#: 把档 B 误判成档 A —— 那正好是这条链最贵的错误方向（该问的没问）。
_STOP = {"的", "了", "一个", "个", "那个", "这个", "在", "和", "与", "是", "有",
         "the", "a", "an", "and", "or", "of", "in", "on", "with", "at", "is", "are",
         "person", "who", "that", "this"}

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def query_terms(query: str) -> list[str]:
    """把查询切成可比较的词。

    ★ 中文整串 + 双字滑窗都要出：整串能匹配整串相同的 label（「奶龙」），
      双字窗才能匹配「奶龙」落在更长的 label 里。只做整串会漏，只做单字会
      把「色」「的」这种噪声带进来。
    """
    terms: list[str] = []
    for tok in _TOKEN.findall((query or "").lower()):
        if tok.isascii():
            if tok not in _STOP and len(tok) > 1:
                terms.append(tok)
        else:
            if tok not in _STOP:
                terms.append(tok)
            for i in range(len(tok) - 1):
                bg = tok[i:i + 2]
                if bg not in _STOP:
                    terms.append(bg)
    seen, out = set(), []
    for t in terms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def parse_synonyms(pairs: Sequence[str]) -> dict[str, list[str]]:
    """``--synonym 奶龙=parachute,stuffed toy`` → ``{"奶龙": ["parachute", ...]}``。

    为什么要有：query 与 label 不同语言时（本项目常态：中文问、LVIS 英文答），
    纯字符串命中率是 0，档 B 会对**每一帧**开火。同义词表是把这个已知映射
    显式写下来的地方 —— 它比偷偷加一层翻译模型可观测（谁加的、加了什么，
    都在命令行上）。
    """
    out: dict[str, list[str]] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise VrsError(
                f"--synonym {pair!r} is not KEY=WORD[,WORD]",
                hint='example: --synonym "奶龙=parachute,stuffed toy,character sculpture"',
                code=2,
            )
        key, _, val = pair.partition("=")
        words = [w.strip().lower() for w in val.split(",") if w.strip()]
        if not key.strip() or not words:
            raise VrsError(f"--synonym {pair!r} has an empty side",
                           hint="both sides must be non-empty.", code=2)
        out.setdefault(key.strip().lower(), []).extend(words)
    return out


def label_matches(label: str, terms: Sequence[str]) -> bool:
    lab = (label or "").lower()
    return any(t in lab or lab in t for t in terms if t)


def _parse_frames_arg(raw: str) -> list[int]:
    if raw is None or not str(raw).strip():
        return []
    out = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise VrsError(f"--frames {raw!r} is not a comma-separated list of ints",
                           hint='example: --frames 850,384', code=2) from None
    return out


def _default_read_frame(store: Any, fid: int) -> Any:
    import cv2

    img = cv2.imread(str(store.frame_image(fid)))
    if img is None:
        raise VrsError(
            f"frame image unreadable: {store.frame_image(fid)}",
            hint="re-run `veason index`; the stored PNG is corrupt.",
        )
    return img


#: 「有名字的」路由集合 —— 即【确实查过、认出了这区域是 query 说的东西】。
#: ★ 这是证据类型白名单，不是置信度门限：名单里的路由代表「有一个模型给出了
#:   它叫什么的判断」，名单外的（含 `*_kept`）代表「没有这个判断」。
#: ★ 新增「确实查过」的路由时必须显式加进这里；默认落【弱证据层】是保守方向
#:   —— 不会让一个未知路由悄悄插到目标前面把强证据埋掉。
_EVIDENCED_ROUTES = frozenset({"yolo_hit", "jev_confirmed"})


def _evidence_tier(route: str) -> int:
    """route → 证据层级。数字小的排前面。

    0 = 有名字的强证据（YOLO 命中 / Jev 确认）；1 = 无证据留着的（``*_kept`` 等）。

    ★ 它只分【证据类型】，不碰【自信度】：层内的先后仍交给 area_ratio，
      所以「绝不看谁更自信」这条纪律原样保留。
    """
    return 0 if str(route) in _EVIDENCED_ROUTES else 1


def run_screen(store: Any, query: str, top_n: int = 6, oov_rule: str = "query_miss",
               *, oov_threshold: float = 0.30, jev_backend: Optional[str] = None,
               axis_backend: Optional[str] = None,
               synonyms: Optional[dict[str, list[str]]] = None,
               frames: Optional[str] = None,
               jev_yes: float = JEV_YES,
               max_jev_regions: int = MAX_JEV_REGIONS,
               load_frame: Optional[Any] = None) -> tuple[dict, dict]:
    """返回 ``(stdout payload, info)``；``info["events"]`` 由 cli 写到 stderr。"""
    if not (query or "").strip():
        raise VrsError("no query given", hint='usage: veason screen <index_dir> --query "..."', code=2)
    if top_n < 1:
        raise VrsError(f"--top-n must be >= 1 (got {top_n})",
                       hint="top-n 是返回几个候选，不是问几个问题。", code=2)
    if oov_rule not in ("query_miss", "score_threshold"):
        raise VrsError(f"unknown --oov-rule {oov_rule!r}",
                       hint="known: query_miss (可解释) | score_threshold (更连续)", code=2)

    meta = store.require()
    t_of = {int(f["frame_id"]): float(f["t"]) for f in meta.get("frames", [])}
    terms = query_terms(query)
    for key, words in (synonyms or {}).items():
        if key in terms:
            terms.extend(words)
    wanted = _parse_frames_arg(frames) if frames else sorted(t_of)

    events: list[dict] = []
    # ★ 一帧的解码结果要在一帧之内复用：档 B 对每个 region 各问一次 Jev，
    #   而同一个 frame 的 PNG 内容不会变 —— 不缓存就会把同一张 720p PNG
    #   解码最多 max_jev_regions 次（48 帧 × 8 region = 最多 384 次无谓解码）。
    img_cache: dict[int, Any] = {}
    base_loader = load_frame or _default_read_frame

    def _cached_loader(store_: Any, fid_: int) -> Any:
        key = int(fid_)
        if key not in img_cache:
            img_cache[key] = base_loader(store_, key)
        return img_cache[key]

    jev = None
    jev_error = None
    try:
        jev = get_backend("jev", jev_backend)
    except BackendUnavailable as exc:
        jev_error = exc
        events.append({"event": "screen_jev_unavailable", "error": exc.error,
                       "hint": exc.hint,
                       "consequence": "档 B/C 的帧一律保留（route=jev_unavailable_kept）："
                                      "Jev 没接线不等于目标不在。"})

    candidates: list[dict] = []
    n_yolo_hit = n_jev_frames = n_dropped = n_jev_model_calls = 0

    for fid in wanted:
        if fid not in t_of:
            raise VrsError(
                f"frame_id {fid} is not in this index",
                hint="run `veason frames <index_dir>` for the frame ids this index samples.",
            )
        regions = store.require_regions(fid).get("regions", [])
        matched = [r for r in regions if label_matches(r.get("label", ""), terms)]
        max_score = max([float(r.get("score", 0.0)) for r in regions], default=0.0)
        # 两条 OOV 判据的差别：query_miss 只看「有没有一个 label 叫得出 query 的名字」，
        # 是可解释的硬判据；score_threshold 连「叫得出名字但分很低」也算 OOV ——
        # 更连续，但门限本身要被标定（0.30 是推断值）。
        oov = not matched if oov_rule == "query_miss" else max_score < oov_threshold
        if matched and not oov:
            best = max(matched, key=lambda r: (float(r.get("score", 0.0)), -int(r["idx"])))
            n_yolo_hit += 1
            candidates.append({
                "frame_id": int(fid), "timestamp": round(t_of[fid], 3),
                "area_ratio": round(_area_ratio(best, meta), 6),
                "region_idx": int(best["idx"]), "route": "yolo_hit",
                "score": round(float(best.get("score", 0.0)), 4),
                "label": str(best.get("label", "")),
            })
            continue

        # ---- 档 B / C：这里才花钱 ----
        tier = "B" if regions else "C"
        n_jev_frames += 1
        if jev is None:
            kept = _fallback_region(regions)
            candidates.append({
                "frame_id": int(fid), "timestamp": round(t_of[fid], 3),
                "area_ratio": round(_area_ratio(kept, meta), 6) if kept else 0.0,
                "region_idx": int(kept["idx"]) if kept else -1,
                "route": "jev_unavailable_kept", "tier": tier,
                "note": (jev_error.error if jev_error else "jev unavailable"),
            })
            continue

        hit = _ask_jev(jev, store, fid, regions, query, tier, _cached_loader,
                       jev_yes=jev_yes, max_regions=max_jev_regions)
        n_jev_model_calls += hit["calls"]
        if hit["yes"] is not None:
            candidates.append({
                "frame_id": int(fid), "timestamp": round(t_of[fid], 3),
                "area_ratio": round(_area_ratio(hit["region"], meta), 6) if hit["region"] else 0.0,
                "region_idx": int(hit["region"]["idx"]) if hit["region"] else -1,
                "route": "jev_confirmed", "tier": tier,
                "jev_yes": round(float(hit["yes"]), 4),
            })
        else:
            # ★ 只有走到这里才剔：YOLO 没给名字、Jev 明确说不在。
            n_dropped += 1

    # ★ 排序分两级：**先按证据类型分层，层内再按面积**（「面积大 = 目标更可能
    #   看得清」这条理由在【同一层内】成立，所以它不是被丢掉，而是被限定在层内）。
    #
    # 为什么必须先分层 —— `route` 表达的是【证据类型】，不是【自信度】：
    #   yolo_hit / jev_confirmed = 检测器/校验器【认出了这区域是 query 说的东西】（有名字）
    #   *_kept（如 jev_unavailable_kept）= 压根没问过（Jev 没接线），保守留着（没有证据）
    # 两者证据强度差一个量级，不该在同一层比面积。整池按面积排的必然后果：
    # 未命中帧挑的是【画面里最大的现存区域】（通常是水面/沙地等背景），而背景面积
    # 必然大于目标框 ⇒ 背景永远排在强证据前面，top-n 全是背景
    # （实测：frame 850 是 yolo_hit 却被排到 rank=20）。
    candidates.sort(key=lambda c: (_evidence_tier(c.get("route", "")),
                                   -c["area_ratio"], -float(c.get("score", 0.0)),
                                   c["frame_id"]))
    kept_before = len(candidates)
    shown = candidates[:top_n]
    if kept_before > len(shown):
        events.append({
            "event": "screen_top_n_truncated",
            "kept": kept_before, "shown": len(shown), "top_n": top_n,
            "hint": "top-n 只截断【展示】，被截掉的帧是高分帧之外的大面积帧；"
                    "要看全量就加大 --top-n（它不改变提问次数，只改变返回长度）。",
        })

    payload = {
        "candidates": shown,
        "screened": {
            "total_frames": len(wanted),
            "yolo_hit": n_yolo_hit,
            "jev_called": n_jev_frames,
            "jev_model_calls": n_jev_model_calls,
            "dropped": n_dropped,
            "kept": kept_before,
            "shown": len(shown),
            "oov_rule": oov_rule,
        },
        "query_terms": terms,
    }
    return payload, {"events": events}


def _area_ratio(region: Optional[dict], meta: dict) -> float:
    """面积占比 = 确定性计算，不问模型（规格 §2.3 纪律①）。

    优先用索引里已经算好的 ``area_ratio``；没有就自己算 —— 两者必须一致，
    不一致说明 region 表被人动过，那我们算的这一份为准。
    """
    if not region:
        return 0.0
    if "area_ratio" in region:
        return float(region["area_ratio"])
    w = float(meta.get("width") or 1) or 1.0
    h = float(meta.get("height") or 1) or 1.0
    x1, y1, x2, y2 = (float(v) for v in region["bbox"])
    return max(0.0, (x2 - x1) * (y2 - y1)) / (w * h)


def _fallback_region(regions: list[dict]) -> Optional[dict]:
    if not regions:
        return None
    return max(regions, key=lambda r: (float(r.get("area_ratio", 0.0)), -int(r["idx"])))


def _ask_jev(jev: Any, store: Any, fid: int, regions: list[dict], query: str, tier: str,
             load_frame: Optional[Any], *, jev_yes: float, max_regions: int) -> dict:
    """问 Jev。返回 ``{"yes": float|None, "region": dict|None, "calls": int}``。

    ``yes is None`` = Jev 明确说不在（**只有这个状态允许剔帧**）。

    ★★ 档 B **一次 N 选一**，绝不退化成「逐个问是不是」：

    ```
    把 4 个框分别问它「这个是目标吗？」 → 它 74% 都答「是」
    把 4 个框摆在【同一道题】的候选里，让它挑一个 → 才有比较压力
    ```
    （`资料/Jev用法-20261013.md` §6 纪律②；架构 §2.3 也已写明这条被实测否掉。）

    ★ 档 B **不用概率门限**：N 选一的答案就是「它挑了哪个」，而概率在这类任务上
      不可信（实测判对 0.746 / 判错 0.708，几乎分不开）。给赢家再加一道门限，
      等于把 N 选一好不容易拿到的比较压力又扔掉，退回「高置信地错」。
      存在性（档 C）是二选一，才需要一个切点，且那个切点标注为未标定。
    """
    def frame():
        if load_frame is not None:
            return load_frame(store, fid)
        return _default_read_frame(store, fid)

    if tier == "C":
        ans = jev.exists(frame(), query)
        return {"yes": ans if (ans is not None and ans >= jev_yes) else None,
                "region": None, "calls": 1, "form": "existence-2way"}

    order = sorted(regions, key=lambda r: (-float(r.get("area_ratio", 0.0)), int(r["idx"])))
    order = order[:max_regions]
    picked = jev.choose(frame(), order, query)
    if not picked or picked.get("region_idx") is None:
        # 它选了「都不是」（前提为假那一档）。
        return {"yes": None, "region": None, "calls": 1, "form": "choose-n",
                "none_score": (picked or {}).get("score")}
    idx = int(picked["region_idx"])
    region = next((r for r in order if int(r["idx"]) == idx), None)
    if region is None:
        # 后端返回了一个不在候选表里的编号：这是件违约，不是「不在」。
        return {"yes": None, "region": None, "calls": 1, "form": "choose-n",
                "bad_pick": idx}
    return {"yes": float(picked.get("score") or 0.0), "region": region,
            "calls": 1, "form": "choose-n", "options": len(order) + 1}
