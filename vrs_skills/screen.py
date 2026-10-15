"""① 初筛：索引里的 YOLOE 主力，query 词零命中的帧才上 Jev。

## OOV 判据（新契约②，已定案）

> **OOV = query 词【agent 改写后】零命中。**

这里**没有**相似度门限那条路。旧版还有一个 ``score_threshold``（最高分 < 0.30 也算
OOV），0.30 是推断值、没法标定，且一旦 agent 选它就按未标定的数决定整条链的走向。
现在只有一条可解释的硬判据：把（agent 已改写的）query 切成词，去比 region 的
``label``；一帧里一个都没命中 ⇒ 这帧走 Jev。

依据（实测）：整句表达式直接喂开放词表检测器时，288 个框只有 94 个（32.6%）是
真目标 —— 因为 "A prompt describes appearance, not relationships"。所以表达式
必须先被 agent 改写/拆词，再由本命令做**字面**匹配，不做语义猜测。

## 三档（每帧只落一档，台账加起来正好是总帧数）

===== ============================ ==========================================
档    条件                          动作
===== ============================ ==========================================
A     label 命中 query 词           直接用，不上 Jev
B     YOLO 有框但 label 未命中       上 Jev：问「这个区域是不是 query 说的那个」
C     YOLO 零框                    上 Jev：问存在性（none 档）
===== ============================ ==========================================

★ 档 B 不是补丁：开放词表检测器在卡通/非自然物体上**标签错、框对**（本仓 dragon
  索引里那只黄色奶龙被贴成 ``parachute`` / ``stuffed toy``），所以「标签没命中」
  ≠「目标不在」，必须有人兜。

★ 高召回：**只有 Jev 明确说「不在」才剔帧**。Jev 不肯回答（``None``）与 Jev 说
  「不在」（显式低分）是两件事，前者保留 —— 把「模型没答」当成「目标不在」正是
  这条链最贵的方向。

★ 本命令**不加载 YOLO**：「YOLO 检出」= 索引时那次 prompt-free 检测已经落在盘上的
  region 表。索引不知道 query，它产出的正是「检测器在 query 未知时说看到了什么」。
  真要用 query 词重跑开放词表检测走 `vrs detect`。

## Jev 是真模型（新契约①）

``JevModel`` 直接加载组服务器上的 RSI-Jev v6.1-VL（``rsijev.Decider``）。没有
「可注入后端」注册表，也没有 ``auto -> unavailable`` 的静默降级：模型加载失败就是
一条硬 ``VrsError``，因为整条 skill 跑在 GPU 服务器上，Jev 本该在。``jev=`` 形参
是**测试缝**（传假件验路由），生产路径不传。
"""
from __future__ import annotations

import os
import re
from typing import Any, Optional, Sequence

from .errors import VrsError
from .video import load_bgr, parse_frame_ids, region_area_ratio

#: 档 B 逐区域问 Jev 的上限。区域多到几十个时「每次都是一笔钱」会失控，
#: 且 >6 个候选进打分器就崩。截断记在 ``screened.max_jev_regions`` 里，不静默。
MAX_JEV_REGIONS = 8

#: 判 Jev「说是」的门限（RSI-Jev 的 ``noul`` 已是概率，0.5 是二分类切点）。
JEV_YES = 0.5

#: Jev 载入用哪个 checkpoint（HF repo id、路径或别名，如 ``v6.1-vl-4b``）。
JEV_MODEL_ENV = "VRS_JEV_MODEL"
DEFAULT_JEV_MODEL = "v6.1-vl-4b"

#: 切词时要丢的**虚词**。
#: ★ 实义词（person/vehicle/…）绝不能进这张表：``query_terms("person")`` 返回空
#:   会导致「每帧都判 OOV」，把最该用索引标签直接命中的那道题推给 Jev。
_STOP = {"的", "了", "一个", "个", "那个", "这个", "在", "和", "与", "是", "有", "把",
         "被", "给", "对", "从", "到", "就", "都", "也", "很", "又", "还", "而", "或",
         "the", "a", "an", "and", "or", "of", "in", "on", "with", "at", "is", "are",
         "was", "were", "to", "for", "by", "that", "this", "those", "these", "which",
         "who", "whom", "whose", "it", "its"}

_TOKEN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


def query_terms(query: str) -> list[str]:
    """把（agent 改写后的）查询切成可比较的**实义词**。

    ★ 中文整串 + 双字滑窗都要出：整串能匹配同名 label（「奶龙」），双字窗才能
      匹配「奶龙」落在更长的 label 里。只做整串会漏，只做单字会把噪声带进来。
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

    query 与 label 不同语言时（本项目常态：中文问、英文 label），纯字面命中率是 0，
    档 B 会对每一帧开火。同义词表把这条已知映射**显式**写下来，比偷偷加一层翻译
    模型可观测（谁加的、加了什么，都在命令行上）。
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


# ---------------------------------------------------------------------------
# Jev：真的 RSI-Jev（rsijev.Decider），不是可注入后端
# ---------------------------------------------------------------------------

def _draw_box(img_bgr: Any, bbox: Sequence[int], label: int):
    """在帧上画一个编号框，返回 PIL RGB 图（RSI-Jev 的问题里好指认）。"""
    import cv2
    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    dr = ImageDraw.Draw(pil)
    x1, y1, x2, y2 = (int(v) for v in bbox[:4])
    dr.rectangle([x1, y1, x2, y2], outline=(230, 40, 40), width=4)
    dr.rectangle([x1, max(0, y1 - 26), x1 + 34, y1], fill=(230, 40, 40))
    dr.text((x1 + 8, max(0, y1 - 22)), str(label), fill=(255, 255, 255))
    return pil


def _draw_boxes(img_bgr: Any, boxes: Sequence[Sequence[int]],
                labels: Optional[Sequence[str]] = None):
    """一次把多个框画上编号，返回 PIL RGB 图（供一次 N 选一）。"""
    import cv2
    from PIL import Image, ImageDraw

    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    for i, bbox in enumerate(boxes, 1):
        x1, y1, x2, y2 = (int(v) for v in bbox[:4])
        dr.rectangle([x1, y1, x2, y2], outline=(230, 40, 40), width=4)
        tag = str(labels[i - 1]) if labels else str(i)
        dr.rectangle([x1, max(0, y1 - 26), x1 + 10 + 11 * len(tag), y1],
                     fill=(230, 40, 40))
        dr.text((x1 + 6, max(0, y1 - 22)), tag, fill=(255, 255, 255))
    return pil


class JevModel:
    """组服务器上的 RSI-Jev v6.1-VL：存在性 / 候选确认。

    ★ 它**不产生坐标**，只回答「是 / 不是 / 不肯答」。``exists``/``confirm`` 返回
      ``float``（P(yes)）或 ``None``（模型不肯回答 —— 调用方必须当作「不知道」，
      不许当作「不在」）。
    """

    def __init__(self, model: Optional[str] = None, *, device: Optional[str] = None,
                 dtype: Optional[str] = None) -> None:
        try:
            from rsijev import Decider
        except Exception as exc:  # pragma: no cover - 服务器外才会发生
            raise VrsError(
                "rsijev is not importable, so the Jev model cannot run",
                hint="this skill must run on the GPU box (`$HOME/envs/vrs`), where "
                     "`rsijev` and the RSI-Jev weights are installed.",
                code=3,
            ) from exc
        model = model or os.environ.get(JEV_MODEL_ENV) or DEFAULT_JEV_MODEL
        model = _resolve_jev_model(model)
        try:
            self._decider = Decider(model, device=device, dtype=dtype)
        except Exception as exc:
            raise VrsError(
                f"RSI-Jev checkpoint {model!r} could not be loaded: {exc}",
                hint=f"check {JEV_MODEL_ENV} / the `rsijev` install on the GPU box.",
                code=3,
            ) from exc
        self.name = f"rsi-jev:{getattr(self._decider, 'name', model)}"

    def _noul(self, pil_image: Any, instructions: str) -> Optional[float]:
        a = self._decider.decide(
            "<image>",
            {"q": {"type": "noul", "instructions": instructions}},
            images=[pil_image],
        )
        slot = (a or {}).get("q") or {}
        v = slot.get("noul")
        return None if v is None else float(v)

    def exists(self, img_bgr: Any, query: str) -> Optional[float]:
        """这帧里有没有 query 说的东西？``None`` = 不肯答。"""
        from PIL import Image
        import cv2

        pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return self._noul(pil, f"Is the {clean_query(query)} present in this image?")

    def _pick(self, pil_image: Any, instructions: str,
              options: Sequence[str]) -> Optional[dict]:
        """N 选一（``choice`` 型）。返回 ``{option: P}``，模型不肯答返回 ``None``。

        ★ 为什么档 B 必须用这个而不是 ``_noul``：实测同一张画了编号框的图，
        ``noul`` 问「这个框里是不是 X」给 0.146（判不是），而 ``choice`` 让它在
        ``[X, none]`` 里挑给 0.726（挑中 X）。``noul`` 适合**全图存在性**，
        不适合**指向某个区域** —— 有指代对象时得让模型在候选里挑。
        """
        opts = [str(o) for o in options]
        a = self._decider.decide(
            "<image>",
            {"q": {"type": "choice", "instructions": instructions,
                   "criteria": {o: None for o in opts}}},
            images=[pil_image],
        )
        slot = (a or {}).get("q") or {}
        got = slot.get("choice")
        probs = slot.get("probabilities")
        return {
            "choice": None if got is None else str(got),
            "probs": {str(k): float(v) for k, v in (probs or {}).items()},
        }

    def confirm_many(self, img_bgr: Any, bboxes: Sequence[Sequence[int]],
                     query: str) -> Optional[int]:
        """一次摆出 N 个编号框，让模型挑【哪个是 query 说的那个】。

        返回被挑中的框的**下标**（0-based），模型挑 ``none`` 或不肯答返回 ``None``。

        ★ 为什么必须是「一次多选」：逐个问时框越大越容易被判"是"（大框天然包含目标），
        而按面积降序问 + 第一个"是"就返回 ⇒ 最大那个坏框必然胜出（22697 实测）。
        一次摆出来让模型在**同一屏里横向比较**，它才有机会挑那个框得最准的。
        """
        if not bboxes:
            return None
        keys = [str(i + 1) for i in range(len(bboxes))]
        res = self._pick(
            _draw_boxes(img_bgr, bboxes, keys),
            f"Which numbered box contains the {clean_query(query)}? "
            f"If none of them does, choose none.",
            keys + ["none"],
        )
        if res is None:
            return None
        got = res.get("choice")
        if got is None or got == "none":
            return None
        try:
            return int(got) - 1
        except (TypeError, ValueError):
            return None

    def confirm(self, img_bgr: Any, bbox: Sequence[int], query: str) -> Optional[float]:
        """框里的是不是 query 说的那个？``None`` = 不肯答。"""
        target = clean_query(query)
        res = self._pick(
            _draw_box(img_bgr, bbox, 1),
            "Look at the red box labeled 1. What is inside it?",
            [target, "none"],
        )
        if res is None:
            return None
        # ★ 读【离散选择】，不拿 0.5 去切选项间的相对概率：
        #   choice 的 probabilities 是「这几个选项怎么分配」，不是「目标存在的概率」，
        #   拿二分类切点切它是量错东西。选中 = 是，选 none = 不是。
        return 1.0 if res.get("choice") == target else 0.0


#: 默认断网找权重。★ 权重明明在本地却让 Decider 去 HF 下载，会卡在
#: ``Fetching 17 files: 0%``（实测踩过）。置 0 可恢复联网行为。
OFFLINE_ENV = "VRS_JEV_OFFLINE"


def _resolve_jev_model(model: str) -> str:
    """把 Jev 的「名字」解析成**本地目录路径**（找不到就原样返回，让上游报硬错）。

    ``Decider`` 拿到名字会去 HuggingFace 拉权重；拿到已存在的目录路径才用本地那份。
    解析顺序：显式路径 → ``$VRS_JEV_PATH`` → ``paths.DEFAULT_JEV_DIR``。
    """
    from .paths import DEFAULT_JEV_DIR

    if os.path.isdir(model):                       # 调用方直接给了路径
        return model
    for cand in (os.environ.get("VRS_JEV_PATH"), str(DEFAULT_JEV_DIR)):
        if cand and os.path.isdir(cand):
            if os.environ.get(OFFLINE_ENV, "1") != "0":
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            return cand
    return model                                   # 本地没有 ⇒ 交给上游报硬错


#: 塞进问句模板前要剥掉的冠词。★ 只剥【句首】的，句子内部的 the 是内容。
_ARTICLES = ("the ", "a ", "an ")


def clean_query(term: str) -> str:
    """把指代表达式清洗成能塞进 ``Is the {X} present...`` 的名词短语。

    实测依据（22697 "the primate."）：不清洗时拼出 ``Is the the primate. present...``，
    Jev 给 0.459（判不在）；清洗成 ``primate`` 后 0.757（判在）。**双重冠词 + 句尾标点
    足以把判断翻过 0.5 那条线。**

    清的是英文题面通用的噪声，不对任何具体名词做特判：

    * 句尾标点（``. , ; : ! ?``）
    * 句首冠词（``the/a/an``，可重复剥 —— 题面常自带 "the"）
    * 首尾空白

    剥完若为空（题面就是一个 "the"），原样返回 —— 宁可问得怪，也不要凭空造词。
    """
    t = (term or "").strip().rstrip(".,;:!?)]}").strip()
    changed = True
    while changed and t:
        changed = False
        low = t.lower()
        for art in _ARTICLES:
            if low.startswith(art):
                t = t[len(art):].strip()
                changed = True
                break
    return t or (term or "").strip()


def default_jev(*, model: Optional[str] = None, device: Optional[str] = None,
                dtype: Optional[str] = None) -> JevModel:
    """装载真 Jev。加载失败抛出可行动的 ``VrsError``（不静默降级）。"""
    return JevModel(model=model, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _fallback_region(regions: list[dict], meta: dict) -> Optional[dict]:
    if not regions:
        return None
    return max(regions, key=lambda r: (region_area_ratio(r, meta), -int(r["idx"])))


def _ask_jev(jev: Any, img: Any, regions: list[dict], query: str, tier: str, meta: dict,
             *, jev_yes: float, max_regions: int) -> dict:
    """问 Jev。返回 ``{"yes", "region", "refused", "calls"}``。

    ``yes is None and not refused`` = Jev 明确说不在（**只有这个状态允许剔帧**）。
    ``refused`` = 至少一次调用模型不肯回答，且没有一次说是 ⇒ 保留该帧。
    """
    calls = 0
    if tier == "C":
        calls += 1
        ans = jev.exists(img, query)
        if ans is None:
            return {"yes": None, "region": None, "refused": True, "calls": calls}
        if ans >= jev_yes:
            return {"yes": ans, "region": None, "refused": False, "calls": calls}
        return {"yes": None, "region": None, "refused": False, "calls": calls}

    order = sorted(regions, key=lambda r: (-region_area_ratio(r, meta), int(r["idx"])))
    pool = order[:max_regions]

    # ★ 一次 N 选一（设计原意）。逐个问会掉进「框大者胜」的陷阱：大框天然包含目标，
    #   按面积降序问、第一个"是"就返回 ⇒ 最大的坏框必然赢（22697 实测）。
    many = getattr(jev, "confirm_many", None)
    if many is not None and pool:
        calls += 1
        picked = many(img, [r["bbox"] for r in pool], query)
        if picked is not None and 0 <= picked < len(pool):
            return {"yes": 1.0, "region": pool[picked], "refused": False, "calls": calls}

    # 退路：模型没做多选（或测试替身没有 confirm_many）⇒ 逐个问。
    refused = False
    for r in pool:
        calls += 1
        ans = jev.confirm(img, r["bbox"], query)
        if ans is None:
            refused = True
            continue
        if ans >= jev_yes:
            return {"yes": ans, "region": r, "refused": False, "calls": calls}
    return {"yes": None, "region": None, "refused": refused, "calls": calls}


def run_screen(store: Any, query: str, top_n: int = 6, *,
               synonyms: Optional[dict[str, list[str]]] = None,
               frames: Optional[str] = None,
               jev_yes: float = JEV_YES,
               max_jev_regions: int = MAX_JEV_REGIONS,
               jev: Optional[Any] = None) -> tuple[dict, dict]:
    """返回 ``(stdout payload, info)``；``info["events"]`` 由 cli 写到 stderr。

    ``jev`` 只给测试用；生产路径传 ``None``，本函数装载真 RSI-Jev。
    """
    if not (query or "").strip():
        raise VrsError("no query given", hint='usage: vrs screen <index_dir> --query "..."', code=2)
    if top_n < 1:
        raise VrsError(f"--top-n must be >= 1 (got {top_n})",
                       hint="top-n 是返回几个候选，不是问几个问题。", code=2)

    meta = store.require()
    t_of = {int(f["frame_id"]): float(f.get("t") or 0.0) for f in meta.get("frames", [])}
    terms = query_terms(query)
    for key, words in (synonyms or {}).items():
        if key in terms:
            terms.extend(words)
    if not terms:
        raise VrsError(
            f"query {query!r} has no content words after the rewrite",
            hint="把表达式改写成能被 label 匹配的实义词（可配 --synonym 显式映射），"
                 "否则「零命中」这条判据对每一帧都成立。",
            code=2,
        )
    wanted = parse_frame_ids(frames) if frames else sorted(t_of)

    events: list[dict] = []
    # ★ Jev 懒加载：整道题都走档 A 时不该为了「没问的模型」占一张卡。
    jev_obj = jev

    candidates: list[dict] = []
    n_yolo_hit = n_jev_called = n_dropped = n_refused = n_jev_region_calls = 0

    for fid in wanted:
        if fid not in t_of:
            raise VrsError(
                f"frame_id {fid} is not in this index",
                hint="run `vrs frames <index_dir>` for the frame ids this index samples.",
            )
        regions = store.require_regions(fid).get("regions", [])
        matched = [r for r in regions if label_matches(r.get("label", ""), terms)]

        if matched:
            best = max(matched, key=lambda r: (float(r.get("score", 0.0)), -int(r["idx"])))
            n_yolo_hit += 1
            candidates.append({
                "frame_id": int(fid), "timestamp": round(t_of[fid], 3),
                "area_ratio": round(region_area_ratio(best, meta), 6),
                "region_idx": int(best["idx"]), "route": "yolo_hit",
                "score": round(float(best.get("score", 0.0)), 4),
                "label": str(best.get("label", "")),
            })
            continue

        # ---- 档 B / C：query 词在这一帧零命中，这里才花钱 ----
        tier = "B" if regions else "C"
        if jev_obj is None:
            jev_obj = default_jev()
        n_jev_called += 1
        img = load_bgr(store, fid)          # 一帧只解码一次（不再做无上限的全局缓存）
        hit = _ask_jev(jev_obj, img, regions, query, tier, meta,
                       jev_yes=jev_yes, max_regions=max_jev_regions)
        n_jev_region_calls += hit["calls"]

        if hit["yes"] is not None:
            reg = hit["region"]
            candidates.append({
                "frame_id": int(fid), "timestamp": round(t_of[fid], 3),
                "area_ratio": round(region_area_ratio(reg, meta), 6) if reg else 0.0,
                "region_idx": int(reg["idx"]) if reg else -1,
                "route": "jev_confirmed", "tier": tier,
                "jev_yes": round(float(hit["yes"]), 4),
            })
        elif hit["refused"]:
            # ★ 模型不肯答 ≠ 目标不在。保留，并让读数里看得见这次不肯答。
            n_refused += 1
            reg = _fallback_region(regions, meta)
            candidates.append({
                "frame_id": int(fid), "timestamp": round(t_of[fid], 3),
                "area_ratio": round(region_area_ratio(reg, meta), 6) if reg else 0.0,
                "region_idx": int(reg["idx"]) if reg else -1,
                "route": "jev_refused_kept", "tier": tier,
                "note": f"{getattr(jev_obj, 'name', 'jev')} refused to answer",
            })
        else:
            # 只有走到这里才剔：label 没命中且 Jev 明确说不在。
            n_dropped += 1

    # 排序只决定 top-n 截掉谁，不决定谁被剔。面积大 = 目标更可能看得清
    # （规格 §6.2「低可见」用的就是面积），所以面积优先，绝不看「谁更自信」。
    candidates.sort(key=lambda c: (-c["area_ratio"], -float(c.get("score", 0.0)),
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

    absent = bool(n_yolo_hit == 0 and n_jev_called > 0 and n_dropped == n_jev_called)
    return {
        "candidates": shown,
        "absent": absent,
        "screened": {
            "total_frames": len(wanted),
            "yolo_hit": n_yolo_hit,
            "jev_called": n_jev_called,
            "jev_region_calls": n_jev_region_calls,
            "dropped": n_dropped,
            "refused": n_refused,
            "kept": kept_before,
            "shown": len(shown),
            "max_jev_regions": max_jev_regions,
            "oov_rule": "query_miss",
            "jev": getattr(jev_obj, "name", "not-loaded"),
        },
        "query_terms": terms,
    }, {"events": events}
