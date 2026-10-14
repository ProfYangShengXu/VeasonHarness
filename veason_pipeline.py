#!/usr/bin/env python
"""VRS 五环编排 —— 使用 skill 的 agent（Hermes）跑这一个脚本，走完整条链。

    python veason_pipeline.py --video <mp4> --query "<表达式>" [--index <dir>]

## 这个脚本是什么 / 不是什么

★ 它是 **agent 的作业指导书**，不是又一个 CLI。
  它把「agent 要在哪几步做决定」显式写出来（`--decide-*` 那些），
  其余全交给 `veason` CLI（确定性、可复现）。

```
⓪ ★ rewrite agent 改写 query  ★ 脚本在这里【停下来等 agent】 → 外观名词短语/synonym
① screen    初筛选关键帧     YOLO 主力 + OOV 上 Jev        → 候选帧集（高召回）
② locate    切分             Veason-R1，只定位不选帧       → 每帧 bbox
③ evidence  打标签 + 路由    动作/遮挡/面积/文字/空间       → 证据包（带 src）
④ ★ decide   agent 选帧      ★ 脚本在这里【停下来等 agent】 → frame_id + region_idx
⑤ reflect   分路反思         换帧/放大/换轴，★ 不设次数上限   → 拿新信息再选
                                （唯一的正常终止：没有新测量可拿）
```

## ★ 它会在两处停下来等 agent（⓪ 改写 与 ④ 选帧，这是设计，不是缺陷）

`--rewrite-from <file>` / `--synonym` 给 agent 的【改写】；没给就打印改写请求并退出码 3。

★ 为什么必须有 ⓪：开放词表检测器「A prompt describes appearance, not relationships」
（YOLOE 官方 Limitations）。整句/关系式表达式喂进去会零命中，而**这是静默的**
（照样出候选、照样往下走，只是 yolo_hit=0）。所以改写是【agent 的活】，且必须
留下可观测的 stage 事件 —— 见 `stage_rewrite`。

`--decide-from <file>` 读 agent 的选择；没给就打印**决策请求**并退出码 3。
这个形态对应 skill 的铁律：**agent 只排序 + 确认，不产生坐标**。
（agent 自己也不会输入坐标 —— 它给 `frame_id` + `region_idx`，坐标由 CLI 查。）

## 参数里的「待拍板」默认值

它们来自 `资料\\VRS-架构-v2-20261014.md` §9，我按倾向设了默认、全部可覆盖：

    --oov-rule        query_miss（默认，可解释） | score_threshold
    --pad-to          8（Veason-R1 一次吃 8 帧，候选不足就补齐）
    --crop-source     yolo（默认） | veason（更准但要多跑一轮）
    --reflect-first   low_visibility（默认，零成本确定性） | by_measurement
    --fuse-reflect    1000（★ 保险丝，不是轮次上限；0 = 不设）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

EXIT_NEEDS_DECISION = 3
#: Veason-R1 官方推理口径：一次 8 帧、每帧 560×560（`batch_eval_keyframe_sft_dataset_.py`）
VEASON_FRAMES = 8


def veason(*args: str) -> dict:
    """跑一条 `veason` 命令，拿它 stdout 的单个 JSON。"""
    cmd = [sys.executable, str(Path(__file__).with_name("veason.py")), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # ★ 子进程 stderr 一律转发到本进程 stderr（降级/成本事件因此不再被吞掉）。
    #   只在 rc != 0 时拼接会漏掉所有 rc == 0 的降级读数
    #   （screen_jev_unavailable / screen_top_n_truncated / evidence_axis_unavailable …），
    #   那些正是「声明有产、但没有送达路径」的典型。见 AGENTS.md §7⑬。
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    if proc.returncode != 0:
        raise SystemExit(f"veason {' '.join(args)} 失败：{proc.stderr[:600]}")
    return json.loads(proc.stdout)


def _synonym_args(raw) -> list[str]:
    """把改写产物的两种写法归一成 ``["KEY=WORD[,WORD]", ...]``（screen 的 --synonym 形态）。

    两种写法（其余一律报错，不猜）：
      ``["奶龙=parachute,stuffed toy"]``      —— 与 CLI 同形
      ``{"奶龙": ["parachute", "stuffed toy"]}`` —— KEY → 词表映射
    """
    out: list[str] = []
    if isinstance(raw, dict):
        for key, words in raw.items():
            ws = [words] if isinstance(words, str) else list(words)
            ws = [str(w).strip() for w in ws if str(w).strip()]
            if not str(key).strip() or not ws:
                raise SystemExit(f"--rewrite-from 的条目 {key!r} 有一边是空的（两边都要非空）")
            out.append(f"{str(key).strip()}=" + ",".join(ws))
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, str) or "=" not in item:
                raise SystemExit(
                    "--rewrite-from 的 synonym 列表里每一项都要是 'KEY=WORD[,WORD]'，"
                    f"收到 {item!r}")
            out.append(item)
    else:
        raise SystemExit("--rewrite-from 里 synonym(s) 必须是字符串列表或 KEY→词表 对象，"
                         f"收到 {type(raw).__name__}")
    return out


def load_rewrite(path: str) -> list[str]:
    """读 agent 的第 ⓪ 步改写产物。形态与 --decide-from 一致：一个 JSON 文件，路径过 paths.resolve。

    ★ 校验口径只留一处：最后交给 ``screen.parse_synonyms`` 过一遍，格式错就在这里报。
    """
    resolved = _resolve(path)
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"--rewrite-from 读不到 {resolved}：{exc}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--rewrite-from 不是合法 JSON（{resolved}）：{exc}") from None
    if not isinstance(data, dict):
        raise SystemExit(f"--rewrite-from 的内容必须是一个 JSON 对象，收到 {type(data).__name__}")
    if ("synonym" in data) == ("synonyms" in data):
        raise SystemExit(
            '--rewrite-from 必须且只能含 "synonym"（列表）或 "synonyms"（映射）之一；'
            '例：{"synonym": ["奶龙=parachute,stuffed toy"]}')
    args = _synonym_args(data["synonym"] if "synonym" in data else data["synonyms"])
    from veason_harness.errors import VrsError
    from veason_harness.screen import parse_synonyms

    try:
        parse_synonyms(args)
    except VrsError as exc:
        raise SystemExit(f"--rewrite-from 里的 synonym 不合法：{exc} "
                         f"{getattr(exc, 'hint', '')}") from None
    return args


def stage_rewrite(synonym: list[str], rewrite_from: str, query: str,
                  no_rewrite: bool) -> tuple[list[str], dict, bool]:
    """⓪ ★ agent 的第 ⓪ 步：把 query 改写成外观名词短语（作为 --synonym 交给 screen）。

    返回 ``(synonyms, stage_payload, must_stop)``。

    ★ 三条路径，全都不静默：
      · ``--rewrite-from`` / ``--synonym`` 给了 ⇒ rewritten=True，synonyms 带着走；
      · 没给且没 ``--no-rewrite`` ⇒ rewritten=False + ``pipeline_rewrite_missing`` 事件，
        且 ``must_stop=True``（脚本停下来要 agent 去改写，不允许静默 yolo_hit=0）；
      · 没给但显式 ``--no-rewrite`` ⇒ 同样打事件，但放行（agent 已声明不需要改写）。
    """
    if synonym and rewrite_from:
        raise SystemExit("--synonym 与 --rewrite-from 只能给一个："
                         "手工传用 --synonym，agent 走文件用 --rewrite-from。")
    if no_rewrite and (synonym or rewrite_from):
        raise SystemExit("--no-rewrite 与 --synonym/--rewrite-from 互斥："
                         "要么给改写，要么显式声明不改写。")

    if rewrite_from:
        synonyms = load_rewrite(rewrite_from)
        source = f"--rewrite-from {rewrite_from}"
    elif synonym:
        synonyms = list(synonym)
        source = "--synonym"
    else:
        synonyms = []
        source = ""

    if synonyms:
        from veason_harness.screen import parse_synonyms

        mapping = parse_synonyms(synonyms)
        payload = {
            "stage": "rewrite", "rewritten": True, "source": source, "query": query,
            "synonyms": synonyms,
            "appearance_phrases": [w for words in mapping.values() for w in words],
            "note": "agent 已把 query 改写成外观名词短语；screen 的 OOV 判据用这些词。",
        }
        return synonyms, payload, False

    payload = {
        "stage": "rewrite", "rewritten": False,
        "event": "pipeline_rewrite_missing", "query": query,
        "consequence": "没有改写 ⇒ OOV 判据用的是原始 query 词。开放词表检测器「describes "
                       "appearance, not relationships」，整句/关系式表达式零命中（yolo_hit=0），"
                       "全部落档 B —— 这个失败本来是静默的，这里把它显式化。",
        "hint": "要改写给 --rewrite-from <file> 或 --synonym \"KEY=WORD,...\"；"
                "确实不用改写（query 已是外观名词短语）就显式传 --no-rewrite。",
    }
    return [], payload, not no_rewrite


def stage_screen(index_dir: str, query: str, top_n: int, oov_rule: str,
                 synonyms: list[str] | None = None) -> dict:
    """① 初筛：YOLO 主力，OOV 才上 Jev。``synonyms`` = 第 ⓪ 步改写出来的词。"""
    args = ["screen", index_dir, "--query", query, "--top-n", str(top_n),
            "--oov-rule", oov_rule]
    for pair in synonyms or []:
        args += ["--synonym", pair]
    return veason(*args)


def stage_locate(index_dir: str, frames: list[int], query: str, pad_to: int) -> dict:
    """② 切分：Veason-R1 定位。★ 只定位，不选帧。

    ★ 补齐到 8：它一次吃 8 帧，候选不足时补齐比分批便宜（分批要多一次前向）。

    ⚠ 这里的 ``ids.append(ids[-1])`` 会把最后一个候选重复若干次，而官方口径要
      8 个**不同**帧。``veason locate`` 自己会先去重、再用索引里**没被要过**的帧
      补齐（并且把补进来的帧记在 ``padded`` 里、不混进答案），所以这里的粗糙
      补齐不会漏到模型输入上——但 ``--frames`` 里出现的重复 id 是没意义的，
      真正去重的地方在 locate。判据见 tests/test_five_rings.py::test_locate_dedupes_the_pipelines_repeat_padding。
    """
    if pad_to < 1:
        raise SystemExit(f"--pad-to 必须是 >=1 的整数（收到 {pad_to}）："
                         "它是一次前向的帧数，0 会在取模时除零。")
    ids = list(frames)
    if not ids:
        raise SystemExit("① 初筛返回了 0 个候选帧：没有帧可以交给 ② 定位。"
                         "先看 screen 的 screened 读数与 stderr 事件，再决定是放宽 "
                         "--oov-rule 还是回头修初筛。")
    while len(ids) % pad_to:
        ids.append(ids[-1])
    return veason("locate", index_dir, "--frames", ",".join(map(str, ids)), "--query", query)


def stage_evidence(index_dir: str, frames: list[int], query: str, axes: str) -> dict:
    """③ 按轴路由小模型。面积走纯计算（零模型）；其余按轴选件。"""
    return veason(
        "evidence", index_dir,
        "--frames", ",".join(map(str, frames)),
        "--query", query,
        "--axes", axes,
    )


def stage_decide(candidates: list[dict], evidence: dict, decided_from: str | None) -> dict | None:
    """④ ★ agent 选帧 —— 脚本在这里停下。

    ★ 为什么让它停：agent 的活被压到最小（排序 + 确认），
      它读的是 ③ 的结构化证据（**索引式**：帧号/秒/region idx/属性值），
      不带辩护式文字（「第 3 帧清晰可见因为…」会触发 anchoring）。

    ⚠ 原来这里打印的 ``candidates`` 装的是 ``evidence["evidence"]``，而证据条目里
      **没有 region_idx**（规格 §4.3 的形状就是没有）—— agent 照它回不了
      ``{"frame_id": N, "region_idx": M}``。所以决策请求里必须同时给
      ① 的候选（它带 region_idx），证据作为附件。
    """
    if not decided_from:
        print(json.dumps({
            "needs_decision": True,
            "what": "从候选帧里选一个，给出 frame_id + region_idx",
            "candidates": candidates,
            "evidence": evidence.get("evidence", []),
            "rules": [
                "只排序 + 确认，不生成坐标",
                "有确定性证据（面积/相似度/IoU）时优先信它，不信模型的自信度",
                "候选不足时宁可多留，不要早剔",
            ],
            "reply_with": "--decide-from <file>，文件里写 {\"frame_id\": N, \"region_idx\": M}",
        }, ensure_ascii=False, indent=1))
        return None
    # ★ Windows/WSL 两种路径形式都要能吃（这是全 CLI 唯一不能只靠 Path() 的入口：
    #   从 WSL 调 Windows Python 时 /mnt/c/... 不是 Windows 能打开的路径）。
    return json.loads(_resolve(decided_from).read_text(encoding="utf-8"))


def _resolve(p: str):
    """交给 veason_harness.paths 这一处翻译，别在这里自造第二套路径逻辑。"""
    from veason_harness import paths as vrs_paths

    return vrs_paths.resolve(p)


def validate_decision(index_dir: str, decision: dict) -> dict:
    """④ 的出口校验：坐标由 CLI 查，所以这里必须确认那个框真的存在。

    不校验的后果：``{"frame_id": 99999}`` 会一路 exit 0 走到最终答案，
    而链上没有任何一步能发现它 —— 与「结构上产不出不存在的框」这条不变量相抵。
    """
    from veason_harness.store import IndexStore

    store = IndexStore(_resolve(index_dir))
    store.require()
    if not isinstance(decision, dict):
        raise SystemExit(f"--decide-from 的内容必须是一个 JSON 对象，收到 {type(decision).__name__}")
    for key in ("frame_id", "region_idx"):
        if key not in decision:
            raise SystemExit(f"--decide-from 缺少 {key!r}；需要 "
                             '{"frame_id": N, "region_idx": M}')
    fid, ridx = int(decision["frame_id"]), int(decision["region_idx"])
    store.require_frame(fid)
    if ridx >= 0:
        # region_idx = -1 是 ① 明确留给「Jev 零检出帧」的哨兵值，不是坐标。
        store.get_region(fid, ridx)
    return {"frame_id": fid, "region_idx": ridx}


def stage_reflect(index_dir: str, state: dict, round_no: int, crop_source: str = "yolo") -> dict:
    """⑤ 分路反思：按问题类型路由，每轮必须拿【新信息】。

    ``--crop-source`` 在这里落地（待拍板③）：它决定「放大」剪的是哪个框。

    ★ state 必须走**文件**，不能走命令行。实测：真实索引上 candidates(6) +
      evidence(42 条) 的 JSON 已经超过 Windows 的 ~32K 命令行上限，表现是
      ``FileNotFoundError: [WinError 206] 文件名或扩展名太长``（Linux 上是 E2BIG）。
      规格 §6.3 写 ``--state state.json`` 不是可选项 —— 内联只在玩具状态上成立。
    """
    fd, path = tempfile.mkstemp(suffix=".json", prefix="veason-state-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)
        return veason("reflect", index_dir, "--round", str(round_no),
                   "--state", path, "--crop-source", crop_source)
    finally:
        try:
            os.unlink(path)
        except OSError:  # pragma: no cover - 临时文件被清理是正常的
            pass


def _merge_new_info(state: dict, reflected: dict) -> None:
    """把这一轮的新测量并回 state，否则下一轮会把同一件事再报一次「新」。

    ★ 判据（AGENTS.md §7⑬「声明必须有送达路径」）：reflect 报的 ``new_info`` 若
      只出现在 stdout、不回写 state，``axes_already_measured`` 就永不更新，
      「换轴」那一类会每轮都把同一根轴当新测量 ⇒ 循环只能靠保险丝停下，
      与「唯一正常终止 = no_new_information」相抵。

      * ``frames`` → ``seen_frames``（换帧判据）
      * 带 ``axis`` 的条目 → ``evidence``（换轴判据读 ``axes_already_measured``）
    """
    state["seen_frames"] = sorted(set(state.get("seen_frames", []))
                                  | set(reflected.get("frames", [])))
    new_evidence = [i for i in reflected.get("new_info", []) if i.get("axis")]
    if new_evidence:
        state["evidence"] = list(state.get("evidence", [])) + new_evidence


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="mp4 路径")
    ap.add_argument("--query", required=True, help="指代表达式")
    ap.add_argument("--index", default="", help="已有索引目录；不给就先 veason index 建")
    ap.add_argument("--frames", type=int, default=48, help="建索引时的抽帧数")
    ap.add_argument("--top-n", type=int, default=6, help="初筛给几个候选（★ 实测 >3 个进打分器就崩，>6 更要小心）")
    ap.add_argument("--axes", default="area,action,occlusion", help="③ 问哪些轴")
    ap.add_argument("--decide-from", default="", help="④ agent 的选择（不给则打印决策请求并退出码 3）")
    ap.add_argument("--synonym", action="extend", nargs="+", default=[], metavar="KEY=WORD[,WORD]",
                    help="★ ⓪ agent 改写的手工形态：query 词 ↔ 开放词表 label 的显式映射，"
                         '可重复。例：--synonym "奶龙=parachute,stuffed toy"')
    ap.add_argument("--rewrite-from", default="",
                    help='★ ⓪ agent 改写的文件形态（形态同 --decide-from）：'
                         '{"synonym": ["奶龙=parachute,stuffed toy"]} 或 '
                         '{"synonyms": {"奶龙": ["parachute", "stuffed toy"]}}')
    ap.add_argument("--no-rewrite", action="store_true",
                    help="显式声明「不改写」（query 已是外观名词短语）。仍打 rewrite 事件，"
                         "不静默；与 --synonym/--rewrite-from 互斥。")
    ap.add_argument("--fuse-reflect", "--max-reflect", dest="fuse_reflect", type=int, default=1000,
                    help="★ 反射保险丝，不是轮次上限（默认 1000；0 = 不设）。"
                         "正常终止是 reflect 报 no_new_information")
    ap.add_argument("--no-reflect", action="store_true",
                    help="关掉 ⑤ 整环（反证对照臂用；注意 --fuse-reflect 0 是「不设保险丝」"
                         "不是「不反思」）")
    # ---- 四个待拍板的默认值（架构 §9）----
    ap.add_argument("--oov-rule", default="query_miss", choices=("query_miss", "score_threshold"))
    ap.add_argument("--pad-to", type=int, default=VEASON_FRAMES)
    ap.add_argument("--crop-source", default="yolo", choices=("yolo", "veason"))
    ap.add_argument("--reflect-first", default="low_visibility", choices=("low_visibility", "by_measurement"))
    args = ap.parse_args(argv)

    index_dir = args.index
    if not index_dir:
        # ⚠ 这里原来是 `veason index <video> --frames N --json`，而 `index` 根本没有
        #   `--json` 参数（它本来总打 JSON）⇒ 不给 --index 时 rc=2 直接死。
        built = veason("index", args.video, "--frames", str(args.frames))
        index_dir = built["index_dir"]

    # ⓪ ★ agent 的第 0 步：改写 query。不给就停下要它 —— 绝不静默地 yolo_hit=0。
    rewrite_synonyms, rewrite_payload, must_stop = stage_rewrite(
        args.synonym, args.rewrite_from, args.query, args.no_rewrite)
    print(json.dumps(rewrite_payload, ensure_ascii=False))
    if must_stop:
        print(json.dumps({
            "needs_rewrite": True,
            "what": "把 query 改写成开放词表检测器能匹配的【外观名词短语】，写进文件后重跑",
            "query": args.query,
            "why": "整句/关系式表达式会让检测器零命中；不改写就跑，yolo_hit 恒为 0（且静默）。",
            "reply_with": '--rewrite-from <file>，文件里写 '
                          '{"synonym": ["奶龙=parachute,stuffed toy"]}',
        }, ensure_ascii=False, indent=1))
        return EXIT_NEEDS_DECISION

    # ① 初筛
    screen = stage_screen(index_dir, args.query, args.top_n, args.oov_rule, rewrite_synonyms)
    frames = [c["frame_id"] for c in screen["candidates"]]
    print(json.dumps({"stage": "screen", "count": len(frames), "screened": screen.get("screened")}, ensure_ascii=False))

    # ② 切分
    located = stage_locate(index_dir, frames, args.query, args.pad_to)
    print(json.dumps({"stage": "locate", "frames": frames, "n_boxes": len(located.get("locate", []))}, ensure_ascii=False))

    # ③ 证据
    evidence = stage_evidence(index_dir, frames, args.query, args.axes)
    print(json.dumps({"stage": "evidence", "n": len(evidence.get("evidence", []))}, ensure_ascii=False))

    # ④ ★ agent 在这里做决定
    #   ★ 必须把 ① 的候选喂进去：证据条目里没有 region_idx，agent 照它回不了
    #     {"frame_id": N, "region_idx": M}。
    decision = stage_decide(screen["candidates"], evidence, args.decide_from or None)
    if decision is None:
        return EXIT_NEEDS_DECISION
    # ★ 出口校验：坐标由 CLI 查，所以要确认那个框真的存在（否则 99999 会 exit 0）。
    decision = validate_decision(index_dir, decision)

    # ⑤ 分路反思：★ 不设次数上限。正常终止只有一个 —— reflect 报「没有新测量可拿」。
    #    ``--fuse-reflect`` 是保险丝（默认 1000），只在「循环有 bug、每轮都假装有
    #    新信息」时兜住，触发时明说是保险丝。
    state = {"query": args.query, "candidates": screen["candidates"],
             "evidence": evidence.get("evidence", []), "decision": decision,
             # ★ 「放大」的裁剪框来源（待拍板③）：yolo = 用 ① 的 region 框（默认，
             #   它现在就有）；veason = 用 ② 的框（更准，但要等 ② 跑完）。两者都
             #   得先落在 state 里，reflect 才知道自己能用哪个。
             "locate": located.get("locate", []), "crop_source": args.crop_source,
             "seen_frames": frames}          # ★ 「换帧」要能真的换【没看过的】帧
    if args.no_reflect:
        # ★ 反证对照臂：关掉 ⑤。规格 §10 要求「开/关两臂」，判据只看最终结果。
        #   ⚠ 注意别用 `--fuse-reflect 0` 来关 —— 0 的语义是【不设保险丝】，
        #     不是【不反思】；两者混用会让对照臂悄悄跑成开臂。
        print(json.dumps({"stage": "reflect-disabled", "reflect": False,
                          "note": "反证对照臂：⑤ 整环关掉。"}, ensure_ascii=False))
    else:
        round_no = 1
        while True:
            if args.fuse_reflect and round_no > args.fuse_reflect:
                print(json.dumps({
                    "stage": "reflect-fuse", "fuse": True, "reason": "fuse_round_limit",
                    "rounds_completed": round_no - 1,
                    "note": "★ 这是保险丝，不是正常终止。正常终止是 reason=no_new_information"
                            "（没有新测量可拿）；跑到这里说明每轮都在报新信息却停不下来。",
                }, ensure_ascii=False))
                break
            reflected = stage_reflect(index_dir, state, round_no, args.crop_source)
            print(json.dumps({"stage": f"reflect-{round_no}",
                              "new_info": reflected.get("new_info", []),
                              "terminated": reflected.get("terminated", False),
                              "reason": reflected.get("reason")}, ensure_ascii=False))
            if not reflected.get("new_info"):
                break                        # ★ 没有新测量 ⇒ 停（重排旧信息不算反思）
            _merge_new_info(state, reflected)
            round_no += 1

    print(json.dumps({"stage": "answer", "decision": decision}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
