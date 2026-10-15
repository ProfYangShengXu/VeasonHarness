#!/usr/bin/env python
"""VRS 五环编排 —— agent 跑这一个脚本，走完整条链。

    python vrs_pipeline.py --video <mp4> --query "<表达式>" [--index <dir>]

## 这个脚本是什么 / 不是什么

★ 它是 **agent 的作业指导书**，不是又一个 CLI。它把「agent 要在哪几步做决定」显式
  写出来（`--decide-from`），其余全交给 `vrs` CLI（确定性、可复现）。

```
① screen    初筛选关键帧     label 命中直接用，零命中才上 Jev   → 候选帧集（高召回）
② locate    切分             Veason-R1，只定位不选帧，分批≤8     → 每帧 bbox
③ evidence  打标签 + 路由     动作/遮挡/面积/文字/空间            → 证据包（带 src）
④ ★ decide   agent 选帧      ★ 脚本在这里【停下来等 agent】      → frame_id + region_idx
⑤ reflect   分路反思         换帧/放大/换轴，每轮必须带新测量     → 拿新信息再选（无上限）
```

## 这一版落地的四条拍板

* **① 真模型**：①②④ 直接用服务器上的 Veason-R1 / RSI-Jev，没有可注入后端。
* **② OOV = query 词零命中**：不再有 `--oov-rule`；判据只有一条。
* **③ 分批不补齐**：`--pad-to` 已删。② 一批最多 8 帧，最后一批短就是短。
* **④ 放大用 Veason 的框**：`--crop-source` 已删，只有 Veason；①②不能再并行。
* **⑤ 反思无轮次上限**：`--max-reflect` 已删，循环到 `terminated`（拿不出新测量）为止。

## ★ 它在第 ④ 步会停下来（这是设计，不是缺陷）

`--decide-from <file>` 读 agent 的选择；没给就打印**决策请求**并退出码 3。
agent 只排序 + 确认，不产生坐标：它给 `frame_id` + `region_idx`，坐标由 CLI 查。
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


def vrs(*args: str) -> dict:
    """跑一条 `vrs` 命令，拿它 stdout 的单个 JSON。

    ★ 子进程的 stderr 一律转发到本进程的 stderr（降级/成本事件因此不再被吞掉）。
      只把 rc != 0 的 stderr 拼进 SystemExit 会让 rc == 0 的降级读数永远查不出来。
    """
    cmd = [sys.executable, str(Path(__file__).with_name("vrs.py")), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
        sys.stderr.flush()
    if proc.returncode != 0:
        raise SystemExit(f"vrs {' '.join(args)} 失败：{proc.stderr[:600]}")
    return json.loads(proc.stdout)


def _resolve(p: str):
    from vrs_skills import paths as vrs_paths

    return vrs_paths.resolve(p)


def stage_screen(index_dir: str, query: str, top_n: int) -> dict:
    """① 初筛：label 命中直接用，query 词零命中的帧才上 Jev。"""
    return vrs("screen", index_dir, "--query", query, "--top-n", str(top_n))


def stage_locate(index_dir: str, frames: list[int], query: str, batch_size: int = 8) -> dict:
    """② 切分：Veason-R1 定位。★ 只定位，不选帧；分批 ≤8，不补齐。"""
    if not frames:
        raise SystemExit("① 初筛返回了 0 个候选帧：没有帧可以交给 ② 定位。"
                         "先看 screen 的 screened 读数与 stderr 事件。")
    return vrs("locate", index_dir, "--frames", ",".join(map(str, frames)),
               "--query", query, "--batch-size", str(batch_size))


def stage_evidence(index_dir: str, frames: list[int], query: str, axes: str) -> dict:
    """③ 按轴路由。面积/数量纯计算（零模型）；其余读 `vrs senses`/`vrs relate` 的真测量。"""
    return vrs("evidence", index_dir, "--frames", ",".join(map(str, frames)),
               "--query", query, "--axes", axes)


def stage_decide(candidates: list[dict], evidence: dict, decided_from: str | None) -> dict | None:
    """④ ★ agent 选帧 —— 脚本在这里停下。

    ★ 决策请求里必须同时给 ① 的候选（它带 region_idx）和 ③ 的证据：证据条目本身
      没有 region_idx，agent 照它回不了 `{"frame_id": N, "region_idx": M}`。
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
    return json.loads(_resolve(decided_from).read_text(encoding="utf-8"))


def validate_decision(index_dir: str, decision: dict) -> dict:
    """④ 的出口校验：坐标由 CLI 查，所以这里必须确认那个框真的存在。

    不校验的后果：``{"frame_id": 99999}`` 会一路 exit 0 走到最终答案，而链上没有任何
    一步能发现它 —— 与「结构上产不出不存在的框」这条不变量相抵。
    """
    from vrs_skills.store import IndexStore

    store = IndexStore(_resolve(index_dir))
    store.require()
    if not isinstance(decision, dict):
        raise SystemExit(f"--decide-from 的内容必须是一个 JSON 对象，收到 {type(decision).__name__}")
    for key in ("frame_id", "region_idx"):
        if key not in decision:
            raise SystemExit(f"--decide-from 缺少 {key!r}；需要 {{\"frame_id\": N, \"region_idx\": M}}")
    fid, ridx = int(decision["frame_id"]), int(decision["region_idx"])
    store.require_frame(fid)
    if ridx >= 0:
        # region_idx = -1 是 ① 明确留给「Jev 零检出/不肯答」的哨兵值，不是坐标。
        store.get_region(fid, ridx)
    return {"frame_id": fid, "region_idx": ridx}


def stage_reflect(index_dir: str, state: dict, round_no: int) -> dict:
    """⑤ 分路反思：每轮必须带新测量。★ 没有轮次上限。

    ★ state 必须走**文件**：真实索引上 state 的 JSON 超过命令行上限（Windows ~32K，
      Linux E2BIG），内联只在玩具状态上成立。
    """
    fd, path = tempfile.mkstemp(suffix=".json", prefix="vrs-state-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, ensure_ascii=False)
        return vrs("reflect", index_dir, "--round", str(round_no), "--state", path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _merge_new_info(state: dict, reflected: dict) -> None:
    """把这一轮的新测量并回 state，否则下一轮会把同一件事再报一次「新」。

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
    ap.add_argument("--index", default="", help="已有索引目录；不给就先 vrs index 建")
    ap.add_argument("--frames", type=int, default=48, help="建索引时的抽帧数")
    ap.add_argument("--top-n", type=int, default=6, help="初筛给几个候选")
    ap.add_argument("--axes", default="area,action,occlusion", help="③ 问哪些轴")
    ap.add_argument("--decide-from", default="", help="④ agent 的选择（不给则打印决策请求并退出码 3）")
    args = ap.parse_args(argv)

    index_dir = args.index
    if not index_dir:
        built = vrs("index", args.video, "--frames", str(args.frames))
        index_dir = built["index_dir"]

    # ① 初筛
    screen = stage_screen(index_dir, args.query, args.top_n)
    frames = [c["frame_id"] for c in screen["candidates"]]
    print(json.dumps({"stage": "screen", "count": len(frames),
                      "absent": screen.get("absent"),
                      "screened": screen.get("screened")}, ensure_ascii=False))
    if screen.get("absent"):
        print(json.dumps({"stage": "answer", "absent": True}, ensure_ascii=False))
        return 0

    # ② 切分
    located = stage_locate(index_dir, frames, args.query)
    print(json.dumps({"stage": "locate", "frames": frames,
                      "n_boxes": located.get("n_boxes")}, ensure_ascii=False))

    # ③ 证据
    evidence = stage_evidence(index_dir, frames, args.query, args.axes)
    print(json.dumps({"stage": "evidence", "n": len(evidence.get("evidence", []))}, ensure_ascii=False))

    # ④ ★ agent 在这里做决定
    decision = stage_decide(screen["candidates"], evidence, args.decide_from or None)
    if decision is None:
        return EXIT_NEEDS_DECISION
    decision = validate_decision(index_dir, decision)

    # ⑤ 分路反思（无轮次上限；终止 = 拿不出新测量）
    state = {"query": args.query, "candidates": screen["candidates"],
             "evidence": evidence.get("evidence", []), "decision": decision,
             "seen_frames": frames, "locate": located.get("locate", []),
             "absent": screen.get("absent")}
    round_no = 0
    while True:
        round_no += 1
        reflected = stage_reflect(index_dir, state, round_no)
        print(json.dumps({"stage": f"reflect-{round_no}",
                          "action": reflected.get("action"),
                          "new_info": reflected.get("new_info", []),
                          "terminated": reflected.get("terminated"),
                          "reason": reflected.get("reason")}, ensure_ascii=False))
        if reflected.get("terminated") or not reflected.get("new_info"):
            break
        _merge_new_info(state, reflected)

    print(json.dumps({"stage": "answer", "decision": decision,
                      "reflect_rounds": round_no - 1}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
