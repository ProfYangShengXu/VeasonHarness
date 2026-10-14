"""② 切分：Veason-R1。★ 只定位，不选帧。

## 为什么这条界线必须画在这里（规格 §0/§3，实测依据）

```
Veason-R1 定位   平均 IoU 0.786 / 中位 0.914 / 83% 的框 IoU>0.5   ← 很强
Veason-R1 选帧   0.111（随机 0.125），33/36 都选帧 0                ← 等于没有
```

所以「挑哪一帧」不归它管（那是 ① + ④ 的活），它只回答「你让我看的这一帧里，
目标在哪」。本命令**不接受也不需要**「哪一帧最好」这种问题。

## 官方推理口径（规格 §3，逐行抄自 batch_eval_keyframe_sft_dataset_.py）

* 一次吃 **8 帧**；每张图前加 ``(帧号 s)`` 前缀（是**帧号**不是秒）；
* 图缩到 560×560；框在 560 空间里，要乘 ``width/560`` 反算回原图；
* 贪心解码，``<think>…</think><answer>{"keyframe_timestamp":…,"bbox_2d_list":[…]}</answer>``；
* ★ ``keyframe_timestamp`` 必须落在采样帧集合里 —— 官方 reward 里
  ``if keyframe_id not in frame_ids: return 0``，静默 fallback 到 0 会把
  一个格式错误伪装成「选了第 0 帧」的合法答案。

## 补齐：本命令自己补，且补的是【真·新帧】

pipeline 的 ``stage_locate`` 用 ``ids.append(ids[-1])`` 补齐（复制最后一帧）。
这在本机是错的：官方口径要 8 个**不同**帧，重复同一个 id 等于把一份信息
数了 8 遍，还会把「6 真候选 + 2 个重复」伪装成 8 帧的输入。所以这里：
1. 先去掉重复 id（保留首次出现顺序）—— pipeline 的补齐因此被抵消；
2. 不足 ``--pad-to`` 时，从索引里挑**与最后一个候选时间最近、且还没被要过**的帧；
3. 补进来的帧记在 ``padded``，**不进** ``locate`` 输出（调用方要的是它要的那几帧）。
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

from .backends import BackendUnavailable, get_backend
from .errors import VrsError

#: Veason-R1 一次前向吃几帧（`num_sampled_frames=8`）。
VEASON_FRAMES = 8


def parse_frame_ids(raw: str) -> list[int]:
    if raw is None or not str(raw).strip():
        raise VrsError(
            "no --frames given",
            hint='usage: veason locate <index_dir> --frames 850,384 --query "..."',
            code=2,
        )
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise VrsError(
                f"--frames {raw!r} is not a comma-separated list of ints",
                hint="frame ids are integers, e.g. --frames 850,384",
                code=2,
            ) from None
    if not out:
        raise VrsError(f"--frames {raw!r} contains no frame ids",
                       hint="pass at least one id, e.g. --frames 850", code=2)
    return out


def dedupe(ids: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def pad_with_unseen(ids: list[int], available: Sequence[int], pad_to: int) -> list[int]:
    """补齐到 ``pad_to`` 的倍数，补的是**没被要过**的帧。

    选题策略：按 int 距离最后一个候选帧最近优先，同距取小 —— 完全确定性，
    不需要模型，也不需要随机种子。离得近的帧在外观上更可能还看得见目标，
    这对 ② 是有用的（对 ④ 的「换帧」不一定，那条要求**没看过**，见 reflect.py）。
    """
    if pad_to < 1:
        raise VrsError(f"--pad-to must be >= 1 (got {pad_to})",
                       hint="Veason-R1 一次吃 8 帧，所以默认 8。", code=2)
    out = list(ids)
    pool = [f for f in available if f not in set(out)]
    if not pool:
        # 索引里的帧全被要过了：只能重复，但把这件事说出来。
        while len(out) % pad_to:
            out.append(out[-1])
        return out
    anchor = out[-1]
    pool.sort(key=lambda f: (abs(f - anchor), f))
    k = 0
    while len(out) % pad_to:
        out.append(pool[k % len(pool)])
        k += 1
    return out


def batches(ids: Sequence[int], size: int) -> list[list[int]]:
    return [list(ids[i:i + size]) for i in range(0, len(ids), size)]


def run_locate(store: Any, frames: str, query: str, *, pad_to: int = VEASON_FRAMES,
               veason_backend: Optional[str] = None,
               load_frame: Optional[Any] = None) -> tuple[dict, dict]:
    if not (query or "").strip():
        raise VrsError("no query given", hint='usage: veason locate <index_dir> --frames 1,2 --query "..."', code=2)
    meta = store.require()
    available = [int(f["frame_id"]) for f in meta.get("frames", [])]
    for fid in parse_frame_ids(frames):
        store.require_frame(fid)          # 给出「已知帧号」的可行动提示

    requested = dedupe(parse_frame_ids(frames))
    filled = pad_with_unseen(requested, available, pad_to)
    padded = [f for f in filled if f not in set(requested)]
    # ★ 送进模型的是 **filled**（请求帧 + 填充帧），不是 requested。
    #   只送 requested 的话最后一批会短于 pad_to —— 而「一次吃 8 帧」正是
    #   官方口径的一部分，短批会让模型看到的帧数不是 8。填充帧的答案照样
    #   丢弃（不进 locate 输出），它们的作用只是把一次前向填满。
    plan = list(filled)

    events: list[dict] = []
    empty = {int(f): [] for f in requested}
    try:
        backend = get_backend("veason", veason_backend)
    except BackendUnavailable as exc:
        # ★ 不编框。拿不到定位就交空表 + 一条 stderr 事件，
        #   让「不知道」在读数里长得像不知道 —— 编一个框出来才是灾难。
        events.append({
            "event": "locate_backend_unavailable", "error": exc.error, "hint": exc.hint,
            "consequence": "每帧 bbox_2d_list 返回空表，没有任何坐标被编造出来。",
        })
        return _payload(requested, empty, padded, pad_to, src=None), {"events": events}

    result: dict[int, list[list[int]]] = {}
    for batch_ids in batches(plan, pad_to):
        imgs = [_read_frame(store, fid, load_frame) for fid in batch_ids]
        result.update(backend.locate(imgs, batch_ids, query))

    # 只对「我们根本没送过」的帧报异常。填充帧是**送过**的，后端返回它们的框是
    # 正常的（只是不该进答案）—— 把正常行为报成异常会让 stderr 变噪声。
    unknown = sorted(set(result) - set(filled))
    if unknown:
        # 打包帧不该出现在答案里；出现了说明后端没守约定，说清楚而不是静默丢。
        events.append({"event": "locate_returned_unrequested_frames",
                       "frames": unknown,
                       "note": "这些帧只是用来把一次前向填满 8 帧的填充帧，不是调用方要的。"})
    payload = _payload(requested, {f: result.get(f, []) for f in requested},
                       padded, pad_to, src=getattr(backend, "name", None))
    return payload, {"events": events}


def _payload(requested: Sequence[int], boxes: dict[int, list[list[int]]],
             padded: Sequence[int], pad_to: int, *, src: Optional[str]) -> dict:
    return {
        "locate": [
            {"frame_id": int(f),
             "bbox_2d_list": [[int(v) for v in b] for b in boxes.get(f, [])],
             # iou_hint 留 null：它需要【第二次测量】才有意义（比如 ③ 的 mask 或
             # 人工 GT）。拿本轮的框自己跟自己算 IoU 是恒等，不是提示。
             "iou_hint": None}
            for f in requested
        ],
        "src": src or "unavailable",
        "batch_size": pad_to,
        "padded": [int(f) for f in padded],
        "n_boxes": sum(len(boxes.get(f, [])) for f in requested),
    }


def _read_frame(store: Any, fid: int, load_frame: Optional[Any]) -> Any:
    if load_frame is not None:
        return load_frame(store, fid)
    import cv2

    img = cv2.imread(str(store.frame_image(fid)))
    if img is None:
        raise VrsError(f"frame image unreadable: {store.frame_image(fid)}",
                       hint="re-run `veason index`; the stored PNG is corrupt.")
    return img
