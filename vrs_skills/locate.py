"""② 切分：Veason-R1。★ 只定位，不选帧。

## 为什么这条界线必须画在这里（实测依据）

```
Veason-R1 定位   平均 IoU 0.786 / 中位 0.914 / 83% 的框 IoU>0.5   ← 很强
Veason-R1 选帧   0.111（随机 0.125），33/36 都选帧 0                ← 等于没有
```

所以「挑哪一帧」不归它管（那是 ① + ④ 的活），它只回答「你让我看的这一帧里，
目标在哪」。

## 官方推理口径（实跑自 ``exp_veason_batch.py``）

* 一次吃 **8 帧**；每张图前加 ``(帧号)`` 前缀（是**帧号**不是秒）；
* 图缩到 560×560；框在 560 空间里，要乘 ``width/560`` 反算回原图；
* ``<think>…</think><answer>{"keyframe_timestamp":…,"bbox_2d_list":[…]}</answer>``；
* ``keyframe_timestamp`` 必须落在这一批的帧集合里 —— 官方 reward 里
  ``if keyframe_id not in frame_ids: return 0``，静默 fallback 到 0 会把一个格式
  错误伪装成「选了第 0 帧」的合法答案，所以这里**报错而不兜底**；
* 官方 eval 的 ``system_prompt`` 是 ``"You are a helpful assistant."``（不是 ``None``）。

## 分批，不补齐（新契约③）

旧版会 ``--pad-to 8``：候选不足时用索引里没被要过的帧凑满一次前向。**已去掉。**
一批最多 8 帧，最后一批短就是短。理由：补齐帧会把「6 个真候选」伪装成 8 帧输入，
而且补齐帧的答案必须丢弃 —— 那等于让模型多做一次它不负责的判断。

## 真模型（新契约①）

``VeasonR1`` 直接加载组服务器 ``~/models/Veason-R1`` 的 Qwen2.5-VL。没有可注入
后端、没有 ``auto -> unavailable``：拿不到定位就是硬 ``VrsError``，绝不编框。
``veason=`` 是测试缝。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional, Sequence

from .errors import VrsError
from .video import load_bgr, parse_frame_ids

#: Veason-R1 一次前向吃几帧（官方 eval ``num_sampled_frames=8``）。
VEASON_FRAMES = 8

#: 官方推理调用方传的 system prompt（``multi_gpu_eval_keyframe_sft_mod_reason_vos.py``）。
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

VEASON_DIR_ENV = "VRS_VEASON_DIR"

_TAIL = (
    " \nYou are given {num} frames uniformly sampled from a 1 FPS video. Each image is "
    "preceded by its timestamp.\nGiven the following expression: '{q}', identify the "
    "timestamp where the target is best represented. Then, locate the target in that "
    "frame by predicting the bounding box(es).\nFirst, output the thinking process in "
    "<think> </think> tags, and then output the result in the following format within "
    "<answer> </answer> tags."
)

_ANSWER = re.compile(r"<answer>\s*(\{.*?\})\s*</answer>", re.S)


def dedupe(ids: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def batches(ids: Sequence[int], size: int = VEASON_FRAMES) -> list[list[int]]:
    if size < 1:
        raise VrsError(f"batch size must be >= 1 (got {size})",
                       hint="Veason-R1 一次 8 帧；分批大小由 --batch-size 管。", code=2)
    return [list(ids[i:i + size]) for i in range(0, len(ids), size)]


def _bboxes_from_answer(answer: str, frame_ids: Sequence[int],
                        sizes: dict[int, tuple[int, int]]) -> dict[int, list[list[int]]]:
    """把 560 空间的框反算回原图坐标，并强制 keyframe 在本批帧集合里。"""
    out: dict[int, list[list[int]]] = {int(f): [] for f in frame_ids}
    m = _ANSWER.search(answer or "")
    if not m:
        return out
    try:
        payload = json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise VrsError(
            f"Veason-R1 returned an <answer> block that is not JSON: {m.group(1)[:200]!r}",
            hint="the model answered but the payload is malformed; re-run once, and if it "
                 "repeats, the checkpoint or the prompt template is not what this caller expects.",
        ) from exc
    key = payload.get("keyframe_timestamp")
    boxes = payload.get("bbox_2d_list") or []
    if key is None:
        return out
    key = int(key)
    if key not in out:
        raise VrsError(
            f"Veason-R1 answered keyframe_timestamp={key}, which is not one of the "
            f"frames in this batch: {sorted(out)}",
            hint="the answer and the caller are in different id spaces (frame index vs "
                 "sample index); a silent fallback would forge a legal-looking answer.",
        )
    w, h = sizes[key]
    xf, yf = w / 560.0, h / 560.0
    for b in boxes:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        out[key].append([int(round(x1 * xf)), int(round(y1 * yf)),
                         int(round(x2 * xf)), int(round(y2 * yf))])
    return out


class VeasonR1:
    """组服务器上的 Veason-R1（Qwen2.5-VL）。★ 只定位，不选帧。"""

    FRAME_RESIZE = 560

    def __init__(self, model_dir: Optional[str] = None,
                 system_prompt: Optional[str] = None,
                 device: str = "cuda") -> None:
        self.model_dir = model_dir or os.environ.get(VEASON_DIR_ENV) or os.path.expanduser(
            "~/models/Veason-R1")
        self.system_prompt = DEFAULT_SYSTEM_PROMPT if system_prompt is None else system_prompt
        self.device = device
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as exc:  # pragma: no cover - 服务器外才会发生
            raise VrsError(
                "torch/transformers not importable, so Veason-R1 cannot run",
                hint="this skill must run on the GPU box (`$HOME/envs/vrs`).",
                code=3,
            ) from exc
        if not os.path.isdir(self.model_dir):
            raise VrsError(
                f"Veason-R1 checkpoint directory not found: {self.model_dir}",
                hint=f"set {VEASON_DIR_ENV} to the checkpoint path on the GPU box.",
                code=3,
            )
        try:
            self._proc = AutoProcessor.from_pretrained(self.model_dir)
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_dir, dtype=torch.bfloat16, device_map=self.device).eval()
        except Exception as exc:
            raise VrsError(
                f"Veason-R1 could not be loaded from {self.model_dir}: {exc}",
                hint="check the checkpoint is complete (two safetensors shards) and the "
                     "GPU has room (bf16 ≈ 8 GB).",
                code=3,
            ) from exc
        self.name = f"veason-r1:{os.path.basename(self.model_dir)}"

    def locate(self, frames_bgr: Sequence[Any], frame_ids: Sequence[int],
               query: str) -> dict[int, list[list[int]]]:
        import torch
        from PIL import Image

        content: list[dict] = []
        imgs = []
        sizes: dict[int, tuple[int, int]] = {}
        for i, (fid, img) in enumerate(zip(frame_ids, frames_bgr)):
            if i:
                content.append({"type": "text", "text": ", "})
            h, w = img.shape[:2]
            sizes[int(fid)] = (w, h)
            content.append({"type": "text", "text": f"({int(fid)}s)"})
            content.append({"type": "image"})
            imgs.append(Image.fromarray(img[:, :, ::-1]).resize(
                (self.FRAME_RESIZE, self.FRAME_RESIZE), Image.BILINEAR))
        content.append({"type": "text",
                        "text": _TAIL.format(num=len(imgs), q=str(query).lower().strip("."))})
        msgs = []
        if self.system_prompt is not None:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": content})
        text = self._proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = self._proc(text=[text], images=imgs, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)
        ans = self._proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return _bboxes_from_answer(ans, [int(f) for f in frame_ids], sizes)


def default_veason(*, model_dir: Optional[str] = None,
                   system_prompt: Optional[str] = None) -> VeasonR1:
    return VeasonR1(model_dir=model_dir, system_prompt=system_prompt)


def run_locate(store: Any, frames: str, query: str, *,
               batch_size: int = VEASON_FRAMES,
               veason: Optional[Any] = None) -> tuple[dict, dict]:
    """返回 ``(stdout payload, info)``；``veason`` 只给测试。"""
    if not (query or "").strip():
        raise VrsError("no query given",
                       hint='usage: vrs locate <index_dir> --frames 1,2 --query "..."', code=2)
    meta = store.require()
    requested = dedupe(parse_frame_ids(frames))
    for fid in requested:
        store.require_frame(fid)

    events: list[dict] = []
    if veason is None:
        veason = default_veason()

    result: dict[int, list[list[int]]] = {}
    for batch_ids in batches(requested, batch_size):
        imgs = [load_bgr(store, fid) for fid in batch_ids]
        result.update(veason.locate(imgs, batch_ids, query))

    unknown = sorted(set(result) - set(requested))
    if unknown:
        events.append({"event": "locate_returned_unrequested_frames", "frames": unknown,
                       "note": "后端返回了没被请求的帧；输出只保留请求帧。"})
    boxes = {f: result.get(f, []) for f in requested}
    payload = {
        "locate": [
            {"frame_id": int(f),
             "bbox_2d_list": [[int(v) for v in b] for b in boxes.get(f, [])],
             # iou_hint 留 null：它需要【第二次测量】才有意义。拿本轮的框自己跟
             # 自己算 IoU 是恒等，不是提示。
             "iou_hint": None}
            for f in requested
        ],
        "src": getattr(veason, "name", "veason-r1"),
        "batch_size": batch_size,
        "n_frames": len(requested),
        "n_boxes": sum(len(boxes.get(f, [])) for f in requested),
    }
    return payload, {"events": events}
