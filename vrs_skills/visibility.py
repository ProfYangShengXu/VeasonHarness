# -*- coding: utf-8 -*-
"""visibility 轴：这条 region 是不是一个「完整可见的目标」。

与 occlusion 的区别：
  occlusion  读 depth 的 rank_in_frame / mean_rel —— 量的是「近不近」
  visibility 读掩码几何 + 面积相对量     —— 量的是「露出多少、完不完整」

★★ 为什么不合成单一分数：契约要求「测量出来，不是问出来」。三个分量各自可复算、
   可追责、可单独反驳；合成一个 score 就把它们的证据抹平了。下游（MLLM）拿到的
   是分量的集合，由它按题面决定哪个分量重要。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

AXIS = "visibility"

#: 掩码读出来后缓存在这里（避免同一次 evidence 里重复解 npz）。
_CACHE: dict[tuple[str, int], Optional[dict]] = {}


def _mask_entry(store: Any, fid: int, ridx: int) -> Optional[dict]:
    key = (str(getattr(store, "root", "")), int(fid))
    if key in _CACHE:
        arr = _CACHE[key]
    else:
        try:
            arr = store.load_masks(int(fid)) or {}
        except Exception:
            arr = {}
        _CACHE[key] = arr
    # ★ load_masks 返回的是 {region_idx: mask_2d}（store.py:257），不是 list。
    #   当成 list 迭代会拿到 key（int）⇒ ndim=0 ⇒ 整个轴静默返回空。
    if not isinstance(arr, dict):
        return None
    m = arr.get(ridx)
    if m is None:
        return None
    m = np.asarray(m).astype(bool)
    h, w = m.shape
    if not m.any():
        return None
    # 贴边：四条边各自的命中率，取最大（贴一条边就算「伸到画面外」）
    edge_touch = float(max(m[0].mean(), m[-1].mean(), m[:, 0].mean(), m[:, -1].mean()))
    ys, xs = np.where(m)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    # ★ box_fill = 框里有多少像素是目标 = |掩码 ∩ 框| / |框|（值域 [0,1]）。
    #   regions JSON 里没存这个数，现算 —— 它是「框内被别的东西挤占」的直接测量。
    box_px = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    inside = int(m[y0:y1 + 1, x0:x1 + 1].sum())
    return {
        "area_px": int(m.sum()),
        "area_ratio": float(m.sum()) / float(h * w),
        "bbox": [x0, y0, x1, y1],
        "box_fill": round(inside / float(box_px), 4),
        "edge_touch": round(edge_touch, 4),
        "touches_frame": bool(edge_touch > 0.0),
        "frame_hw": [h, w],
    }


def read(regions: list[dict], meta: dict, senses: Optional[dict], relations: Optional[dict],
         query: str, store: Any = None, frame_id: Optional[int] = None) -> tuple[dict, str]:
    """返回 ({region_idx: {分量...}}, src)。"""
    if store is None or frame_id is None:
        return {}, "sam2"

    # 同索引的面积基准：用中位数（稳健，不被场景框拉偏）
    ratios: list[float] = []
    for r in regions:
        try:
            ratios.append(float(r.get("area_ratio") or 0.0))
        except Exception:
            continue
    base = float(np.median(ratios)) if ratios else 0.0
    if base <= 0:
        base = 0.0

    out: dict[int, dict] = {}
    for r in regions:
        ridx = r.get("idx")
        if ridx is None:
            continue
        ridx = int(ridx)
        st = _mask_entry(store, int(frame_id), ridx)
        if st is None:
            continue
        rel = (st["area_ratio"] / base) if base > 0 else None
        fill = r.get("fill_ratio")
        out[ridx] = {
            "area_ratio": round(st["area_ratio"], 6),
            "rel_area": None if rel is None else round(rel, 4),
            "edge_touch": st["edge_touch"],
            "touches_frame": st["touches_frame"],
            "box_fill": st["box_fill"],
            "index_fill_ratio": None if fill is None else round(float(fill), 4),
            # 被画面切掉 = 贴边 + 相对面积不大（场景框是贴边且巨大，不在此列）
            "truncated": bool(st["touches_frame"] and rel is not None and rel < 1.0),
            "bbox": st["bbox"],
        }
    return out, "sam2+computed"
