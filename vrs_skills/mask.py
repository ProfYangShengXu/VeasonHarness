"""③ 掩码打标：把 locate 给的框，用 SAM2 抠成像素级掩码，并算出可审计的元数据。

## 它在链路里的位置

```
① screen   初筛（真 RSI-Jev 判 OOV）
② locate   Veason-R1 给框（只定位，不选帧）
③ mask     ★ 本模块：框 → 掩码 → 元数据      ← 测量，不是问模型
④ reflect  MLLM 拿【图 + 这些元数据】决定下一步（放大 / 换帧 / 换轴）
```

## 为什么要单独一环

Veason 给的是**框**，而框里混着背景 —— 实测那只猴子框内填充率只有 **0.71**，
即框里有 **29% 不是目标**。这个数决定了下一步该「放大那块重看」还是「换个问法」。
它是一个**测量问题**，不该交给模型去猜（设计 §6.2：判据必须测量出来，不是问出来）。

## 元数据（全部可复算）

| 字段 | 含义 |
|---|---|
| ``area_px`` / ``area_ratio`` | 掩码像素数 / 占全图比例 |
| ``fill_ratio`` | **\|掩码 ∩ 框\| ÷ \|框\|** —— 框里有多少像素是目标，值域 [0,1] |
| ``inside_box_px`` / ``outside_box_px`` | 掩码在框内 / 伸出框外的像素数 |
| ``bbox`` | 掩码的外接框（与输入框对照可看出框偏了多少） |
| ``score`` | SAM2 自己的 IoU 预测，**如实记录，不作判据** |
| ``frame_size`` | 掩码的 (h, w)，用于反算坐标 |
| ``mask_source`` | 这次用的哪档权重，如实标注 |

★ **反直觉但必须记住**：``fill_ratio`` 单看会骗人 ——
把框放大 3 倍，SAM2 会把背景也当成目标抠进去，fill 反而从 0.71 升到 0.91。
所以判断「框准不准」要 **area / bbox 偏移 / outside 几个量一起看**，
这也正是这一环的产物要连同**原图**一起交给 ④ 的原因。

★ 加载失败一律硬 ``VrsError``（模型该在 GPU box 上），**不静默退回用框代替掩码**。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from .errors import VrsError
from .paths import DEFAULT_SAM2_SMALL, DEFAULT_SAM2_TINY

#: SAM2 权重搜索顺序（先小后大）。路径全部来自 paths.py —— 唯一的路径层。
CKPTS: tuple[tuple[Path, str], ...] = (
    (DEFAULT_SAM2_TINY, "sam2.1/sam2.1_hiera_t.yaml"),
    (DEFAULT_SAM2_SMALL, "sam2.1/sam2.1_hiera_s.yaml"),
)


def mask_metadata(mask: np.ndarray, box: Sequence[float],
                  score: float) -> dict[str, Any]:
    """从掩码 + 输入框算出全部元数据（纯计算，可复算）。"""
    m = np.ascontiguousarray(np.asarray(mask).astype(bool))
    h, w = m.shape
    n = int(m.sum())
    if n == 0:
        return {"area_px": 0, "area_ratio": 0.0, "fill_ratio": 0.0,
                "inside_box_px": 0, "outside_box_px": 0, "bbox": None,
                "score": round(float(score), 4), "frame_size": [h, w], "empty": True}
    ys, xs = np.where(m)
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    # 框裁剪到图内（越界框不该把分母算大）
    cx1, cy1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
    cx2, cy2 = min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
    box_px = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    # ★ fill_ratio = 框里有多少像素是目标 = |掩码 ∩ 框| / |框|，值域 [0,1]。
    #   拿掩码总面积除以框面积是错的（掩码会伸出框外 ⇒ 算出 >1 的值）。
    inside = int(m[cy1:cy2, cx1:cx2].sum()) if box_px else 0
    return {
        "area_px": n,
        "area_ratio": round(n / float(h * w), 6),
        "fill_ratio": round(inside / float(box_px), 4) if box_px else None,
        "inside_box_px": inside,
        "outside_box_px": n - inside,
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "score": round(float(score), 4),
        "frame_size": [h, w],
        "empty": False,
    }


class Sam2Segmenter:
    """懒加载的 SAM2 图像分割器。加载失败 = 硬错。"""

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None,
                 ckpt: Optional[str] = None) -> None:
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:  # pragma: no cover - 只在缺依赖时走
            raise VrsError(
                f"SAM2 不可用：{exc}",
                hint="在 GPU box 上装 SAM-2，并把 sam2.1_hiera_{tiny,small}.pt 放到 "
                     "paths.MODELS_DIR；依赖还要补 portalocker / iopath / hydra-core / timm。",
                code=2,
            ) from exc
        self._build = build_sam2
        self._Pred = SAM2ImagePredictor
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.ckpt = Path(ckpt) if ckpt else self._find_ckpt(
            Path(model_dir) if model_dir else None)
        self.cfg = self._cfg_for(self.ckpt.name)
        self.name = f"sam2.1-hiera-{self.ckpt.stem.rsplit('_', 1)[-1]}"
        self._pred = None

    @staticmethod
    def _find_ckpt(root: Optional[Path] = None) -> Path:
        """按 paths.py 的常量找权重；显式给了 root 就只在那底下找。"""
        cands = ([root / p.name for p, _ in CKPTS] if root else [p for p, _ in CKPTS])
        for p in cands:
            if p.exists():
                return Path(p)
        raise VrsError(
            "SAM2 权重没找到（找过 " + " / ".join(str(p) for p in cands) + "）",
            hint="把 sam2.1_hiera_{tiny,small}.pt 放到 paths.MODELS_DIR（仓库 models/），"
                 "或在 GPU box 上 ln -s 过去。",
            code=2,
        )

    @staticmethod
    def _cfg_for(fname: str) -> str:
        for f, cfg in CKPTS:
            if f.name == fname:
                return cfg
        raise VrsError(f"没有 {fname} 对应的 SAM2 config",
                       hint=f"已知：{[f.name for f, _ in CKPTS]}", code=2)

    def _load(self) -> Any:
        if self._pred is None:
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=str(_sam2_config_dir()), version_base=None):
                model = self._build(self.cfg, str(self.ckpt), device=self.device)
            self._pred = self._Pred(model)
        return self._pred

    def segment(self, frame_rgb: Any, box: Sequence[float]) -> dict[str, Any]:
        """一张图 + 一个框 → 掩码 + 元数据。

        ``frame_rgb`` 是 **RGB** ndarray (H,W,3)。返回里带 ``__mask__`` 键存布尔掩码本身，
        调用方负责落盘（``store.save_masks``）后删掉该键再序列化 —— 掩码不是 JSON。
        """
        pred = self._load()
        # ★ SAM2 官方 ``set_image`` 要的就是 **RGB**（HWC, ndarray, 0-255）：
        #   "The input image to embed in RGB format. The image should be in HWC
        #    format if np.ndarray"。
        #   之前这里转成了 BGR（多发一次翻转）—— 属于接线错误，实测掩码仍能用，
        #   但那是因为 SAM2 对色偏有一定鲁棒性，不是它该吃的输入。
        #   本矩阵只保证连续（负 stride 会让 torch 崩），不改通道。
        img = np.ascontiguousarray(np.asarray(frame_rgb))
        pred.set_image(img)
        b = np.asarray([float(v) for v in box[:4]])[None, :]
        masks, scores, _ = pred.predict(box=b, multimask_output=False)
        m = np.ascontiguousarray(np.asarray(masks[0]).astype(bool))
        meta = mask_metadata(m, box, float(scores[0]))
        meta["mask_source"] = self.name
        meta["input_box"] = [int(round(float(v))) for v in box[:4]]
        meta["__mask__"] = m
        return meta


def _sam2_config_dir() -> Path:
    """SAM2 的 hydra config 目录（随包安装，在 site-packages/sam2/configs）。"""
    import sam2

    return Path(sam2.__file__).resolve().parent / "configs"


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False
