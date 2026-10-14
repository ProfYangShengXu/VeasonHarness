"""①–④ 的模型后端口：真件 / 假件，按名字注入。

## 机制为什么保留（口径已更新）

原来抽这一层是为了「本机 CPU 跑不动大模型，所以把推理赶到别的机器」。
★ 现在部署口径变了：**整个 skill 直接跑在组服务器上**（8×RTX3090），
模型和 skill 在同一台机器上。所以这里**不再有跨机调用 / HTTP 代理那一层** ——
真件就是**同进程加载**的真件，`fake` 仍然是给测试用的确定性替身。

抽象仍然要留，因为两件事没变：
1. 测试必须能在**没有 GPU、没有 8.15GB 权重**的地方验编排（`fake`）；
2. 件没接线时必须是**一个可 catch 的降级**，而不是让整条命令 exit 1
   （① 的高召回纪律要求「Jev 不可用」≠「目标不在」）。

## ★ Jev 的形态是【N 选一】，不是【逐个判是/不是】

这条是实测否掉的，不是偏好（`资料/Jev用法-20261013.md` §6 纪律②）：

```
✗ 把 4 个框分别问它「这个是目标吗？」 → 它 74% 都答「是」
✓ 把 4 个框摆在【同一道题】的候选里，让它挑一个
   独立判断没有比较压力，它就会一律放行。
```

所以 `JevBackend` 暴露的是 ``choose(frame, regions, query)``（一次前向挑一个），
而不是 ``confirm(frame, bbox, query)`` 的循环。**前者是唯一的正路**；
把它退化成逐区域提问，就等于把这个件换成一个「全说 yes」的常数。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

from .errors import VrsError

#: 真件的权重目录可以用环境变量挪走（服务器上的路径和仓库布局不一定一样）。
MODELS_DIR_ENV = "VEASON_MODELS_DIR"
DEVICE_ENV = "VEASON_DEVICE"
# ★ 环境变量：新名优先，旧名（VRS_）保留作别名 —— 改名不硬切。
BACKEND_ENV = "VEASON_{kind}_BACKEND"
LEGACY_BACKEND_ENV = "VRS_{kind}_BACKEND"


class BackendUnavailable(VrsError):
    """件没接线（缺权重 / 缺依赖 / 缺设备）。

    继承 ``VrsError`` 是为了让调用方**能选择降级**：① 的高召回纪律要求
    「Jev 不可用」不能等于「目标不在」，所以这个异常必须可以被 catch，
    而不是把整条命令打死。
    """

    def __init__(self, what: str, hint: str) -> None:
        super().__init__(f"{what} is not available", hint=hint, code=3)
        self.what = what


def models_dir() -> Any:
    from .paths import MODELS_DIR

    return (os.environ.get(MODELS_DIR_ENV) or os.environ.get("VRS_MODELS_DIR")
            or MODELS_DIR)


# ---------------------------------------------------------------------------
# 协议（只放方法，不放数据字段 —— PEP 544：数据协议会让 issubclass 不可用）
# ---------------------------------------------------------------------------

@runtime_checkable
class JevBackend(Protocol):
    """Visual Jev：在给定选项上打分的 forced-choice 件。

    ★ 它**不产生坐标**，也**不解释**；它只在候选表里挑一个最像的。
      候选池的召回是上游的责任 —— 选项里没有正确答案它不会报警，会给一个
      很自信的错误答案（实测答错时仍有 0.7~0.98）。
    """

    name: str

    def exists(self, frame_rgb: Any, query: str) -> Optional[float]:
        """存在性（none 档）：候选 ``present / not present``，返回 present 的分数。"""

    def choose(self, frame_rgb: Any, regions: Sequence[dict],
               query: str) -> Optional[dict]:
        """候选【N 选一】：把 regions 编成选项，**一次前向**挑一个。

        返回 ``{"region_idx": int | None, "score": float, "scores": {...}}``；
        ``region_idx is None`` = 它选了「都不是」（前提为假那一档）。
        """


@runtime_checkable
class VeasonBackend(Protocol):
    """② 定位。★ 只定位，不选帧 —— 调用方给帧号，它给框。"""

    name: str

    def locate(self, frames_rgb: Sequence[Any], frame_ids: Sequence[int],
               query: str) -> dict[int, list[list[int]]]:
        """一次 ≤8 帧，返回 ``{frame_id: [bbox, ...]}``。"""


@runtime_checkable
class AxisBackend(Protocol):
    """③ 一个轴一个件。★ 每个件只对它负责的那个轴出数。"""

    name: str

    def supports(self, axis: str) -> bool: ...

    def measure(self, axis: str, frame_rgb: Any, regions: list[dict],
                masks: dict, query: str) -> Optional[list[dict]]:
        """返回该轴的 ``[{region_idx, value, src}, ...]``；``None`` = 不支持或没得测。"""


# ---------------------------------------------------------------------------
# 注册表：照 huggingface_hub 的 PROVIDERS → get_provider_helper 那套形状
# ---------------------------------------------------------------------------

_FACTORIES: dict[str, dict[str, Callable[..., Any]]] = {"jev": {}, "veason": {}, "axis": {}}


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        _FACTORIES[kind][name] = fn
        return fn

    return deco


def registered(kind: str) -> list[str]:
    return sorted(_FACTORIES[kind])


def get_backend(kind: str, spec: Optional[str] = None, **kwargs: Any) -> Any:
    """``spec`` > 环境变量 ``VRS_<KIND>_BACKEND`` > ``"auto"``。

    ★ ``auto`` 现在解析成 **local（同进程真件）**，因为部署已经变成「skill 跑在
      服务器上」。本机没权重时 local 会抛 ``BackendUnavailable``，调用方按降级
      处理 —— 于是同一份代码在服务器上真跑、在没有 GPU 的机器上仍然可测。
    """
    if kind not in _FACTORIES:
        raise VrsError(f"unknown backend kind: {kind}", hint=f"known: {sorted(_FACTORIES)}", code=2)
    spec = (spec or os.environ.get(BACKEND_ENV.format(kind=kind.upper()), "")
            or os.environ.get(LEGACY_BACKEND_ENV.format(kind=kind.upper()), "") or "auto").strip()
    if spec == "auto":
        spec = "senses-cache" if kind == "axis" else "local"
    factory = _FACTORIES[kind].get(spec)
    if factory is None:
        raise BackendUnavailable(
            f"{kind} backend {spec!r}",
            hint=f"registered {kind} backends: {registered(kind)}; "
                 f"set {BACKEND_ENV.format(kind=kind.upper())} or pass the backend explicitly.",
        )
    return factory(**kwargs)


# ---------------------------------------------------------------------------
# 假件：确定性、零依赖。■ 它验证的是【编排与归因】，不是模型精度。
# ---------------------------------------------------------------------------

def _digest(*parts: Any) -> float:
    """把任意输入压成 [0,1) 的确定性数。

    用 sha256 而不是 ``hash()``：PYTHONHASHSEED 会把 str 的 hash 随机化，
    那样「同一输入两次跑出不同读数」—— 反证对照两臂就不可比了。
    """
    raw = json.dumps([str(p) for p in parts], ensure_ascii=False, sort_keys=True)
    n = int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12], 16)
    return (n % 10000) / 10000.0


@register("jev", "fake")
class FakeJev:
    """测试替身（Fowler: Fake —— 有可工作实现，走捷径）。

    ★ 它的返回值**没有语义**，只保证再跑一次是同一个数。
      任何拿它当精度读数的结论都是错的；它只用来验路由、计数、降级分支。
    ★ 它照着真件的形态实现（``choose`` 一次挑一个），所以「逐区域提问」那种
      退化写法在测试里也会立刻暴露。
    """

    name = "fake-jev"

    def __init__(self, yes_rate: float = 0.5, pick_rate: float = 0.5,
                 answers: Optional[dict] = None, **_kw: Any) -> None:
        self.yes_rate = float(yes_rate)
        self.pick_rate = float(pick_rate)
        self.answers = dict(answers or {})

    def _answer(self, key: str, rate: float) -> float:
        if key in self.answers:
            return float(self.answers[key])
        return round(0.5 + rate / 2, 4) if _digest(rate, key) < rate else round(0.5 - rate / 2, 4)

    def exists(self, frame_rgb: Any, query: str) -> Optional[float]:
        return self._answer(f"exists|{query}|{getattr(frame_rgb, 'shape', '')}", self.yes_rate)

    def choose(self, frame_rgb: Any, regions: Sequence[dict],
               query: str) -> Optional[dict]:
        if not regions:
            return None
        key = "choose|" + query + "|" + ",".join(str(list(r["bbox"])) for r in regions)
        scores = {str(r["idx"]): round(_digest(key, r["idx"]), 4) for r in regions}
        scores["none"] = round(_digest(key, "none"), 4)
        best = max(scores, key=lambda k: scores[k])
        if best == "none" or _digest(key, "none-wins") > self.pick_rate:
            return {"region_idx": None, "score": scores["none"], "scores": scores}
        return {"region_idx": int(best), "score": scores[best], "scores": scores}


@register("veason", "fake")
class FakeVeason:
    """测试替身：优先给「query 词命中的最大 region」，否则给最大 region。

    这样下游拿到的框**一定落在真实 region 表里**（有 region 表时），
    不会因为假件而产出一个不存在的框。
    """

    name = "fake-veason"

    def __init__(self, regions_of: Optional[Callable[[int], list[dict]]] = None,
                 match: Optional[Callable[[dict], bool]] = None, **_kw: Any) -> None:
        self.regions_of = regions_of or (lambda fid: [])
        self.match = match or (lambda r: False)

    def locate(self, frames_rgb: Sequence[Any], frame_ids: Sequence[int],
               query: str) -> dict[int, list[list[int]]]:
        out: dict[int, list[list[int]]] = {}
        for fid, img in zip(frame_ids, frames_rgb):
            regions = self.regions_of(fid)
            if regions:
                pool = [r for r in regions if self.match(r)] or regions
                pool = sorted(pool, key=lambda r: -float(r.get("area_ratio", 0.0)))
                out[fid] = [list(map(int, pool[0]["bbox"]))]
                continue
            # 没有 region 表可依（CLI 里就是这种情形）时给一个**确定性**的居中框，
            # 让下游拿到的是坐标而不是空表。★ 这个框是替身编的，不是测量：
            #   它的 src 永远是 fake-veason，读数里不许把它当模型结论。
            h, w = (img.shape[0], img.shape[1]) if hasattr(img, "shape") else (100, 100)
            out[fid] = [[w // 4, h // 4, (3 * w) // 4, (3 * h) // 4]]
        return out


@register("axis", "fake")
class FakeAxis:
    name = "fake-axis"

    def __init__(self, **_kw: Any) -> None:
        # ★ 必须吃掉 **_kw：get_backend 会把 store=... 一起传进来（axis 类件的
        #   构造需要一个索引句柄）。不吃的话报的是 "takes no arguments" ——
        #   一个与真实原因（构造签名不一致）脱节的错误。
        pass

    def supports(self, axis: str) -> bool:
        return True

    def measure(self, axis: str, frame_rgb: Any, regions: list[dict],
                masks: dict, query: str) -> Optional[list[dict]]:
        return [
            {"region_idx": int(r["idx"]), "axis": axis,
             "value": {f"{axis}_probe": round(_digest(axis, r["bbox"]), 4)},
             "src": "fake-axis"}
            for r in regions
        ]


# ---------------------------------------------------------------------------
# 不可用：显式的一个件，而不是 None。
# 让「没接线」在读数里看得见（jev_called=0 且 route=*_unavailable_kept），
# 而不是悄悄退化成一个看起来正常的数。
# ---------------------------------------------------------------------------

@register("jev", "unavailable")
def _jev_unavailable(**_kw: Any) -> Any:
    raise BackendUnavailable(
        "Jev backend",
        hint=f"Visual Jev needs {os.environ.get(MODELS_DIR_ENV, '<models>')}/visual-jev "
             "and a CUDA device; pass --jev-backend unavailable explicitly to force "
             "the high-recall degraded path, or --jev-backend fake to exercise routing offline.",
    )


@register("veason", "unavailable")
def _veason_unavailable(**_kw: Any) -> Any:
    raise BackendUnavailable(
        "Veason-R1 backend",
        hint=f"Veason-R1 (3B, 8.15 GB, qwen2_5_vl) weights + the qwen2_5_vl runtime are "
             f"needed; set {MODELS_DIR_ENV} to the weights root, or pass "
             "--veason-backend fake to exercise the deterministic orchestration offline.",
    )


@register("axis", "unavailable")
def _axis_unavailable(**_kw: Any) -> Any:
    raise BackendUnavailable(
        "axis backend",
        hint="no cached senses/relations for these frames and no model backend wired; "
             "either run `veason senses <index_dir>` / `veason relate <index_dir> <frame_id> ...` "
             "once (their output is reused, not recomputed), or pass --axis-backend fake.",
    )


# ---------------------------------------------------------------------------
# ★ 真件（同进程加载）。部署已经变成「skill 跑在组服务器的 8×RTX3090 上」，
#   所以这里没有 HTTP / 跨机那一层：模型和 skill 在同一台机器里。
#   ⚠ 这两个类在本机（Windows + 纯 CPU + 无权重）**未验** —— 见交付说明。
# ---------------------------------------------------------------------------

def _device(explicit: Optional[str] = None) -> str:
    dev = explicit or os.environ.get(DEVICE_ENV) or ""
    if dev:
        return dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - 只有没装 torch 时才走到
        return "cpu"


#: Visual Jev 的候选上限（源码：超过 K_MAX 直接 raise，不截断）。
JEV_K_MAX = 16


@register("jev", "local")
class LocalJev:
    """Visual Jev = ``Qwen/Qwen3-VL-4B-Instruct`` + LoRA ``guanxuyu/visual-jev-4b-answer-sft``。

    用法照 `资料/Jev用法-20261013.md` §2「方式三」：
    ``VDM(...)`` → 挂 PeftModel → ``prepare_group`` → ``run_prefix_share_batch``
    → 从 ``out["lm_option_logits"]`` 读字母上的概率。

    三条硬约束（踩了会报错或静默出错）：

    1. **候选 2~16**，超过直接 raise（源码拒绝截断）；
    2. **必须挂适配器**，否则是原装 Qwen3-VL，不做 forced-choice；
    3. ★ **一次问一批题** —— 它的整套设计围绕共享前缀；但**跨帧不能共享**
       （图不同），所以这里的粒度是「一帧一题」。
    """

    name = "visual-jev-4b-answer-sft"

    def __init__(self, base_model: str = "Qwen/Qwen3-VL-4B-Instruct",
                 adapter: str = "guanxuyu/visual-jev-4b-answer-sft",
                 device: Optional[str] = None, max_pixels: int = 200704,
                 weights_dir: Optional[str] = None, **_kw: Any) -> None:
        hints = (f"needs `torch` + `peft` + the Visual-Jev `vdm` package on sys.path, "
                 f"the adapter {adapter}, and a CUDA/MPS device. Set "
                 f"{MODELS_DIR_ENV} to a local copy of the weights if the hub is unreachable.")
        try:
            import torch  # noqa: F401
            from peft import PeftModel  # noqa: F401
            from vdm.models.vdm_model import VDM  # noqa: F401
        except ImportError as exc:
            raise BackendUnavailable("Visual Jev", f"{exc}; {hints}") from exc
        self._hints = hints
        self.device = _device(device)
        self.base_model = base_model
        self.adapter = adapter
        self.max_pixels = max_pixels
        self._model = None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        import torch
        from peft import PeftModel
        from vdm.models.vdm_model import VDM

        model = VDM(self.base_model, device=self.device, dtype=torch.bfloat16, with_heads=False)
        # 图像 token 预算：调小省显存（笔记里的默认值）。
        model.processor.image_processor.max_pixels = self.max_pixels
        model.backbone = PeftModel.from_pretrained(model.backbone, self.adapter).eval()
        self._model = model
        return model

    def _ask(self, frame_rgb: Any, instruction: str,
             candidates: Sequence[str]) -> Optional[dict[str, float]]:
        """一次前向，读选项字母上的概率。"""
        opts = [str(c) for c in candidates]
        if not 2 <= len(opts) <= JEV_K_MAX:
            raise VrsError(
                f"Visual Jev takes 2..{JEV_K_MAX} options, got {len(opts)}",
                hint="it rejects rather than truncates; narrow the candidate pool upstream.",
                code=2,
            )
        import numpy as np
        from PIL import Image

        model = self._load()
        image = Image.fromarray(np.ascontiguousarray(np.asarray(frame_rgb)[:, :, ::-1]))  # BGR -> RGB
        questions = [{"qtype": "choice", "instruction": instruction, "candidates": opts}]
        group = model.prepare_group(image, questions, shared_context="")
        import torch

        with torch.inference_mode():
            out = model.run_prefix_share_batch(group)
        logits = out["lm_option_logits"][0, :group.n_options[0]]
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        return {o: float(p) for o, p in zip(opts, probs)}

    def exists(self, frame_rgb: Any, query: str) -> Optional[float]:
        scores = self._ask(frame_rgb, f"Is there {query} in this image?", ["present", "not present"])
        return None if scores is None else scores.get("present")

    def choose(self, frame_rgb: Any, regions: Sequence[dict],
               query: str) -> Optional[dict]:
        """★ N 选一：把候选框编号画在图上，让它在编号之间挑一个。

        编号是**外部画上去的**（笔记 §5 测试 B：「编号是外部画上去的，正是候选池
        的做法」）—— 不画编号的话模型无从指认「第几个框」，只能回答文字描述，
        那又退化成自由生成。
        """
        regions = list(regions)[:JEV_K_MAX - 1]
        if not regions:
            return None
        marked = _mark_regions(frame_rgb, regions)
        # 选项 = 编号 + 显式的「都不是」。★ 前提为假那一档不用额外造：
        #   把选项写进候选表它自己会选（笔记 §5）。
        candidates = [str(int(r["idx"])) for r in regions] + ["none of them"]
        instruction = (f"Which numbered region is {query}? "
                       "If none of them is, choose 'none of them'.")
        scores = self._ask(marked, instruction, candidates)
        if scores is None:
            return None
        best = max(scores, key=lambda k: scores[k])
        if best == "none of them":
            return {"region_idx": None, "score": scores[best], "scores": scores}
        return {"region_idx": int(best), "score": scores[best], "scores": scores}


def _mark_regions(frame_rgb: Any, regions: Sequence[dict]) -> Any:
    """在每个 region 的框上画编号标记（候选池的做法）。"""
    import cv2

    canvas = frame_rgb.copy()
    for r in regions:
        x1, y1, x2, y2 = (int(v) for v in r["bbox"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 0), 3, cv2.LINE_AA)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
        label = str(int(r["idx"]))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        ty = max(th + 6, y1)
        cv2.rectangle(canvas, (x1, ty - th - 8), (x1 + tw + 10, ty + 2), (0, 0, 0), -1)
        cv2.putText(canvas, label, (x1 + 5, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


@register("veason", "local")
class LocalVeason:
    """Veason-R1（3B / 8.15 GB / qwen2_5_vl），**同进程**定位。

    ★ 官方推理口径（规格 §3，逐行抄自 `batch_eval_keyframe_sft_dataset_.py`）：

    * 一次吃 **8 帧**；每张图前加 ``(帧号 s)`` 前缀 —— 是**帧号**不是秒；
    * 图缩到 560×560；框在 560 空间里，乘 ``width/560, height/560`` 反算回原图；
    * ``do_sample=False``（贪心，确定性）、``max_new_tokens=1024``；
    * 输出 ``<think>…</think><answer>{"keyframe_timestamp":…,"bbox_2d_list":[…]}</answer>``；
    * ★ ``keyframe_timestamp`` 必须落在采样帧集合里 —— 官方 reward 里
      ``if keyframe_id not in frame_ids: return 0``。静默 fallback 到 0 会把一个
      格式错误伪装成「选了第 0 帧」的合法答案，所以这里**报错**。

    ⚠ 规格写 ``system_prompt = None``，官方仓库传 ``"You are a helpful assistant."``。
      两版不等价（传 None 会得到字面量 "None"）。默认服从规格，参数化留给接线时定。
    """

    name = "veason-r1-3b"
    FRAME_RESIZE = 560
    NUM_FRAMES = 8

    def __init__(self, model_dir: Optional[str] = None, device: Optional[str] = None,
                 system_prompt: Optional[str] = None, **_kw: Any) -> None:
        from pathlib import Path

        self.model_dir = Path(model_dir) if model_dir else Path(models_dir()) / "Veason-R1"
        self.device = _device(device)
        self.system_prompt = system_prompt
        self._model = None
        self._processor = None
        if not self.model_dir.exists():
            raise BackendUnavailable(
                "Veason-R1 weights",
                hint=f"expected them under {self.model_dir}; set {MODELS_DIR_ENV} to the "
                     "weights root (the 3B release is 8.15 GB), or pass --veason-backend fake.",
            )

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._processor
        try:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as exc:
            raise BackendUnavailable(
                "the qwen2_5_vl runtime",
                hint=f"{exc}; install transformers with Qwen2.5-VL support on the GPU box.",
            ) from exc
        self._processor = AutoProcessor.from_pretrained(str(self.model_dir))
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        ).to(self.device).eval()
        return self._model, self._processor

    def locate(self, frames_rgb: Sequence[Any], frame_ids: Sequence[int],
               query: str) -> dict[int, list[list[int]]]:
        import numpy as np

        model, processor = self._load()
        sizes = {int(f): (int(img.shape[1]), int(img.shape[0]))
                 for f, img in zip(frame_ids, frames_rgb)}
        content = []
        for fid, img in zip(frame_ids, frames_rgb):
            content.append({"type": "text", "text": f"({int(fid)}s)"})
            content.append({"type": "image", "image": _resized_560(img, self.FRAME_RESIZE)})
        content.append({"type": "text", "text": query})
        # ★ 没给 system_prompt 时用官方口径 —— 否则模型退化成视频描述，不产出 <answer>。
        sys_prompt = self.system_prompt or VEASON_TAIL.format(
            num=len(list(frame_ids)), q=query)
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": content}]

        import torch

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[c["image"] for c in content
                                                if c["type"] == "image"], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, do_sample=False, max_new_tokens=1024)
        generated = out[:, inputs["input_ids"].shape[1]:]
        raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return parse_veason_answer(raw, list(frame_ids), sizes)


# ★ Veason-R1 官方任务口径（逐字取自 ReVOS 官方推理脚本的 TAIL）：
#   不传 prompt 时模型会退化成「视频描述」，不会输出 <answer> JSON。
VEASON_TAIL = (
    " \nYou are given {num} frames uniformly sampled from a 1 FPS video. "
    "Each image is preceded by its timestamp.\n"
    "Given the following expression: '{q}', identify the timestamp where the target "
    "is best represented. Then, locate the target in that frame by predicting the "
    "bounding box(es).\n"
    "First, output the thinking process in <think> </think> tags, and then output the "
    "result in the following format within <answer> </answer> tags."
)


def _resized_560(img: Any, size: int) -> Any:
    import cv2
    import numpy as np

    return np.ascontiguousarray(
        cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)[:, :, ::-1])  # BGR->RGB


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)


def parse_veason_answer(raw: str, frame_ids: Sequence[int],
                        sizes: dict[int, tuple[int, int]]) -> dict[int, list[list[int]]]:
    """把模型原始输出解析回原图坐标。

    单独拎出来是为了**能被单测钉住**：这条解析链上有三个静默出错点
    （没有 <answer>、key 不在采样帧里、框在 560 空间），本机没有权重也能测。
    """
    out: dict[int, list[list[int]]] = {int(f): [] for f in frame_ids}
    m = _ANSWER.search(raw or "")
    if not m:
        raise VrsError(
            "Veason-R1 returned no <answer> block",
            hint="the raw output is in the traceback path; check the chat template and the "
                 "sampled-frame prefix `(Ns)` before trusting any box.",
        )
    try:
        payload = json.loads(m.group(1).strip())
    except json.JSONDecodeError as exc:
        raise VrsError(
            f"Veason-R1 <answer> is not JSON ({exc}): {m.group(1)[:200]!r}",
            hint="format compliance dropped; the official reward would score this 0.",
        ) from exc
    key = payload.get("keyframe_timestamp")
    if key is None:
        return out
    key = int(key)
    if key not in out:
        # ★ 不回退到 0。官方 reward 里 error 与「选了第 0 帧」都是 0 分，
        #   但一个是格式错、一个是合法答案 —— 混起来就再也查不出来了。
        raise VrsError(
            f"Veason-R1 answered keyframe_timestamp={key}, which is not one of the sampled "
            f"frames {sorted(out)}",
            hint="the answer and the ground truth are in different id spaces (frame index vs "
                 "sample index); re-read the caller's frame sampling before trusting any box.",
        )
    w, h = sizes.get(key, (560, 560))
    xf, yf = w / 560.0, h / 560.0
    for b in payload.get("bbox_2d_list") or []:
        x1, y1, x2, y2 = (float(v) for v in b[:4])
        out[key].append([int(round(x1 * xf)), int(round(y1 * yf)),
                         int(round(x2 * xf)), int(round(y2 * yf))])
    return out


# ---------------------------------------------------------------------------
# ③ 的本地件：先读【已经算过】的那次测量（veason senses / veason relate 的产物）。
# 这不省事 —— 它保证 src 指向真正产出该值的那个模型：
# 值来自哪个字段，src 就写哪个模型（见 evidence.py 的 FIELD_SRC）。
# ---------------------------------------------------------------------------

@register("axis", "senses-cache")
class SensesCacheAxis:
    name = "senses-cache"

    def __init__(self, store: Any = None, **_kw: Any) -> None:
        self.store = store

    def supports(self, axis: str) -> bool:
        return axis in ("attribute", "relation", "action", "occlusion", "text")

    def measure(self, axis: str, frame_rgb: Any, regions: list[dict],
                masks: dict, query: str) -> Optional[list[dict]]:
        return None  # 真正的取数在 evidence.py：它才知道索引在哪、字段叫什么
