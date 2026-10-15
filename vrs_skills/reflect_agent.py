"""reflect_agent.py —— 让 MLLM 看着【图 + 元数据 + 动作菜单】自己选下一步动作。

## 这一环在干什么

reflect 把三种动作（放大 / 换帧 / 换轴）**实现**好了，但「选哪个」以前是代码写死的。
本模块把它交给 MLLM：模型拿到

```
① 当前帧的图             —— 它得真看得见
② 元数据（measurements）  —— 面积 / 框内填充率 / 掩码外接框 / 已测轴 / 已看帧
③ 动作菜单               —— 每种动作产出什么新测量、代价多大
```

然后回答一句话：**下一步做哪个，为什么**。

## 规则（不是提示词技巧，是硬约束）

* 只能从菜单里挑（别的答案一律拒收，退回确定性兜底）。
* 选的动作必须**产得出新测量**——铁律：写不出 measurement 的不算新信息。
* 模型答不出 / 格式不对 / 超时 ⇒ **退回确定性兜底**，不把链子跑挂。
  （「模型坏产物禁抛异常 ⇒ 兑具名失败」）

## 为什么不用它来替代确定性 plan

那条兜底有实测依据（放大是主力、换帧权重低），是**资产**。
留着它，两条路才能**对照**——到底 MLLM 选得比硬编码好还是差，是个可测的问题。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from .errors import VrsError

#: ★ 白名单只有一份真相：reflect 那边定义，这边引用。
#: 上一版这里自己写死了三个动作、而提示词允许 ``stop`` ⇒ 模型按提示词答 stop
#: 被这里判成坏产物，连累到 reflect 的放行逻辑根本没机会执行（实测踩过）。
from .reflect import AGENT_KINDS, AGENT_STOP  # noqa: E402

ALLOWED_KINDS = tuple(AGENT_KINDS) + (AGENT_STOP,)

#: 默认口径：硅基流动的 Qwen3.8-27B（实测认图 1.9s，比本地 Qwen3-VL-8B 快 12 倍）。
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"

SYSTEM_PROMPT = """你是一个视频目标定位 agent 的决策环节。

上一环已经给你了：当前帧的画面、若干候选框、以及每个候选框的测量元数据。
现在要你决定**下一步做哪个动作**，才能拿到**新的测量**（把同一个信息重排一遍不算）。

可选动作只有三个：
{menu}

规则（必须遵守）：
1. 只能从上面三个里选，不能自创。
2. 你选的动作必须能产出**新的测量** —— 说不出「会多拿到什么测量」的动作无效。
3. 如果当前信息已经足够判断，或三个动作都拿不到新测量，就回答 "stop"。
4. 不要复述数据，直接给结论。

只输出一行 JSON：
{{"kind": "zoom|new_frame|new_axis|stop", "reason": "一句话说清为什么，以及能多拿到什么测量"}}"""


class MllmChooser:
    """可调用对象：``chooser(state, measurements, menu) -> (kind, reason)``。

    给 ``run_reflect(..., chooser=...)`` 用。任何失败都退回确定性兜底
    （``fallback`` 回调，由调用方给；没给就返回 ``("", "mllm_failed")`` 让它走兜底）。
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 api_key: Optional[str] = None, timeout: float = 180.0,
                 retries: int = 2,
                 fallback: Optional[Any] = None, frame_provider: Optional[Any] = None,
                 verbose: bool = False) -> None:
        self.model = model
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or os.environ.get("VRS_MLLM_KEY") or os.environ.get(
            "SILICONFLOW_API_KEY")
        self.timeout = float(timeout)
        #: ★ 超时/网络抖动**不算模型拒答**：重试 retries 次再退回兜底。
        #: 模型好好答了一次「不在菜单里」才算真拒答，一次都不重试。
        self.retries = max(0, int(retries))
        self.fallback = fallback
        self.frame_provider = frame_provider
        self.verbose = verbose
        self.calls = 0
        self.last_reason = ""

    # ── 给模型看的三样东西 ────────────────────────────────────────
    @staticmethod
    def _menu_text(menu: list[dict]) -> str:
        return "\n".join(
            f"- {a['kind']}：{a['what']}\n"
            f"    产出：{a['produces']}\n"
            f"    代价：{a['cost']}\n"
            f"    何时有用：{a['when_useful']}"
            for a in menu
        )

    @staticmethod
    def _state_text(state: dict, m: dict) -> str:
        """元数据摘要。★ 只放测量出来的东西，不放任何模型打分。"""
        seen = state.get("seen_frames") or []
        cands = state.get("candidates") or []
        rows = []
        for c in cands[:8]:
            rows.append({
                "frame": c.get("frame_id"),
                "rank": c.get("rank"),
                "route": c.get("route"),
                "area_ratio": c.get("area_ratio"),
                "box": c.get("bbox"),
                "mask": c.get("mask"),          # ← ③ 产出的掩码元数据（若有）
            })
        return json.dumps({
            "query": state.get("query"),
            "看过的帧": seen[:16],
            "候选": rows,
            "测量": {
                "low_visibility": m.get("low_visibility"),
                "ambiguity": m.get("ambiguity"),
                "distractor": m.get("distractor"),
                "temporal": m.get("temporal"),
                "axes_already_measured": m.get("axes_already_measured"),
                "candidate_count": m.get("candidate_count"),
            },
        }, ensure_ascii=False, indent=1)

    # ── 调用模型 ─────────────────────────────────────────────────
    def _ask(self, state: dict, m: dict, menu: list[dict],
             image_path: Optional[str]) -> Optional[tuple[str, str]]:
        try:
            import requests
        except ImportError:  # pragma: no cover
            return None
        if not self.api_key:
            self.last_reason = "no_api_key"
            return None

        content: list[dict] = [{
            "type": "text",
            "text": (f"【当前状态与测量元数据】\n{self._state_text(state, m)}\n\n"
                     f"请决定下一步动作。"),
        }]
        if image_path and os.path.exists(image_path):
            import base64

            with open(image_path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            content.insert(0, {"type": "image_url",
                               "image_url": {"url": f"data:image/png;base64,{b64}"}})

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": SYSTEM_PROMPT.format(menu=self._menu_text(menu))},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": 300,
        }
        txt = None
        last = ""
        for attempt in range(self.retries + 1):
            try:
                r = requests.post(f"{self.base_url}/chat/completions",
                                  headers={"Authorization": f"Bearer {self.api_key}",
                                           "Content-Type": "application/json"},
                                  json=payload, timeout=self.timeout)
                self.calls += 1
                if r.status_code != 200:
                    last = f"http_{r.status_code}"
                    # 4xx 是请求本身的问题，重试没意义
                    if 400 <= r.status_code < 500:
                        break
                    continue
                txt = r.json()["choices"][0]["message"]["content"]
                break
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}"
                if self.verbose:
                    print(f"[chooser] 第 {attempt + 1} 次失败：{last}，"
                          f"{'重试' if attempt < self.retries else '不再重试'}")
                continue
        if txt is None:
            self.last_reason = last or "unknown"
            return None

        if self.verbose:
            print(f"[chooser] 原始回答：{txt[:300]}")
        match = re.search(r"\{.*?\}", txt, re.S)
        if not match:
            self.last_reason = "no_json"
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            self.last_reason = "bad_json"
            return None
        kind = str(obj.get("kind") or "").strip()
        reason = str(obj.get("reason") or "").strip()[:300]
        if kind not in ALLOWED_KINDS:
            self.last_reason = f"kind_not_allowed:{kind}"
            return None
        self.last_reason = "ok"
        return (kind, reason or "mllm_choice")

    # ── 主入口 ───────────────────────────────────────────────────
    def __call__(self, state: dict, m: dict, menu: list[dict]) -> tuple[str, str]:
        img = None
        if self.frame_provider is not None:
            try:
                img = self.frame_provider(state, m)
            except Exception:  # noqa: BLE001
                img = None
        picked = self._ask(state, m, menu, img)
        if picked is None:
            # ★ 模型坏产物 ⇒ 退回确定性兜底，不抛异常、不把链子跑挂。
            if self.fallback is not None:
                return self.fallback(state, m, menu)
            return ("", f"mllm_failed:{self.last_reason}")
        return picked
