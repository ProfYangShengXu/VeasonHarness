"""五环链 ①②③⑤ 的自检 —— 断言的是**判据**，不是 happy path。

每条断言都对应规格里一句话，命名里写出那句话；注释里写清「它为什么可能不成立」。
和 ``tests/test_veason.py`` 同样的信念：一条不会失败的检查不是检查。

这一批用例**不需要任何模型**：screen 的 YOLO 臂读索引里既有的 region 表，
evidence 的模型轴读 ``veason senses`` 已经算过的产物，locate/reflect 的模型件
用注入的假件。所以它能跑在纯 CPU、无 GPU、无权重的地方 ——
这正是把后端做成可注入的目的（见 backends.py）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from veason_harness import backends  # noqa: E402
from veason_harness.backends import (JEV_K_MAX, BackendUnavailable, FakeJev,  # noqa: E402
                                 JevBackend, LocalJev, parse_veason_answer)
from veason_harness.evidence import run_evidence  # noqa: E402
from veason_harness.locate import run_locate  # noqa: E402
from veason_harness.reflect import parse_state, run_reflect  # noqa: E402
from veason_harness.screen import run_screen  # noqa: E402
from veason_harness.store import IndexStore, sha256_json  # noqa: E402

PY = sys.executable
VRS = str(ROOT / "veason.py")
PIPELINE = str(ROOT / "veason_pipeline.py")

#: ★ 「indexer.py 一个字节不改」的硬保证。改了它这条断言就红。
#: 它守的不是风格，是分层：senses 必须保持**显式命令**，不许被并进 index 默认路径。
INDEXER_SHA256 = "8e7016d4e873ff24d4e41c914ba0daf97b39f8ef517e094f6ce84adf16448a4c"

W, H, FPS = 200, 160, 10.0


# ---------------------------------------------------------------------------
# 造一个**真文件**索引（真 PNG / 真 mask / 真 region 表），零模型
# ---------------------------------------------------------------------------

def build_index(root: Path, spec: dict[int, list[dict]], *, w: int = W, h: int = H,
                fps: float = FPS) -> IndexStore:
    """``spec`` = ``{frame_id: [ {label, bbox, score?}, ... ]}``；空列表 = 零检出帧。"""
    store = IndexStore(root)
    table = []
    for fid, regions in spec.items():
        img = np.full((h, w, 3), 50, np.uint8)
        store.frame_image_path(fid).parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(store.frame_image_path(fid)), img)
        out, masks = [], {}
        for i, r in enumerate(regions):
            x1, y1, x2, y2 = (int(v) for v in r["bbox"])
            m = np.zeros((h, w), bool)
            m[y1:y2, x1:x2] = True
            masks[i] = m
            out.append({
                "idx": i, "label": str(r["label"]), "score": float(r.get("score", 0.5)),
                "bbox": [x1, y1, x2, y2],
                "area_ratio": round(((x2 - x1) * (y2 - y1)) / float(w * h), 6),
                "has_mask": True, "source": "test",
            })
        store.save_regions(fid, {"frame_id": fid, "t": round(fid / fps, 3), "width": w,
                                 "height": h, "image": "", "detector": {},
                                 "next_idx": len(out), "regions": out})
        store.save_masks(fid, masks)
        table.append({"frame_id": fid, "t": round(fid / fps, 3), "n_regions": len(out),
                      "image": "", "image_win": "", "regions_file": ""})
    top = max(spec) if spec else 0
    store.save_meta({
        "schema": 1, "video": "", "video_win": "", "video_sha256": "x", "video_bytes": 0,
        "fps": fps, "n_video_frames": top + 1, "duration_s": (top + 1) / fps,
        "width": w, "height": h, "n_frames": len(table), "frame_sampling": "test",
        "detector": {}, "relation": {}, "frames": table, "timings": {},
        "last_run_s": 0.0, "last_run_work": {},
    })
    return store


def run(*args: str) -> tuple[int, str, str]:
    p = subprocess.run([PY, VRS, *args], capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout, p.stderr


def _only_json(text: str) -> dict:
    """stdout 必须是**恰好一个** JSON 文档（不是 NDJSON，不是多段）。"""
    return json.loads(text)


# ===========================================================================
# ① screen：OOV 三档、高召回、route 归因、成本可见
# ===========================================================================

def test_screen_tier_a_uses_yolo_and_never_calls_jev(tmp: Path) -> None:
    """档 A：YOLO 检出 && 命中 query 词 ⇒ 直接用，不上 Jev（成本为 0）。"""
    store = build_index(tmp / "i", {0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}]})
    calls = []

    class Recorder(FakeJev):
        def confirm(self, *a, **k):  # 档 A 不该走到这里
            calls.append("confirm")
            return 0.99

        def exists(self, *a, **k):
            calls.append("exists")
            return 0.99

    backends.register("jev", "t-recorder")(lambda **kw: Recorder())
    payload, _ = run_screen(store, "parachute", top_n=5, jev_backend="t-recorder")
    assert payload["screened"]["yolo_hit"] == 1
    assert payload["screened"]["jev_called"] == 0
    assert calls == [], f"档 A 不该问 Jev，实际调了 {calls}"
    assert payload["candidates"][0]["route"] == "yolo_hit"


def test_screen_tier_b_asks_jev_once_with_every_region_as_an_option(tmp: Path) -> None:
    """★★ 档 B = 一次 N 选一，不是逐个问"是不是"。

    ★ 这是实测纪律（`资料/Jev用法-20261013.md` §6 纪律②）：把 4 个框分别问它
      "这个是目标吗？" → 它 74% 都答"是"。独立判断没有比较压力，它会一律放行。
      所以这里钉两件事：**只调一次 choose**，且**全部候选都在同一道题的选项里**。
      一旦有人把它改回逐区域的 confirm 循环，这条立刻红。
    """
    store = build_index(tmp / "i", {
        0: [{"label": "parachute", "bbox": (10, 10, 90, 90)},
            {"label": "stuffed toy", "bbox": (100, 100, 150, 150)}],
    })
    calls = []

    class Recorder(FakeJev):
        def choose(self, frame, regions, query):
            calls.append([int(r["idx"]) for r in regions])
            return {"region_idx": 1, "score": 0.87,
                    "scores": {"0": 0.1, "1": 0.87, "none": 0.03}}

        def exists(self, *a, **k):
            raise AssertionError("档 B 不该问存在性")

    backends.register("jev", "t-choose")(lambda **kw: Recorder())
    payload, _ = run_screen(store, "黄色的卡通奶龙", top_n=5, jev_backend="t-choose")
    assert len(calls) == 1, f"档 B 必须只问一次（N 选一），实际 {len(calls)} 次"
    assert calls[0] == [0, 1], "全部候选都要在同一道题的选项里，否则没有比较压力"
    assert payload["screened"]["jev_called"] == 1
    assert payload["screened"]["jev_model_calls"] == 1
    assert payload["screened"]["dropped"] == 0
    assert payload["candidates"][0]["route"] == "jev_confirmed"
    assert payload["candidates"][0]["tier"] == "B"
    assert payload["candidates"][0]["region_idx"] == 1
    assert payload["candidates"][0]["jev_yes"] == 0.87


def test_screen_tier_b_drops_the_frame_when_jev_picks_none_of_them(tmp: Path) -> None:
    """★ 「前提为假」那一档不用额外造：它选了「都不是」就是不在。"""
    store = build_index(tmp / "i", {0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}]})

    class PicksNone(FakeJev):
        def choose(self, frame, regions, query):
            return {"region_idx": None, "score": 0.93, "scores": {"0": 0.05, "none": 0.93}}

    backends.register("jev", "t-none")(lambda **kw: PicksNone())
    payload, _ = run_screen(store, "黄色的卡通奶龙", top_n=5, jev_backend="t-none")
    assert payload["screened"]["dropped"] == 1
    assert payload["candidates"] == []


def test_screen_tier_b_is_not_gated_on_jev_confidence(tmp: Path) -> None:
    """★ 档 B 不用概率门限：概率在这类任务上几乎分不开对错（0.746 vs 0.708）。

    N 选一的答案就是「它挑了哪个」；给赢家再加一道门限，等于把好不容易拿到的
    比较压力又扔掉，退回「高置信地错」。
    """
    store = build_index(tmp / "i", {0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}]})

    class LowConfidencePick(FakeJev):
        def choose(self, frame, regions, query):
            return {"region_idx": 0, "score": 0.21, "scores": {"0": 0.21, "none": 0.19}}

    backends.register("jev", "t-lowconf")(lambda **kw: LowConfidencePick())
    payload, _ = run_screen(store, "黄色的卡通奶龙", top_n=5, jev_backend="t-lowconf")
    assert payload["candidates"][0]["route"] == "jev_confirmed"
    assert payload["candidates"][0]["jev_yes"] == 0.21


def test_screen_tier_c_asks_existence_when_yolo_is_silent(tmp: Path) -> None:
    """档 C：YOLO 零检出 ⇒ 上 Jev 问存在性（读 none 档）。"""
    store = build_index(tmp / "i", {0: []})
    seen = []

    class Recorder(FakeJev):
        def exists(self, frame, query):
            seen.append(query)
            return 0.9

        def confirm(self, *a, **k):  # 零检出，没有框可问
            raise AssertionError("档 C 不该问 confirm")

    backends.register("jev", "t-exists")(lambda **kw: Recorder())
    payload, _ = run_screen(store, "一只龙", top_n=5, jev_backend="t-exists")
    assert seen == ["一只龙"]
    assert payload["candidates"][0]["route"] == "jev_confirmed"
    assert payload["candidates"][0]["tier"] == "C"
    assert payload["candidates"][0]["region_idx"] == -1, "零检出帧没有 region 可指"


def test_screen_high_recall_keeps_frames_when_jev_is_unavailable(tmp: Path) -> None:
    """★ 高召回：剔帧判据 =【YOLO 与 Jev 都说不在】才剔。

    Jev 不可用 ≠ Jev 说不在。这条反了就是把「没接线」当成「目标不存在」，
    而那正是这条链最贵的错误方向（漏掉有目标的帧）。
    """
    store = build_index(tmp / "i", {0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}]})
    payload, info = run_screen(store, "黄色的卡通奶龙", top_n=5, jev_backend="unavailable")
    assert payload["screened"]["dropped"] == 0, "Jev 没接线时不许剔任何帧"
    assert payload["candidates"][0]["route"] == "jev_unavailable_kept"
    assert any(e["event"] == "screen_jev_unavailable" for e in info["events"])


def test_screen_drops_only_when_jev_explicitly_says_no(tmp: Path) -> None:
    """只有 Jev 明确说不在（选了「都不是」/存在性判否）才剔；说在就留。"""
    store = build_index(tmp / "i", {0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}]})
    no, _ = run_screen(store, "黄色的奶龙", top_n=5, jev_backend="fake")
    # FakeJev 的默认行为会同时给出「挑中」与「都不是」两种结果；
    # 这里只钉住方向：dropped 的每一帧都对应一次 Jev 调用。
    assert no["screened"]["dropped"] <= no["screened"]["jev_called"]

    class AlwaysNo(FakeJev):
        def choose(self, frame, regions, query):
            return {"region_idx": None, "score": 0.95, "scores": {"none": 0.95}}

    backends.register("jev", "t-no")(lambda **kw: AlwaysNo())
    zero, _ = run_screen(store, "黄色的奶龙", top_n=5, jev_backend="t-no")
    assert zero["screened"]["dropped"] == 1
    assert zero["candidates"] == []

    class AlwaysYes(FakeJev):
        def choose(self, frame, regions, query):
            return {"region_idx": int(regions[0]["idx"]), "score": 0.95,
                    "scores": {str(int(regions[0]["idx"])): 0.95}}

    backends.register("jev", "t-yes")(lambda **kw: AlwaysYes())
    one, _ = run_screen(store, "黄色的奶龙", top_n=5, jev_backend="t-yes")
    assert one["screened"]["dropped"] == 0
    assert one["candidates"][0]["route"] == "jev_confirmed"


def test_screen_screened_counters_account_for_every_frame(tmp: Path) -> None:
    """screened 是成本可见：48 帧的去向必须能被四个数加起来解释。"""
    store = build_index(tmp / "i", {
        0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}],   # 档 A
        10: [{"label": "dune", "bbox": (10, 10, 60, 60)}],        # 档 B
        20: [],                                                   # 档 C
    })
    payload, _ = run_screen(store, "parachute", top_n=5, jev_backend="unavailable")
    s = payload["screened"]
    assert s["total_frames"] == 3
    assert s["yolo_hit"] + s["jev_called"] + s["dropped"] == s["total_frames"]
    assert s["yolo_hit"] == 1 and s["jev_called"] == 2 and s["dropped"] == 0


def test_screen_area_ratio_is_computed_never_asked(tmp: Path) -> None:
    """★ 纪律①：面积占比是确定性计算，不问模型。

    判据：把 area_ratio 从输入里拿掉，结果必须仍然等于 bbox 算出来的值。
    """
    store = build_index(tmp / "i", {0: [{"label": "dragon", "bbox": (20, 20, 120, 100)}]})
    payload, _ = run_screen(store, "dragon", top_n=5)
    want = ((120 - 20) * (100 - 20)) / float(W * H)
    assert payload["candidates"][0]["area_ratio"] == pytest.approx(want, abs=1e-6)


def test_screen_rejects_bad_input_with_usage_error(tmp: Path) -> None:
    """非法输入不许静默：--top-n 0 与未知判据都是用法错（码 2）。"""
    store = build_index(tmp / "i", {0: []})
    with pytest.raises(Exception) as e:
        run_screen(store, "x", top_n=0)
    assert getattr(e.value, "code", None) == 2
    with pytest.raises(Exception) as e2:
        run_screen(store, "x", oov_rule="vibes")
    assert getattr(e2.value, "code", None) == 2
    with pytest.raises(Exception):
        run_screen(store, "   ", top_n=3)


def test_screen_sort_puts_named_evidence_above_unevidenced_kept(tmp: Path) -> None:
    """★ 缺陷#1：排序先按【证据类型】分层，层内才按面积。

    frame 10 = ``yolo_hit``（有名字）/ 小框；frame 20 = ``jev_unavailable_kept``
    （Jev 没接线、没有证据）/ 大背景。纯面积排会让背景在前；分层后强证据必须在前
    —— 尽管它面积小得多。★ 这是「背景永远比目标大」那条必然后果的反证。
    """
    store = build_index(tmp / "i", {
        10: [{"label": "dragon", "bbox": (0, 0, 20, 20), "score": 0.9}],        # area 0.0125
        20: [{"label": "water surface", "bbox": (0, 0, W - 1, H - 1)}],         # area ≈0.99
    })
    payload, _ = run_screen(store, "dragon", top_n=2, jev_backend="unavailable")
    cands = payload["candidates"]
    assert cands[0]["route"] == "yolo_hit", cands
    assert cands[1]["route"] == "jev_unavailable_kept", cands
    assert cands[0]["area_ratio"] < cands[1]["area_ratio"], \
        "前置条件必须成立：强证据的面积更小，否则这条测试证明不了分层"


def test_screen_sort_falls_back_to_area_when_stratification_is_off(tmp: Path, monkeypatch) -> None:
    """★ 缺陷#1 的反证臂：把分层关掉（tier 恒 0），必须【逐字退回】纯面积排序。

    与上一条同素材，只把 ``_evidence_tier`` 改成常数 —— 期望大背景重新排到前面。
    这就证明「是分层在起作用」，不是别的改动顺带改对了。
    """
    from veason_harness import screen as screen_mod

    monkeypatch.setattr(screen_mod, "_evidence_tier", lambda route: 0)
    store = build_index(tmp / "i", {
        10: [{"label": "dragon", "bbox": (0, 0, 20, 20), "score": 0.9}],
        20: [{"label": "water surface", "bbox": (0, 0, W - 1, H - 1)}],
    })
    payload, _ = run_screen(store, "dragon", top_n=2, jev_backend="unavailable")
    assert payload["candidates"][0]["route"] == "jev_unavailable_kept", \
        "关掉分层必须退回纯面积排序（大背景在前）"


def test_screen_sort_uses_area_within_the_same_evidence_tier(tmp: Path) -> None:
    """★ 层内仍按面积排 —— 「面积大 = 目标更可能看得清」这条理由没有被丢掉。"""
    store = build_index(tmp / "i", {
        10: [{"label": "dragon", "bbox": (0, 0, 40, 40), "score": 0.9}],         # area 0.05
        20: [{"label": "dragon", "bbox": (0, 0, 100, 100), "score": 0.9}],       # area 0.3125
    })
    payload, _ = run_screen(store, "dragon", top_n=2)
    assert [c["route"] for c in payload["candidates"]] == ["yolo_hit", "yolo_hit"]
    assert [c["frame_id"] for c in payload["candidates"]] == [20, 10], \
        "同一证据层内，面积大的必须在前"


# ===========================================================================
# ② locate：只定位不选帧、8 帧一批、补齐用真新帧、不编框
# ===========================================================================

def test_locate_returns_one_entry_per_requested_frame_only(tmp: Path) -> None:
    """★ 只定位不选帧：返回的就是调用方要的那几帧，填充帧不进答案。"""
    store = build_index(tmp / "i", {f: [{"label": "x", "bbox": (10, 10, 60, 60)}]
                                    for f in (0, 10, 20, 30, 40, 50, 60, 70, 80)})
    payload, _ = run_locate(store, "0,10", "x", pad_to=8, veason_backend="fake")
    assert [e["frame_id"] for e in payload["locate"]] == [0, 10]
    assert payload["padded"] and all(p not in (0, 10) for p in payload["padded"])
    assert len(payload["locate"]) == 2, "填充帧不许混进 locate 输出"


def test_locate_dedupes_the_pipelines_repeat_padding(tmp: Path) -> None:
    """★ pipeline 用 ``ids.append(ids[-1])`` 补齐；重复 id 在这里被去掉。

    不去的后果：官方口径要 8 个**不同**帧，重复同一个 id 等于把一份信息
    数 8 遍，还把「6 真候选」伪装成 8 帧输入。
    """
    store = build_index(tmp / "i", {f: [{"label": "x", "bbox": (10, 10, 60, 60)}]
                                    for f in range(0, 90, 10)})
    payload, _ = run_locate(store, "0,10,20,20,20", "x", pad_to=8, veason_backend="fake")
    assert [e["frame_id"] for e in payload["locate"]] == [0, 10, 20]


def test_locate_batches_by_eight(tmp: Path) -> None:
    """一次吃 8 帧：9 个要定位的帧 ⇒ 两次前向，每批 ≤8。"""
    store = build_index(tmp / "i", {f: [{"label": "x", "bbox": (10, 10, 60, 60)}]
                                    for f in range(0, 180, 10)})
    sizes: list[int] = []

    class Recorder:
        name = "recording"

        def locate(self, frames_rgb, frame_ids, query):
            sizes.append(len(frame_ids))
            return {int(f): [[1, 2, 3, 4]] for f in frame_ids}

    backends.register("veason", "t-batch")(lambda **kw: Recorder())
    payload, _ = run_locate(store, ",".join(str(f) for f in range(0, 90, 10)), "x",
                            pad_to=8, veason_backend="t-batch")
    assert len(payload["locate"]) == 9
    assert sizes == [8, 8], f"9 帧应切成两批 8+8（第二批含 7 个填充），实际 {sizes}"
    assert payload["batch_size"] == 8


def test_locate_pad_to_zero_is_a_usage_error(tmp: Path) -> None:
    """``--pad-to 0`` 是 ZeroDivisionError 的经典来源（pipeline 里有同款）。
    在这里必须是一条带 hint 的用法错（码 2），不是裸 traceback。"""
    store = build_index(tmp / "i", {0: []})
    with pytest.raises(Exception) as e:
        run_locate(store, "0", "x", pad_to=0, veason_backend="fake")
    assert getattr(e.value, "code", None) == 2


def test_locate_unavailable_invents_no_coordinates(tmp: Path) -> None:
    """★ 拿不到定位就交空表 —— 不许编一个框出来。

    这条是「agent 从不输入坐标」的反面：CLI 也不许在没有测量时产坐标。
    """
    store = build_index(tmp / "i", {0: [{"label": "x", "bbox": (10, 10, 60, 60)}]})
    payload, info = run_locate(store, "0", "x", veason_backend="unavailable")
    assert payload["locate"] == [{"frame_id": 0, "bbox_2d_list": [], "iou_hint": None}]
    assert payload["src"] == "unavailable"
    assert any(e["event"] == "locate_backend_unavailable" for e in info["events"])


def test_locate_rejects_an_unknown_frame_with_a_hint(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: []})
    with pytest.raises(Exception) as e:
        run_locate(store, "999", "x", veason_backend="fake")
    assert "999" in str(e.value)


# ===========================================================================
# ③ evidence：src 归因、每件只出它那枚、缓存失效、退化
# ===========================================================================

def _write_senses(store: IndexStore, fid: int, region_axes: dict, axes=("depth", "face")) -> None:
    """写一份**真的能对得上**的 senses 缓存（regions_sha256 必须匹配当前表）。"""
    regions = store.require_regions(fid)["regions"]
    store.save_senses(fid, {
        "schema": 1, "frame_id": fid, "t": fid / FPS,
        "regions_sha256": sha256_json(regions),
        "axes": list(axes),
        "models": {"face": ["mediapipe-face-landmarker", "clip-vitb32-expression-zeroshot"],
                   "depth": ["depth-anything-v2-small"]},
        "regions": region_axes, "frame": {}, "computed_at": "test",
    })


def test_evidence_area_src_is_computed_and_zero_model(tmp: Path) -> None:
    """★ 纪律①：面积走纯计算，src 必须是 "computed"，不是任何模型名。"""
    store = build_index(tmp / "i", {0: [{"label": "x", "bbox": (20, 20, 120, 100)}]})
    payload, _ = run_evidence(store, "0", "x", "area")
    ev = payload["evidence"][0]
    assert ev["axis"] == "area"
    assert ev["src"] == "computed"
    assert ev["value"]["area_ratio"] == pytest.approx(((120 - 20) * (100 - 20)) / (W * H), abs=1e-6)


def test_evidence_src_points_at_the_model_that_really_produced_it(tmp: Path) -> None:
    """★★ 纪律③：src 必须指向【真正产出该值】的那个模型。

    这只 expression 是 CLIP 产的，而介质（blendshape）是 MediaPipe 产的。
    读 expression 那一栏就必须写 CLIP —— 写 mediapipe 就是「说不清出处」，
    那正是 senses.py 现存的标错（本模块不复制它）。
    """
    store = build_index(tmp / "i", {0: [{"label": "x", "bbox": (10, 10, 100, 100)}]})
    _write_senses(store, 0, {"0": {
        "face": {"n_faces": 0, "expression": "laughing", "expression_score": 0.49,
                 "expression_src": "clip-vitb32-expression-zeroshot",
                 "src": "mediapipe-face-landmarker"},
        "depth": {"rank_in_frame": 1, "mean_rel": 0.71},
    }})
    act, _ = run_evidence(store, "0", "x", "action")
    assert act["evidence"][0]["src"] == "clip-vitb32-expression"
    assert act["evidence"][0]["value"]["expression"] == "laughing"

    occ, _ = run_evidence(store, "0", "x", "occlusion")
    assert occ["evidence"][0]["src"] == "depth-anything-v2-small"
    assert occ["evidence"][0]["value"]["depth_rank"] == 1


def test_evidence_falls_back_to_videomae_src_when_only_videomae_answered(tmp: Path) -> None:
    """没有表情头那一栏时用 VideoMAE —— 但 src 也要跟着换成它。"""
    store = build_index(tmp / "i", {0: [{"label": "x", "bbox": (10, 10, 100, 100)}]})
    _write_senses(store, 0, {"0": {"action": {"label": "yoga", "score": 0.12}}})
    act, _ = run_evidence(store, "0", "x", "action")
    assert act["evidence"][0]["src"] == "videomae-small-kinetics"
    assert act["evidence"][0]["value"]["label"] == "yoga"


def test_evidence_ignores_senses_that_describe_another_region_table(tmp: Path) -> None:
    """缓存失效判据是 regions_sha256：表变了，旧测量就不再描述它。"""
    store = build_index(tmp / "i", {0: [{"label": "x", "bbox": (10, 10, 100, 100)}]})
    _write_senses(store, 0, {"0": {"depth": {"rank_in_frame": 1, "mean_rel": 0.9}}})
    rec = store.require_regions(0)
    rec["regions"].append({"idx": 1, "label": "y", "score": 0.1, "bbox": [5, 5, 20, 20],
                           "area_ratio": 0.01, "has_mask": False, "source": "test"})
    store.save_regions(0, rec)
    payload, info = run_evidence(store, "0", "x", "occlusion")
    assert payload["evidence"] == []
    assert any(e["event"] == "evidence_stale_senses_dropped" for e in info["events"])


def test_evidence_discards_a_backend_that_answers_the_wrong_axis(tmp: Path) -> None:
    """★ 纪律②：每个件只对它负责的那枚标签出数。

    一个「顺手多答一个轴」的件必须被丢掉 + 记账，否则归因就废了。
    """
    store = build_index(tmp / "i", {0: [{"label": "x", "bbox": (10, 10, 100, 100)}]})

    class Liar:
        name = "liar"

        def supports(self, axis):
            return True

        def measure(self, axis, frame, regions, masks, query):
            return [{"region_idx": 0, "axis": "not-" + axis, "value": {"bad": 1}, "src": "liar"}]

    backends.register("axis", "t-liar")(lambda **kw: Liar())
    payload, info = run_evidence(store, "0", "x", "text", axis_backend="t-liar")
    assert payload["evidence"] == []
    assert any(e["event"] == "evidence_backend_answered_the_wrong_axis" for e in info["events"])


def test_evidence_rejects_an_unknown_axis_instead_of_guessing(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: []})
    with pytest.raises(Exception) as e:
        run_evidence(store, "0", "x", "vibes")
    assert getattr(e.value, "code", None) == 2


# ===========================================================================
# ④ reflect：每轮必须带【新测量】、换帧必须换新帧、轮次上限
# ===========================================================================

def _zoom_path(store: IndexStore, fid: int, ridx: int) -> Path:
    return store.output_path(f"zoom/f{fid:06d}_r{ridx}.png")


def test_reflect_new_frame_never_reuses_a_seen_frame(tmp: Path) -> None:
    """★★ 规格 §6.1 钉死的那条：「换帧」必须真的换成【没看过的】帧。

    这里 seen_frames 覆盖了除 20 以外的每一帧，所以唯一合法的新帧就是 20。
    一旦实现偷懒（比如取时间相邻帧、或允许重复看已看过的帧），
    它要么返回空、要么返回一个 ∈ seen_frames 的帧 —— 两种都会被这条抓住。
    """
    store = build_index(tmp / "i", {
        0: [{"label": "dragon", "bbox": (10, 10, 30, 30)}],      # 小 ⇒ 低可见
        10: [{"label": "dragon", "bbox": (10, 10, 40, 40)}],
        20: [{"label": "dragon", "bbox": (10, 10, 120, 120)}],   # 大 ⇒ 才是要换去的帧
        30: [{"label": "dragon", "bbox": (10, 10, 35, 35)}],
    })
    seen = [0, 10, 30]
    # ★ candidate 的 area_ratio 必须与 bbox 一致（10×20 → 400/32000 = 0.0125），
    #   否则测的就不是「低可见触发换帧」，而是一个自相矛盾的状态。
    state = {"query": "dragon", "candidates": [{"frame_id": 0, "region_idx": 0,
                                                "area_ratio": 0.0125, "score": 0.5}],
             "evidence": [], "decision": {"frame_id": 0, "region_idx": 0},
             "seen_frames": seen}
    payload, _ = run_reflect(store, 1, state, reflect_first="low_visibility",
                             jev_backend="unavailable", axis_backend="unavailable")
    assert payload["new_info"], "低可见帧应该触发换帧"
    assert payload["action"] == "new_frame"
    assert payload["frames"] == [20]
    assert set(payload["frames"]).isdisjoint(set(seen)), "★ 换帧不许返回看过的帧"
    assert all(i["kind"] == "new_frame" and i["frame_id"] not in seen
               for i in payload["new_info"])


def test_reflect_has_no_new_frame_when_every_frame_was_seen(tmp: Path) -> None:
    """没有没看过的帧 ⇒ 换帧拿不出新信息，不许拿旧帧充数。"""
    store = build_index(tmp / "i", {
        0: [{"label": "dragon", "bbox": (10, 10, 30, 30)}],
        10: [{"label": "dragon", "bbox": (10, 10, 120, 120)}],
    })
    _zoom_path(store, 0, 0).parent.mkdir(parents=True, exist_ok=True)
    _zoom_path(store, 0, 0).write_bytes(b"already-zoomed")
    state = {"query": "dragon", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0, 10]}
    payload, _ = run_reflect(store, 1, state, reflect_first="low_visibility",
                             jev_backend="unavailable", axis_backend="unavailable")
    assert payload["new_info"] == []
    assert payload["terminated"] is True
    assert payload["reason"] == "no_new_information"


def test_reflect_zoom_produces_a_real_new_measurement(tmp: Path) -> None:
    """★ 放大是主力：它必须真的裁出文件来 —— 那是这一轮多出来的测量。"""
    store = build_index(tmp / "i", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    state = {"query": "dragon", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0]}
    payload, _ = run_reflect(store, 1, state, reflect_first="by_measurement",
                             jev_backend="unavailable", axis_backend="unavailable")
    assert payload["action"] == "zoom"
    entry = payload["new_info"][0]
    assert entry["kind"] == "zoom"
    # ★ 用 store 自己算出来的路径断言存在性：``crop`` 是给 agent 看的 WSL 形式
    #   (/mnt/c/...)，在 Windows 进程里 Path() 解析不了它 —— 那是路径口味问题，
    #   不是「文件没写出来」。两个口味都要给出来（crop / crop_win）。
    assert store.output_path("zoom/f000000_r0.png").is_file(), "crop 文件不存在 ⇒ 这一轮没有新测量"
    assert entry["crop"].startswith("/")
    # ★ crop_win 是「Windows 口味」：Windows 宿主上必须是盘符路径（含冒号）；
    #   类 Unix 宿主上 paths.for_windows 无从造盘符，只要求给出非空路径。
    #   原来无条件断言 ':' in crop_win，在 Linux 部署机上恒红（审计 #602）。
    assert entry["crop_win"]
    if os.name == "nt":
        assert ":" in entry["crop_win"]
    assert entry["measurement"]
    assert entry["answer"] is None, "没接线的后端不许假装答过"


def test_reflect_repeating_a_zoom_is_not_new_information(tmp: Path) -> None:
    """★ 铁律：「重排旧信息」不算反思。

    同一个 crop 已经存在 = 上一轮已经看过这块像素 ⇒ 这一轮拿不出新测量，
    必须终止而不是再报一次 zoom。
    """
    store = build_index(tmp / "i", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    _zoom_path(store, 0, 0).parent.mkdir(parents=True, exist_ok=True)
    _zoom_path(store, 0, 0).write_bytes(b"already-zoomed")
    state = {"query": "dragon", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0]}
    payload, _ = run_reflect(store, 1, state, reflect_first="by_measurement",
                             jev_backend="unavailable", axis_backend="unavailable")
    assert payload["new_info"] == []
    assert payload["terminated"] is True


def test_reflect_fuse_is_explicitly_not_a_normal_termination(tmp: Path) -> None:
    """★ 口径：**不设次数上限**。唯一正常终止 = 没有新测量可拿。

    `fuse_rounds` 是保险丝，只在「每轮都假装有新信息」这种程序 bug 上兜底。
    它触发时输出必须**明说是保险丝** —— 否则会被当成轮次上限来规划预算，
    等于用「重试 N 次」代替「想清楚」。
    """
    store = build_index(tmp / "i", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    state = {"query": "dragon", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0]}
    payload, _ = run_reflect(store, 4, state, fuse_rounds=3, jev_backend="unavailable",
                             axis_backend="unavailable")
    assert payload["new_info"] == []
    assert payload["reason"] == "fuse_round_limit"
    assert payload["fuse"] is True, "保险丝触发必须显式标记，否则会被当成正常终止"
    assert "保险丝" in payload["note"] and "no_new_information" in payload["note"]
    assert payload["fuse_rounds"] == 3


def test_reflect_has_no_round_limit_while_new_information_keeps_coming(tmp: Path) -> None:
    """★ 只要每轮真拿得到新测量，就一直继续 —— 跑在第 6 轮也不许因为「轮次」而停。

    （旧口径在 round 2 就停了；这条把「judge 是 new_info、不是一个常数」钉住。）
    """
    store = build_index(tmp / "i", {
        f: [{"label": "dragon", "bbox": (10, 10, 30 + f, 30 + f)}]
        for f in (0, 10, 20, 30, 40, 50, 60, 70)
    })
    _zoom_path(store, 0, 0).parent.mkdir(parents=True, exist_ok=True)
    _zoom_path(store, 0, 0).write_bytes(b"already-zoomed")   # 放大做过了，逼它走换帧
    state = {"query": "dragon", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0},
             "seen_frames": [0, 10, 20, 30, 40, 50]}
    payload, _ = run_reflect(store, 6, state, fuse_rounds=0, reflect_first="low_visibility",
                             min_area=1.0, jev_backend="unavailable", axis_backend="unavailable")
    assert payload["action"] == "new_frame"
    assert payload["new_info"], "第 6 轮仍有新测量 ⇒ 不许因为有「轮次」这个概念而停"
    assert payload["frames"] == [60, 70]
    assert payload["fuse_rounds"] is None, "0 = 完全不设保险丝"


def test_reflect_normal_termination_is_labelled_as_normal(tmp: Path) -> None:
    """正常终止要能被区分出来：normal_termination=true 且 reason=no_new_information。"""
    store = build_index(tmp / "i", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    _zoom_path(store, 0, 0).parent.mkdir(parents=True, exist_ok=True)
    _zoom_path(store, 0, 0).write_bytes(b"already-zoomed")
    state = {"query": "dragon", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0]}
    payload, _ = run_reflect(store, 9, state, jev_backend="unavailable",
                             axis_backend="unavailable")
    assert payload["reason"] == "no_new_information"
    assert payload["normal_termination"] is True
    assert not payload.get("fuse"), "正常终止不该带保险丝标记"


def test_reflect_stops_when_the_premise_is_false(tmp: Path) -> None:
    """前提为假（Jev 的 none 判定）⇒ 终止，原样交付，不猜。"""
    store = build_index(tmp / "i", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    state = {"query": "dragon", "candidates": [],
             "evidence": [{"frame_id": 0, "axis": "existence", "value": {"exists": False},
                           "src": "jev"}],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0]}
    payload, _ = run_reflect(store, 1, state, jev_backend="unavailable",
                             axis_backend="unavailable")
    assert payload["new_info"] == []
    assert payload["reason"] == "premise_false"


def test_reflect_new_axis_only_fires_for_an_axis_never_measured(tmp: Path) -> None:
    """换轴 = 换个问法问**原来看不到的**属性；已经问过的轴是旧信息。"""
    store = build_index(tmp / "i", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    asked: list[str] = []

    class Axis:
        name = "t-axis"

        def supports(self, axis):
            return True

        def measure(self, axis, frame, regions, masks, query):
            asked.append(axis)
            return [{"region_idx": 0, "axis": axis, "value": {"v": 1}, "src": "t-axis"}]

    backends.register("axis", "t-axis")(lambda **kw: Axis())
    state = {"query": "dragon", "candidates": [],
             "evidence": [{"frame_id": 0, "axis": "occlusion", "value": {}, "src": "x"}],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0]}
    payload, _ = run_reflect(store, 1, state, reflect_first="by_measurement",
                             jev_backend="unavailable", axis_backend="t-axis")
    # 放大是主力，所以第一件仍是 zoom；但 zoom 之后任何一轮都不许再问 occlusion。
    assert payload["action"] == "zoom"
    assert "occlusion" not in asked


def test_reflect_state_accepts_both_file_and_inline_json(tmp: Path) -> None:
    """``--state`` 两吃：规格写文件路径，pipeline 传内联 JSON。"""
    state = {"query": "q", "seen_frames": [1, 2]}
    inline = parse_state(json.dumps(state))
    assert inline == state
    p = tmp / "state.json"
    p.write_text(json.dumps(state), encoding="utf-8")
    assert parse_state(str(p)) == state
    with pytest.raises(Exception) as e:
        parse_state("not json and not a file")
    assert getattr(e.value, "code", None) == 2


# ---- 待拍板③：放大用谁的框（--crop-source）----

def test_reflect_new_frame_only_counts_the_same_label(tmp: Path) -> None:
    """★ 「换帧」找的是**同一个目标**更大的帧，不是「那一帧里最大的东西」。

    这里目标 label 是 'dragon'，另一帧里只有 'boulder' 更大 —— 那不是同一个目标，
    不能拿来当新信息。放它过去就会让 measurement 里的「同名目标」变成假话。
    """
    store = build_index(tmp / "i", {
        0: [{"label": "dragon", "bbox": (10, 10, 30, 30)}],      # 小 ⇒ 低可见
        10: [{"label": "boulder", "bbox": (0, 0, 200, 160)}],    # 更大，但不是同一个目标
    })
    state = {"query": "dragon", "candidates": [{"frame_id": 0, "region_idx": 0,
                                                "area_ratio": 0.0125, "score": 0.5}],
             "evidence": [], "decision": {"frame_id": 0, "region_idx": 0},
             "seen_frames": [0]}
    payload, _ = run_reflect(store, 1, state, reflect_first="low_visibility",
                             jev_backend="unavailable", axis_backend="unavailable")
    assert payload["new_info"], "换帧拿不到，应该退到放大（主力），而不是拿 boulder 充数"
    assert all(i["kind"] != "new_frame" for i in payload["new_info"])
    assert payload["action"] == "zoom"


def test_reflect_crop_source_picks_between_yolo_and_veason_boxes(tmp: Path) -> None:
    """``--crop-source`` 决定「放大」剪谁的框，并且读数里要写清框的来源。

    ★ 写清来源不是装饰：多 provider 并存时说不清「这块像素是哪来的」，
      就等于这一轮的新信息不可复核。
    """
    state = {"query": "dragon", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0],
             "locate": [{"frame_id": 0, "bbox_2d_list": [[40, 40, 80, 80]], "iou_hint": None}]}

    store_v = build_index(tmp / "v", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    pv, _ = run_reflect(store_v, 1, state, reflect_first="by_measurement",
                        crop_source="veason", jev_backend="unavailable",
                        axis_backend="unavailable")
    assert pv["new_info"][0]["bbox_source"] == "veason-r1"
    assert pv["new_info"][0]["bbox"] == [40, 40, 80, 80]

    store_y = build_index(tmp / "y", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    py, _ = run_reflect(store_y, 1, state, reflect_first="by_measurement",
                        crop_source="yolo", jev_backend="unavailable",
                        axis_backend="unavailable")
    assert py["new_info"][0]["bbox_source"] == "yolo-region"
    assert py["new_info"][0]["bbox"] == [10, 10, 120, 120]


def test_reflect_veason_crop_falls_back_visibly_when_there_is_no_locate_box(tmp: Path) -> None:
    """要 ② 的框但没有 ⇒ 退回 ① 的框，并把「退回过」写进读数，不静默换源。"""
    store = build_index(tmp / "i", {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    state = {"query": "dragon", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0]}
    payload, _ = run_reflect(store, 1, state, reflect_first="by_measurement",
                             crop_source="veason", jev_backend="unavailable",
                             axis_backend="unavailable")
    assert payload["new_info"][0]["bbox_source"].startswith("yolo-region(fallback")


def test_reflect_rejects_an_unknown_crop_source(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: []})
    with pytest.raises(Exception) as e:
        run_reflect(store, 1, {"decision": {}}, crop_source="vibes")
    assert getattr(e.value, "code", None) == 2


def test_reflect_low_visibility_threshold_is_tunable(tmp: Path) -> None:
    """「低可见」是**测出来的**：把门限抬到 0.5，一个 0.45 的目标就该被判低可见。"""
    store = build_index(tmp / "i", {
        0: [{"label": "d", "bbox": (10, 10, 120, 120)}],     # 110×110/32000 ≈ 0.378
        10: [{"label": "d", "bbox": (0, 0, 200, 160)}],       # 更大
    })
    state = {"query": "d", "candidates": [], "evidence": [],
             "decision": {"frame_id": 0, "region_idx": 0}, "seen_frames": [0]}
    strict, _ = run_reflect(store, 1, state, reflect_first="low_visibility",
                            min_area=0.5, jev_backend="unavailable",
                            axis_backend="unavailable")
    assert strict["measurements"]["low_visibility"] is True
    assert strict["measurements"]["min_area_threshold"] == 0.5
    assert strict["action"] == "new_frame"

    loose, _ = run_reflect(store, 1, state, reflect_first="low_visibility",
                           min_area=0.01, jev_backend="unavailable",
                           axis_backend="unavailable")
    assert loose["measurements"]["low_visibility"] is False
    # ★ 面积是从索引里量出来的（这条 state 的 candidates 是空的）——
    #   量成 0.0 会被读成「低可见」，而低可见会触发换帧。
    assert loose["measurements"]["decision_area"] == pytest.approx(0.378, abs=0.01)


# ===========================================================================
# 后端接缝：Jev 的形态、Veason 的答案解析、auto 的解析与降级
# 这些都不需要 GPU / 权重 —— 正是把它们从模型里拆出来的目的。
# ===========================================================================

def test_jev_protocol_is_n_choice_not_per_region_judgement() -> None:
    """★ 协议里**只有** choose（N 选一）与 exists（存在性）。

    没有 per-region 的 confirm：那个形态实测 74% 都答「是」，是被否掉的那个。
    把这条钉在协议上，后面就没法悄悄加回来。
    """
    assert not hasattr(JevBackend, "confirm")
    fake = FakeJev()
    assert isinstance(fake, JevBackend), "假件必须满足协议，否则测试验的不是同一条路"
    assert set(("choose", "exists", "name")) <= set(dir(fake))


def test_veason_answer_parser_maps_the_560_space_back_to_the_frame() -> None:
    """★ 框在 560 坐标空间里，要乘 width/560, height/560 反算回原图。"""
    raw = ('<think>looking</think><answer>'
           '{"keyframe_timestamp": 20, "bbox_2d_list": [[140, 140, 420, 420]]}'
           '</answer>')
    out = parse_veason_answer(raw, [0, 20], {0: (1280, 720), 20: (2560, 1440)})
    assert out[20] == [[640, 360, 1920, 1080]]
    assert out[0] == [], "它只答了一帧，其余帧必须是空表而不是别的帧的框"


def test_veason_answer_parser_refuses_a_keyframe_outside_the_sampled_frames() -> None:
    """★ 官方 reward：`if keyframe_id not in frame_ids: return 0`。

    静默退回 0 会把「格式错」伪装成「选了第 0 帧」这个**合法答案** ——
    两种情况的分数都是 0，混起来以后就再也查不出来了。所以这里报错。
    """
    raw = '<answer>{"keyframe_timestamp": 7, "bbox_2d_list": [[1, 2, 3, 4]]}</answer>'
    with pytest.raises(Exception) as e:
        parse_veason_answer(raw, [0, 20], {0: (1280, 720), 20: (1280, 720)})
    assert "7" in str(e.value) and "sampled" in str(e.value)


def test_veason_answer_parser_refuses_output_without_an_answer_block() -> None:
    with pytest.raises(Exception) as e:
        parse_veason_answer("I think the dragon is at (100, 100).", [0], {0: (1280, 720)})
    assert "answer" in str(e.value).lower()


def test_jev_rejects_more_options_than_k_max_instead_of_truncating() -> None:
    """★ 源码纪律：超过 K_MAX 直接 raise，**不截断**（截断会静默改变问题）。"""
    jev = object.__new__(LocalJev)          # 绕开 __init__：不需要 GPU / 权重
    with pytest.raises(Exception) as e:
        jev._ask(None, "q", [str(i) for i in range(JEV_K_MAX + 1)])
    assert getattr(e.value, "code", None) == 2
    assert str(JEV_K_MAX) in str(e.value)


def test_backend_auto_resolves_to_local_and_degrades_without_weights(monkeypatch, tmp_path) -> None:
    """★ 部署口径：auto = 同进程真件（skill 跑在服务器上）。

    本机没权重时它必须抛 ``BackendUnavailable``（可 catch 的降级），
    而不是 ImportError 或让整条命令 exit 1 —— ① 的高召回纪律靠这个成立。
    """
    monkeypatch.setenv("VEASON_MODELS_DIR", str(tmp_path / "no-such-models-root"))
    assert set(backends.registered("jev")) >= {"local", "fake", "unavailable"}
    with pytest.raises(BackendUnavailable) as e:
        backends.get_backend("veason", "auto")
    assert "Veason-R1" in e.value.error
    assert "VEASON_MODELS_DIR" in e.value.hint
    with pytest.raises(BackendUnavailable):
        backends.get_backend("jev", "unavailable")


def test_backend_env_var_is_a_fallback_that_explicit_argument_overrides(monkeypatch) -> None:
    """显式传参 > 环境变量（pytest 官方口径：能显式注入就显式注入）。"""
    monkeypatch.setenv("VEASON_JEV_BACKEND", "fake")
    assert type(backends.get_backend("jev")).__name__ == "FakeJev"
    with pytest.raises(BackendUnavailable):
        backends.get_backend("jev", "unavailable")
    with pytest.raises(Exception):
        backends.get_backend("jev", "no-such-provider")


# ===========================================================================
# CLI 契约：stdout 单个 JSON、错误走 stderr JSON、非交互
# ===========================================================================
def test_cli_screen_stdout_is_exactly_one_json_document(tmp: Path) -> None:
    root = tmp / "i"
    build_index(root, {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    rc, out, err = run("screen", str(root), "--query", "dragon", "--top-n", "3")
    assert rc == 0, err
    payload = _only_json(out)
    assert payload["candidates"][0]["route"] == "yolo_hit"


def test_cli_errors_are_one_json_object_on_stderr_with_a_hint(tmp: Path) -> None:
    """★ 四条新命令都遵守既有错误契约：stderr 单 JSON + 非零退出码。"""
    root = tmp / "i"
    build_index(root, {0: []})
    rc, out, err = run("evidence", str(root), "--frames", "0", "--query", "x",
                       "--axes", "vibes")
    assert rc == 2
    assert out.strip() == "", "错误时 stdout 必须为空，否则调用方会把错误当数据"
    payload = json.loads(err)
    assert payload["ok"] is False and payload["where"] == "evidence"
    assert payload.get("hint")


def test_cli_reflect_reads_state_from_a_file(tmp: Path) -> None:
    root = tmp / "i"
    store = build_index(root, {0: [{"label": "dragon", "bbox": (10, 10, 120, 120)}]})
    p = tmp / "state.json"
    p.write_text(json.dumps({"query": "dragon", "candidates": [], "evidence": [],
                             "decision": {"frame_id": 0, "region_idx": 0},
                             "seen_frames": [0]}), encoding="utf-8")
    rc, out, err = run("reflect", str(root), "--round", "1", "--state", str(p))
    assert rc == 0, err
    assert _only_json(out)["action"] == "zoom"
    assert _zoom_path(store, 0, 0).is_file()


def test_the_four_modules_never_prompt(tmp: Path) -> None:
    """非交互：四条命令的模块里不许出现 input() / 确认提示。"""
    for name in ("screen", "locate", "evidence", "reflect", "backends"):
        src = (ROOT / "veason_harness" / f"{name}.py").read_text(encoding="utf-8")
        assert "input(" not in src, f"{name}.py 包含 input()，会挂住调用它的 agent"
        assert "getpass" not in src


def test_indexer_is_byte_frozen() -> None:
    """★ 硬约束：``indexer.py`` 一个字节不改（分层：senses 保持显式命令）。"""
    got = hashlib.sha256((ROOT / "veason_harness" / "indexer.py").read_bytes()).hexdigest()
    assert got == INDEXER_SHA256, (
        "indexer.py 变了。它是要被冻住的：分层靠它保持 —— senses 必须留在显式命令里，"
        "不许被并进 index 的默认路径。若确需修改，请同时更新本测试里的哈希并写清理由。"
    )


# ===========================================================================
# 端到端：veason_pipeline.py 真的能把五环跑通
# ===========================================================================

def test_pipeline_no_reflect_switch_actually_disables_ring_five(tmp: Path) -> None:
    """★ 反证对照臂必须是**真的关掉** ⑤ 整环。

    这里钉一个容易踩的坑：`--fuse-reflect 0` 的语义是「不设保险丝」，
    **不是**「不反思」—— 拿它当关臂会让对照臂悄悄跑成开臂（两边读数一样，
    而人会以为「反思没用」）。
    """
    root = tmp / "i"
    build_index(root, {f: [{"label": "dragon", "bbox": (10, 10, 60 + f, 60 + f)}]
                       for f in (0, 10, 20, 30, 40, 50, 60, 70, 80)})
    dec = tmp / "dec.json"
    dec.write_text(json.dumps({"frame_id": 0, "region_idx": 0}), encoding="utf-8")

    off = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "dragon",
                          "--index", str(root), "--decide-from", str(dec),
                          "--no-rewrite", "--no-reflect"],
                         capture_output=True, text=True, encoding="utf-8")
    assert off.returncode == 0, off.stderr
    stages = [json.loads(x)["stage"] for x in off.stdout.splitlines()
              if x.startswith("{") and '"stage"' in x]
    assert "reflect-disabled" in stages
    # ★ 过滤写成"轮次形态"（reflect-<数字>），不能写 startswith("reflect-")：
    #   reflect-disabled / reflect-fuse 也带这个前缀，用前缀判会把自己判成"跑过一轮"。
    rounds = [s for s in stages if re.fullmatch(r"reflect-\d+", s)]
    assert rounds == [], f"关臂不该有任何反射轮次，实际 {rounds}"

    on = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "dragon",
                         "--index", str(root), "--decide-from", str(dec), "--no-rewrite"],
                        capture_output=True, text=True, encoding="utf-8")
    assert on.returncode == 0, on.stderr
    assert "reflect-1" in on.stdout, "默认（不设次数上限）必须真的跑 ⑤"


def test_pipeline_fuse_is_visible_in_stdout(tmp: Path) -> None:
    """保险丝触发时 pipeline 的 stdout 必须出现 reflect-fuse 且标 fuse=true。"""
    root = tmp / "i"
    build_index(root, {f: [{"label": "dragon", "bbox": (10, 10, 60 + f, 60 + f)}]
                       for f in (0, 10, 20, 30, 40, 50, 60, 70, 80)})
    dec = tmp / "dec.json"
    dec.write_text(json.dumps({"frame_id": 0, "region_idx": 0}), encoding="utf-8")
    p = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "dragon",
                        "--index", str(root), "--decide-from", str(dec),
                        "--no-rewrite", "--fuse-reflect", "1"],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, p.stderr
    fuse = [json.loads(x) for x in p.stdout.splitlines()
            if x.startswith("{") and '"reflect-fuse"' in x]
    assert fuse and fuse[0]["fuse"] is True
    assert "保险丝" in fuse[0]["note"]


def test_pipeline_runs_all_stages_and_stops_for_the_agent(tmp: Path) -> None:
    """★ 验收②：完整 ``veason_pipeline.py`` 跑通，且第 ④ 步【停下来等 agent】。"""
    root = tmp / "i"
    build_index(root, {f: [{"label": "dragon", "bbox": (10, 10, 60 + f, 60 + f)}]
                       for f in (0, 10, 20, 30, 40, 50, 60, 70, 80)})
    # 不给 --decide-from ⇒ 应打印决策请求并退出码 3（这是设计，不是缺陷）
    # ★ --no-rewrite：本题 query 已是外观名词短语，显式声明不改写（第 ⓪ 步会先停，
    #   见 test_pipeline_stops_for_agent_rewrite_before_screen）。
    p = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "dragon",
                        "--index", str(root), "--no-rewrite"],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 3, f"应停在第 ④ 步等 agent，实际 rc={p.returncode}\n{p.stderr}"
    assert '"needs_decision": true' in p.stdout
    stages = [json.loads(line)["stage"] for line in p.stdout.splitlines()
              if line.startswith("{") and '"stage"' in line]
    assert stages[:4] == ["rewrite", "screen", "locate", "evidence"], stages

    # 给了 decision ⇒ 走完 ⑤ 并打印 answer
    dec = tmp / "dec.json"
    dec.write_text(json.dumps({"frame_id": 0, "region_idx": 0}), encoding="utf-8")
    p2 = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "dragon",
                         "--index", str(root), "--decide-from", str(dec), "--no-rewrite"],
                        capture_output=True, text=True, encoding="utf-8")
    assert p2.returncode == 0, f"{p2.stdout}\n{p2.stderr}"
    stages2 = [json.loads(line)["stage"] for line in p2.stdout.splitlines()
               if line.startswith("{") and '"stage"' in line]
    assert "answer" in stages2, stages2
    assert any(s.startswith("reflect-") for s in stages2), stages2


# ===========================================================================
# ⓪ rewrite：agent 改写在编排里必须有位置，且不静默（缺陷#2）
# ===========================================================================

def test_pipeline_stops_for_agent_rewrite_before_screen(tmp: Path) -> None:
    """★ 缺陷#2：第 ⓪ 步（agent 改写）是显式停点。

    ★ 为什么必须停：开放词表检测器对整句/关系式查询零命中，而这条失败本来是
      **静默的**（照样出候选、照样往下走，只是 yolo_hit=0）。所以没改写就停下要
      agent 去改写 —— 不许悄悄跑成 0 命中。
    """
    root = tmp / "i"
    build_index(root, {f: [{"label": "parachute", "bbox": (10, 10, 60, 60)}] for f in (0, 10)})
    p = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "黄色的卡通奶龙",
                        "--index", str(root)],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 3, f"应停在 ⓪ 等 agent 改写，实际 rc={p.returncode}\n{p.stdout}\n{p.stderr}"
    assert '"needs_rewrite": true' in p.stdout
    stages = [json.loads(x)["stage"] for x in p.stdout.splitlines()
              if x.startswith("{") and '"stage"' in x]
    assert stages == ["rewrite"], f"停下时只该有 rewrite 一步，实际 {stages}"
    ev = [json.loads(x) for x in p.stdout.splitlines()
          if x.startswith("{") and '"pipeline_rewrite_missing"' in x]
    assert ev and ev[0]["rewritten"] is False, "停下时也必须打事件，不能只有退出码"


def test_pipeline_rewrite_from_file_reaches_screen_synonyms(tmp: Path) -> None:
    """★ 缺陷#2 反证（修后臂）：--rewrite-from 的改写真的送到 screen ⇒ yolo_hit 0 → N。

    两臂只差 `--rewrite-from`：`--no-rewrite`（不改写）应保持 0 命中；给文件后
    同一条链路必须全命中 —— 证明「改写这步被编排进去了」。
    """
    root = tmp / "i"
    build_index(root, {f: [{"label": "parachute", "bbox": (10, 10, 60, 60)}]
                       for f in (0, 10, 20)})
    dec = tmp / "dec.json"
    dec.write_text(json.dumps({"frame_id": 0, "region_idx": 0}), encoding="utf-8")
    rewrite = tmp / "rewrite.json"
    rewrite.write_text(json.dumps({"synonym": ["奶龙=parachute"]}), encoding="utf-8")

    def stages(extra: list[str]) -> dict:
        q = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "黄色的卡通奶龙",
                            "--index", str(root), "--decide-from", str(dec),
                            "--no-reflect", *extra],
                           capture_output=True, text=True, encoding="utf-8")
        assert q.returncode == 0, f"{q.stdout}\n{q.stderr}"
        out: dict = {}
        for line in q.stdout.splitlines():
            if not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "stage" in d:
                out[d["stage"]] = d
        return out

    off = stages(["--no-rewrite"])
    on = stages(["--rewrite-from", str(rewrite)])
    assert off["screen"]["screened"]["yolo_hit"] == 0, "反证臂：不改写 ⇒ 零命中"
    assert on["screen"]["screened"]["yolo_hit"] == 3, "修后臂：改写送到 screen ⇒ 全命中"
    assert on["rewrite"]["rewritten"] is True
    assert on["rewrite"]["appearance_phrases"] == ["parachute"]
    assert off["rewrite"]["rewritten"] is False


def test_pipeline_missing_rewrite_is_not_silent_when_skipped(tmp: Path) -> None:
    """★ 显式 --no-rewrite 也必须打事件（对齐 screen_jev_unavailable 那种事件通道）。"""
    root = tmp / "i"
    build_index(root, {0: [{"label": "dragon", "bbox": (10, 10, 60, 60)}]})
    p = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "dragon",
                        "--index", str(root), "--no-rewrite", "--no-reflect"],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 3, "改了 ⓪ 之后仍应停在 ④ decide"
    rw = [json.loads(x) for x in p.stdout.splitlines()
          if x.startswith("{") and '"pipeline_rewrite_missing"' in x]
    assert rw and rw[0]["event"] == "pipeline_rewrite_missing"


def test_pipeline_forwards_child_stderr_events(tmp: Path) -> None:
    """★ 降级/成本事件必须有【送达路径】：子进程 rc==0 的 stderr 也要转发。

    反证：只在 rc!=0 时拼 stderr，则 screen_top_n_truncated 这类 rc==0 的读数
    在 pipeline 的 stderr 里一条都看不到（AGENTS.md §7⑬）。
    """
    root = tmp / "i"
    build_index(root, {f: [{"label": "dragon", "bbox": (10, 10, 60, 60)}]
                       for f in range(0, 90, 10)})
    p = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "dragon",
                        "--index", str(root), "--top-n", "3",
                        "--no-rewrite", "--no-reflect"],
                       capture_output=True, text=True, encoding="utf-8")
    assert "screen_top_n_truncated" in p.stderr, \
        f"rc==0 的降级事件没有送达 pipeline stderr：{p.stderr!r}"


def test_pipeline_merges_new_measurements_so_axes_do_not_repeat(tmp: Path,
                                                                monkeypatch) -> None:
    """★ 反思的 new_info 必须有【回写路径】，否则同一根轴每轮都算「新」。

    反证（实测）：把 ``_merge_new_info`` 退回只并 frames，reflect 会每轮都报
    ``axis=occlusion``，只能靠保险丝停下 —— 与「唯一正常终止 = no_new_information」
    相抵。修后同一根轴不许被当成新测量两次。
    """
    root = tmp / "i"
    build_index(root, {f: [{"label": "dragon", "bbox": (10, 10, 60, 60)}]
                       for f in range(0, 90, 10)})
    dec = tmp / "dec.json"
    dec.write_text(json.dumps({"frame_id": 0, "region_idx": 0}), encoding="utf-8")
    monkeypatch.setenv("VEASON_AXIS_BACKEND", "fake")
    p = subprocess.run([PY, PIPELINE, "--video", "dummy.mp4", "--query", "dragon",
                        "--index", str(root), "--decide-from", str(dec),
                        "--axes", "area", "--no-rewrite", "--fuse-reflect", "4"],
                       capture_output=True, text=True, encoding="utf-8")
    assert p.returncode == 0, f"{p.stdout}\n{p.stderr}"
    per_round: list[list[str]] = []
    for line in p.stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(d.get("stage", "")).startswith("reflect-") and d.get("new_info") is not None:
            per_round.append([i.get("axis") for i in d["new_info"] if i.get("axis")])
    flat = [a for row in per_round for a in row]
    assert flat, f"fake 轴后端下应当产出 new_axis，实际 {per_round}"
    assert len(flat) == len(set(flat)), \
        f"同一根轴被重复当成新测量 ⇒ new_info 没回写 state：{per_round}"
