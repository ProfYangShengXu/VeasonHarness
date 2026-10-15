"""五环链 ①②③④ 的自检 —— 断言的是**判据**，不是 happy path。

每条断言都对应一句契约；命名里写出那句话，注释里写清「它为什么可能不成立」。
和 ``tests/test_vrs.py`` 同样的信念：一条不会失败的检查不是检查。

这一批用例**不需要任何模型**：① 的 YOLO 臂读索引里既有的 region 表，③ 的模型轴读
``vrs senses`` 已算过的缓存，②④ 的模型件用**显式测试缝**（``veason=`` / ``jev=``）。
这不是「可注入后端」：生产路径的默认值是真模型，测试只是把参数传死。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vrs_skills.evidence import read_axis, run_evidence  # noqa: E402
from vrs_skills.errors import VrsError  # noqa: E402
from vrs_skills.locate import _bboxes_from_answer, batches, run_locate  # noqa: E402
from vrs_skills.reflect import parse_state, run_reflect  # noqa: E402
from vrs_skills.screen import query_terms, run_screen  # noqa: E402
from vrs_skills.store import IndexStore, sha256_json  # noqa: E402

PY = sys.executable
VRS = str(ROOT / "vrs.py")
PIPELINE = str(ROOT / "vrs_pipeline.py")

#: ★ 「indexer.py 一个字节不改」的硬保证（它守的是分层：senses 是显式命令）。
INDEXER_SHA256 = "8e7016d4e873ff24d4e41c914ba0daf97b39f8ef517e094f6ce84adf16448a4c"

W, H, FPS = 200, 160, 10.0


# ---------------------------------------------------------------------------
# 造一个**真文件**索引（真 PNG / 真 mask / 真 region 表），零模型
# ---------------------------------------------------------------------------

def build_index(root: Path, spec: dict[int, list[dict]], *, w: int = W, h: int = H,
                fps: float = FPS, senses: dict[int, dict] | None = None) -> IndexStore:
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
    for fid, payload in (senses or {}).items():
        store.save_senses(fid, payload)
    return store


def _senses_for(store: IndexStore, fid: int, axes: dict) -> dict:
    """造一份 schema 正确、regions_sha256 对得上的 senses 缓存。"""
    regions = store.require_regions(fid)["regions"]
    return {"schema": 2, "frame_id": fid, "t": 0.0,
            "regions_sha256": sha256_json(regions), "axes": list(axes),
            "models": {}, "regions": axes, "frame": {}, "computed_at": "test"}


class FakeJev:
    """测试替身：``None`` = 不肯答，float = P(yes)。"""

    name = "fake-jev"

    def __init__(self, exists=None, confirm=None):
        self._exists, self._confirm = exists, confirm
        self.calls: list = []

    def exists(self, img, query):
        self.calls.append(("exists", query))
        return self._exists(query) if callable(self._exists) else self._exists

    def confirm(self, img, bbox, query):
        self.calls.append(("confirm", tuple(bbox)))
        return self._confirm(bbox, query) if callable(self._confirm) else self._confirm


class FakeVeason:
    """测试替身：记录每批的帧数，返回一个确定性框。"""

    name = "fake-veason"

    def __init__(self, box=(10, 10, 90, 90)):
        self.box, self.batch_sizes = box, []

    def locate(self, frames, frame_ids, query):
        self.batch_sizes.append(list(frame_ids))
        return {int(f): [list(self.box)] for f in frame_ids}


def run(*args: str) -> tuple[int, str, str]:
    p = subprocess.run([PY, VRS, *args], capture_output=True, text=True, encoding="utf-8")
    return p.returncode, p.stdout, p.stderr


def _state(store, query, decision, *, locate=None, candidates=None, evidence=None,
           seen=None, absent=None) -> dict:
    s = {"query": query, "decision": decision,
         "candidates": candidates if candidates is not None else [
             {"frame_id": decision["frame_id"], "region_idx": decision["region_idx"],
              "area_ratio": 0.2, "score": 0.9}],
         "evidence": evidence or [], "seen_frames": seen or [decision["frame_id"]],
         "locate": locate if locate is not None else [
             {"frame_id": decision["frame_id"], "bbox_2d_list": [[10, 10, 90, 90]]}]}
    if absent is not None:
        s["absent"] = absent
    return s


# ===========================================================================
# ① screen
# ===========================================================================

def test_screen_tier_a_never_loads_jev(tmp: Path) -> None:
    """档 A：label 命中 ⇒ 直接用，一次 Jev 也不问（cost 0）。"""
    store = build_index(tmp / "i", {0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}]})
    jev = FakeJev(exists=0.99, confirm=0.99)
    payload, _ = run_screen(store, "parachute", top_n=5, jev=jev)
    assert payload["screened"]["yolo_hit"] == 1
    assert payload["screened"]["jev_called"] == 0
    assert jev.calls == []
    assert payload["candidates"][0]["route"] == "yolo_hit"


def test_screen_tier_b_confirms_the_yolo_box(tmp: Path) -> None:
    """档 B：有框但 label 未命中 ⇒ 上 Jev，且问的是 YOLO 那个框。"""
    store = build_index(tmp / "i", {0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}]})
    jev = FakeJev(confirm=0.87)
    payload, _ = run_screen(store, "黄色奶龙", top_n=5, jev=jev)
    assert payload["screened"]["jev_called"] == 1
    assert payload["candidates"][0]["route"] == "jev_confirmed"
    assert payload["candidates"][0]["tier"] == "B"
    assert payload["candidates"][0]["jev_yes"] == 0.87
    assert jev.calls and jev.calls[0][1] == (10, 10, 90, 90)


def test_screen_tier_c_asks_existence(tmp: Path) -> None:
    """档 C：零检出 ⇒ 上 Jev 问存在性。"""
    store = build_index(tmp / "i", {0: []})
    jev = FakeJev(exists=0.9)
    payload, _ = run_screen(store, "dragon", top_n=5, jev=jev)
    assert payload["screened"]["jev_called"] == 1
    assert payload["candidates"][0]["route"] == "jev_confirmed"
    assert payload["candidates"][0]["tier"] == "C"
    assert payload["candidates"][0]["region_idx"] == -1


def test_screen_refusal_is_not_absence(tmp: Path) -> None:
    """★ 建模#494：Jev 返回 None（不肯答）≠ 说不在。不肯答必须保留该帧。"""
    store = build_index(tmp / "i", {0: []})
    kept, _ = run_screen(store, "dragon", jev=FakeJev(exists=None))
    assert kept["screened"]["dropped"] == 0
    assert kept["screened"]["refused"] == 1
    assert kept["candidates"][0]["route"] == "jev_refused_kept"
    assert kept["candidates"][0]["region_idx"] == -1
    # 显式低分才是「不在」，且两者输出**不同形**（旧版把它们压成同一个）。
    gone, _ = run_screen(store, "dragon", jev=FakeJev(exists=0.05))
    assert gone["screened"]["dropped"] == 1
    assert gone["candidates"] == []
    assert gone["absent"] is True


def test_screen_accounting_is_consistent(tmp: Path) -> None:
    """台账不重不漏：total == yolo_hit + jev_called；kept == total - dropped。"""
    store = build_index(tmp / "i", {
        0: [{"label": "parachute", "bbox": (10, 10, 90, 90)}],
        10: [{"label": "stuffed toy", "bbox": (10, 10, 90, 90)}],
        20: [],
    })
    payload, _ = run_screen(store, "parachute", top_n=10, jev=FakeJev(exists=0.0, confirm=0.0))
    s = payload["screened"]
    assert s["total_frames"] == 3
    assert s["total_frames"] == s["yolo_hit"] + s["jev_called"]
    assert s["kept"] == s["total_frames"] - s["dropped"]
    assert s["max_jev_regions"] == 8          # #496: 截断上限必须出现在读数里


def test_screen_jev_region_calls_are_capped(tmp: Path) -> None:
    """档 B 逐区域问，但一次最多问 max_jev_regions 个（>6 个进打分器就崩）。"""
    regions = [{"label": f"obj{i}", "bbox": (5, 5, 30 + i, 30 + i)} for i in range(10)]
    store = build_index(tmp / "i", {0: regions})
    jev = FakeJev(confirm=0.0)
    payload, _ = run_screen(store, "dragon", max_jev_regions=4, jev=jev)
    assert payload["screened"]["jev_region_calls"] == 4
    assert len(jev.calls) == 4


def test_screen_zero_hit_rule_not_a_score_threshold(tmp: Path) -> None:
    """契约②：OOV 是「query 词零命中」，不是「最高分低于某门限」。

    证据：索引里有一个高分框（0.99）但 label 完全不匹配 query ⇒ 仍然走 Jev。
    """
    store = build_index(tmp / "i", {0: [{"label": "parachute", "score": 0.99,
                                        "bbox": (10, 10, 90, 90)}]})
    payload, _ = run_screen(store, "xylophone", jev=FakeJev(confirm=0.8))
    assert payload["screened"]["oov_rule"] == "query_miss"
    assert payload["screened"]["jev_called"] == 1
    assert payload["candidates"][0]["route"] == "jev_confirmed"


def test_screen_content_word_person_is_not_a_stopword(tmp: Path) -> None:
    """★ 建模#495：'person' 是实义词。丢掉它会让 'person' 的每一帧都判 OOV。"""
    assert "person" in query_terms("a person")
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (10, 10, 90, 90)}]})
    payload, _ = run_screen(store, "person", jev=FakeJev())
    assert payload["screened"]["yolo_hit"] == 1
    assert payload["screened"]["jev_called"] == 0


def test_screen_rejects_a_query_with_no_content_words(tmp: Path) -> None:
    """全是虚词的 query 无法用「零命中」判据，必须报错而不是把每帧判 OOV。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (10, 10, 90, 90)}]})
    with pytest.raises(VrsError):
        run_screen(store, "的 了 和", jev=FakeJev())


# ===========================================================================
# ② locate
# ===========================================================================

def test_locate_batches_without_padding(tmp: Path) -> None:
    """契约③：分批、不补齐。5 帧 + batch 8 ⇒ 一次前向 5 帧，输出没有 padded。"""
    store = build_index(tmp / "i", {f: [{"label": "x", "bbox": (5, 5, 40, 40)}]
                                    for f in (0, 10, 20, 30, 40)})
    v = FakeVeason()
    payload, _ = run_locate(store, "0,10,20,30,40", "dragon", veason=v)
    assert v.batch_sizes == [[0, 10, 20, 30, 40]]
    assert "padded" not in payload and "pad_to" not in payload
    assert payload["n_frames"] == 5
    assert payload["batch_size"] == 8


def test_locate_last_batch_may_be_short(tmp: Path) -> None:
    """7 帧 + batch 3 ⇒ [3,3,1]：最后一批短就是短，不补齐。"""
    ids = [0, 10, 20, 30, 40, 50, 60]
    store = build_index(tmp / "i", {f: [{"label": "x", "bbox": (5, 5, 40, 40)}] for f in ids})
    v = FakeVeason()
    run_locate(store, ",".join(map(str, ids)), "x", batch_size=3, veason=v)
    assert v.batch_sizes == [[0, 10, 20], [30, 40, 50], [60]]


def test_locate_dedupes_requested_ids(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: [{"label": "x", "bbox": (5, 5, 40, 40)}],
                                   10: [{"label": "x", "bbox": (5, 5, 40, 40)}]})
    v = FakeVeason()
    payload, _ = run_locate(store, "0,10,10,0", "x", veason=v)
    assert v.batch_sizes == [[0, 10]]
    assert [i["frame_id"] for i in payload["locate"]] == [0, 10]


def test_bboxes_rescale_from_560_and_reject_unseen_keyframe() -> None:
    ans = '<answer>{"keyframe_timestamp": 12, "bbox_2d_list": [[0,0,560,560]]}</answer>'
    out = _bboxes_from_answer(ans, [12, 13], {12: (1120, 560), 13: (1120, 560)})
    assert out[12] == [[0, 0, 1120, 560]]
    # ★ keyframe 不在这一批里 ⇒ 报错，不静默 fallback（那会伪造一个合法答案）。
    with pytest.raises(VrsError):
        _bboxes_from_answer(ans, [1, 2], {1: (10, 10), 2: (10, 10)})


def test_batches_rejects_zero_size() -> None:
    with pytest.raises(VrsError):
        batches([1, 2], 0)


# ===========================================================================
# ③ evidence
# ===========================================================================

def test_evidence_area_is_computed_zero_model(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 120, 120)}]})
    payload, _ = run_evidence(store, "0", "person", "area")
    e = payload["evidence"][0]
    assert e["axis"] == "area" and e["src"] == "computed"
    assert e["value"]["area_ratio"] == pytest.approx(0.3125, abs=1e-3)


def test_evidence_reads_cached_axis_with_true_src(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (10, 10, 90, 90)}]})
    store.save_senses(0, _senses_for(store, 0, {
        "0": {"face": {"expression": "laughing", "expression_score": 0.42}}}))
    payload, _ = run_evidence(store, "0", "laughing", "action")
    e = payload["evidence"][0]
    assert e["axis"] == "action"
    assert e["src"] == "clip-vitb32-expression"
    assert e["value"]["expression"] == "laughing"


def test_evidence_missing_axis_is_an_event_not_a_fabricated_value(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (10, 10, 90, 90)}]})
    payload, info = run_evidence(store, "0", "person", "occlusion")
    assert payload["evidence"] == []
    assert info["events"][0]["event"] == "evidence_axis_unavailable"
    assert info["events"][0]["axis"] == "occlusion"


def test_evidence_drops_stale_senses(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (10, 10, 90, 90)}]})
    stale = _senses_for(store, 0, {"0": {"depth": {"rank_in_frame": 1, "mean_rel": 0.7}}})
    stale["regions_sha256"] = "deadbeef"
    store.save_senses(0, stale)
    payload, info = run_evidence(store, "0", "person", "occlusion")
    assert payload["evidence"] == []
    assert any(e["event"] == "evidence_stale_senses_dropped" for e in info["events"])


# ===========================================================================
# ④ reflect
# ===========================================================================

def test_reflect_zoom_uses_veason_box(tmp: Path) -> None:
    """契约④：裁剪框来自 Veason（state['locate']），不是 YOLO 的 region 框。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    st = _state(store, "person", {"frame_id": 0, "region_idx": 0},
                locate=[{"frame_id": 0, "bbox_2d_list": [[5, 5, 150, 150]]}])
    payload, _ = run_reflect(store, 1, st, jev=FakeJev(confirm=None))
    zoom = [i for i in payload["new_info"] if i["kind"] == "zoom"]
    assert zoom, payload
    assert zoom[0]["bbox"] == [5, 5, 150, 150]        # Veason 的框
    assert zoom[0]["bbox_source"] == "veason-r1"
    assert Path(zoom[0]["crop"]).is_file()


def _no_senses(*a, **k):
    """测试缝：不让 reflect 去真跑 senses 模型。"""
    return None


def test_reflect_zoom_without_veason_box_is_not_a_silent_yolo_fallback(tmp: Path) -> None:
    """拿不到 Veason 框 ⇒ 报事件、不产 crop，绝不退回 YOLO。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    st = _state(store, "person", {"frame_id": 0, "region_idx": 0}, locate=[])
    payload, info = run_reflect(store, 1, st, jev=FakeJev(), senses_runner=_no_senses)
    assert payload["terminated"] is True
    assert payload["reason"] == "no_new_information"
    assert all(i["kind"] != "zoom" for i in payload["new_info"])
    assert any(e["event"] in ("reflect_zoom_unavailable", "reflect_zoom_target_skipped")
               for e in info["events"])


def test_reflect_zoom_skips_only_the_target_without_a_veason_box(tmp: Path) -> None:
    """★ 真机发现：一个候选缺 Veason 框，不能让整次放大丢掉决策帧的 crop。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}],
                                    20: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    cands = [{"frame_id": 0, "region_idx": 0, "area_ratio": 0.50, "score": 0.90},
             {"frame_id": 20, "region_idx": 0, "area_ratio": 0.48, "score": 0.88}]
    st = _state(store, "person", {"frame_id": 0, "region_idx": 0}, candidates=cands,
                locate=[{"frame_id": 0, "bbox_2d_list": [[5, 5, 150, 150]]}])
    payload, info = run_reflect(store, 1, st, jev=FakeJev(), senses_runner=_no_senses)
    zoom = [i for i in payload["new_info"] if i["kind"] == "zoom"]
    assert [z["frame_id"] for z in zoom] == [0], "决策帧的 crop 不该因为另一个目标缺框而丢"
    assert any(e["event"] == "reflect_zoom_target_skipped" and e["frame_id"] == 20
               for e in info["events"])


def test_reflect_crop_is_keyed_by_query(tmp: Path) -> None:
    """★ 建模#503：同一 (fid, region) 换 query 是一次新测量，不能因旧 crop 判「无新信息」。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    locate = [{"frame_id": 0, "bbox_2d_list": [[5, 5, 150, 150]]}]
    dec = {"frame_id": 0, "region_idx": 0}
    p1, _ = run_reflect(store, 1, _state(store, "a person", dec, locate=locate),
                        jev=FakeJev(), senses_runner=_no_senses)
    p2, _ = run_reflect(store, 1, _state(store, "a cat", dec, locate=locate),
                        jev=FakeJev(), senses_runner=_no_senses)
    c1 = [i for i in p1["new_info"] if i["kind"] == "zoom"][0]["crop"]
    c2 = [i for i in p2["new_info"] if i["kind"] == "zoom"][0]["crop"]
    assert c1 != c2 and Path(c1).is_file() and Path(c2).is_file()
    # 同一 query 再跑 = 旧信息，不再产 zoom。
    p3, _ = run_reflect(store, 2, _state(store, "a person", dec, locate=locate),
                        jev=FakeJev(), senses_runner=_no_senses)
    assert all(i["kind"] != "zoom" for i in p3["new_info"])


def test_reflect_has_no_round_limit(tmp: Path) -> None:
    """契约⑤：没有轮次上限。第 5 轮不该因 --round 大而被判 round_limit。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    st = _state(store, "person", {"frame_id": 0, "region_idx": 0})
    payload, _ = run_reflect(store, 5, st, jev=FakeJev(), senses_runner=_no_senses)
    assert payload.get("reason") != "round_limit"


def test_reflect_terminates_on_absent_premise(tmp: Path) -> None:
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    st = _state(store, "dragon", {"frame_id": 0, "region_idx": 0}, absent=True)
    payload, _ = run_reflect(store, 1, st, jev=FakeJev(), senses_runner=_no_senses)
    assert payload["terminated"] is True
    assert payload["reason"] == "premise_false"


def test_reflect_new_axis_reads_a_cache_axis(tmp: Path) -> None:
    """换轴：读一个此前没进过 evidence 的缓存轴，产出一条带 measurement 的新信息。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    store.save_senses(0, _senses_for(store, 0, {
        "0": {"depth": {"rank_in_frame": 1, "mean_rel": 0.8}}}))
    st = _state(store, "person", {"frame_id": 0, "region_idx": 0},
                locate=[{"frame_id": 0, "bbox_2d_list": [[5, 5, 150, 150]]}])
    # 第一轮：zoom 是主力，先产出 crop。
    p1, _ = run_reflect(store, 1, dict(st), jev=FakeJev(), senses_runner=_no_senses)
    assert any(i["kind"] == "zoom" for i in p1["new_info"])
    # 第二轮：crop 已存在 ⇒ zoom 不再产新信息，落到换轴。
    p2, _ = run_reflect(store, 2, dict(st), jev=FakeJev(), senses_runner=_no_senses)
    infos = p2["new_info"]
    assert infos, p2
    assert infos[0]["kind"] == "new_axis"
    assert infos[0]["axis"] == "occlusion"
    assert infos[0]["measurement"]


def test_reflect_new_axis_degrades_when_a_model_is_missing(tmp: Path) -> None:
    """某个轴的小模型没装 ⇒ 记事件、跳过、不把整条链打死（其余轴仍试）。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    st = _state(store, "person", {"frame_id": 0, "region_idx": 0},
                locate=[{"frame_id": 0, "bbox_2d_list": [[5, 5, 150, 150]]}])

    def boom(*a, **k):
        raise RuntimeError("No module named 'rapidocr_onnxruntime'")

    run_reflect(store, 1, dict(st), jev=FakeJev(), senses_runner=_no_senses)  # 先把 zoom 做掉
    payload, info = run_reflect(store, 2, dict(st), jev=FakeJev(), senses_runner=boom)
    assert payload["terminated"] is True
    assert payload["reason"] == "no_new_information"
    bad = [e for e in info["events"] if e["event"] == "reflect_axis_unavailable"]
    assert bad and "rapidocr" in bad[0]["error"]


def test_reflect_rejects_bad_decision_types(tmp: Path) -> None:
    """★ 建模#514：agent 手写 state ⇒ 类型错误必须变成 VrsError，不是 TypeError。"""
    store = build_index(tmp / "i", {0: [{"label": "person", "bbox": (20, 20, 60, 60)}]})
    bad = _state(store, "person", {"frame_id": "abc", "region_idx": 0})
    with pytest.raises(VrsError):
        run_reflect(store, 1, bad, jev=FakeJev())
    bad2 = _state(store, "person", {"frame_id": None, "region_idx": 0})
    with pytest.raises(VrsError):
        run_reflect(store, 1, bad2, jev=FakeJev())


def test_reflect_state_parses_file_and_inline(tmp: Path) -> None:
    p = tmp / "s.json"
    p.write_text(json.dumps({"query": "q", "decision": {"frame_id": 0, "region_idx": 0}}),
                 encoding="utf-8")
    assert parse_state(str(p))["query"] == "q"
    assert parse_state('{"query": "q2"}')["query"] == "q2"


# ===========================================================================
# 契约面：命令行的旗子、README、分层
# ===========================================================================

def test_removed_flags_are_gone() -> None:
    _, screen_help, _ = run("screen", "--help")
    _, locate_help, _ = run("locate", "--help")
    _, reflect_help, _ = run("reflect", "--help")
    for gone in ("--oov-rule", "--oov-threshold", "--jev-backend"):
        assert gone not in screen_help, gone
    for gone in ("--pad-to", "--veason-backend"):
        assert gone not in locate_help, gone
    for gone in ("--max-rounds", "--crop-source", "--jev-backend", "--axis-backend"):
        assert gone not in reflect_help, gone
    _, ev_help, _ = run("evidence", "--help")
    assert "--axis-backend" not in ev_help


def test_pipeline_flags_are_gone() -> None:
    """删掉的旗子必须是**无法解析**的，而不只是从文档里消失。"""
    for gone in ("--pad-to", "--crop-source", "--max-reflect", "--oov-rule"):
        p = subprocess.run([PY, PIPELINE, gone, "8", "--video", "x", "--query", "y"],
                           capture_output=True, text=True)
        assert p.returncode != 0, f"{gone} 仍被 argparse 接受"
        assert "unrecognized" in (p.stderr or "").lower() or "error" in (p.stderr or "").lower()


def test_no_injectable_backend_module_and_no_cpu_only_claim() -> None:
    assert not (ROOT / "vrs_skills" / "backends.py").exists()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "**CPU only.**" not in readme and "No CUDA anywhere" not in readme


def test_indexer_is_byte_frozen() -> None:
    import hashlib

    got = hashlib.sha256((ROOT / "vrs_skills" / "indexer.py").read_bytes()).hexdigest()
    assert got == INDEXER_SHA256
