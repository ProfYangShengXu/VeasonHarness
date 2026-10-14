---
name: veason-harness
description: 用一句话在视频里找目标（第几帧、画面哪里）时用。先建索引，再走五环链拿 region id，别自己猜坐标。
---

# VeasonHarness：让 agent 在视频里"指哪打哪"

**何时用**：有视频，有一个自然语言描述（"那只爬上去又滑下来的熊猫"），要回答
**"这件事发生在第几帧、目标在画面哪个位置"**。

**不要用它做**：视频理解/摘要/问答（那是 LVLM 的活）。这里只做**定位**。

---

## 第 0 步：确认它装好了

```bash
python -m veason_harness --help      # 零安装路径，clone 下来就有
veason --help                        # 或者 pip install -e . 之后
```

两条都通就行。**不需要 GPU 也能建索引** —— 只有 ② locate 需要模型权重。

---

## 第 1 步：建索引（每个视频一次）

```bash
python -m veason_harness index <video.mp4> --frames 16
# → {"index_dir": "...", "frames": [{"frame_id": 357, "t": 29.75, ...}], ...}
```

★ **`frame_id` 是视频解码帧号，不是"我抽的第 N 帧"**。`--frames 16` 在 110 帧的视频上
会给出 `0, 7, 15, 22, …` 这样的 id。后面所有命令都用这个 id。

★ **索引是无提示词的**：建的时候不知道你要找什么。这是刻意的 ——
用评测表达式去建索引，指标就自己泄给自己了。

## 第 2 步：走五环链

```bash
python veason_pipeline.py --video <video.mp4> --query "<指代表达>" \
       --synonym "<外观名词短语>" [--index <dir>] [--decide-from <file>]
```

```
⓪ rewrite   ★ 它在这里【停下来等你】—— 你必须给出外观名词短语
① screen    初筛候选帧
② locate    在候选帧上精细定位（需要模型）
③ evidence  多轴取证
④ decide    ★ 由你选：哪一帧、哪个 region_idx
⑤ reflect   没新信息就停（唯一正常出口）
```

**⓪ 是硬性的**：不给改写结果就退出码 3 并打 `pipeline_rewrite_missing`。
不要试图绕过它 —— 因为绕过之后的 `yolo_hit = 0` 和"画面里真没有这个东西"
长得一模一样，两种情况处置完全不同。

## 第 3 步：要答案

```bash
veason answer <index_dir> <frame_id> <region_idx> [<region_idx> ...]
```

★ **`answer` 是唯一把 id 变成框的命令。** 别的命令都只回 `frame_id` + `region_idx`，
所以**你引用一个自己没看过的区域在结构上不可能发生**。

交付前自检：

```bash
veason verify <index_dir> <frame_id> <region_idx>   # 退出码 4 = 对不上
```

---

## 别做的事

- ❌ **别自己编坐标**。你没有坐标，只有 id。`answer` 会把 id 转成框。
- ❌ **别跳过 ⓪**。缺改写是硬停，不是警告。
- ❌ **别把降级读数当模型读数**。看 `route` 字段：
  `*_kept` 结尾说明模型**根本没被调用**（权重缺失），结果是"如实标注的降级"。
- ❌ **别并发跑同一个索引目录上的写命令**。没有跨进程锁，会丢更新。
- ❌ **别把空结果当失败**。`[]` + 退出码 0 = "确实没有"，这是合法答案。
- ❌ **别用 `recall` 的阈值当通用阈值**。那是标定出来的，换素材要重标。

## 排错

| 现象 | 处置 |
|---|---|
| `veason: command not found` | 用 `python -m veason_harness`（零安装路径） |
| 全部 `route: *_kept` | 模型权重缺失 ⇒ 走了降级。设 `VEASON_MODELS_DIR` 指向权重根目录 |
| `pipeline_rewrite_missing` | ⓪ 步没拿到改写结果。补 `--synonym` 或 `--rewrite-from` |
| `BackendUnavailable` | 该后端要的依赖没装（`tools/setup_detect.sh`），或权重不在 |
| `verify` 退出码 4 | 交付的答案与当前区域表对不上 —— 表被改过了 |
| 退出码 2 + hint | 参数/前置条件问题，按 hint 修 |
