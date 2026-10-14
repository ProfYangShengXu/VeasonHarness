# VeasonHarness

**给 LLM agent 的一双眼睛**：喂它一段视频和一句话（"那只爬上去又滑下来的熊猫"），
它告诉你**这件事发生在第几帧、目标在画面哪个位置**。

名字由来：`Veason` 是这套流程里做定位的那只模型（Veason-R1），`Harness` 是它外面的那圈
脚手架 —— 模型负责"看"，脚手架负责"别让它瞎看"。**两者分开才是这个项目的重点**：
模型换掉、脚手架不动；脚手架改口径、模型不用重训。

```
视频 ──► 建索引(一次) ──► 五环链(每个问题跑一遍) ──► 一个框

        YOLOE 无提示词              ① screen   初筛出少数候选帧
        理解画面的分区             ② locate   在候选帧上做精细定位
                                    ③ evidence 从不同角度取证
                                    ④ decide  选一帧、选一个区域
                                    ⑤ reflect 没新发现就停
```

**它不做理解。** 理解交给外面的 agent；这里只做**确定性计算 + 可审计的中间产物**。

---

## 实测跑分

不是"我们相信它能用"，是**真跑出来的数字**。数据来自 ReVOS 验证集（真实视频 + 真实指代表达），
模型是 Veason-R1（3B，8.15 GB，bf16），单张 RTX 3090，每题 8 帧一次前向。

```
题                       类型              耗时    选帧      判定
① 跑→跳→两滚→停在垫上    时序/动作链       21.5s   第 16 帧   ✗ 框偏左约 26 px
② 背对镜头的黑熊         目标中途消失      11.1s   第 0 帧    ✓ 框住目标
③ "the only man."       被遮挡 + 极简     10.6s   第 0 帧    ✓ 框住目标
④ 属大熊猫科的动物       4 实例 + 要常识   11.9s   第 0 帧    ✓ 给出 2 个框
⑤ "the primate."        对照臂（简单）     8.2s   第 0 帧    ✓ 框住目标（框只占画面 1.5%）
                                                           ⇒ 4/5
```

**⑤ 最值得看** —— 那个视频的主体是两只跑动的斑马，而题面问的"灵长类"是**左下角一只
不起眼的小猴**。模型的框是 `[17, 318, 91, 399]`，真值点是 `[60, 354]`：**完整包住，而且框只
占画面 1.5%**（不是撒一张大网蒙过去）。

**④ 也值得看** —— 题目描述的是"属大熊猫科的动物"（要常识推出是熊猫），画面里有 4 个实例，
模型给了 2 个框，两个都命中了真值点。

**① 差在哪** —— 选帧是对的（题目说"最后停在垫子上"，它选了最后一帧），但框的中心偏左约
26 px、没能包住真值点。**这是"帧的判断对、像素级定位差一点"**，和 ②③ 的失败模式不同。

> ⚠️ 这 5 题是**故意挑的难例**（见下方"实验任务报告"），**不代表整体精度**。
> 想看基线对比，看同一节里的常数策略得分。

---

## 架构设计

### 一、分成两层：脚手架 + 模型

```
┌─────────────────────────── 脚手架（确定性，可测） ───────────────────────────┐
│                                                                             │
│   veason_pipeline.py   编排：在【哪些点】需要 agent 做决定，显式写出来           │
│   veason_harness/cli.py 装配点：唯一的依赖注入处                              │
│                                                                             │
│   ① screen   ─ ② locate ─ ③ evidence ─ ④ decide ─ ⑤ reflect                 │
│                                                                             │
│   索引层  store.py / video.py / paths.py   区域表只追加、不重排                │
│   契约层  errors.py                        stdout 一个 JSON，stderr 一个 JSON  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                     只通过 backend 协议接触（可替换）
                                    ▼
┌─────────────────────────── 模型（可换，接线在 backends.py）──────────────────┐
│  YOLOE 11s-seg      无提示词建索引 + 可提示词重查                               │
│  Veason-R1 (3B)     关键帧定位                                                │
│  CLIP / depth / pose / relation   证据轴                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

**为什么这条线要画在中间**：模型是会换的（换权重、换版本、换供应商），而**编排逻辑不该跟着变**。
`backends.py` 是唯一的接线层，`auto` 解析成"同进程真件"，权重缺失时抛**可捕获**的
`BackendUnavailable`，调用方按降级处理 —— 于是**同一份代码在有 GPU 的机器上真跑、在没有 GPU
的机器上仍然可测**。

### 二、五环：每一环的输入输出都是落盘的 JSON

| 环 | 干什么 | 为什么单列一环 |
|---|---|---|
| ⓪ rewrite | agent 把自然语言问题改写成外观名词短语 | 检测器只认外观、不认关系（YOLOE 官方原话：*a prompt describes appearance, not relationships*）。**这一步缺了就直接停**，不静默跑成"零命中" |
| ① screen | 从全部帧里初筛出少数候选 | 后面每一环都贵，先砍数量 |
| ② locate | 在候选帧上做精细定位 | 唯一产框的一环 |
| ③ evidence | 从深度/关系/表情/动作等轴取证 | 单一视角会误判，多轴可以互相证伪 |
| ④ decide | 选一帧、选一个区域 | **这是 agent 的自由度所在**，不是规则 |
| ⑤ reflect | 检查"这一轮有没有拿到新信息" | 没有新信息就停 —— 这是唯一的正常出口 |

### 三、三条不显然的设计判断

**1. `route` 是证据类型，不是自信度。**

初版把候选按"区域面积"排序，注释写着"面积大 = 目标看得清，绝不看谁更自信"。实测直接翻车：

```
48 个候选，YOLO 命中的 11 帧全被埋在 rank 20 以后
真目标（frame 850）排第 20，而 top-6 全是 area≈0.87 的【近乎整个画面】的框
```

因为**背景永远比目标大** —— 按面积排，背景永远赢。修法不是调权重，而是**分两级**：
先按**证据类型**分层（有名字的 `yolo_hit` / `jev_confirmed` 是强证据层，`*_kept` 是弱层），
**层内**再按面积（"面积大 = 看得清"这条理由在层内是成立的）。
关掉分层做反证，输出与修前**逐字相同**。

**2. 反思不设次数上限，靠"有没有新信息"收敛。**

反思如果靠一个常数兜底（`MAX_ROUNDS = 2`），那停下来的原因是"跑够了次数"，不是"想清楚了"。
改成：**每一轮必须产出新的测量值**（每条 `new_info` 都要写出 `measurement` 字段），
于是"把旧信息重排一遍"在结构上就产不出 `new_info` ⇒ 循环必然收敛。
`fuse_rounds` 保留为**保险丝**（默认 1000）而不是正常出口，触发时明确标注"这是保险丝"。
实测：第 500 轮仍然只因 `no_new_information` 而停。

**3. 缺改写是硬停，不是警告。**

⓪ 步如果没拿到改写结果，流程**退出码 3 并打出 `pipeline_rewrite_missing` 事件**。
不给它"打条警告继续跑"的选项 —— 因为继续跑的结果是 `yolo_hit = 0`，
**和"画面里真没有这个东西"长得一模一样**。两种情况的处置完全不同，混起来就再也查不出来了。
（想显式跳过可以传 `--no-rewrite`，但事件照打。）

### 四、对外的硬契约

```
· 非交互      没有任何 input()、任何确认提示
· stdout     一个 JSON 文档；错误走 stderr，也是一个 JSON，带 hint，退出码非零
· 区域 id    在同一个索引内是永久的：region #3 在后续每次调用里都是同一个框
· 答案形态   agent 永远不打坐标 —— 它从工具输出里挑 frame_id + region_idx，
             answer 是唯一把 id 变成框的命令
```

最后一条是关键：**它让"引用一个自己没看过的区域"在结构上不可能，而不只是不推荐。**

---

## 实验任务报告

### 一、为什么要专门设计题目

选帧类任务的指标有个陷阱：**"永远选第 0 帧"这条死规则本身得分就很高**。
我们第一版挑的 5 道题，**每一帧上目标都在场** ⇒ "选对帧"这一项恒为真 ⇒
死规则拿了 5/5，**这个指标没有任何区分度**。

**判据（写在前面）：下"模型不会做 X"这个结论之前，先算常数策略的得分。**
死规则 ≈ 模型 ⇒ 你测的是数据的先验，不是模型的能力。

### 二、题目怎么挑的

从 ReVOS ground 训练集 24,068 条里筛**目标不全程在场**的（5,142 条），再按失败模式分桶：

| 题 | 失败模式 | 目标在场 | 为什么挑它 |
|---|---|---|---|
| ① | 时序/动作链 | 15/18 帧，第 3 帧才出现 | 题目是"跑过去→跳→翻两滚→停在垫子上"，**必须理解动作的先后**，不是认物体 |
| ② | 目标中途消失 | 91/110 帧，后半段没了 | 背对镜头的黑熊 —— 目标"在画面里"但**看不到正面** |
| ③ | 被遮挡 + 极简表达 | 86/110 帧（有断档） | 题面只有 "the only man."，**没有任何外观线索**，只能靠唯一性消歧 |
| ④ | 多实例 + 要常识 | 100/180 帧 | "属大熊猫科的动物"需要**先知道大熊猫是啥**，画面里有 4 个实例 |
| ⑤ | 对照臂（简单） | 9/9 帧全场 | **用来判断"系统是不是根本不能用"** |

**⑤ 的作用**：如果 ①~④ 全挂而 ⑤ 过 ⇒ 失败是**任务难度**；如果 ⑤ 也挂 ⇒ **先修系统**，
别谈瓶颈。这次 ⑤ 过了，所以前面几道的结果才可解释。

### 三、判据（三级，逐级收紧）

```
① 选对帧      frame_id 必须在"该帧目标在场"里
② 框得准      bbox 含该帧的真值点
③ 框不是撒网   bbox 面积 ≤ 画面 50%

命中 = ① ∧ ② ∧ ③
```

第三条是**必须的**：没有它，"给一个全画面的大框"永远满足前两条。

### 四、读数与失败解剖

```
① 21.5s  选帧对  ✗ 框偏左 26 px        ← 帧的判断对、像素级定位差
② 11.1s  选帧对  ✓ 框住（2.4% 面积）
③ 10.6s  选帧对  ✓ 框住（5.2% 面积）
④ 11.9s  选帧对  ✓ 2 个框都命中（24.7% 面积）
⑤  8.2s  选帧对  ✓ 框住（1.5% 面积，真值点 [60,354] 在框 [17,318,91,399] 内）
```

**四种失败模式，这次只踩到第一种**：帧判断对但框偏一点。
另外三种（选错帧 / 帧上目标不在场 / 框撒太大）本次**没有触发** —— 这是下次要扩大样本去压的。

### 五、这轮暴露的三个缺陷（都是"本机测不出来"的类型）

这轮是**第一次把代码放到真 GPU 上跑**，暴露的三个问题有共同特征：
**本机是 CPU 环境，所以它们在本机永远测不出来。**

| 缺陷 | 本机为什么测不出来 | 修法 |
|---|---|---|
| `tensor.numpy()` 崩 | CPU 上 tensor 在内存里，`.numpy()` 正常 | `.cpu().numpy()`（同一个文件里其它 3 处**本来就写对了**，就漏了这一处） |
| 负 stride 报错 | 同上，且 pytorch CPU 路径更宽容 | `np.ascontiguousarray()` 包住翻转结果（仓库里另一处**本来就写对了**） |
| 模型答非所问 | 权重缺失时走降级路径，压根没进模型 | 接上官方 prompt 口径 —— 不接的话模型会退化成"视频描述" |

**第三个最值得记**：同一份权重、同一个视频，

```
没有 prompt：  "The video depicts zebras running across a grassy field…"   ← 视频描述
接上 prompt：  选第 0 帧 + 框 [17,318,91,399]                              ← 正是那只猴
```

**换个提问口径，行为完全不同** —— 而我们差点把这个当成"模型不行"。

> **推广**：接真模型时，"跑通了"和"跑对了"是两件事。
> **先确认它真的进了模型**（看是否走了降级路径），再看它答得对不对。

### 六、复现

```bash
# 建索引（CPU 就能跑）
python -m veason_harness index <video.mp4> --frames 16

# 五环链（需要 GPU + 权重）
python veason_pipeline.py --video <video.mp4> --query "<指代表达>" --synonym "<外观短语>"
```

题目来自 ReVOS 验证集，可自行下载；**本仓库不含任何数据集**（见下方"数据与许可"）。

---

---

## 0. Cheat sheet (copy-paste)

```bash
# Windows shim:      veason.cmd <args>
# WSL shim:          ./veason.sh <args>
# Direct:            <python312> veason.py <args>

cd <你的克隆目录>

cd <你的克隆目录>
#   -> {"index_dir": "...", "frames": [{"frame_id":357,"t":29.75,...}], ...}

IDX=<你的克隆目录>/index/people-detection-18ffe867

./veason.sh frames  "$IDX"                 # which frames exist
./veason.sh regions "$IDX" 357             # candidate regions of frame 357  -> LIST
./veason.sh seg     "$IDX" 357 0           # overlay image -> {"out": "<abs path>", ...}
./veason.sh crop    "$IDX" 357 0           # zoom in on one region
./veason.sh answer  "$IDX" 357 0 1 2       # official VRS payload  -> 2 keys only
./veason.sh verify  "$IDX" 357 0 1 2       # re-check the answer (exit 4 = mismatch)

./veason.sh detect  "$IDX" 357 "a red dress" "a person"   # open-vocabulary re-query
./veason.sh relate  "$IDX" 357 "riding" "holding"         # relation vocab AT INFERENCE
./veason.sh recall  "$IDX" --label "person" --area-min 0.1 --top-k 8 --explain   # multi-route recall (RRF)
./veason.sh senses  "$IDX"                                # compute the face/action/depth/text/pose axes
./veason.sh recall  "$IDX" --label "yellow" "dragon" --expression laughing --explain   # a state, from disk
./veason.sh show    "$IDX" 357                            # whole frame as PNG
```

Paths may be given in either WSL form (`/mnt/c/...`) or Windows form
(`C:\...`); both are accepted everywhere. Every path the CLI *prints* comes in
both forms (`out` / `out_win`, `image` / `image_win`): the `*_win` variant is
for Windows tools, the other for a WSL shell.

---

## 1. Commands and their JSON

### `veason index <video> [--frames N] [--out DIR] [--conf C] [--topk K] [--with-relations]`

Decodes `N` frames (evenly spaced, endpoints inclusive) and runs the
**prompt-free** YOLOE over each (its built-in 4,585-class LVIS vocabulary —
no caller vocabulary is involved, so nothing leaks from an evaluation
expression).  It writes the frame images, region tables and masks.

**Relations are not computed here by default.**  An index build cannot know
which predicates will be asked about, and RelateAnything is the expensive
model (~0.93 s/frame vs the detector's ~0.29 s/frame), so relation graphs are
backfilled on demand by `recall --relation` and stored per vocabulary.
`--with-relations` restores the old compute-everything behaviour for callers
that really want a full default-vocabulary graph in the index.

`--frames` is the cost knob and the cost is linear; default `16`.
Default `--out` is `<repo>/index` (not the shell's cwd), so the command behaves
the same from any directory.

```jsonc
{
  "index_dir": "/mnt/c/.../index/people-detection-18ffe867",  // pass this to every other command
  "index_dir_win": "C:\\...\\people-detection-18ffe867",
  "video": "/mnt/c/.../people-detection.mp4",
  "video_sha256": "18ffe8672d74...",   // cache key: same video => same index dir
  "n_frames": 16,                      // frames actually indexed
  "n_video_frames": 596,               // frames in the file
  "fps": 12.0,
  "duration_s": 49.667,
  "width": 768, "height": 432,
  "reused": false,                     // true => nothing was recomputed
  "frames": [                          // frame table, also returned by `veason frames`
    {"frame_id": 357, "t": 29.75, "n_regions": 12,
     "image": "/mnt/.../frames/f000357.png",
     "image_win": "C:\\...\\frames\\f000357.png",
     "regions_file": "/mnt/.../regions/f000357.json"}
  ],
  "detector": {"model": "yoloe-11s-seg-pf.pt", "mode": "prompt-free-lvis4585",
               "conf": 0.25, "imgsz": 640, "pipeline": 2},
  "relation": {"model": "relsgg-vits16plus", "mode": "on-demand", "index_vocabulary": null},
  "timings": {...},                    // see §4 — measured cost, not a guess
  "size_mb": 5.25
}
```

With `--with-relations`, `relation.mode` is `"index-time"`,
`relation.index_vocabulary` names the default predicate list, and `timings`
also has `model_load_relater_s` / `relate_s_per_frame`.  Without it,
`timings.frames_related` is always present and `0` — the key is not omitted,
so "no relations" is an explicit fact rather than a missing field.

**Reusing an index.** The index directory is named `<video-stem>-<sha256[:8]>`,
and the cache key of each frame is its decoded video id. Re-running `index` on
the same video reuses every frame whose detector settings still match, so:

* same command again → `"reused": true`, ~0.4 s, no model loaded;
* `--frames 32` on top of a 16-frame index → only the *newly sampled* ids are
  computed (note: the 16- and 32-frame grids only share their endpoints);
* changing `--conf` (or the pipeline version) invalidates the detector output
  and rebuilds those frames.

### `veason frames <index_dir>`

`{"index_dir", "index_dir_win", "video", "fps", "duration_s", "n_frames", "frames": [...]}`
— same `frames` entries as `index`. `n_regions` is re-read from disk, so it
reflects any later `detect`.

### `veason regions <index_dir> <frame_id>`

A **JSON list** (not an object) of the frame's candidate regions, in
`idx` order:

```jsonc
[{"idx": 0, "label": "person", "score": 0.9632,
  "bbox": [182, 109, 322, 401],        // integer pixels, xyxy, origin top-left
  "area_ratio": 0.123216,              // bbox area / frame area
  "has_mask": true,                    // false => `seg` will degrade to box-only
  "source": "yoloe-pf-lvis4585"}]      // or "yoloe-prompted" after `detect`
```

`idx` is the position in the table and never changes. Two calls return
byte-identical output (acceptance C).

### `veason detect <index_dir> <frame_id> <phrase> [<phrase> ...] [--conf 0.10]`

Runs the **promptable** YOLOE with exactly the phrases you pass (open
vocabulary, e.g. `"red dress"`), then merges the hits into the frame's table.

**Returns the newly added regions only** (same shape as `regions`), so:

* nothing matches → `[]`, exit `0` (an empty result is a result, not an error);
* a hit duplicates an existing region (same label, IoU ≥ 0.8) → `[]`, because
  the existing `idx` is reused — repeated `detect` is idempotent;
* a hit has a *different* label than the existing region on those pixels →
  it is a new hypothesis and gets the next `idx`;
* a table that changed invalidates the frame's relation graph.

### `veason relate <index_dir> <frame_id> <relation> [<relation> ...] [--topk 20] [--min-score 0.0]`

Runs RelateAnything with the predicates you pass — the phrases are encoded
into the vocabulary head before inference (`RelateAnything.set_vocabulary`),
so this is not a filter over a fixed list.  A *new* vocabulary means a new
model load (~14 s here); `recall --relation` pays it once and reuses the
resulting graph on later calls.

```jsonc
[{"subject": 0, "relation": "riding", "object": 6, "score": 0.0539}]
```

`subject`/`object` are **region idxs**, not positions in the list you passed.
A frame with fewer than 2 regions yields `[]` (RelateAnything cannot form a
pair). If the table exceeds 60 regions only the 60 highest-scoring ones are
used, and a JSON `warning` is written to stderr — the cap is never silent.
Each frame's file holds one graph per predicate vocabulary (`sets`), so a
`relate` with vocabulary A does not destroy a `recall` backfill with
vocabulary B.

### `veason senses <index_dir> [--axes AXIS…] [--force]` / `veason senses <index_dir> --frame-id N`

The **state / action / space / text / pose** layer: five axes that the detector
cannot express, computed once per frame by small local models and cached in
`senses/f<frame_id>.json`.  Like relation graphs, this is *not* part of `index`:
`veason index` stays at 0.29–0.60 s/frame, and the senses pass (≈3–9 s/frame
depending on axes) is opt-in.  `veason recall` reads the files and loads no model.

```jsonc
// veason senses <idx> --frame-id 850  (one region's slice of the real file)
"regions": {"0": {
  "face":   {"n_faces": 0, "expression": "laughing", "expression_score": 0.4949,
             "expression_scores": {"laughing": 0.4949, "smiling": 0.1514, ...},
             "expression_src": "clip-vitb32-expression-zeroshot",
             "src": "mediapipe-face-landmarker"},
  "action": {"label": "yoga", "score": 0.1166, "top": [...], "src": "videomae-small-kinetics",
             "note": "clip-level label over the 16 sampled frames centred here; copied to every region"},
  "depth":  {"mean_rel": 0.8408, "p50_rel": 0.8518, "rank_in_frame": 1,
             "src": "depth-anything-v2-small"},
  "track":  {"track_id": 2, "dx_per_s": -4.491, "dy_per_s": -5.988, "src": "bytetrack"},
  "pose":   {"kpt17": [[x,y,conf] × 17], "arms_raised": false, "score": 0.63, "src": "yolo11n-pose"},
  "text":   []
}}
```

* `face` has **two providers**.  MediaPipe Face Landmarker supplies the
  calibrated blendshapes (`jaw_open`, `mouth_smile_l/r`, `cheek_squint_l/r`,
  `eye_squint_l/r`, plus every other blendshape, snake_cased).  The zero-shot
  expression head supplies `expression` / `expression_scores` from a **fixed
  9-prompt bank** over the mask-aware region crop; it is the provider that
  reads a stylised cartoon face, which is outside BlazeFace's domain.  Each
  block says which one produced the numbers.
* `action` is **clip-level** (VideoMAE sees the 16 sampled frames centred on
  this one) and is copied to every region of the frame; the block says so.
* `depth` is relative inverse depth, min-max normalised **per frame**, so
  `mean_rel` is only comparable within a frame; `rank_in_frame` (1 = nearest)
  is the scale-free fact.
* `pose.track` gives the ByteTrack id and the centre velocity (`dx_per_s` /
  `dy_per_s`) against that track's previous sampled frame.
* A frame's file records the `regions_sha256` it was computed against.  A later
  `detect` that changes the table invalidates the file (same rule as
  relations); `recall` then simply has no senses for that frame.
* `--axes` recomputes only the named axes and **merges** them into the existing
  file, so `veason senses --axes depth` never throws away an earlier face pass.
* `--frame-id N` prints the stored file for one frame and computes nothing.

### `veason recall <index_dir> [--label W...] [--relation P...] [--area-min 0.05] [--frame-min N] [--frame-max M] [--top-k 12] [--no-clip] [--no-face] [--face-smile-min F] [--jaw-open-min F] [--expression WORD] [--expression-min F] [--action W...] [--action-min F] [--depth-near] [--depth-min F] [--text W...] [--text-min F] [--track ID] [--arms-raised] [--explain]`

Retrieval over the index.  **It takes structured fields, not natural
language** — the LLM turns "the laughing yellow dragon" into
`--label yellow "cartoon character" --relation "smiling at"` before calling.
The CLI never does NLU, so a query is reproducible and its evidence is
auditable.  `--label` / `--relation` accept several words each and may be
repeated.

Five routes over the region table, each named after the field it reads:

| route | field it uses | cost |
|---|---|---|
| `label` | `region.label` | free text match (phrase-coverage scored) |
| `relation` | `relations/f*.json` | RelateAnything, **backfilled on demand** |
| `semantic` | region crop (mask-aware) | CLIP ViT-B/32 image/text cosine |
| `geometry` | `bbox` / `area_ratio` / frame range | free filter + ranking |
| `cooccurrence` | relation edges | free: neighbours of an already-hit region |

Six more routes read the per-frame **sense files** written by `veason senses` — they
load no model, because the models ran once at senses time:

| route | sense field | flag |
|---|---|---|
| `face` | `face.mouth_smile_l/r`, `face.jaw_open` | `--face-smile-min F` / `--jaw-open-min F` |
| `expression` | `face.expression_scores` | `--expression WORD` / `--expression-min F` |
| `action` | `action.label` / `action.score` | `--action WORD…` / `--action-min F` |
| `depth` | `depth.mean_rel` / `depth.rank_in_frame` | `--depth-near` / `--depth-min F` |
| `text` | `text[].s` | `--text WORD…` / `--text-min F` |
| `pose` | `track.track_id` / `pose.kpt17` | `--track ID` / `--arms-raised` |

`--no-face` switches the whole face axis off (landmarker *and* expression head);
the other five routes are active only when their flag is given. A sense route on
an index that has no `senses/` contributes nothing and `recall` writes a
`recall_senses_unavailable` event to stderr naming `veason senses` as the fix.

Ranks are fused with **Reciprocal Rank Fusion**, `score = Σ 1/(60 + rank)`
(Cormack et al., SIGIR 2009).  RRF consumes ranks, not scores, so a cosine in
[-1, 1] and a detector confidence in [0, 1] need no hand-tuned weights.
`--explain` adds each route's rank and raw score, which is how an LLM can see
*which* route a candidate came from:

```jsonc
[{"frame_id": 850, "region_idx": 0, "score": 0.031,
  "hits": ["sem:0.28", "area:0.61"],          // the `why`, per route
  "label": "parachute", "bbox": [196,92,1100,717], "t": 14.167, "area_ratio": 0.613064,
  "routes": {"semantic": {"rank": 3, "raw": 0.2794}, "geometry": {"rank": 7, "raw": 0.6131}}}]
```

* **Empty is a result.**  No route above its threshold → `[]` and exit `0`.
  A "closest thing" is never invented.  The semantic route has a floor
  (`--sem-min`, default `0.24`, calibrated so `purple elephant` on the demo
  clips stays empty at ~0.20 while `yellow dragon` passes at ~0.27).
* **The relation route backfills per frame and per vocabulary.**  Only frames
  the cheap routes already consider plausible (`--relate-max-frames`, default
  16) are related, and a graph is reused only when both the region table hash
  and the predicate vocabulary match.  A JSON ledger of what was computed vs
  reused is written to **stderr** (`{"event": "recall_relation_backfill", ...}`),
  because stdout is the result list.
* `--no-clip` skips the semantic route entirely (no model load); `--frame-min` /
  `--frame-max` bound `frame_id`; `--relation-min` floors the relation scores.

### `veason show <index_dir> <frame_id> [--out PATH]`
### `veason crop <index_dir> <frame_id> <region_idx> [--out PATH]`

Write a PNG and return `{"out": "<wsl path>", "out_win": "<windows path>", "bbox": [...]}`
(`crop` adds `region_idx`, `label`). Default location is
`<index_dir>/outputs/`.

### `veason seg <index_dir> <frame_id> <region_idx> [--out PATH]`  ← the deliverable picture

Original frame + translucent mask fill + contour, plus the `#idx label` tag.
With `--out` you choose the file; without it, `<index_dir>/outputs/`.

```jsonc
{"out": "/mnt/.../outputs/seg_f000357_r0.png",
 "out_win": "C:\\...\\seg_f000357_r0.png",
 "bbox": [182, 109, 322, 401],
 "mask_source": "yoloe-seg",     // or "box-only" when the region has no mask
 "region_idx": 0, "label": "person"}
```

`mask_source` is **never** faked: if there is no usable mask the command draws
a box and says `"box-only"`. `sam2` is not implemented; the field can only take
`yoloe-seg` or `box-only` today.

### `veason answer <index_dir> <frame_id> <region_idx> [<region_idx> ...]`

Exactly the official VRS output — **only these two keys, nothing else**:

```jsonc
{"keyframe_timestamp": 30,                    // INTEGER SECONDS, not a frame id
 "bbox_2d_list": [[182,109,322,401], [366,166,457,383], [434,195,582,430]]}
```

* `keyframe_timestamp = floor(t + 0.5)` where `t = frame_id / fps`.
  **Not `round()`**: Python rounds halves to even, so `round(2.5) == 2` and a
  frame at 2.5 s would land a second early while a frame at 3.5 s would not.
* Order of `bbox_2d_list` = order of the `region_idx` arguments.
* Multiple instances = multiple `region_idx` arguments.

A snapshot (the answer plus the region-table hash it was built from) is written
to `<index_dir>/answers/f<frame_id>_r<idx-...>.json`.

### `veason verify <index_dir> <frame_id> <region_idx> [...]`

Re-checks the *delivered* answer against the region table **as it is now**, so
the check can actually fail. Exit `0` when everything matches, `4` when it does
not, `2` when no answer snapshot exists yet.

```jsonc
{"ok": true, "frame_id": 357, "region_idxs": [0,1,2],
 "snapshot": "/mnt/.../answers/f000357_r0-1-2.json",
 "checks": [
   {"check": "regions_file_unchanged_since_answer", "ok": true, "snapshot_sha256": "...", "current_sha256": "..."},
   {"check": "answer_reproducible_from_current_regions", "ok": true,
    "per_idx": [{"region_idx": 0, "snapshot_bbox": [182,109,322,401],
                 "regions_bbox": [182,109,322,401], "match": true}]},
   {"check": "keyframe_timestamp_consistent", "ok": true, "snapshot": 30, "current": 30},
   {"check": "mask_within_region_bbox", "ok": true, "tolerance_px": 2, "per_idx": [...]}
 ]}
```

On failure the payload gains `error` and `hint` (both also present, with
`codes`/`checks`, on stderr-free stdout — `verify` is the one command whose
failure payload is on stdout, because the caller needs to read *which* check
failed).

The last check is an independent fact: the segmentation mask must live inside
the box it belongs to. It exists because it once did not — see §7.

### Errors

`stdout` stays empty; `stderr` gets one JSON object; the process exits non-zero.

```jsonc
{"ok": false, "where": "seg",
 "error": "region_idx 99 does not exist on frame 357 (valid: 0..11)",
 "hint": "run `veason regions <index_dir> 357` to list the candidates."}
```

Codes: `2` usage / wrong arguments, `3` runtime or environment,
`4` verification failed. Add `--debug` **before** the subcommand to get a real
traceback when debugging the CLI itself.

---

## 2. What lands on disk

```
index/people-detection-18ffe867/
  index.json                  meta: video hash, probe, frame table, timings
  frames/f000357.png          extracted frame (the source of truth for artifacts)
  regions/f000357.json        candidate table (append-only)
  masks/f000357.npz           bool masks, keyed by region idx
  relations/f000357.json      relation graphs, one per predicate vocabulary (optional)
  senses/f000357.json         face/action/depth/text/pose fields (optional, `veason senses`)
  answers/f000357_r0-1-2.json answer snapshot used by `verify`
  outputs/seg_*.png crop_*.png show_*.png
```

`regions/`, `masks/` and `senses/` are the files whose change can invalidate other
artifacts: a relation graph carries the `regions_sha256` it was scored against, a
senses file carries the same hash (a `detect` that appends a region makes it
stale), and `answers/` keeps a copy of the delivered boxes, which is what makes
`verify` meaningful.

A `relations/f*.json` file is a container: `sets[<vocab sha>]` is one graph per
predicate vocabulary, each with its own `edges`/`info`.  The top-level
`relation`/`edges` mirror the newest set (so a flat file written by an older
`relate` still reads), and a changed `regions_sha256` invalidates the whole
container.  Being keyed on *both* hashes is what makes the on-demand backfill
safe: two vocabularies never alias each other, and `filter`-only reuse is
impossible.

---

## 3. Models, dependencies, setup

Everything below is already installed here; the commands are for rebuilding.

| what | where / how |
|---|---|
| YOLOE prompt-free `yoloe-11s-seg-pf.pt` (28 MB) | GitHub release `ultralytics/assets` v8.3.0 → `models/` |
| YOLOE promptable `yoloe-11s-seg.pt` (28 MB) | same release → `models/` |
| RelateAnything `maelic/relsgg-vits16plus` (310 MB) | **hf-mirror.com**, not huggingface.co |
| MobileCLIP text tower `mobileclip_blt.ts` (572 MB) | GitHub release, needed by `detect` only |
| OpenAI CLIP `ViT-B-32.pt` (338 MB) | `openaipublic.azureedge.net` (auto-downloaded by `veason recall`, cached in `models/ultralytics_weights/clip/`) |
| MediaPipe Face Landmarker `face_landmarker.task` (3.6 MB) | Google storage → `models/mediapipe/`; pip `mediapipe` |
| VideoMAE-small Kinetics `pytorch_model.bin` (88 MB) | **hf-mirror.com** `MCG-NJU/videomae-small-finetuned-kinetics` → `models/videomae-small-kinetics/` |
| Depth Anything V2-Small `model.onnx` (99 MB) | **hf-mirror.com** `onnx-community/depth-anything-v2-small` → `models/depth-anything-v2-small/` |
| yolo11n-pose `yolo11n-pose.pt` (6.3 MB) | GitHub release `ultralytics/assets` v8.4.0 → `models/ultralytics_weights/` |
| RapidOCR ONNX | `pip install rapidocr-onnxruntime` — models ship inside the 27 MB wheel |
| `relsgg` package | `pip install -e third_party/RelateAnything --no-deps` |
| `clip` (ultralytics fork) | `tools/setup_detect.sh` — see below |
| `lap` (1.4 MB) | ByteTrack's assignment solver; ultralytics auto-installs `lap>=0.5.12` on first `track` |

```bash
bash tools/setup_models.sh             # YOLOE ×2 + RelateAnything + relsgg
bash tools/setup_detect.sh             # click fork + mobileclip_blt.ts (veason detect only)
<py312> tools/fetch_round3_models.py   # round-3 weights (urllib, not curl)
pip install mediapipe rapidocr-onnxruntime
```

Why the setup scripts do unusual things:

* **`huggingface.co` is unreachable here** (curl times out); `hf-mirror.com`
  answers. `RelateAnything.from_pretrained()` would hang, so the CLI loads the
  local checkpoint with `from_checkpoint()` instead.
* **`relsgg` is installed with `--no-deps`.** Its `pyproject.toml` pins
  `transformers~=5.14`, `numpy~=2.5`, `Pillow~=12.3`; this machine has
  5.9.0 / 2.2.4 / 11.1.0. A plain `pip install -e .` would try to move all
  three under a working ultralytics + torch install. The code imports cleanly
  against what is installed.
* **`clip` cannot be installed by ultralytics' auto-installer** (it shells out
  to `git.exe`, which does not read the Windows system proxy at
  `127.0.0.1:7890`; Python's `urllib`/`pip` do). `tools/setup_detect.sh`
  downloads the tarball through `urllib` and installs it locally.
* **`mobileclip_blt.ts` is 572 MB** and is only required by `veason detect`.
  If it is missing, `detect` fails with a hint naming the setup script;
  `index`/`regions`/`seg`/`answer` are unaffected.  No model or asset here
  exceeds 1 GB, and no `HTTP_PROXY`/`HTTPS_PROXY` is ever set.
* **The semantic route uses OpenAI CLIP, not the MobileCLIP already on disk.**
  `mobileclip_blt.ts` is in fact the *full* MobileCLIP (149.7M params,
  `image_encoder` + `text_encoder`), but it was traced for the text tower only:
  the image encoder carries no scripted `forward`, so it cannot embed a region
  crop (verified by loading the TorchScript module and calling it).  CLIP is
  already installed for `detect` and its 338 MB weights download once into
  `models/ultralytics_weights/clip/`, so the semantic route adds no new
  *dependency* — only a weight file.  `--no-clip` runs recall with no extra
  model at all.

Python: `<你的 python>`
(3.12.4). Installed: torch 2.14.0+cpu, torchvision 0.29.0, ultralytics
8.4.150, transformers 5.9.0, opencv 4.13.0, numpy 2.2.4, Pillow 11.1.0,
onnxruntime 1.26.0. `timm` / `open_clip` were **not** needed and are not
installed.

---

## 4. Measured cost (Intel Ultra 5 125H, CPU only, 16 GB)

Fixed process/import cost is ~0.2 s. Per-command wall time, cold process:

| command | measured | notes |
|---|---:|---|
| `index --frames 48` (fresh, 1280×720, **detect only**) | **39.8 s** | 2.7 s detector load + 0.44 s/frame detect + ~16 s decode/PNG write; `frames_related: 0` |
| `index --frames 48 --with-relations` (fresh, same video) | **91.7 s** | adds the relation phase: 10.2 s relater load + 0.92 s/frame |
| `index` (same video, same settings) | **~0.4 s** | everything reused, no model loaded |
| `recall` label/geometry only (`--no-clip`) | **0.21 s** | pure disk reads |
| `recall` with the semantic route (96 crops) | **~15 s** | ~3.5 s CLIP load + ~9 s crop/encode |
| `recall --relation` first call, 8 frames | **~21 s** | ~13 s relater load + 0.92 s/frame; later frames reuse at ~0 s |
| `detect <fid> <phrase>` | **8.5 s** | ~7 s is loading the 572 MB MobileCLIP tower; ~1.1 s inference |
| `relate <fid> <phrase...>` | **16.8 s** | ~14 s model load, ~0.9 s/frame |
| `seg` | 0.36 s | |
| `show` / `crop` | 0.30 / 0.27 s | |
| `frames` / `regions` / `answer` / `verify` | 0.20–0.29 s | pure disk reads |

**Layering, measured.**  On the 48-frame demo clip the same build costs 39.8 s
with detection only and 91.7 s with `--with-relations` — the relation phase is
~52 s, i.e. **57 % of the build**, and the index does not need it: relation
graphs are query-dependent and `recall` backfills only the frames it considers
(8 of 48 in the acceptance run).  Note the 39.8 s floor is not the detector
alone (0.44 s/frame ≈ 21 s); decoding 48 frames and writing 48 PNGs is ~16 s on
this machine, so "remove the relation cost" saves ~52 s, it does not shrink the
build to ~14 s.

**The senses pass is a third, separate layer.**  It is ~8.8 s/frame, i.e. ~30x
the detector, so it is opt-in and never runs during `index`:

| stage | s/frame (16-frame face clip / 48-frame dragon) | notes |
|---|---:|---|
| `index` (prompt-free detect only) | 0.289–0.60 | decode + detector + PNG |
| `senses --axes face` | 1.65 | MediaPipe ≈0.02 s/frame + the CLIP expression head |
| `senses --axes text` | 1.31 | RapidOCR, whole frame |
| `senses --axes pose` | 0.33 | yolo11n-pose + ByteTrack |
| `senses --axes action` | 2.10 | VideoMAE-small, one 16-frame clip per frame |
| `senses --axes depth` | 2.32–2.62 | Depth Anything V2-Small ONNX at 518 short side |
| `senses` (all five, dragon 48 frames) | 8.79 | wall 421.8 s for 48 frames |
| `recall` sense routes | 0.25 / call | pure disk reads, **no model loaded** |

```
index (detect only)     ≈  3 s  + 0.83 s × --frames
index --with-relations  ≈ 13 s  + 1.75 s × --frames
```

Raise `--frames` when keyframe recall matters; the index is reused, so a
larger re-index only pays for the frames that are new.

---

## 5. Design decisions (and why)

1. **Answers are indexes, not coordinates.** `answer` is the only writer of
   boxes; every other command hands back `frame_id` + `region_idx`. The
   delivered box cannot be one the agent never inspected.
2. **Index with a prompt-free vocabulary.** The 4,585-class LVIS head is baked
   into the weights, so `index` never learns what is being searched for. A
   vocabulary built from the evaluation expressions would make its own metrics
   meaningless — that is the leak this design avoids. Caller phrases enter the
   system in exactly one place: `detect`.
3. **Region identity is append-only.** `idx` = position in the table; `detect`
   appends and reuses, never reorders. Verified byte-for-byte across calls.
4. **A box is stored once.** `regions[].bbox` is integer pixels written at
   index time, and `seg`/`answer`/`verify`/`crop` all read *that* array.
5. **`verify` compares two facts, not one fact with itself.** The last
   `answer` snapshot is checked against the live region table (plus the
   independent mask-vs-box check), so rebuilding or editing the table makes
   the check fail — demonstrated in acceptance D.
6. **Cost is measured, not assumed.** Every `index` run records per-frame
   timings with its sample count; a 1-frame top-up is not allowed to overwrite
   the benchmark from a 16-frame run.
7. **Empty is a result.** `detect` returning `[]` exits 0 and prints `[]`.
   The CLI never "helpfully" invents a region to avoid an empty answer.
8. **Layer the models by what a build can know.** An index build cannot know
   the query, so it runs only the detector; relation graphs are query-dependent
   and are backfilled per frame on demand, keyed on *both* the region-table hash
   and the predicate vocabulary. Computing all 35 predicates for every frame at
   index time was the most expensive model applied to the blindest decision.
9. **Recall consumes structured fields, not prose.** NLU happens in the LLM
   (which is good at it); the CLI stays pure, deterministic computation with no
   embedded semantics, so a query's `hits` can be audited route by route.
10. **Fuse ranks, not scores.** `recall` combines routes with RRF
    (`1/(k + rank)`, k = 60) because the routes' scores have different scales
    and meanings; summing them would smuggle in a hand-tuned weight.
11. **State is a stored field, not a judgement call.** "laughing" is a number
    (`face.expression_scores.laughing`) computed once by a small local model, so
    it can be thresholded, ranked and audited like any other route.  A query
    that needs it no longer depends on the LLM noticing it, and the models stay
    behind the opt-in `veason senses` pass.

---

## 6. Tests

```bash
<python312> tests/test_veason.py           # 91 checks, synthetic index, no models
<python312> tests/test_veason.py --slow     # 95 checks: also the model-backed reject paths
```

The suite is written so that each check has been seen to **fail** on the thing
it guards: a 2 px bbox tamper makes `verify` exit 4, a mask that escapes its
box fails `mask_within_region_bbox`, `keyframe_timestamp(2.5) == 3` is asserted
precisely because `round(2.5) == 2`, and `detect`'s empty answer is only
trustworthy because a control case (`"a person"`) does add a region.

The round-2 guards were mutation-tested, as the originals were: making
`find_relation_set` vocabulary-blind fails 2 checks, and replacing RRF with a
raw-score sum fails 2 more (`RRF top is the 3-route region, not any route's
rank-1`; `score is sum of 1/(k+rank)`) — so the new checks are not decorative.

The round-3 guards cover the new fields without loading a model: the synthetic
index gets hand-written `senses/*.json`, and 30 checks assert each sense route
surfaces the right frame and names itself in `hits`, that `--no-face` switches
the face axis off, that a senses file whose `regions_sha256` no longer matches
is ignored, and that `fuse` cannot silently drop a route it was given.

---

## 7. Known limitations / gotchas

* **`frame_id` is a decoded video frame index**, not "the 4th frame I asked
  for". `t = frame_id / fps`. This is what makes re-indexing with a different
  `--frames` reuse the cache, but it means `--frames 8` gives ids like
  `0, 85, 170, …` rather than `0..7`.
* **Prompt-free LVIS labels are noisy on this footage.** `people-detection.mp4`
  yields `person` on 8 of 16 sampled frames (13 boxes) but also `playroom`,
  `bathroom mirror` and similar scene words for whole-room boxes. That is the
  model, not a bug: `detect` exists precisely to refine a region with the
  right phrase.
* **`detect` works, and still often returns `[]`.** `"person"` fires at 0.94 on
  frame 40, `"mirror"` at 0.29, but `"shoe"` / `"a red dress"` find nothing at
  `--conf 0.10` — the open-vocabulary head is not a general text search engine.
  Lower `--conf` before concluding "not present".
* **`relate` never returns an empty list when ≥ 2 regions exist.** It returns
  the top-k triplets for whatever vocabulary you give it, with low scores if
  there is no support. Judge by `score`, not by presence.
* **Relation cap: 60 regions per frame.** Beyond that only the highest-scoring
  60 are used, and a stderr `warning` is emitted. The frame table still lists
  them all; `answer` on a truncated region is unaffected (it does not use the
  relation graph).
* **`verify` needs an `answer` snapshot.** Running it before `answer` is exit
  `2` with a hint, not a silent pass.
* **Re-running `answer` overwrites the snapshot.** If the region table changed
  after a submission, `answer` re-derives from the *current* table — that is
  the documented recovery path, but it means the previous snapshot is gone.
* **No cross-process lock.** Two concurrent `veason index` / `veason detect` runs on
  the same index directory can lose an `index.json` update (`n_regions` /
  `frames`); the per-frame files themselves are written atomically and cannot
  be corrupted. Call one mutating command at a time. The same applies to two
  `recall --relation` calls with *different* vocabularies: each appends its
  graph to the same per-frame container with a read-modify-write, so a race can
  drop one set (the region files and the other set survive). The affected work
  is a cache miss on the next call, not wrong data.
* **`recall` thresholds are calibrated, not universal.** `--sem-min 0.24` was
  fitted on this cartoony footage (true matches 0.25–0.29, absent objects
  0.18–0.21). Different material may need a different floor; too low and
  "purple elephant" stops returning `[]`, too high and real attributes drop out.
  The relation route has no reject semantics at all (`relate` returns top-k for
  any vocabulary), so `--relation` alone can never produce an empty result —
  only the semantic/label routes have a threshold that can.
* **The semantic route costs ~15 s per call** (CLIP load + up to
  `--sem-max-crops` 96 region crops, largest first). Crop embeddings are not
  cached on disk yet; `--no-clip` and `--sem-max-crops N` are the levers.
* **Candidate frames for the relation backfill are elected by the cheap
  routes.** With a label/semantic hit they are the top `--relate-max-frames`
  frames; with no cheap signal at all (e.g. `--relation` alone and `--no-clip`)
  they fall back to the largest-area frames, which is a heuristic — raise
  `--relate-max-frames` if the target may be small or early.
* **`recall` has no temporal smoothing.** A target spread over several frames
  comes back as per-frame hits; grouping them into a time range is the caller's
  job.
* **`vtest.avi` cannot be decoded here.** `cv2.VideoCapture` reports
  `isOpened == False` for it with this OpenCV build (a codec issue, not a path
  issue — `people-detection.mp4` and
  `head-pose-face-detection-female-and-male.mp4` both open fine). The CLI
  explains this and suggests re-encoding with ffmpeg.
* **`sam2` is not implemented.** `mask_source` is only ever `yoloe-seg` or
  `box-only`.
* **MediaPipe Face Landmarker is blind to stylised cartoon faces.** On
  `dragon.mp4` it reports **0 faces on all 48 frames** (and 0 on
  `people-detection.mp4`, where the faces are ~40 px); lowering the detection
  confidence to 0.05, cropping to a CLIP-located face ROI and sweeping
  0.1–2.0x scale all failed. On `head-pose-face-detection-female-and-male.mp4`
  it works (2 faces/frame, ~20 ms/frame). So `--face-smile-min` / `--jaw-open-min`
  are a real route on real faces but contribute nothing on the cartoon; the
  `expression` route (below) is what carries the cartoon.
* **The expression head is a fixed-bank CLIP classifier, not a face model.**
  It exists precisely because the landmarker cannot see this face. Its bank is
  9 prompts and its `--expression-min` default (0.35) is fitted to `dragon.mp4`;
  like `--sem-min` it is not universal.
* **VideoMAE/Kinetics labels are not trustworthy on the cartoon.** `dragon.mp4`
  gets `yoga` (0.12) / `walking the dog` (0.22) — lowest-confidence nonsense.
  It is a real field and it is honest about being clip-level, but do not read
  the cartoon's action from it.
* **RapidOCR finds phantom text.** It reports 0.85–0.91-confidence boxes on
  `people-detection.mp4` frames that contain no text (`口`), and `8` / `·` on
  cartoon frames. Treat `text[]` as "shapes that look like text", and raise
  `--text-min` if that matters.
* **`mean_rel` is per-frame only.** Depth Anything is a *relative* model and
  min-max normalisation is per frame, so a region's `mean_rel` in frame 1 says
  nothing about frame 2. Use `rank_in_frame` for anything cross-frame.
* **Senses are a per-frame read-modify-write with no cross-process lock**,
  exactly like the relation container: two concurrent `veason senses` runs on one
  index can lose one frame's update (a cache miss, not wrong data), and a
  `detect` racing a `veason senses` run leaves that frame's senses stale — which
  `recall` refuses to use (hash mismatch), so the affected work is a recompute.
* Default `--out` is resolved against the **repository**, not the shell's cwd,
  for direct calls; the WSL shim exports `VEASON_CWD` so a relative `--out` from a
  WSL shell resolves against that shell's cwd.

---

# 附：怎么装、怎么用、边界在哪

## 快速开始

### 零安装（clone 下来就能跑）

```bash
git clone https://github.com/<你的用户名>/VeasonHarness.git
cd VeasonHarness

# 建索引（CPU 就能跑，不需要 GPU、不需要下大模型）
python -m veason_harness index  your_video.mp4 --frames 16

# 看有哪些帧 / 有哪些区域
python -m veason_harness frames  index/your_video-<hash>
python -m veason_harness regions index/your_video-<hash> 357
```

### 装成命令

```bash
pip install -e .
veason index your_video.mp4 --frames 16
```

### 要跑精细定位（② locate）才需要的东西

```bash
bash tools/setup_models.sh     # 检测器权重（YOLOE + relation）
bash tools/setup_detect.sh     # YOLOE 的文本编码器（离线装 ultralytics 的 CLIP fork）

# 定位模型（Veason-R1）与视觉 Jev 需要自备，放到 $VEASON_MODELS_DIR（默认 ./models）
#   Veason-R1/      qwen2_5_vl 形态，约 8.15 GB
```

**没有 GPU 也能用**：`auto` 会解析成"同进程真件"，权重缺失时抛**可捕获**的
`BackendUnavailable`，调用方降级 —— 降级路径会**如实标注**（`route: *_kept`），不会假装是真结果。

> ⚠️ 降级读数**不能当模型读数用**。判据：看 `route` 字段。
> 走到 `jev_unavailable_kept` 说明模型根本没被调用。

---

## 给 agent 用

**这个项目的主要用户是 agent，不是人。** 所以仓库里带了一份 `SKILL.md`，
告诉 agent 怎么发现它、怎么用、什么时候该停下来问人。

```bash
# Hermes Agent
cp -r SKILL.md ~/AppData/Local/hermes/skills/<分类>/veason-harness/

# Claude Code
cp -r SKILL.md .claude/skills/veason-harness/

# dsh (DeepSeek Harness)
#   ★ frontmatter 只认 name + description，本仓的 SKILL.md 已只带这两个
mkdir -p ~/.dsh/skills/veason-harness && cp SKILL.md ~/.dsh/skills/veason-harness/
```

装上之后，agent 才知道有这条路可走：**别自己猜坐标，用 `answer` 走 id**。

---

## 数据与许可

**本仓库不含任何数据集、任何模型权重。** 这两类是永久性的仓库垃圾，而且多数有独立的许可。

| 部分 | 许可 |
|---|---|
| 本仓库的代码 | MIT（见 `LICENSE`） |
| ReVOS 数据集 | 见其官方发布页，**本仓库不包含、不转发** |
| YOLOE / MobileCLIP / CLIP fork | 各自的原许可（由 `tools/setup_*.sh` 从官方源获取） |
| Veason-R1 | 见其官方发布页，**需自行获取** |

用第三方数据训练/评测产出的模型，其许可**不随本仓库的 MIT 而放宽**。

---

## 设计出处与致谢

这套流程不是凭空来的，几个关键判断来自这些工作：

- **Veason-R1** —— ② locate 这一环用的定位模型；其"8 帧一次前向、每帧加 `(帧号 s)` 前缀、
  560 缩放"的推理口径直接决定了解析逻辑。
- **YOLOE** —— ① screen 的主力。选它是为了**无提示词建索引**（见"设计决策"第 2 条）；
  它文档里那句 *a prompt describes appearance, not relationships* 直接催生了 ⓪ 步的存在。
- **ReVOS / MeViS / Ref-YouTube-VOS / MOSE / UVO / OVIS** —— 验证数据的来源，
  也是"目标不全程在场"这个筛选条件的出处。

方法论上受两条影响最深：

- **"先算常数策略的得分"** —— 任何"模型不会做 X"的结论之前，先做一遍死规则。
- **"声明必须有送达路径"** —— 没有读者的声明只是注释。这条直接导致了 `route` 字段、
  `pipeline_rewrite_missing` 事件、以及降级必须如实标注这些设计。

---

## 不做什么

明确画清边界，省得你试了才发现：

- **不做理解**。自然语言到结构化字段的那一步在 agent 侧完成，CLI 不内置语义。
- **不做端到端的"喂视频出答案"**。中间产物（候选帧、区域表、证据轴）**全部落盘**，
  因为出错时必须能定位到**是哪一环的判断错了**。
- **不做跨进程锁**。同一个索引目录上并发跑两个写命令会丢更新，请一次跑一个。
- **不做"智能补全"**。查不到就是空结果且退出码 0，不会"贴心地"造一个区域出来。
- **不做训练/微调**。这是推理侧的脚手架。
- **不打包权重和数据**。理由见上一节。

---

## 许可证

MIT。见 `LICENSE`。

代码与数据分开授权，**数据部分的许可不因代码的 MIT 而改变**。
