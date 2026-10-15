# 用 Docker 复现 VeasonHarness

镜像里装的是**环境**，不是**权重** —— 权重合计 20GB+，打进 layer 会让镜像没法分发。
运行时挂载到 `/models` 即可。

## 0. 一句话

```bash
docker build -f docker/Dockerfile -t veason-harness:latest .
docker run --rm -v /path/to/models:/models veason-harness:latest --help
```

## 1. 这个镜像开箱能跑什么

| 能力 | 需要什么 | 说明 |
|---|---|---|
| 单元测试 | 无 | `docker run --rm --entrypoint python veason-harness -m pytest -q tests/` |
| 建索引（`vrs index`） | YOLOE 权重（~50MB） | CPU 就能跑，不需要 GPU |
| `vrs frames/regions/show/crop` | 一个已有索引 | 纯读 IO |
| 完整五环（①②③） | GPU + Veason-R1/Jev/SAM2 权重 | 见 §3 |
| ④ 决策 | 一个支持图像的 LLM API key | 见 §4 |

## 2. 装权重

权重全部落在 `models/`（已在 `.gitignore` / `.dockerignore` 里，不会进仓库也不会进镜像）。

```bash
# ① 两个 YOLOE 检查点 + relsgg（国内走 hf-mirror）
bash tools/setup_models.sh

# ② 证据轴的小模型（depth / videomae / mediapipe / pose）
python tools/fetch_round3_models.py

# ③ 三个大件（需自行获取，放到位即可）
#    models/Veason-R1/                  Veason-R1，约 8.15 GB
#    models/RSI-Jev-v6.1-vl-4b/         初筛用的 Jev，约 9.1 GB
#    models/sam2.1_hiera_tiny.pt        SAM-2 tiny，约 156 MB
#    models/sam2.1_hiera_small.pt       SAM-2 small，约 73 MB
```

> SAM-2 的两个 `.pt` 从 `facebookresearch/sam2` 的 release 下载；
> `Veason-R1` 与 `RSI-Jev` 请按各自发布页面获取。**本仓库不含任何权重。**

## 3. 跑完整五环

```bash
docker run --rm -it --gpus all \
  -v /path/to/models:/models \
  -v /path/to/videos:/data \
  -v /path/to/out:/out \
  veason-harness:latest \
  index /data/clip.mp4 --frames 16 --out /out/idx
```

`--gpus all` 需要宿主机装了 `nvidia-container-toolkit`。
没装 GPU 支持时，`vrs index` 的 CPU 路径仍然可用（YOLOE 在 CPU 上慢，但能出结果）。

## 4. 决策那一步

`④ decide` 不调本地模型 —— 它把「测量表 + 候选帧图像」交给一个 MLLM，
由后者选帧并给出理由。这一步需要你自己提供一个支持图像输入的 API key：

```
DEEPSEEK_API_KEY=...          # deepseek-flash（实测能吃图）
# 或任何 OpenAI 兼容的视觉模型
```

> 注意：`deepseek-v4-pro` **不吃图**（实测返回"无法判断"），要用 `deepseek-flash`。

## 5. 用 compose（推荐）

```bash
# 先把 .env.example 复制成 .env，填上 models 路径和 key
cp docker/.env.example .env
docker compose -f docker/docker-compose.yml run --rm vrs --help
```

## 6. 验证装对了

```bash
# CLI 能否起来（build 时已经自检过一次）
docker run --rm veason-harness:latest --help

# 单元测试（不需要权重）
docker run --rm --entrypoint python veason-harness:latest -m pytest -q tests/

# 有权重时的冒烟：建一个小索引
docker run --rm -v /path/to/models:/models -v /path/to/videos:/data \
  veason-harness:latest index /data/<某个短 mp4> --frames 8 --out /out/smoke
```

## 7. 已知问题（都实测踩过）

### torch 被装成 CUDA 13 版

```
症状：pip list 里出现 nvidia-cudnn-cu13 / nvidia-nccl-cu13，容器里 torch.version.cuda == "13.0"
原因：pip 命令里只要出现依赖 torch 的包（timm / ultralytics / transformers），
      pip 就会顺手从【默认源】拉一个 torch —— 那是 CUDA 13 版
修法：torch 必须【第一个装】且带 --index-url .../whl/cu126。
      本 Dockerfile 已经这么排了；你改依赖时别把顺序打乱。
```

### download.pytorch.org 下到一半断连

```
症状：build 卡在 "Downloading torch-2.14.0+cu126-...whl (869.1 MB)" 十几分钟不动，
      日志 mtime 不再更新，容器里 pip 进程 CPU 时间几乎为 0。
★ 关键：这种静默挂起【不会触发 --timeout】，加 --retries 也救不了。
修法（推荐第 3 种）：
  ① 多试几次（碰运气）
  ② 换镜像源：
       mirrors.aliyun.com/pytorch-wheels/cu126
       mirror.sjtu.edu.cn/pytorch-wheels/cu126
  ③ 提前用 curl 断点续传扒 wheel 到 docker/wheels/，build 会自动优先用它：
       mkdir -p docker/wheels && cd docker/wheels
       B=https://download.pytorch.org/whl/cu126
       curl -C - -O $B/torch-2.14.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl
       curl -C - -O $B/torchvision-0.29.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl
     断了就重跑同一条命令，-C - 会接着下。
```

### 构建上下文过大

```
症状：docker build 开头 "transferring context: 367MB"
原因：仓库里有图片/权重/索引
修法：.dockerignore 已经排掉了 models/ index/ 和编译缓存。别把权重放进仓库。
```

## 8. 构建验证状态

```
已实测通过：
  · base image（nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04）能拉下来
  · apt 层（python3.10 / ffmpeg / libgl1 / libglib2.0-0）装得上
  · 依赖解析正确 —— torch 走 cu12 线（cuda-toolkit==12.6.3 / nvidia-cublas-cu12 /
    cuda-bindings<13,>=12.9.4），torch wheel 落在 torch-2.14.0+cu126

未完成：
  · 完整 build 未跑完 —— 卡在 torch 主 wheel（869MB）下载静默挂起，
    用上面的 §7 第 ③ 种方式可绕过。诚实起见标注在这里。
```

## 9. 已知的边界

```
· 镜像基于 CUDA 12.6 runtime（对齐验证环境：Ubuntu 20.04 / Python 3.10 /
  torch 2.14.0+cu126 / transformers 5.17.0 / ultralytics 8.4.177）
· 只装了 CPU 侧也需要的依赖；`mediapipe` / `rapidocr_onnxruntime` / `relsgg`
  这三个轴的可选件【没有装】—— 缺件时对应轴会被跳过并记事件，不影响主链路
· 权重一律不打包；镜像大小 ≈ CUDA runtime + Python 依赖（数 GB）
```
