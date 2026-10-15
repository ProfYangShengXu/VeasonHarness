# syntax=docker/dockerfile:1
#
# VeasonHarness —— 复现环境
#
# 设计：镜像里只装【环境】，不装【权重】。
#   权重加起来 20GB+（Veason-R1 8.15G / RSI-Jev 9.1G / SAM2 1.5G / 其余轴 ~0.5G），
#   塞进 layer 会让镜像没法分发。运行时用 volume 挂到 /models。
#
#   所以这个镜像开箱能跑的是：单元测试 + 建索引（CPU）+ `vrs` 的纯计算子命令；
#   要跑完整五环（①②③ 需要 GPU + 大权重），把权重挂进来即可。
#
# 用法见 docker/README.md。

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# ---------- 运行镜像 ----------
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VRS_MODELS_DIR=/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev \
        python3-pip \
        git curl ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用 layer 缓存：代码改动不必重装依赖）
COPY pyproject.toml README.md ./

# ★ 顺序铁律：torch 必须【第一个装】，且必须用 cu126 源。
#   踩过两次：只要 pip 命令里出现任何依赖 torch 的包（timm / ultralytics / transformers），
#   pip 就会顺手从默认源拉一个 torch —— 实测拉成 CUDA 13 版（cudnn-cu13/nccl-cu13），
#   白下 2GB 且和宿主 CUDA 12.6 对不上。
#   先把 cu126 的 torch 装掉，后面的包看到 "Requirement already satisfied" 就不会再拉。
#
# ★ 网络：download.pytorch.org 上的大 wheel（torch 主包 869MB）在国内经常下到一半断连，
#   而且断在半路时 pip 不会触发 timeout（卡在 socket read 上）。三种绕法，任选：
#     ① 直接用官方源多试几次（--retries 打不过 socket 静默挂起，效果有限）
#     ② 换镜像：mirrors.aliyun.com/pytorch-wheels/cu126 或 mirror.sjtu.edu.cn/pytorch-wheels/cu126
#     ③ ★ 推荐：用 curl 断点续传先把 wheel 扒到 docker/wheels/，build 时会自动优先用它：
#          mkdir -p docker/wheels && cd docker/wheels
#          B=https://download.pytorch.org/whl/cu126
#          curl -C - -O $B/torch-2.14.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl
#          curl -C - -O $B/torchvision-0.29.0%2Bcu126-cp310-cp310-manylinux_2_28_x86_64.whl
#        中断了就把同一条命令再跑一遍，-C - 会接着下。
COPY docker/wheels/ /tmp/wheels/
RUN if ls /tmp/wheels/*.whl >/dev/null 2>&1; then \
        echo "==> 用 docker/wheels/ 里预下的 wheel 装 torch"; \
        python -m pip install --no-index --find-links /tmp/wheels "torch==2.14.0" "torchvision==0.29.0"; \
    else \
        echo "==> docker/wheels/ 是空的，从 pytorch.org 拉"; \
        python -m pip install \
            --index-url https://download.pytorch.org/whl/cu126 \
            --timeout 180 --retries 15 \
            "torch==2.14.0" "torchvision==0.29.0"; \
    fi

RUN python -m pip install \
        --timeout 180 --retries 15 \
        "numpy==2.2.6" \
        "pillow==12.3.0" \
        "scipy==1.15.3" \
        "opencv-python-headless" \
        "onnxruntime==1.23.2" \
        "pycocotools==2.0.11" \
        "einops==0.8.2" \
        "timm==1.0.30" \
        "safetensors==0.8.0" \
        "iopath==0.1.10" \
        "portalocker==4.4.0" \
        "hydra-core" \
        "omegaconf" \
    && python -m pip install \
        --timeout 180 --retries 15 \
        "transformers==5.17.0" \
        "accelerate==1.15.0" \
        "peft==0.21.0" \
        "ultralytics==8.4.177" \
    && python -m pip install "pytest" "pytest-timeout"

# SAM-2（官方 repo 的 pip 包名是 sam2；官方 wheel 在 PyPI 上叫 sam-2）
RUN python -m pip install "sam-2" || \
    (git clone --depth 1 https://github.com/facebookresearch/sam2.git /opt/sam2 && \
     python -m pip install -e /opt/sam2)

# 再放代码
COPY . .

# 权重目录（挂载点）
RUN mkdir -p /models /data /out
VOLUME ["/models", "/data", "/out"]

# 冒烟自检：不依赖权重，验证包能 import、CLI 能起来
RUN python -c "import numpy, torch, cv2, onnxruntime; print('core ok')" \
    && python vrs.py --help >/dev/null && echo "cli ok"

ENTRYPOINT ["python", "vrs.py"]
CMD ["--help"]
