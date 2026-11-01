# FlashVSR WebUI —— 通用镜像（RunPod / Vast.ai / AutoDL / 阿里云 都能直接跑）
# 构建： docker build -f deploy/docker/Dockerfile -t flashvsr-webui .
# 运行： docker run --gpus all -p 7860:7860 -v $PWD/data:/data flashvsr-webui
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    FVSR_PORT=7860 \
    FVSR_DATA=/data \
    FVSR_ENGINE=real

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git ffmpeg curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/flashvsr-webui
COPY . .

# 官方代码 + 依赖（torch 用 cu124 轮子）
RUN git clone --filter=blob:none https://github.com/OpenImagingLab/FlashVSR_Plus.git third_party/FlashVSR_Plus \
 && python3 -m venv .venv \
 && .venv/bin/pip install -q -U pip wheel \
 && .venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cu124 torch torchvision \
 && .venv/bin/pip install -q -r third_party/FlashVSR_Plus/requirements.txt || true
RUN .venv/bin/pip install -q flask

# 权重（6.4 GB/模式）在构建时拉会让镜像巨大，建议运行时挂载到 /weights：
#   -v /host/ckpts:/opt/flashvsr-webui/third_party/FlashVSR_Plus/ckpts
VOLUME ["/data", "/opt/flashvsr-webui/third_party/FlashVSR_Plus/ckpts"]

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s CMD curl -sf http://127.0.0.1:7860/api/hardware || exit 1
CMD ["bash", "deploy/entrypoint.sh"]
