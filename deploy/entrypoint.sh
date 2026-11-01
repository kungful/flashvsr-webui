#!/usr/bin/env bash
# 通用启动脚本（裸机 / 任意云厂商 / 容器都能用）
#   FVSR_PORT   监听端口（默认 7860）
#   FVSR_DATA   产物与上传目录（默认 ./data，**别放在代码目录里**，否则重新部署会覆盖）
#   FVSR_ENGINE real | mock
#   FVSR_REPO   官方 FlashVSR_Plus 路径（默认 ./third_party/FlashVSR_Plus）
#   FVSR_PY     跑推理的解释器（默认本脚本的 python）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export FVSR_PORT="${FVSR_PORT:-7860}"
export FVSR_DATA="${FVSR_DATA:-$ROOT/data}"
export FVSR_ENGINE="${FVSR_ENGINE:-real}"
export FVSR_REPO="${FVSR_REPO:-$ROOT/third_party/FlashVSR_Plus}"
export FVSR_PY="${FVSR_PY:-$( [ -x "$ROOT/.venv/bin/python" ] && echo "$ROOT/.venv/bin/python" || command -v python3 )}"
export FVSR_UPSTREAM="${FVSR_UPSTREAM:-}"
mkdir -p "$FVSR_DATA/outputs" "$FVSR_DATA/uploads"

echo "engine=$FVSR_ENGINE port=$FVSR_PORT data=$FVSR_DATA"
echo "repo=$FVSR_REPO"
exec "$FVSR_PY" "$ROOT/app.py"
