#!/usr/bin/env bash
# 一键准备运行环境：克隆官方 FlashVSR+（固定 commit）+ 建 venv + 装依赖 + 拉权重
#
#   ./scripts/bootstrap.sh                 # 装 CPU/自动检测 CUDA 的 torch
#   TORCH_CUDA=cu124 ./scripts/bootstrap.sh
#   ./scripts/bootstrap.sh --no-weights    # 只装代码，权重自己拉
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# 官方仓库固定在实测通过的 commit 上（换版本请改这两个变量并重跑冒烟测试）
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/OpenImagingLab/FlashVSR_Plus.git}"
UPSTREAM_SHA="${UPSTREAM_SHA:-b548a2cc2c92e5cd5bb5f1a41b6b0b0d5f2b6c2b}"
THIRD="$ROOT/third_party/FlashVSR_Plus"
PY="${PYTHON:-python3}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
WANT_WEIGHTS=1
[ "${1:-}" = "--no-weights" ] && WANT_WEIGHTS=0

echo "== 1/4 官方代码 → $THIRD"
mkdir -p "$ROOT/third_party"
if [ -d "$THIRD/.git" ]; then
  git -C "$THIRD" fetch --depth 1 origin main || true
else
  git clone --filter=blob:none "$UPSTREAM_URL" "$THIRD"
fi
git -C "$THIRD" checkout -q main
if [ "$UPSTREAM_SHA" != "HEAD" ]; then
  git -C "$THIRD" checkout -q "$UPSTREAM_SHA" 2>/dev/null \
    || echo "!! 固定 commit $UPSTREAM_SHA 取不到，用 main 最新（$(git -C "$THIRD" rev-parse --short HEAD)）"
fi
echo "   上游 commit: $(git -C "$THIRD" rev-parse --short HEAD)"

echo "== 2/4 venv"
[ -d "$ROOT/.venv" ] || "$PY" -m venv "$ROOT/.venv"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install -q -U pip wheel

echo "== 3/4 依赖"
if ! python -c "import torch, torchvision" 2>/dev/null; then
  echo "   装 torch（$TORCH_CUDA）…"
  python -m pip install -q --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" torch torchvision
fi
python -m pip install -q -r "$THIRD/requirements.txt" || true
python -m pip install -q flask
python - <<'PY'
import torch, flask
print(f"   torch {torch.__version__} cuda={torch.cuda.is_available()} | flask {flask.__version__}")
PY

echo "== 4/4 权重"
if [ "$WANT_WEIGHTS" = "1" ]; then
  if python -c "import sys; sys.path.insert(0,'$THIRD'); import model_io" 2>/dev/null; then
    python - <<PY
import sys, subprocess
sys.path.insert(0, "$THIRD")
# 上游自带 model_io 时优先用它拉权重
try:
    from model_io import download  # noqa: F401
    print("   上游提供 model_io.download —— 按上游 README 的命令拉取 turbo/base")
except Exception:
    print("   未找到 model_io.download")
PY
  fi
  echo "   权重目录应为: $THIRD/ckpts/{turbo,base}/{dit.safetensors,lq_proj_in.safetensors}"
  echo "   没自动下到的话，按上游 README 的 HuggingFace 说明手动放到上面那两个目录"
else
  echo "   跳过（--no-weights）"
fi

cat <<EOF

完成。启动：
  FVSR_ENGINE=real FVSR_REPO=$THIRD FVSR_PY=$ROOT/.venv/bin/python \\
  FVSR_DATA=$ROOT/data FVSR_PORT=7860 \\
  $ROOT/.venv/bin/python $ROOT/app.py

冒烟测试：
  ./scripts/smoke_test.sh
EOF
