#!/bin/bash
# 补齐 / 修复 FlashVSR+ 的权重（镜像里一般已经全套，这里用于换机器或重装后补）
#
#   bash fetch_ckpts.sh            # 缺什么补什么（turbo + base + v1 + TCDecoder）
#   bash fetch_ckpts.sh turbo      # 只补 turbo
#   N=64 bash fetch_ckpts.sh       # 改并行段数（默认 48）
#
# 为什么用 dl_parallel.py 而不是 hf download：
#   本机房到 HuggingFace / hf-mirror 单连接只有 0.13~0.17 MB/s（文件走 Xet），
#   6.4 GB 要 13 小时；48 段并行能到 8.9 MB/s（约 12 分钟）。hf_transfer 同思路，
#   但它依赖 hf 端点/鉴权，这个纯 curl 分段，还能失败换源。
set -uo pipefail
CK=/opt/FlashVSR_Plus/ckpts
PY=/root/miniconda3/bin/python
N="${N:-48}"
WANT="${1:-all}"
HERE="$(cd "$(dirname "$0")" && pwd)"
DL="$HERE/dl_parallel.py"
[ -f "$DL" ] || DL=/root/dl_parallel.py
[ -f "$DL" ] || { echo "找不到 dl_parallel.py（跟本脚本放同一目录，或放 /root/）"; exit 1; }

mkdir -p "$CK/turbo" "$CK/base" "$CK/v1"
HF=https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/main
MIR=https://hf-mirror.com/ruchangyao/FlashVSR_plus/resolve/main

get() {  # get <远端路径> <目标文件> [备用远端路径]
  local p="$1" out="$2" alt="${3:-}"
  [ -s "$out" ] && { echo "✓ 已有 $(basename "$(dirname "$out")")/$(basename "$out")"; return 0; }
  echo "== $p"
  "$PY" "$DL" "$HF/$p" "$out" -n "$N" ${alt:+-r "$MIR/$alt"}
}

case "$WANT" in
  turbo|all)
    get turbo/dit.safetensors          "$CK/turbo/dit.safetensors"          turbo/dit.safetensors
    get turbo/lq_proj_in.safetensors   "$CK/turbo/lq_proj_in.safetensors"   turbo/lq_proj_in.safetensors
    ;;
esac
if [ "$WANT" = "all" ]; then
  get base/dit.safetensors         "$CK/base/dit.safetensors"           base/dit.safetensors
  get base/lq_proj_in.safetensors  "$CK/base/lq_proj_in.safetensors"    base/lq_proj_in.safetensors
  get TCDecoder.ckpt               "$CK/TCDecoder.ckpt"                 TCDecoder.ckpt
fi

echo
echo "现在有的档位（WebUI 会自动只显示权重齐全的档位）："
for m in turbo base v1; do
  if [ -f "$CK/$m/dit.safetensors" ] && [ -f "$CK/$m/lq_proj_in.safetensors" ]; then
    echo "  ✓ $m  $(du -sh "$CK/$m" | cut -f1)"
  else
    echo "  ✗ $m（缺 dit 或 lq_proj_in）"
  fi
done
echo
echo "补完重启 WebUI："
echo "  kill \$(ss -lntp | grep ':7860' | grep -o 'pid=[0-9]*' | cut -d= -f2)   # start.d 会自动拉起"
