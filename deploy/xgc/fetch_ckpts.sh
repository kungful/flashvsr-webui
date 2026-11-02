#!/bin/bash
# 补齐 FlashVSR+ 的 turbo / base 权重（v1 已内置在镜像里）
# 用法（在实例里跑，网络好的时候）：
#   bash /root/fetch_ckpts.sh          # 拉 turbo（6.4G，够用）
#   bash /root/fetch_ckpts.sh all      # turbo + base（各 6.4G）
#
# 实测：huggingface.co / hf-mirror.com 在本机房只有 0.3~0.9 MB/s（文件走 Xet 存储，
# 慢得没法用）；ModelScope 能到 10~80 MB/s，但只有 v1 的权重。所以：
#   网快的时候（或换机房/挂代理）再跑这个脚本补 turbo/base。
set -uo pipefail
CK=/opt/FlashVSR_Plus/ckpts
WANT="${1:-turbo}"          # turbo | all
mkdir -p "$CK/turbo" "$CK/base"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PATH=/root/miniconda3/bin:$PATH

pull() {  # pull <repo_path> <dest_dir>
  local p="$1" d="$2" n; n="$(basename "$p")"
  for base in "${HF_ENDPOINT:-https://hf-mirror.com}" "https://huggingface.co"; do
    echo "== $p  ← $base"
    if curl -fL --retry 3 --retry-delay 3 -m 3600 -o "$d/$n.part" \
        "$base/ruchangyao/FlashVSR_plus/resolve/main/$p"; then
      mv "$d/$n.part" "$d/$n"; echo "   ok $(du -h "$d/$n" | cut -f1)"; return 0
    fi
    rm -f "$d/$n.part"; echo "   失败，换下一个源"
  done
  return 1
}

pull turbo/dit.safetensors "$CK/turbo"
pull turbo/lq_proj_in.safetensors "$CK/turbo"
if [ "$WANT" = "all" ]; then
  pull base/dit.safetensors "$CK/base"
  pull base/lq_proj_in.safetensors "$CK/base"
fi

echo
echo "现在有的档位（WebUI 会自动只显示权重齐全的档位）："
for m in v1 turbo base; do
  [ -f "$CK/$m/dit.safetensors" ] && echo "  ✓ $m" || echo "  ✗ $m（缺 dit.safetensors）"
done
echo
echo "补完记得重启 WebUI： kill \$(ss -lntp | grep ':7860' | grep -o 'pid=[0-9]*' | cut -d= -f2)   # start.d 会自己拉起来"
