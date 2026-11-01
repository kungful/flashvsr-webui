#!/bin/bash
# /scripts/start.d/flashvsr-newui.sh —— FlashVSR 新 UI（模拟引擎版）占住公网端口 7860
export PATH=/root/miniconda3/bin:$PATH
export TZ=Asia/Shanghai
export PYTHONUNBUFFERED=1
export FVSR_PORT=7860
export FVSR_ENGINE=real
export FVSR_UPSTREAM=http://127.0.0.1:7861   # 老 Gradio UI，挂在 /g
export FVSR_DL_BASE=/fvdl
export FVSR_DATA=/root/fvsr-data          # 产物/上传独立存放，换代码不丢、也不复制
mkdir -p /root/logs /root/fvsr-data/outputs /root/fvsr-data/uploads
cd /root/app2 || exit 1
exec 9>/var/lock/flashvsr-newui.lock
flock -n 9 || { echo "[start.d] newui already running, skip" >> /root/logs/fvsr-newui.log; exit 0; }
while true; do
  echo "[start.d] newui launch $(date '+%F %T %Z')" >> /root/logs/fvsr-newui.log
  /root/miniconda3/bin/python /root/app2/app.py >> /root/logs/fvsr-newui.log 2>&1
  echo "[start.d] newui exited rc=$? ; restart in 5s" >> /root/logs/fvsr-newui.log
  sleep 5
done
