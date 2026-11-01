#!/bin/bash
# 一键把 FlashVSR WebUI 部署到仙宫云
#   用法： ./deploy/deploy_xgc.sh            # 没实例就开一台，有就复用
#          IMAGE=<镜像id> GPU="NVIDIA GeForce RTX 4090 D" ./deploy/deploy_xgc.sh
# 前提：~/.xgc_token 已写入；本机有 sshpass / ssh / scp / jq
set -uo pipefail

X=~/.pi/agent/skills/xgc-api/scripts/xgc.py
IMAGE="${IMAGE:-ed73b8ff-1e5c-4f39-9d80-e10ac79bef82}"   # FlashVSR_Plus
GPU="${GPU:-NVIDIA GeForce RTX 4090 D}"
NAME="${NAME:-flashvsr-webui}"
ITYPE="${ITYPE:-private}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"   # 项目根（含 app.py）
PROJ="$(basename "$ROOT")"
SSH_OPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20"

log(){ printf '\n\033[36m== %s\033[0m\n' "$*"; }

# ---------- 1) 实例 ----------
IID=$(python3 "$X" --json instances 2>/dev/null | jq -r '.[0].list[0].id // empty')
if [ -z "$IID" ]; then
  log "1/7 部署实例（镜像 $IMAGE）"
  python3 "$X" deploy --gpu "$GPU" --count 1 --image "$IMAGE" --image-type "$ITYPE" --name "$NAME" --yes | tail -3
  for i in $(seq 1 90); do
    sleep 10
    J=$(python3 "$X" --json instances 2>/dev/null)
    IID=$(echo "$J" | jq -r '.[0].list[0].id // empty')
    ST=$(echo "$J" | jq -r '.[0].list[0].status // ""')
    PR=$(echo "$J" | jq -r '.[0].list[0].progress // 0')
    printf '  [%02d] %s %.0f%%\n' "$i" "$ST" "${PR%%.*}"
    [ "$ST" = "running" ] && break
  done
else
  log "1/7 复用已有实例 $IID"
fi
[ -z "$IID" ] && { echo "没拿到实例 id，停"; exit 1; }

# ---------- 2) 连接信息 ----------
log "2/7 取 SSH 信息"
J=$(python3 "$X" --json instances 2>/dev/null)
PORT=$(echo "$J" | jq -r '.[0].list[0].ssh_port')
DOM=$(echo "$J" | jq -r '.[0].list[0].ssh_domain')
PWF=/tmp/xgc_pw; umask 077; echo "$J" | jq -r '.[0].list[0].password' > "$PWF"
echo "  实例 $IID  ssh -p $PORT root@$DOM"

# ---------- 3) 上传 ----------
log "3/7 打包上传"
tar czf /tmp/fvsr-webui.tgz -C "$(dirname "$ROOT")" \
    --exclude='data/*' --exclude='third_party/*' --exclude='.venv/*' \
    --exclude='__pycache__' --exclude='.git' "$PROJ"
tar tzf /tmp/fvsr-webui.tgz >/dev/null || { echo "!! 打包失败"; exit 1; }
echo "  $(tar tzf /tmp/fvsr-webui.tgz | wc -l) 个文件，$(du -h /tmp/fvsr-webui.tgz | cut -f1)"
for f in /tmp/fvsr-webui.tgz "$HERE/flashvsr-newui.sh" "$HERE/patch_gradio.py" "$HERE/remote_setup.sh"; do
  sshpass -p "$(cat $PWF)" scp $SSH_OPT -P "$PORT" "$f" "root@$DOM":/root/ || exit 1
done
sshpass -p "$(cat $PWF)" ssh $SSH_OPT -p "$PORT" "root@$DOM" 'chmod +x /root/remote_setup.sh'

# ---------- 4) 容器内安装 ----------
log "4/7 容器内安装（flask / 代码 / 自启 / 补丁）"
sshpass -p "$(cat $PWF)" ssh $SSH_OPT -p "$PORT" "root@$DOM" 'bash /root/remote_setup.sh'

# ---------- 5) 公网自检 ----------
log "5/7 公网自检"
B="https://$IID-7860.container.x-gpu.com"
for i in $(seq 1 12); do
  C1=$(curl -s --noproxy '*' -m 15 -o /dev/null -w '%{http_code}' "$B/" || echo 000)
  C2=$(curl -s --noproxy '*' -m 15 -o /dev/null -w '%{http_code}' "$B/g" || echo 000)
  C3=$(curl -s --noproxy '*' -m 15 -o /dev/null -w '%{http_code}' "$B/api/outputs" || echo 000)
  echo "  / -> $C1   /g -> $C2   /api/outputs -> $C3"
  [ "$C1" = "200" ] && [ "$C3" = "200" ] && break
  sleep 8
done

log "6/7 结果"
cat <<EOF
  新 UI      $B/            （模拟引擎，就是你看到那个）
  老 UI      $B/g           （镜像自带的 Gradio，真模型）
  下载通道   $B/fvdl/outputs/<文件名>?dl=1
  实例       $IID   ¥1.59/h，记得不用了销毁：
             python3 $X destroy $IID --yes
EOF

log "7/7 收工"
echo "$IID" > /tmp/fvsr.instance
