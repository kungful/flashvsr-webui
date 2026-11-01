#!/bin/bash
# 在容器内执行：把 FlashVSR 新 UI 装成「公网主入口」，老 Gradio UI 退到 /g
# 需要先 scp 到 /root/：fvsr-webui.tgz flashvsr-newui.sh patch_gradio.py
set -uo pipefail
PY=/root/miniconda3/bin/python
PIP=/root/miniconda3/bin/pip

echo "== 1) flask =="
if ! $PY -c "import flask" 2>/dev/null; then
  $PIP install -q flask -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2
fi
$PY -c "import flask;print('  flask', flask.__version__)"

echo "== 2) 新 UI 代码 =="
rm -rf /root/app2_new && mkdir -p /root/app2_new
tar xzf /root/fvsr-webui.tgz -C /root/app2_new --strip-components=1 || { echo "!! 解包失败"; exit 1; }
mkdir -p /root/fvsr-data/outputs /root/fvsr-data/uploads
# 一次性把老位置的文件搬到独立数据目录（之后数据就不再跟代码目录走了）
for d in /root/app2 /root/app2.old; do
  [ -d "$d/outputs" ] && cp -an "$d/outputs/." /root/fvsr-data/outputs/ 2>/dev/null
  [ -d "$d/uploads" ] && cp -an "$d/uploads/." /root/fvsr-data/uploads/ 2>/dev/null
done
[ -d /root/app2 ] && rm -rf /root/app2.old && mv /root/app2 /root/app2.old
mv /root/app2_new /root/app2
# 清掉代码目录里的产物残留，避免"删了又出现"
rm -rf /root/app2/outputs /root/app2/uploads /root/app2.old2 2>/dev/null
echo "  $(ls /root/app2 | tr '\n' ' ')"

echo "== 3) 开机自启 =="
install -m 755 /root/flashvsr-newui.sh /scripts/start.d/flashvsr-newui.sh
echo "  $(ls /scripts/start.d/ | tr '\n' ' ')"

echo "== 4) 老 Gradio 挪到 7861 + root_path=/g =="
$PY /root/patch_gradio.py

echo "== 5) 重启老 UI（让它加载新端口）=="
O=$(ss -lntp | grep -E ":786[01] " | grep -o "pid=[0-9]*" | cut -d= -f2 | head -1)
if [ -n "${O:-}" ] && [ -r "/proc/$O/cmdline" ] && tr '\0' ' ' < "/proc/$O/cmdline" | grep -q "/root/app/app.py"; then
  kill "$O"; echo "  killed $O"
else
  echo "  7860 上没有老 UI 进程（可能已退出，靠 start.d 自己拉起）"
fi

echo "== 6) 重启新 UI =="
L=$(ps -eo pid,args | grep "start[.]d/flashvsr-newui" | grep -v grep | awk '{print $1}')
[ -n "${L:-}" ] && kill $L
F=$(ss -lntp | grep ":7860" | grep -o "pid=[0-9]*" | cut -d= -f2)
[ -n "${F:-}" ] && kill $F
sleep 10
setsid nohup /scripts/start.d/flashvsr-newui.sh </dev/null >>/root/logs/fvsr-newui.log 2>&1 &
sleep 5

echo "== 7) 自检 =="
ss -lntp | grep -E ":786[01] " || echo "  !! 没有监听"
curl -s -m 10 -o /dev/null -w "  新 UI  /        -> %{http_code}\n" http://127.0.0.1:7860/
curl -s -m 10 -o /dev/null -w "  API    /api/outputs -> %{http_code}\n" http://127.0.0.1:7860/api/outputs
curl -s -m 10 -o /dev/null -w "  Gradio /g       -> %{http_code}\n" http://127.0.0.1:7860/g
curl -s -m 10 http://127.0.0.1:7860/api/outputs | head -c 120; echo
