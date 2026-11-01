#!/usr/bin/env bash
# 冒烟测试：起服务 → 上传示例 → 跑一次 → 检查产物 → 杀掉
#   ./scripts/smoke_test.sh          # 默认 mock 引擎（不需要 GPU）
#   ENGINE=real ./scripts/smoke_test.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SMOKE_PORT="${SMOKE_PORT:-7899}"
PORT="$SMOKE_PORT"   # 不要用外部 PORT（很多环境预置了它）
ENGINE="${ENGINE:-mock}"
TMP="$(mktemp -d)"
PY="${FVSR_PY:-${ROOT}/.venv/bin/python}"
[ -x "$PY" ] || PY=python3

echo "== 启动 $ENGINE 引擎 on :$PORT"
FVSR_ENGINE="$ENGINE" FVSR_PORT="$PORT" FVSR_DATA="$TMP" \
  "$PY" "$ROOT/app.py" >"$TMP/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null; wait $SPID 2>/dev/null' EXIT

for i in $(seq 1 40); do
  sleep 0.5
  curl -sf "http://127.0.0.1:$PORT/api/hardware" >/dev/null && break
done

fail(){ echo "✗ $*"; echo "--- server.log ---"; tail -30 "$TMP/server.log"; exit 1; }

echo "== /api/hardware"
curl -sf "http://127.0.0.1:$PORT/api/hardware" | tee "$TMP/hw.json" >/dev/null || fail "hardware 接口挂了"
grep -q '"engine"' "$TMP/hw.json" || fail "没有 engine 字段"

echo "== 生成示例 → 上传 → 跑一次"
ffmpeg -y -v error -f lavfi -i "testsrc2=size=320x192:rate=12:duration=1" \
  -c:v libx264 -pix_fmt yuv420p "$TMP/in.mp4" 2>/dev/null \
  || ffmpeg -y -v error -f lavfi -i "testsrc2=size=320x192:rate=12:duration=1" \
     -c:v libopenh264 -b:v 2M -pix_fmt yuv420p "$TMP/in.mp4"

FID=$(curl -sf -F "file=@$TMP/in.mp4" "http://127.0.0.1:$PORT/api/upload" | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])') \
  || fail "上传失败"

JID=$(curl -sf -H 'Content-Type: application/json' \
  -d "{\"file_id\":\"$FID\",\"scale\":2,\"model\":\"turbo\"}" \
  "http://127.0.0.1:$PORT/api/job" | python3 -c 'import json,sys;print(json.load(sys.stdin)["job_id"])') \
  || fail "建任务失败"

for i in $(seq 1 240); do
  S=$(curl -sf "http://127.0.0.1:$PORT/api/job/$JID")
  ST=$(echo "$S" | python3 -c 'import json,sys;print(json.load(sys.stdin)["status"])')
  [ "$ST" = "done" ] && break
  [ "$ST" = "error" ] && fail "任务失败: $(echo "$S" | head -c 400)"
  sleep 1
done
[ "$ST" = "done" ] || fail "超时未完成"

OUT=$(echo "$S" | python3 -c 'import json,sys;print(json.load(sys.stdin)["result"]["out"] or "")')
[ -n "$OUT" ] || fail "没有产物"
curl -sf -o /dev/null "http://127.0.0.1:$PORT$OUT" || fail "产物下载失败"
SIZE=$(curl -s -o /dev/null -w '%{size_download}' -r 0-1023 "http://127.0.0.1:$PORT$OUT")
[ "$SIZE" -gt 0 ] || fail "Range 请求没返回数据"

echo "== 产物: $OUT ($SIZE B via Range)"
echo "✓ 冒烟测试通过（$ENGINE）"
