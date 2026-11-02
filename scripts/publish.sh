#!/usr/bin/env bash
# 一条龙发布：建仓库（不存在就建）→ 推 main → 设简介/话题
#
#   GITHUB_TOKEN=ghp_xxx ./scripts/publish.sh          # 非交互
#   ./scripts/publish.sh                               # 交互式，隐藏输入令牌
#   PRIVATE=1 ./scripts/publish.sh                     # 建成私有仓库
#   NAME=flashvsr-webui OWNER=kungful ./scripts/publish.sh
#
# 令牌需要什么权限：
#   classic PAT     → 勾 repo（含 repo:status / public_repo 的父项）
#   fine-grained    → Administration: write（建仓库） + Contents: write（推送）
# 令牌不会被写入磁盘、不会进 git config、不打印。
set -euo pipefail

OWNER="${OWNER:-kungful}"
NAME="${NAME:-flashvsr-webui}"
BRANCH="${BRANCH:-main}"
DESC="${DESC:-FlashVSR（一步扩散流式视频超分）的 WebUI + 多平台部署脚手架}"
TOPICS="${TOPICS-fvsr,video-super-resolution,flashvsr,flask,gradio,upscaling}"
PRIVATE="${PRIVATE:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API="https://api.github.com"

die() { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
ok()  { printf '\033[1;32m✓ %s\033[0m\n' "$*"; }
log() { printf '\033[1;36m==> %s\033[0m\n' "$*"; }

TOKEN="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
if [ -z "$TOKEN" ]; then
  [ -t 0 ] || die "没给 GITHUB_TOKEN，且当前不是交互终端"
  printf '粘贴 GitHub PAT（输入不回显）: '
  read -rs TOKEN; echo
  [ -n "$TOKEN" ] || die "空令牌"
fi

api() { # api METHOD PATH [JSON]  → stdout=响应体
  local m="$1" p="$2" d="${3:-}"
  if [ -n "$d" ]; then
    curl -sS -X "$m" -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" \
      -H "Content-Type: application/json" -d "$d" "$API$p"
  else
    curl -sS -X "$m" -H "Authorization: Bearer $TOKEN" \
      -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" "$API$p"
  fi
}

log "1/4 验证令牌"
WHO=$(api GET /user)
echo "$WHO" | grep -q '"login"' || die "令牌无效：$(echo "$WHO" | head -c 300)"
LOGIN=$(echo "$WHO" | sed -n 's/.*"login": *"\([^"]*\)".*/\1/p' | head -1)
ok "身份 $LOGIN"

log "2/4 检查 ^$OWNER/$NAME 是否存在"
CODE=$(curl -s -o /tmp/fvsr_repo.json -w '%{http_code}' -H "Authorization: Bearer $TOKEN" \
       -H "Accept: application/vnd.github+json" "$API/repos/$OWNER/$NAME")
if [ "$CODE" = 200 ]; then
  ok "仓库已存在，直接推"
else
  [ "$CODE" = 404 ] || die "查询仓库失败 HTTP $CODE：$(head -c 300 /tmp/fvsr_repo.json)"
  log "仓库不存在，创建中"
  BODY=$(printf '{"name":"%s","description":"%s","private":%s,"has_issues":true,"has_wiki":false,"auto_init":false}' \
         "$NAME" "$DESC" "$([ "$PRIVATE" = 1 ] && echo true || echo false)")
  if [ "$LOGIN" = "$OWNER" ]; then
    RES=$(api POST /user/repos "$BODY")
  else
    RES=$(api POST "/orgs/$OWNER/repos" "$BODY")
  fi
  echo "$RES" | grep -q '"full_name"' || die "建仓库失败：$(echo "$RES" | head -c 400)"
  ok "已创建 $OWNER/$NAME"
fi

log "3/4 推送 $BRANCH"
cd "$HERE"
git rev-parse --git-dir >/dev/null 2>&1 || die "$HERE 不是 git 仓库"
[ -z "$(git status --porcelain)" ] || die "工作区有未提交改动，先 commit 再发布"
# 令牌只出现在这一条命令的 URL 里，不落盘、不进 .git/config
git -c credential.helper= push "https://x-access-token:$TOKEN@github.com/$OWNER/$NAME.git" \
    "HEAD:refs/heads/$BRANCH" 2>&1 | sed -E "s#$TOKEN#***#g"
git fetch -q "https://x-access-token:$TOKEN@github.com/$OWNER/$NAME.git" "$BRANCH" 2>/dev/null || true
git branch -q --set-upstream-to="FETCH_HEAD" "$BRANCH" 2>/dev/null || true
ok "已推送到 https://github.com/$OWNER/$NAME/tree/$BRANCH"

log "4/4 设置简介与话题"
api PATCH "/repos/$OWNER/$NAME" "{\"description\":\"$DESC\",\"has_wiki\":false}" >/dev/null || true
NAMES=$(printf '%s' "$TOPICS" | tr ',' '\n' | grep -v '^$' \
        | awk '{printf "%s\"%s\"", (NR>1?",":""), $0}')
[ -n "$NAMES" ] && api PUT "/repos/$OWNER/$NAME/topics" "{\"names\":[$NAMES]}" >/dev/null || true
ok "话题：${TOPICS:-(跳过)}"

cat <<EOF

  ✅ 完成：https://github.com/$OWNER/$NAME
     克隆：git clone https://github.com/$OWNER/$NAME.git
EOF
