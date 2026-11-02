#!/usr/bin/env python3
"""并行分段下载 HuggingFace 大文件 —— 单连接太慢时的救命工具。

为什么需要它：2026-11 实测国内机房到 HuggingFace/ hf-mirror 的单连接只有
0.13~0.17 MB/s（文件走 Xet 存储），6.4 GB 要 13 小时；换成 48 个并行 Range
请求后能到 **8.9 MB/s**（约 12 分钟），快了 60 多倍。hf_transfer 也是这个
思路，区别是这个不依赖 hf 的鉴权/端点，纯 curl 分段，断点续传自己做。

用法：
    python3 dl_parallel.py <url> <输出文件> [-n 48] [-r "备用url"]

    # 例：把 turbo 权重拉到官方目录
    python3 dl_parallel.py \\
      https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/main/turbo/dit.safetensors \\
      /opt/FlashVSR_Plus/ckpts/turbo/dit.safetensors -n 48

特性：
  - 48 段并行（可调），每段独立重试 + 断点续传（重跑同一条命令即续传）
  - 进度按秒打印在 stderr，完成后校验字节数与远端口径一致
  - 段文件在 <输出>.parts/，成功后自动删除；失败可重跑续传
  - 只在最后合并，磁盘峰值需要约 2 倍文件大小
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import subprocess
import sys
import time
import urllib.request


def head_size(urls: list[str]) -> int:
    for u in urls:
        try:
            req = urllib.request.Request(u, method="HEAD", headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=30) as r:
                n = int(r.headers.get("Content-Length") or 0)
                if n > 0:
                    return n
        except Exception:
            continue
    raise SystemExit("拿不到文件大小（所有源 HEAD 都失败）")


def download(urls: list[str], out: str, n: int) -> int:
    size = head_size(urls)
    tmp = out + ".parts"
    os.makedirs(tmp, exist_ok=True)
    chunk = (size + n - 1) // n
    print(f"size={size / 1e9:.2f} GB  segs={n}  chunk={chunk / 1e6:.0f} MB -> {out}", file=sys.stderr)

    def seg(i: int) -> int:
        start, end = i * chunk, min(size - 1, (i + 1) * chunk - 1)
        want = end - start + 1
        f = os.path.join(tmp, f"{i:03d}.part")
        for attempt in range(200):
            have = os.path.getsize(f) if os.path.exists(f) else 0
            if have >= want:
                return want
            if have > want:            # 段文件比预期长，重来
                open(f, "wb").close()
                have = 0
            url = urls[attempt % len(urls)]   # 失败就换源
            subprocess.run(["curl", "-sL", "--max-time", "1200",
                            "-r", f"{start + have}-{end}", "-o", f + ".tmp", url],
                           capture_output=True)
            part = f + ".tmp"
            if os.path.exists(part):
                with open(part, "rb") as src, open(f, "ab") as dst:
                    while True:
                        b = src.read(8 << 20)
                        if not b:
                            break
                        dst.write(b)
                os.remove(part)
            if (os.path.getsize(f) if os.path.exists(f) else 0) >= want:
                return want
            time.sleep(min(2 + attempt, 30))
        return 0

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        futs = [ex.submit(seg, i) for i in range(n)]
        while any(not f.done() for f in futs):
            time.sleep(10)
            got = sum(os.path.getsize(os.path.join(tmp, x))
                      for x in os.listdir(tmp) if x.endswith(".part"))
            print(f"  {got / 1e9:.2f}/{size / 1e9:.2f} GB  "
                  f"{got / max(time.time() - t0, 1) / 1e6:.2f} MB/s  {time.time() - t0:.0f}s",
                  file=sys.stderr)
        ok = sum(1 for f in futs if f.result())

    if ok < n:
        print(f"!! 有 {n - ok} 段没下完，重跑本命令可续传", file=sys.stderr)
        return 1

    with open(out, "wb") as o:
        for i in range(n):
            with open(os.path.join(tmp, f"{i:03d}.part"), "rb") as s:
                shutil.copyfileobj(s, o, 8 << 20)
    got = os.path.getsize(out)
    print(f"FINAL {out} {got / 1e9:.2f} GB  ok={got == size}  {time.time() - t0:.0f}s", file=sys.stderr)
    if got == size:
        shutil.rmtree(tmp, ignore_errors=True)
        return 0
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="并行分段下载大文件（HF 单连接太慢时用）")
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("-n", "--segs", type=int, default=48, help="并行段数（默认 48）")
    ap.add_argument("-r", "--retry-url", action="append", default=[],
                    help="备用源（可多次给），失败时轮换")
    a = ap.parse_args()
    return download([a.url] + a.retry_url, a.out, a.segs)


if __name__ == "__main__":
    sys.exit(main())
