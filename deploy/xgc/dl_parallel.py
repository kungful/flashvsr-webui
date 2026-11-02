#!/usr/bin/env python3
"""并行分段下载 HuggingFace 大文件 —— 单连接太慢时的救命工具。

为什么需要它：2026-11 实测国内机房到 HuggingFace / hf-mirror 的单连接只有
0.13~0.17 MB/s（文件走 Xet 存储），6.4 GB 要 13 小时；换成 48 个并行 Range
请求后能到 **8.9 MB/s**（约 12 分钟），快了 60 多倍。hf_transfer 也是这个
思路，区别是这个不依赖 hf 的鉴权/端点，纯 curl 分段，断点续传自己做。

用法：
    python3 dl_parallel.py <url> <输出文件> [-n 48] [-r "备用url"]

特性：
  - 48 段并行（可调），每段独立重试 + 断点续传（重跑同一条命令即续传）
  - **文件总长度用 Range 探测而不是只信 HEAD**（HEAD 经 CDN/Xet 可能撒谎，
    被骗就会下出"大小自洽但内容被截断"的坏文件 —— 踩过，base 权重就是这么坏的）
  - 完成后做格式校验：.safetensors 真的能解析、.ckpt/.pth 真的能 torch.load
  - 段文件在 <输出>.parts/，成功后自动删除；失败可重跑续传
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


def _head(u: str) -> int:
    try:
        req = urllib.request.Request(u, method="HEAD", headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


def _range_probe(u: str) -> int:
    """GET 1 字节，读 Content-Range: bytes 0-0/<total> —— 比 HEAD 可靠"""
    try:
        req = urllib.request.Request(u, headers={"Range": "bytes=0-0", "User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=30) as r:
            cr = r.headers.get("Content-Range") or ""
            if "/" in cr:
                tail = cr.rsplit("/", 1)[1].strip()
                if tail.isdigit():
                    return int(tail)
    except Exception:
        pass
    return 0


def probe_size(urls: list[str]) -> int:
    """多源、多种方式探长度；截断只会变小，所以取最大值最安全"""
    cands = []
    for u in urls:
        for fn in (_range_probe, _head):
            n = fn(u)
            if n > 0:
                cands.append(n)
    if not cands:
        raise SystemExit("拿不到文件大小（HEAD / Range 全失败）")
    return max(cands)


def validate(out: str, size: int) -> tuple[bool, str]:
    got = os.path.getsize(out)
    if got != size:
        return False, f"字节数不对：{got} != {size}（少了 {(size - got) / 1e6:.1f} MB）"
    if out.endswith(".safetensors"):
        try:
            from safetensors import safe_open
            with safe_open(out, "pt") as f:
                n = len(list(f.keys()))
            return True, f"safetensors 可解析，{n} 个张量"
        except Exception as e:
            return False, f"safetensors 解析失败：{str(e)[:100]}"
    if out.endswith((".ckpt", ".pth", ".bin")):
        try:
            import torch
            torch.load(out, map_location="cpu", weights_only=True)
            return True, "torch.load 通过"
        except Exception as e:
            return False, f"torch.load 失败：{str(e)[:100]}"
    return True, "字节数正确"


def download(urls: list[str], out: str, n: int) -> int:
    size = probe_size(urls)
    tmp = out + ".parts"
    os.makedirs(tmp, exist_ok=True)
    chunk = (size + n - 1) // n
    print(f"size={size / 1e9:.2f} GB  segs={n}  chunk={chunk / 1e6:.0f} MB -> {out}", file=sys.stderr)

    def seg(i: int) -> int:
        start, end = i * chunk, min(size - 1, (i + 1) * chunk - 1)
        want = end - start + 1
        f = os.path.join(tmp, f"{i:03d}.part")
        # 上次跑剩下的段可能比预期长（长度探测变了），清掉
        if os.path.exists(f) and os.path.getsize(f) > want:
            open(f, "wb").close()
        for attempt in range(200):
            have = os.path.getsize(f) if os.path.exists(f) else 0
            if have >= want:
                return want
            url = urls[attempt % len(urls)]      # 失败就换源
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
                # 服务器忽略 Range 会返回整个文件；超长立刻回滚，别污染段
                if os.path.getsize(f) > want:
                    open(f, "wb").close()
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

    good, why = validate(out, size)
    print(f"FINAL {out} {os.path.getsize(out) / 1e9:.2f} GB  校验{'通过' if good else '失败'}：{why}"
          f"  {time.time() - t0:.0f}s", file=sys.stderr)
    if good:
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
