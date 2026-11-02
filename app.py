#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR WebUI — 纯 UI + 模拟推理引擎
------------------------------------
本机没有 CUDA 显卡，所以这里不跑真正的 FlashVSR 扩散模型。
后端做三件真事：
  1. 真实读取上传文件（ffprobe 拿分辨率/帧率/时长/码率/编码）
  2. 真实模拟 FlashVSR 的推理流程（分窗、稀疏注意力、条件解码器）并给出进度/日志/显存估算
  3. 真实用 ffmpeg(lanczos) 生成放大预览，让前后对比滑块看到实际画面

要接真模型，只需把 run_job() 里 simulate_* 那一段换成 FlashVSR 的 infer 调用。
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import urllib.error
import urllib.request

from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, send_from_directory, stream_with_context)

BASE = Path(__file__).resolve().parent
# 产物/上传放独立数据目录（默认在代码目录下，部署时用 FVSR_DATA 指到代码外面去）
DATA = Path(os.environ.get("FVSR_DATA", str(BASE / "data"))).expanduser()
UPLOADS = DATA / os.environ.get("FVSR_UPLOADS", "uploads")
OUTPUTS = DATA / os.environ.get("FVSR_OUTPUTS", "outputs")
UPLOADS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_PREVIEW_W = 3840

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

# 部署在子路径下时（例如 https://host/fv），把前缀带进所有返回的 URL
PREFIX = os.environ.get("FVSR_PREFIX", "").rstrip("/")


def pfx(u: str) -> str:
    return f"{PREFIX}{u}" if u.startswith("/") else u


# 直连下载通道（部署时由反代侧提供，绕开中间反代的整包缓冲，支持 Range/断点续传）
DL_BASE = os.environ.get("FVSR_DL_BASE", "").rstrip("/")

# ---- 推理引擎：mock = 模拟；real = 真调用 FlashVSR+ ----
ENGINE = os.environ.get("FVSR_ENGINE", "mock")


def _guess_repo() -> str:
    """按顺序找官方仓库：环境变量 → 仓库内 third_party → 常见路径"""
    cands = [
        os.environ.get("FVSR_REPO"),
        str(BASE / "third_party" / "FlashVSR_Plus"),
        "/root/FlashVSR_Plus",
        str(Path.home() / "FlashVSR_Plus"),
    ]
    for c in cands:
        if c and (Path(c) / "inference" / "run_stream.py").is_file():
            return c
    return os.environ.get("FVSR_REPO") or str(BASE / "third_party" / "FlashVSR_Plus")


def _guess_py() -> str:
    for c in (os.environ.get("FVSR_PY"), "/root/miniconda3/bin/python",
              str(Path.home() / "miniconda3/bin/python")):
        if c and Path(c).exists():
            return c
    return sys.executable


REPO = _guess_repo()
PYBIN = _guess_py()
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MODES = ("turbo", "base", "v1")


def real_available() -> bool:
    return ENGINE == "real" and (Path(REPO) / "inference" / "run_stream.py").is_file()


def round128(v: float) -> int:
    """模型要求宽高都是 128 的倍数"""
    return max(128, int(round(v / 128.0)) * 128)


def ceil128(v: float) -> int:
    """向上取 128，保证不会小于给定值"""
    return max(128, int(math.ceil(v / 128.0)) * 128)


def size_for_mp(mp: float, sw: int, sh: int) -> tuple[int, int]:
    """按"总像素数 + 素材宽高比"算输出尺寸（不吃裁剪）。

    以前预设写死 2560×1408，竖屏素材套上去会被模型中心裁剪掉大半画面；
    现在只给"几百万像素"这一档，长宽比永远跟素材一致。
    """
    area = max(1.0, float(mp)) * 1e6
    r = sw / max(1, sh)
    h = math.sqrt(area / r)
    w = h * r
    return ceil128(w), ceil128(h)


def target_size(meta: dict, p: dict) -> tuple[int, int]:
    """确定输出宽高。

    优先级：目标像素数(按素材比例) > 显式宽高 > 倍数×原片。
    硬约束：**任何情况下输出都不能比原片小**（曾经因为"显存保护"把 2× 压成 0.6×）。
    """
    sw, sh = int(meta["width"]), int(meta["height"])
    mp = float(p.get("target_mp") or 0)
    tw, th = int(p.get("target_w") or 0), int(p.get("target_h") or 0)
    if mp > 0:
        w, h = size_for_mp(mp, sw, sh)
    elif tw > 0 and th > 0:
        w, h = round128(tw), round128(th)
    else:
        sc = int(p.get("scale", 2)) or 1
        # 倍数模式一律向上取整：2× 就绝不能变成 1.78×
        w, h = ceil128(sw * sc), ceil128(sh * sc)
    if w < sw:
        w = ceil128(sw)
    if h < sh:
        h = ceil128(sh)
    return min(w, 3840), min(h, 3840)


def media_url(kind: str, name: str) -> str:
    """kind = uploads | outputs —— 默认走 /fvdl（支持 Range 的直连通道）"""
    return f"{DL_BASE or '/fvdl'}/{kind}/{name}"


def dl_url(kind: str, name: str) -> str:
    return media_url(kind, name) + "?dl=1"

JOBS: dict[str, dict] = {}
FILES: dict[str, dict] = {}
LOCK = threading.Lock()


# ---------------------------------------------------------------- utils
def run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


_CODECS: dict | None = None


def available_encoders() -> set[str]:
    global _CODECS
    if _CODECS is None:
        p = run(["ffmpeg", "-hide_banner", "-encoders"], timeout=30)
        _CODECS = set(re.findall(r"^\s*V[.A-Z]{5}\s+(\S+)", p.stdout or "", re.M))
    return _CODECS


def pick_video_codec(w: int, h: int) -> tuple[str, list[str], str]:
    """挑一个本机可用、浏览器能直接播的编码器。"""
    enc = available_encoders()
    mbps = max(2.5, min(20.0, (w * h) / 260000))
    if "libx264" in enc:
        return "libx264", ["-preset", "veryfast", "-crf", "20"], ".mp4"
    if "libopenh264" in enc:
        return ("libopenh264",
                ["-b:v", f"{mbps:.1f}M", "-maxrate", f"{mbps * 1.3:.1f}M",
                 "-bufsize", f"{mbps * 2:.1f}M"], ".mp4")
    if "libvpx-vp9" in enc:
        return "libvpx-vp9", ["-crf", "32", "-b:v", "0", "-row-mt", "1"], ".webm"
    if "mpeg4" in enc:
        return "mpeg4", ["-q:v", "3"], ".mp4"
    raise RuntimeError("ffmpeg 没有可用的 h264/vp9 编码器")


def probe(path: Path) -> dict:
    """ffprobe -> 统一元数据"""
    if not have("ffprobe"):
        return {"kind": "unknown", "error": "ffprobe 未安装"}
    p = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ], timeout=120)
    if p.returncode != 0:
        return {"kind": "unknown", "error": p.stderr.strip()[:300]}
    data = json.loads(p.stdout or "{}")
    streams = data.get("streams", [])
    fmt = data.get("format", {})
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if v is None:
        return {"kind": "audio", "duration": float(fmt.get("duration") or 0)}

    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    num, den = v.get("avg_frame_rate", "0/1").split("/")
    fps = (float(num) / float(den)) if float(den or 0) else 0.0
    duration = float(fmt.get("duration") or v.get("duration") or 0)
    kind = "image" if path.suffix.lower() in IMAGE_EXT else "video"
    if kind == "video" and duration <= 0.05:
        kind = "image"

    return {
        "kind": kind,
        "width": w,
        "height": h,
        "fps": round(fps, 3),
        "frames": int(fps * duration) if kind == "video" else 1,
        "duration": round(duration, 3),
        "video_codec": v.get("codec_name", "?"),
        "pix_fmt": v.get("pix_fmt", "?"),
        "bitrate": int(float(fmt.get("bit_rate") or 0)),
        "size": int(fmt.get("size") or path.stat().st_size),
        "audio": bool(a),
        "audio_codec": (a or {}).get("codec_name"),
        "container": fmt.get("format_name", "?"),
    }


def gpu_info() -> dict:
    """尽力探测硬件；本机无卡时前端会显示「仅 UI 模拟」"""
    if have("nvidia-smi"):
        p = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"], timeout=15)
        if p.returncode == 0 and p.stdout.strip():
            name, mem, drv = [x.strip() for x in p.stdout.splitlines()[0].split(",")]
            try:
                import torch  # noqa
                torch_ok = torch.cuda.is_available()
            except Exception:
                torch_ok = False
            return {"cuda": True, "name": name, "vram_mb": int(float(mem)),
                    "driver": drv, "torch": torch_ok}
    return {"cuda": False, "name": "CPU only", "vram_mb": 0,
            "driver": "-", "torch": False}


def vram_estimate(p: dict) -> float:
    """按 FlashVSR 的经验量级估算显存占用（GB），仅用于 UI 参考"""
    tile = p["tile_size"]
    window = p["window_size"]
    scale = p["scale"]
    base = 1.6 + (tile / 256) ** 2 * 1.15 + window / 21 * 0.9
    base *= {2: 1.0, 3: 1.35, 4: 1.8}.get(scale, 1.8)
    if p["dtype"] == "fp32":
        base *= 1.9
    if not p.get("sparse_attention", True):
        base *= 1.45
    return round(base, 2)


# ---------------------------------------------------------------- ffmpeg 预览
def make_image_preview(src: Path, dst: Path, out_w: int, out_h: int) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-vf", f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_int",
         "-frames:v", "1", str(dst)], timeout=300)


def make_video_preview(src: Path, dst: Path, out_w: int, out_h: int,
                       limit: float | None = None) -> None:
    vf = (f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_int,"
          "format=yuv420p")
    codec, extra, ext = pick_video_codec(out_w, out_h)
    if ext != dst.suffix:
        dst = dst.with_suffix(ext)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", vf,
           "-c:v", codec] + extra + ["-movflags", "+faststart", "-an"]
    if limit:
        cmd += ["-t", f"{limit}"]
    cmd.append(str(dst))
    run(cmd, timeout=1800)



# ---------------------------------------------------------------- 真实推理
REAL_LOCK = threading.Lock()      # 同时只允许一个真推理（两个模型抢 24G 必挂）
STALL_TIMEOUT = int(os.environ.get("FVSR_STALL_TIMEOUT", "900"))
def _real_run(job: dict, meta: dict) -> None:
    """调 /root/FlashVSR_Plus 的 inference.run_stream，逐行把 stdout 变成进度+日志"""
    p = job["params"]
    if meta["kind"] != "video":
        raise RuntimeError("真引擎只吃视频，图片请用模拟引擎或直接 ffmpeg")

    # 同一时间只跑一个：两个进程同时占 24G 显存 = 整机卡死（已经踩过两次）
    if not REAL_LOCK.acquire(blocking=False):
        raise RuntimeError("已有真实推理在跑，等它结束再来（24G 显存扛不住两个模型）")
    try:
        _real_infer(job, meta, p)
    finally:
        REAL_LOCK.release()


def _real_infer(job: dict, meta: dict, p: dict) -> None:
    mode = p["model"] if p["model"] in MODES else "turbo"
    tcomp = int(p.get("temporal_compression", 4))
    if tcomp not in (1, 2, 4):
        tcomp = 4

    src = Path(job["src"])
    w, h = target_size(meta, p)
    job["out_w"], job["out_h"] = w, h

    # 不再自动降档（之前把 720×1280 压成 768×1152，比原片还小）。
    # 该拦的在 /api/job 的预检里已经拦了，这里只做提示。
    area = w * h
    src_mp = meta["width"] * meta["height"] / 1e6
    if area <= src_mp * 1e6:
        log(job, f"⚠ 输出 {w}×{h}（{area / 1e6:.2f} MP）不比原片 {meta['width']}×{meta['height']} "
                 f"（{src_mp:.2f} MP）大，等于没放大", "warn")
    elif area > 6.0e6:
        log(job, f"⚠ 输出 {w}×{h} = {area / 1e6:.1f} MP，显存需求很高（预检里应已提示）", "warn")

    out_path = OUTPUTS / f"{job['id']}_sr.mp4"
    cmd = [PYBIN, "-u", "-m", "inference.run_stream",
           "--input", str(src), "--output", str(out_path),
           "--mode", mode, "--height", str(h), "--width", str(w),
           "--temporal-compression", str(tcomp),
           "--seed", str(int(p.get("seed") or 1)),
           "--video-preset", p.get("video_preset", "veryfast")]
    if p.get("low_vram"):
        cmd += ["--cache-offload-mode", "always"]
    k = p.get("attn_kernel") or "auto"
    if k != "auto":
        cmd += ["--attn-kernel", k]
    de = str(p.get("decode_every") or "auto")
    if de not in ("", "auto", "0"):
        cmd += ["--decode-every", de]
    if p.get("color_fix"):
        cmd += ["--cf"]
    if p.get("max_frames"):
        cmd += ["--max-frames", str(int(p["max_frames"]))]

    def bump(pct: float, stage: str | None = None) -> None:
        """进度只增不减（v2：不再出现 78% → 19% 的倒退）"""
        with LOCK:
            if pct > job.get("progress", 0.0):
                job["progress"] = round(min(pct, 99.5), 1)
            if stage:
                job["stage"] = stage

    log(job, f"真引擎 · {mode} · {meta['width']}×{meta['height']} → {w}×{h} · tcomp={tcomp}")
    log(job, "cmd: " + " ".join(cmd), "dim")
    if not p.get("low_vram"):
        log(job, "未开省显存模式：1280×768 这类尺寸峰值会到 23GB 左右；"
                 "如果机器卡死，勾上「省显存模式」再来（会慢 3~5 倍）", "warn")
    bump(2.0, "启动推理进程")

    env = dict(os.environ)
    env["PATH"] = f"/root/miniconda3/bin:/usr/local/cuda/bin:{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("NO_COLOR", None)

    # 先把上次可能残留的推理进程清掉（否则两个模型抢显存直接把机器拖死）
    try:
        subprocess.run(["pkill", "-f", "inference[.]run_stream"], timeout=10)
        time.sleep(1.0)
    except Exception:
        pass

    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    # 里程碑之间让进度条自己慢慢爬，别卡在 12% 不动
    stop_tick = threading.Event()
    CEIL = {"v": 40.0}   # 加载权重阶段最多爬到 40%，之后由里程碑抬升

    def ticker():
        while not stop_tick.wait(1.5):
            with LOCK:
                pr = job.get("progress", 1.0)
                if pr < CEIL["v"]:
                    job["progress"] = round(min(CEIL["v"], pr + max(0.35, (CEIL["v"] - pr) * 0.03)), 1)

    threading.Thread(target=ticker, daemon=True).start()

    peak = {"mb": 0}

    def vram_sampler():
        while not stop_tick.wait(2.0):
            q = run(["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"], timeout=6)
            if q.returncode == 0 and q.stdout.strip():
                try:
                    mb = int(float(q.stdout.splitlines()[0].strip()))
                    if mb > peak["mb"]:
                        peak["mb"] = mb
                        with LOCK:
                            job["vram_peak"] = mb
                except Exception:
                    pass

    threading.Thread(target=vram_sampler, daemon=True).start()
    def watchdog():
        while not stop_tick.wait(10):
            idle = time.time() - job.get("last_activity", time.time())
            if idle > STALL_TIMEOUT and proc.poll() is None:
                log(job, f"⚠ {int(idle)}s 没有任何输出，判定卡死 → 杀掉进程", "err")
                try:
                    proc.kill()
                except Exception:
                    pass
                return

    threading.Thread(target=watchdog, daemon=True).start()

    try:
        for raw in proc.stdout:
            line = ANSI.sub("", raw).strip()
            if not line:
                continue
            log(job, line)
            seg = re.search(r"Decode segment (\d+)/(\d+)", line)
            if seg:
                a, b = int(seg.group(1)), max(1, int(seg.group(2)))
                lo, hi = CEIL["v"], min(84.0, CEIL["v"] + 10)
                bump(lo + (hi - lo) * a / b, f"流式推理 · 解码片段 {a}/{b}")
            elif "Loading model weights" in line:
                bump(5.0, "加载权重（ckpt 在网络盘上，冷启动 1~2 分钟属正常）")
            elif "Model loading complete" in line:
                CEIL["v"] = 42.0
                bump(42.0, "权重就绪")
            elif "Running inference" in line:
                CEIL["v"] = 84.0
                bump(46.0, "流式推理中")
            elif "DiT stream inference complete" in line:
                CEIL["v"] = 88.0
                bump(84.0, "DiT 流式推理完成")
            elif "TinyDecoder complete" in line:
                CEIL["v"] = 91.0
                bump(89.0, "解码器完成")
            elif "Color correction complete" in line:
                CEIL["v"] = 94.0
                bump(92.0, "色彩校正完成")
            elif "Inference complete" in line:
                CEIL["v"] = 99.0
                bump(96.0, "推理完成，写盘")
            elif "Video saved" in line:
                bump(99.0, "封装完成")
    finally:
        stop_tick.set()
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"run_stream 退出码 {rc}")
    if not out_path.exists() or out_path.stat().st_size < 1024:
        raise RuntimeError("没有产出文件")

    # 对比用的“原始”低清版（等比缩到 ≤1920 宽）
    bw = min(meta["width"], 1920) // 2 * 2
    bh = int(round(meta["height"] * bw / meta["width"])) // 2 * 2
    before_path = OUTPUTS / f"{job['id']}_in.mp4"
    try:
        make_video_preview(src, before_path, bw, bh)
    except Exception as e:  # noqa: BLE001
        log(job, f"对比预览生成失败: {e}", "err")

    with LOCK:
        job["progress"] = 100.0
        job["status"] = "done"
        job["stage"] = "完成"
        job["finished"] = round(time.time() - job["t0"], 2)
        job["vram"] = 0
        job["result"] = {
            "out": media_url("outputs", out_path.name),
            "before": media_url("outputs", before_path.name) if before_path.exists() else None,
            "out_dl": dl_url("outputs", out_path.name),
            "before_dl": dl_url("outputs", before_path.name) if before_path.exists() else None,
            "width": w, "height": h,
            "elapsed": job["finished"],
            "engine": "real", "mode": mode, "tcomp": tcomp,
        }
    log(job, f"完成 · {w}×{h} · 耗时 {job['finished']}s · {out_path.stat().st_size / 1048576:.1f} MB"
             + (f" · 峰值显存 {peak['mb'] / 1024:.1f} GB" if peak["mb"] else ""), "ok")
    # 记录一次实测，供下次估算校准
    try:
        with open(DATA / "calib.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "ts": int(time.time()), "mode": mode, "mp": round(w * h / 1e6, 3),
                "tcomp": tcomp, "low_vram": bool(p.get("low_vram")),
                "decode_every": str(p.get("decode_every", "auto")),
                "peak_mb": peak["mb"], "elapsed": job["finished"],
            }) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------- 模拟推理
STAGES = [
    ("load",   "载入 FlashVSR v1.1 权重 (one-step diffusion)", 0.05),
    ("encode", "VAE 编码 + 关键帧抽取", 0.10),
    ("infer",  "稀疏注意力超分（逐窗口 streaming）", 0.62),
    ("decode", "条件解码器 + 时序一致性融合", 0.15),
    ("mux",    "编码输出 / 封装", 0.08),
]


def log(job: dict, msg: str, level: str = "info") -> None:
    with LOCK:
        job["logs"].append({"t": round(time.time() - job["t0"], 2),
                            "level": level, "msg": msg})
        job["logs"] = job["logs"][-400:]
        job["detail"] = msg
        job["last_activity"] = time.time()


def run_job(job_id: str) -> None:
    job = JOBS[job_id]
    try:
        _run_job(job_id, job)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        with LOCK:
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            job["stage"] = "失败"
        log(job, f"任务失败: {type(e).__name__}: {e}", "err")


def _run_job(job_id: str, job: dict) -> None:
    meta = job["meta"]
    if real_available():
        _real_run(job, meta)
        return
    p = job["params"]
    src = Path(job["src"])

    scale = p["scale"]
    out_w = min(meta["width"] * scale, MAX_PREVIEW_W) // 2 * 2
    out_h = int(round(meta["height"] * scale * (out_w / (meta["width"] * scale)))) // 2 * 2
    job["out_w"], job["out_h"] = out_w, out_h

    frames = max(1, meta.get("frames") or 1)
    window = p["window_size"]
    windows = math.ceil(frames / window)
    overlap = max(1, round(window * p["tile_overlap"]))
    vram = vram_estimate(p)
    job["vram"] = vram

    log(job, f"目标分辨率 {out_w}×{out_h} ({scale}×)，{frames} 帧 / {windows} 个窗口")
    log(job, f"tile={p['tile_size']} overlap={overlap}f sparse_ratio={p['sparse_ratio']} "
             f"dtype={p['dtype']} seed={p['seed']}")
    if os.environ.get("FVSR_ENGINE", "mock") == "mock":
        g = gpu_info()
        why = "未检测到 CUDA 设备" if not g["cuda"] else f"检测到 {g['name']}，但引擎配置为 mock（未接真模型）"
        log(job, why + " → 使用模拟引擎，不进行真实扩散推理", "warn")
        log(job, f"模拟引擎按估算显存 {vram} GB 运行", "warn")

    # 总时长：跟帧数/窗口数相关，但保持体验在 14~45 秒
    total = max(14.0, min(45.0, 6.0 + windows * 0.55 + frames * 0.02))
    t_start = time.time()
    stage_idx = 0
    acc = 0.0
    win_done = 0
    log(job, "开始 · stage=load")

    while True:
        elapsed = time.time() - t_start
        frac = min(elapsed / total, 0.999)
        # 追加真实 ffmpeg 时间预算（只占最后的进度尾巴）
        with LOCK:
            job["progress"] = round(frac * 96.0, 1)
            job["eta"] = round(max(0.0, (total - elapsed) / max(frac, 0.02) * (1 - frac)), 1)

        # 阶段推进
        while stage_idx < len(STAGES) and acc + STAGES[stage_idx][2] <= frac:
            acc += STAGES[stage_idx][2]
            stage_idx += 1
            if stage_idx < len(STAGES):
                log(job, f"stage={STAGES[stage_idx][0]} · {STAGES[stage_idx][1]}")
        if stage_idx < len(STAGES):
            with LOCK:
                job["stage"] = STAGES[stage_idx][1]
                job["stage_key"] = STAGES[stage_idx][0]

        # 逐窗口日志
        if STAGES[min(stage_idx, len(STAGES) - 1)][0] == "infer":
            target = int(windows * min(1.0, max(0.0, (frac - 0.15) / 0.62)))
            while win_done < min(target, windows):
                win_done += 1
                if win_done % max(1, windows // 8) == 0 or win_done == windows:
                    f0 = (win_done - 1) * window + 1
                    f1 = min(frames, win_done * window)
                    ms = random.randint(38, 96) * max(1, p["tile_size"] // 256)
                    log(job, f"window {win_done}/{windows}  f{f0}-f{f1}  "
                             f"{ms} ms  "
                             f"显存 {vram + random.uniform(-0.25, 0.35):.2f} GB", "dim")

        if time.time() - t_start > total:
            break
        time.sleep(0.35)

    # ---------------- 真实 ffmpeg 放大预览 ----------------
    with LOCK:
        job["progress"] = 97.0
        job["stage"] = "生成对比预览 (ffmpeg lanczos)"
        job["stage_key"] = "preview"
    log(job, "模拟推理结束，生成前后对比预览…")

    out_base = f"{job_id}_sr"
    before_base = f"{job_id}_in"
    is_img = meta["kind"] == "image"
    src_w, src_h = meta["width"], meta["height"]
    before_w = min(src_w, 1920) // 2 * 2
    before_h = int(round(src_h * before_w / src_w)) // 2 * 2
    out_ext = ".png" if is_img else pick_video_codec(out_w, out_h)[2]
    before_ext = ".png" if is_img else pick_video_codec(before_w, before_h)[2]
    out_path = OUTPUTS / (out_base + out_ext)
    before_path = OUTPUTS / (before_base + before_ext)

    try:
        if is_img:
            make_image_preview(src, out_path, out_w, out_h)
            make_image_preview(src, before_path, before_w, before_h)
        else:
            make_video_preview(src, out_path, out_w, out_h)
            make_video_preview(src, before_path, before_w, before_h)
        log(job, "预览已生成", "ok")
    except Exception as e:  # noqa: BLE001
        log(job, f"预览生成失败: {e}", "err")

    with LOCK:
        job["progress"] = 100.0
        job["status"] = "done"
        job["stage"] = "完成"
        job["finished"] = round(time.time() - job["t0"], 2)
        job["result"] = {
            "out": media_url("outputs", out_path.name) if out_path.exists() else None,
            "before": media_url("outputs", before_path.name) if before_path.exists() else None,
            "out_dl": dl_url("outputs", out_path.name) if out_path.exists() else None,
            "before_dl": dl_url("outputs", before_path.name) if before_path.exists() else None,
            "width": out_w, "height": out_h,
            "elapsed": job["finished"],
        }
    log(job, f"全部完成，耗时 {job['finished']}s", "ok")


# ---------------------------------------------------------------- routes
@app.after_request
def _no_store(resp):
    """HTML/JSON 一律不缓存（否则改了前端还要逼用户 Ctrl+F5）；媒体文件照常缓存"""
    ct = (resp.headers.get("Content-Type") or "")
    if resp.headers.get("Cache-Control") is None and ("html" in ct or "json" in ct):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/")
def index():
    return render_template("index.html", prefix=PREFIX)


@app.get("/api/hardware")
def api_hardware():
    g = gpu_info()
    g["ffmpeg"] = have("ffmpeg")
    g["engine"] = "real" if real_available() else "mock"
    g["modes"] = list(MODES) if real_available() else []
    g["repo"] = REPO
    return jsonify(g)


@app.post("/api/upload")
def api_upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "没有收到文件"}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        return jsonify({"error": f"不支持的类型 {ext}"}), 400

    fid = uuid.uuid4().hex[:10]
    dst = UPLOADS / f"{fid}{ext}"
    f.save(dst)
    meta = probe(dst)
    if meta.get("width", 0) == 0:
        dst.unlink(missing_ok=True)
        return jsonify({"error": meta.get("error", "无法解析该文件")}), 400

    FILES[fid] = {"path": str(dst), "meta": meta, "name": f.filename}
    meta.update({"id": fid, "filename": f.filename,
                 "url": media_url("uploads", dst.name),
                 "size_mb": round(dst.stat().st_size / 1048576, 2)})
    return jsonify(meta)


@app.post("/api/job")
def api_job():
    body = request.get_json(force=True, silent=True) or {}
    fid = body.get("file_id") or ""
    if fid not in FILES:
        return jsonify({"error": "文件不存在，请重新上传"}), 400
    params = {
        "scale": int(body.get("scale", 2)),
        "target_mp": float(body.get("target_mp", 0) or 0),
        "target_w": int(body.get("target_w", 0) or 0),
        "target_h": int(body.get("target_h", 0) or 0),
        "window_size": int(body.get("window_size", 21)),
        "tile_size": int(body.get("tile_size", 256)),
        "tile_overlap": float(body.get("tile_overlap", 0.125)),
        "sparse_ratio": float(body.get("sparse_ratio", 1.5)),
        "dtype": body.get("dtype", "bf16"),
        "seed": int(body.get("seed", 0)),
        "model": body.get("model", "turbo"),
        "temporal_compression": int(body.get("temporal_compression", 4)),
        "max_frames": int(body.get("max_frames", 0) or 0),
        "attn_kernel": body.get("attn_kernel", "auto"),
        "decode_every": body.get("decode_every", "auto"),
        "low_vram": bool(body.get("low_vram", False)),
        "allow_heavy": bool(body.get("allow_heavy", False)),
        "video_preset": body.get("video_preset", "veryfast"),

        "sparse_attention": bool(body.get("sparse_attention", True)),
        "color_fix": bool(body.get("color_fix", True)),
        "face_restore": bool(body.get("face_restore", False)),
        "keep_audio": bool(body.get("keep_audio", True)),
    }
    # 预检：会爆就直接拒绝（allow_heavy 可强开）
    meta_ = FILES[fid]["meta"]
    _w, _h = target_size(meta_, params)
    est = estimate_resources(meta_, allow_heavy=params.get("allow_heavy", False), p={
        "out_w": _w,
        "out_h": _h,
        "temporal_compression": params["temporal_compression"],
        "low_vram": params["low_vram"],
        "decode_every": params["decode_every"],
        "model": params["model"],
        "model_family": "flashvsr-plus",
    })
    if est["level"] == "block" and not params.get("allow_heavy"):
        return jsonify({
            "error": est["verdict"],
            "estimate": est,
            "hint": "；".join(est["tips"]) or "降倍数或开省显存模式",
        }), 409

    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {
        "id": jid, "status": "running", "progress": 0.0, "eta": 0.0,
        "stage": "初始化", "stage_key": "init", "logs": [], "result": None,
        "params": params, "meta": FILES[fid]["meta"], "src": FILES[fid]["path"],
        "name": FILES[fid]["name"], "t0": time.time(), "vram": 0.0,
    }
    threading.Thread(target=run_job, args=(jid,), daemon=True).start()
    return jsonify({"job_id": jid})


@app.get("/api/job/<jid>")
def api_job_status(jid: str):
    job = JOBS.get(jid)
    if not job:
        return jsonify({"error": "not found"}), 404
    with LOCK:
        return jsonify({
            "id": job["id"], "status": job["status"],
            "progress": job["progress"], "eta": job["eta"],
            "stage": job["stage"], "stage_key": job["stage_key"],
            "logs": job["logs"], "result": job["result"],
            "vram": job["vram"], "params": job["params"],
            "detail": job.get("detail", ""),
            "vram_peak": job.get("vram_peak", 0),
            "alive": round(time.time() - job.get("last_activity", time.time()), 1),
            "meta": job["meta"], "name": job["name"],
        })


@app.get("/fvdl/<kind>/<path:name>")
def fvdl(kind: str, name: str):
    """直连下载通道：Range / 断点续传 / ?dl=1 附件下载"""
    base = {"outputs": OUTPUTS, "uploads": UPLOADS}.get(kind)
    if base is None:
        return jsonify({"error": "bad kind"}), 404
    if not (base / name).is_file():
        return jsonify({"error": "not found", "name": name}), 404
    resp = send_from_directory(base, name, conditional=True)
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Cache-Control"] = "public, max-age=86400"
    if request.args.get("dl", "0") in ("1", "true", "yes"):
        resp.headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return resp


# 把原来那个 Gradio UI 反代到 /g（它自己跑在 127.0.0.1:7861，root_path=/g）
UPSTREAM = os.environ.get("FVSR_UPSTREAM", "").rstrip("/")
HOP = {"content-encoding", "transfer-encoding", "connection", "date", "server",
       "content-length", "keep-alive"}


@app.route("/g", defaults={"path": ""},
           methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
@app.route("/g/", defaults={"path": ""},
           methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
@app.route("/g/<path:path>",
           methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
def proxy_gradio(path: str):
    if not UPSTREAM:
        return jsonify({"error": "upstream 未配置（FVSR_UPSTREAM）"}), 404
    # Gradio 的 HTML 里用的是相对路径（./assets/...），必须带尾斜杠才能解析到 /g/assets/
    if path == "" and not request.path.endswith("/") and request.method == "GET":
        return redirect(request.path + "/", code=302)
    url = f"{UPSTREAM}/{path}"
    if request.query_string:
        url += "?" + request.query_string.decode()
    body = request.get_data() if request.method in ("POST", "PUT", "PATCH") else None
    keep = ("content-type", "accept", "accept-language", "range", "if-range",
            "user-agent", "cookie", "referer", "x-requested-with")
    hdrs = {k: v for k, v in request.headers.items() if k.lower() in keep}
    req = urllib.request.Request(url, data=body, headers=hdrs, method=request.method)
    try:
        up = urllib.request.urlopen(req, timeout=900)
    except urllib.error.HTTPError as e:
        up = e
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"upstream 不可达: {e}"}), 502

    def stream():
        try:
            while True:
                chunk = up.read(65536)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                up.close()
            except Exception:
                pass

    out = Response(stream_with_context(stream()), status=getattr(up, "status", 200))
    for k, v in up.headers.items():
        if k.lower() not in HOP:
            out.headers[k] = v
    return out


@app.errorhandler(404)
def _fallback_404(e):
    """没被本应用接住的路径（Gradio 的 /assets /queue /file= /theme.css 之类）转给 7861"""
    p = request.path
    if UPSTREAM and not p.startswith(("/api", "/fvdl", "/media", "/g")):
        return proxy_gradio(p.lstrip("/"))
    return e


@app.get("/media/uploads/<path:name>")
def media_upload(name: str):
    return send_from_directory(UPLOADS, name, conditional=True)


@app.get("/media/outputs/<path:name>")
def media_output(name: str):
    return send_from_directory(OUTPUTS, name, conditional=True)


@app.get("/api/demo")
def api_demo():
    """生成一段内置示例素材，方便没素材时直接点着看"""
    fid = "demo" + uuid.uuid4().hex[:6]
    dst = UPLOADS / f"{fid}.mp4"
    codec, extra, _ = pick_video_codec(640, 360)
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-f", "lavfi", "-i", "mandelbrot=size=1280x720:rate=24:maxiter=512",
           "-t", "5", "-vf", "scale=640:360:flags=bicubic",
           "-c:v", codec] + extra + ["-pix_fmt", "yuv420p", str(dst)]
    p = run(cmd, timeout=180)
    if p.returncode != 0 or not dst.exists():
        return jsonify({"error": "示例生成失败: " + (p.stderr or "")[:200]}), 500

    meta = probe(dst)
    FILES[fid] = {"path": str(dst), "meta": meta, "name": "sample_fractal_360p.mp4"}
    meta.update({"id": fid, "filename": "sample_fractal_360p.mp4",
                 "url": media_url("uploads", dst.name),
                 "size_mb": round(dst.stat().st_size / 1048576, 2)})
    return jsonify(meta)


@app.delete("/api/files/<kind>/<path:name>")
def delete_file(kind: str, name: str):
    base = {"outputs": OUTPUTS, "uploads": UPLOADS}.get(kind)
    if base is None:
        return jsonify({"error": "bad kind"}), 404
    nm = os.path.basename((name or "").strip())
    if not nm or nm.startswith("."):
        return jsonify({"error": "bad name"}), 400
    f = base / nm
    if not f.is_file():
        return jsonify({"error": "not found", "name": nm}), 404
    try:
        f.unlink()
    except OSError as e:
        return jsonify({"error": f"删除失败: {e}"}), 500
    return jsonify({"ok": True, "deleted": nm, "kind": kind})


@app.get("/api/gpu")
def api_gpu():
    """实时显卡占用（真模式下前端会轮询，兼做后端存活信号）"""
    out = {"ok": False}
    if have("nvidia-smi"):
        p = run(["nvidia-smi",
                 "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], timeout=8)
        if p.returncode == 0 and p.stdout.strip():
            try:
                name, util, used, total, temp, power = [
                    x.strip() for x in p.stdout.splitlines()[0].split(",")]
                out = {"ok": True, "name": name, "util": int(float(util)),
                       "used_mb": int(float(used)), "total_mb": int(float(total)),
                       "temp": int(float(temp)), "power": power}
            except Exception:
                pass
    procs = []
    if have("nvidia-smi"):  # 没有 nvidia-smi 时别抛 500，前端轮询会一直报错
        try:
            p = run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"], timeout=8)
        except (OSError, subprocess.SubprocessError):
            p = None
        if p is not None and p.returncode == 0:
            for line in p.stdout.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    procs.append({"pid": int(parts[0]), "mb": int(float(parts[1] or 0))})
    out["procs"] = procs
    return jsonify(out)


@app.delete("/api/outputs")
def clear_outputs():
    n = 0
    for f in OUTPUTS.glob("*"):
        if f.is_file() and not f.name.startswith("."):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return jsonify({"ok": True, "deleted": n})


@app.get("/api/sys")
def api_sys():
    """CPU / 内存占用（用户能看到到底是谁吃满了）"""
    info = {"ok": True}
    try:
        la = os.getloadavg()
        ncpu = os.cpu_count() or 1
        info["cpu_pct"] = round(min(100.0, la[0] / ncpu * 100))
        info["load"] = [round(x, 1) for x in la]
        info["ncpu"] = ncpu
    except Exception:
        pass
    try:
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":", 1)
            mem[k] = int(v.strip().split()[0]) // 1024
        info["mem_total_gb"] = round(mem["MemTotal"] / 1024, 1)
        info["mem_used_gb"] = round((mem["MemTotal"] - mem["MemAvailable"]) / 1024, 1)
        info["mem_pct"] = round((mem["MemTotal"] - mem["MemAvailable"]) / mem["MemTotal"] * 100)
    except Exception:
        pass
    return jsonify(info)


@app.get("/api/outputs")
def api_outputs():
    files = []
    for f in sorted(OUTPUTS.glob("*"), key=lambda x: -x.stat().st_mtime)[:60]:
        if not f.is_file() or f.name.startswith("."):
            continue
        st = f.stat()
        files.append({
            "name": f.name,
            "size": st.st_size,
            "size_h": human_size(st.st_size),
            "mtime": int(st.st_mtime),
            "url": media_url("outputs", f.name),
            "dl": dl_url("outputs", f.name),
            "type": "video" if f.suffix.lower() in {".mp4", ".webm", ".mov"} else "image",
        })
    return jsonify({"files": files, "direct": bool(DL_BASE)})


# ---------------------------------------------------------------- 资源估算
# 系数来自上游 README 的三档实测（2048×1152≈12G / 2560×1408≈15G / 3840×2176≈23.4G）
# 加上本机实测校准（/root/fvsr-data/calib.jsonl）
MODELS = {
    "flashvsr-plus": {
        "label": "FlashVSR+ (turbo/base/v1)",
        "weights_gb": 6.4,
        "mp_coef": 2.0,      # 每 MP 输出占的显存
        "fixed_gb": 1.6,
    },
}

GPU_TOTAL_MB = int(os.environ.get("FVSR_GPU_MB", "24564"))
RAM_TOTAL_GB = float(os.environ.get("FVSR_RAM_GB", "56"))


def estimate_resources(meta: dict, p: dict, allow_heavy: bool = False) -> dict:
    mp = (p["out_w"] * p["out_h"]) / 1e6
    frames = max(1, meta.get("frames") or 1)
    tcomp = int(p.get("temporal_compression", 4)) or 4
    low = bool(p.get("low_vram"))
    decode_auto = str(p.get("decode_every", "auto")) in ("auto", "", "0")
    m = MODELS[p.get("model_family", "flashvsr-plus")]

    vram = m["fixed_gb"] + m["mp_coef"] * mp
    if not low:
        vram += m["weights_gb"] - 1.0        # 权重常驻显存
    else:
        vram *= 0.95
    vram *= {4: 1.0, 2: 1.18, 1: 1.45}.get(tcomp, 1.2)
    if not decode_auto:
        vram *= 0.9

    # 内存：整段载入时要把 (帧数 × 扩张系数) 个 RGB 帧按 float 读进内存
    expand = 4.0 / tcomp
    ram = 2.0 + (m["weights_gb"] if low else 1.5)
    if decode_auto:
        ram += frames * expand * p["out_w"] * p["out_h"] * 3 * 4 / 1e9 * 0.35 / 1.0
        ram *= 1.0
    else:
        ram += 0.8

    # 历史实测校准：取同尺寸区间的最坏值
    worst = None
    try:
        calib = DATA / "calib.jsonl"
        if calib.exists():
            for line in calib.read_text().splitlines()[-200:]:
                r = json.loads(line)
                if abs(r["mp"] - mp) <= 0.35 and r.get("mode") == p.get("model"):
                    w = r["peak_mb"] / 1024.0
                    worst = w if worst is None else max(worst, w)
    except Exception:
        pass

    # 判定用"需求"口径（上游 README 三档实测拟合）：
    # 注意 torch 会把显存 reserve 到接近满，所以"实测保留值"只作参考，不参与判定
    vram_eff = vram
    gpu_total = GPU_TOTAL_MB / 1024.0
    v_ratio = vram_eff / gpu_total
    r_ratio = ram / RAM_TOTAL_GB

    if v_ratio > 0.95 and allow_heavy:
        verdict, level = "⚠ 超过安全线（已强制允许）", "warn"
    elif v_ratio > 0.95:
        verdict, level = "会爆显存（大概率卡死整机）", "block"
    elif r_ratio > 0.92:
        verdict, level = "会爆内存", "block"
    elif v_ratio > 0.80 or r_ratio > 0.75:
        verdict, level = "接近上限，可能卡死", "warn"
    else:
        verdict, level = "安全", "ok"

    tips = []
    if worst:
        tips.append(f"同尺寸上次实测系统峰值 {worst:.1f} GB（torch 会把显存占满，属正常）")
    if level != "ok":
        if v_ratio > 0.85:
            tips.append("降放大倍数（显存几乎只跟输出面积成正比）")
            tips.append("开「省显存模式」(cache-offload)")
            if decode_auto:
                tips.append("把 decode-every 设成 8/16，改成分段解码")
        if r_ratio > 0.75:
            tips.append("设 decode-every={8,16}，别整段载入")
        if p.get("temporal_compression", 4) != 4:
            tips.append("时间压缩调回 4×")

    return {
        "out_w": p["out_w"], "out_h": p["out_h"],
        "mp": round(mp, 2), "vram_gb": round(vram_eff, 1), "vram_pred": round(vram, 1),
        "vram_peak_seen": round(worst, 1) if worst else None,
        "ram_gb": round(ram, 1),
        "gpu_total_gb": round(gpu_total, 1), "ram_total_gb": RAM_TOTAL_GB,
        "vram_pct": round(v_ratio * 100), "ram_pct": round(r_ratio * 100),
        "verdict": verdict, "level": level, "tips": tips,
    }


@app.post("/api/estimate")
def api_estimate():
    body = request.get_json(force=True, silent=True) or {}
    meta = (FILES.get(body.get("file_id") or "", {}) or {}).get("meta")
    if not meta:
        return jsonify({"error": "先上传文件"}), 400
    scale = int(body.get("scale", 2))
    mp = float(body.get("target_mp") or 0)
    tw, th = int(body.get("target_w") or 0), int(body.get("target_h") or 0)
    if mp > 0:
        ow, oh = target_size(meta, {"target_mp": mp})
    elif tw > 0 and th > 0:
        ow, oh = min(round128(tw), 3840), min(round128(th), 3840)
    else:
        ow, oh = target_size(meta, {"scale": scale})
    p = {
        "scale": scale,
        "out_w": ow,
        "out_h": oh,
        "temporal_compression": int(body.get("temporal_compression", 4)),
        "low_vram": bool(body.get("low_vram")),
        "decode_every": body.get("decode_every", "auto"),
        "model": body.get("model", "turbo"),
        "model_family": "flashvsr-plus",
    }
    return jsonify(estimate_resources(meta, p, bool(body.get("allow_heavy"))))


def human_size(n: int) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024.0
    return f"{n:.1f} GB"


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "jobs": len(JOBS), "files": len(FILES)})


# 同时把每条路由再挂一份到 PREFIX 下，这样无论反代是否剥掉前缀都能用
if PREFIX:
    for _r in list(app.url_map.iter_rules()):
        if _r.rule == "/" or _r.rule.startswith(PREFIX):
            continue
        app.add_url_rule(PREFIX + _r.rule, endpoint=_r.endpoint + "__pfx",
                         view_func=app.view_functions[_r.endpoint],
                         methods=sorted((_r.methods or set()) - {"HEAD", "OPTIONS"}))
    app.add_url_rule(PREFIX + "/", endpoint="index__pfx", view_func=index)


if __name__ == "__main__":
    port = int(os.environ.get("FVSR_PORT", "7861"))
    print(f"FlashVSR WebUI → http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
