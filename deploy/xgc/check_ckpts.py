#!/usr/bin/env python3
"""核对 FlashVSR+ 权重是否齐、大小对不对（下完先跑这个，比 `--mode` 报错好查）。

    python3 deploy/xgc/check_ckpts.py                 # 默认查 third_party/FlashVSR_Plus/ckpts
    python3 deploy/xgc/check_ckpts.py /opt/FlashVSR_Plus/ckpts
"""
import pathlib
import sys

# 十进制 GB，来自 HF 仓库实测；容差 0.2 GB
WANT = {
    "turbo/dit.safetensors": 5.68,
    "turbo/lq_proj_in.safetensors": 1.15,
    "base/dit.safetensors": 5.64,
    "base/lq_proj_in.safetensors": 1.15,
    "v1/dit.safetensors": 5.68,
    "v1/lq_proj_in.safetensors": 0.58,
    "TCDecoder.ckpt": 0.19,
    "Wan2.1_VAE.pth": 0.51,
}

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "third_party/FlashVSR_Plus/ckpts")
print(f"ckpts: {root.resolve()}")
modes = {}
for rel, gb in WANT.items():
    p = root / rel
    if not p.is_file():
        print(f"  ✗ {rel:34s} MISSING（期望 {gb:.2f} GB）")
        continue
    size = p.stat().st_size / 1e9
    ok = abs(size - gb) < 0.2
    print(f"  {'✓' if ok else '!'} {rel:34s} {size:.2f} GB" + ("" if ok else f"（期望 {gb:.2f}，可能没下完）"))
    if rel.count("/") == 1:
        m = rel.split("/")[0]
        modes.setdefault(m, True)
        modes[m] &= ok

print("\n可用的档位：", ", ".join(sorted(m for m, ok in modes.items() if ok)) or "（无，连 TCDecoder 都没齐）")
print("缺的档位：", ", ".join(sorted(m for m, ok in modes.items() if not ok)) or "无")
sys.exit(0 if all(modes.values()) else 1)
