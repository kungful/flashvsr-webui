# 从零重建（机器/镜像全丢时照这个做）

前提：一台有 NVIDIA 卡的机器，驱动正常，约 **15 GB 空闲磁盘**，出网能到 GitHub + HuggingFace。

## 0. 需要的东西

| 东西 | 从哪来 | 大小 |
|---|---|---|
| 本仓库 | `git clone <repo>` | ~100 KB |
| 官方 FlashVSR+ 代码 | `scripts/bootstrap.sh` 自动克隆（固定 commit） | ~30 MB |
| Python 依赖 | bootstrap 装（torch cu124 + requirements） | ~3 GB |
| 权重 turbo / base | HuggingFace [`ruchangyao/FlashVSR_plus`](https://huggingface.co/ruchangyao/FlashVSR_plus) | 各 6.4 GB |
| `TCDecoder.ckpt` | 同一 HF 仓库（**turbo/base 共用**） | ~0.5 GB |
| 可选 v1 权重 | [`JunhaoZhuang/FlashVSR-v1.1`](https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1) | 6.4 GB |
| 可选 VAE 解码器 | `ckpts/Wan2.1_VAE.pth`（仅 `--decoder vae` 需要） | ~0.5 GB |

目录必须是这个形状（`ckpts/` 在官方仓库根下）：

```
third_party/FlashVSR_Plus/ckpts/
├── turbo/{dit.safetensors, lq_proj_in.safetensors}
├── base/{dit.safetensors,  lq_proj_in.safetensors}
├── TCDecoder.ckpt
└── Wan2.1_VAE.pth        # 可选
```

## 1. 装环境（10 分钟）

```bash
git clone <repo> flashvsr-webui && cd flashvsr-webui
./scripts/bootstrap.sh          # 官方代码 + .venv + torch + requirements + flask
```

## 2. 拉权重（国内用 hf-mirror，别直连）

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U "huggingface_hub[cli]"

CK=third_party/FlashVSR_Plus/ckpts
mkdir -p $CK/turbo $CK/base

# turbo
hf download ruchangyao/FlashVSR_plus dit.safetensors   --local-dir /tmp/w/turbo
hf download ruchangyao/FlashVSR_plus lq_proj_in.safetensors --local-dir /tmp/w/turbo
# base（同一 HF 仓库里另一组文件，按仓库 actual 文件名下载）
hf download ruchangyao/FlashVSR_plus --local-dir /tmp/w/all   # 全量拉一次最省事
# 再按文件名分到 turbo/ 与 base/，TCDecoder.ckpt 放 ckpts/ 根

python - <<'PY'
import glob, os, shutil, pathlib
ck = pathlib.Path("third_party/FlashVSR_Plus/ckpts")
print("ckpts 现状：")
for p in sorted(ck.rglob("*")):
    if p.is_file(): print(f"  {p.relative_to(ck)}  {p.stat().st_size/1e9:.2f} GB")
PY
```

> 权重文件名与分组以 HF 仓库页面为准；只要最终满足上面的目录形状，`--mode turbo|base` 就能跑。
> 时间紧可以先只拉 `turbo` + `TCDecoder.ckpt`（约 7 GB），UI 里 base 会加载失败但不影响 turbo。

## 3. 起服务

```bash
FVSR_ENGINE=real FVSR_PORT=7860 FVSR_DATA=$PWD/data \
FVSR_REPO=$PWD/third_party/FlashVSR_Plus FVSR_PY=$PWD/.venv/bin/python \
deploy/entrypoint.sh
```

或者 `make real`，或者 `make docker-up`（权重挂载进容器）。

**必须对外暴露 7860**（云厂商控制台的 HTTP 端口 / 安全组 / `-p 7860:7860`）。

## 4. 验证

```bash
curl -s localhost:7860/api/hardware        # engine 应为 real，modes 有 turbo/base/v1
./scripts/smoke_test.sh                    # mock 链路（不需要 GPU，验证 UI/上传/产物/下载）
```

真推理验证：上传一段 5 秒小视频 → 选 2× turbo → 看日志里出现
`Model loading complete / Running inference / DiT stream inference complete`，
产物分辨率应为原片的 128 对齐倍数且**不小于原片**。

## 5. 成功之后 —— 立刻存镜像（这次没做，就是丢的原因）

仙宫云按量实例**必须先关机才能存镜像**（`saveimage` 直接调会回 `请关机后执行储存镜像操作`），
关机顺序还能省钱（GPU 一停就不再按小时计费）：

```bash
X=~/.pi/agent/skills/xgc-api/scripts/xgc.py
python3 $X shutdown <实例id> --release-gpu --yes    # GPU 停止计费，只留磁盘 ¥0.00003/GB/h
# 等 status 从 shutdown_backing_up → shutdown（约 3~4 分钟）
python3 $X saveimage <实例id> --name "flashvsr-webui-$(date +%m%d)" --yes
# 等 xgc.py images 里出现 status=available（约几分钟）
python3 $X destroy <实例id> --yes                   # 镜像保留，实例和数据盘清掉
```

注意：`name` 参数在实测中**没生效**，镜像会存成实例 id 的 UUID，存完自己认一下 id。
做好的那个镜像见 [`IMAGE-XGC.md`](IMAGE-XGC.md)（`08da873c-f3d5-4805-a649-53e6d753967e`）。

## 常见坑

见 [`PITFALLS.md`](PITFALLS.md)。重建时最容易踩的三个：

1. 宽高必须 128 倍数（`3840×2176` 而不是 `3840×2160`）
2. 权重目录层级错 → 报 `FileNotFoundError: ckpts/base/dit.safetensors`
3. 端口没放行 → 外面 403/404，容器内 `curl 127.0.0.1:7860` 却是好的
