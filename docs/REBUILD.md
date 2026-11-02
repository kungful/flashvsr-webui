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

## 2. 拉权重（**必须并行下**，单连接会等到天亮）

2026-11 实测：本机房到 HF / hf-mirror 的单连接只有 **0.13~0.17 MB/s**（文件走 Xet 存储，
`hf download` 默认就是单连接，6.4 GB 要 13 小时）；hf-mirror 多连接还会直接 429。
**48 段并行能到 8.9 MB/s（12 分钟）**。工具有两个：

```bash
# A) 一条命令补齐（内部就是 48 段并行 + 失败换源 + 断点续传）
bash deploy/xgc/fetch_ckpts.sh            # turbo + base + TCDecoder
N=64 bash deploy/xgc/fetch_ckpts.sh all   # 并发再拉高
bash deploy/xgc/fetch_ckpts.sh turbo      # 只补 turbo

# B) 单独下一个文件
python3 deploy/xgc/dl_parallel.py \
  https://huggingface.co/ruchangyao/FlashVSR_plus/resolve/main/turbo/dit.safetensors \
  third_party/FlashVSR_Plus/ckpts/turbo/dit.safetensors -n 48 \
  -r https://hf-mirror.com/ruchangyao/FlashVSR_plus/resolve/main/turbo/dit.safetensors
```

v1 权重还有个更快的来源：ModelScope `AI-ModelScope/FlashVSR`（10~80 MB/s，2 分钟下完），
但要把 `diffusion_pytorch_model_streaming_dmd.safetensors` 改名为 `dit.safetensors`、
把 `LQ_proj_in.ckpt` **转成** `lq_proj_in.safetensors`（`model_io` 按后缀选加载器，直接改名会走错分支）。
转换方法见 [`IMAGE-XGC.md`](IMAGE-XGC.md)。

拉完先核对大小，再起服务：

```bash
python3 deploy/xgc/check_ckpts.py          # 或直接 ls -l ckpts/*/
```

期望（十进制 GB）：turbo/dit 5.68、turbo/lq_proj_in 1.15、base/dit 5.64、
base/lq_proj_in 1.15、v1/dit 5.68、v1/lq_proj_in 0.58、TCDecoder 0.19、Wan2.1_VAE 0.51。

> 时间紧可以先只拉 `turbo` + `TCDecoder.ckpt`（约 6.8 GB，12 分钟），
> WebUI 会自动只显示权重齐全的档位。

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
