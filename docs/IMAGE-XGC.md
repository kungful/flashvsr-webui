# 仙宫云镜像（直接用，别再装环境）

| 镜像 | id | 大小 | 权重 | 状态 |
|---|---|---|---|---|
| **完整版（推荐）** | `d1dd208b-6ebd-4028-a766-1b9e2fa9b301` | 40.6 GB | turbo + base + v1 全齐 | 2026-11-02 |
| 精简版 | `08da873c-f3d5-4805-a649-53e6d753967e` | 26.9 GB | 只有 v1 | 2026-11-02 |

> 平台不给 `saveimage` 的 `name` 生效，两个镜像在控制台里都显示成 UUID 或「临时镜像 xxxxx」。
> **要改名自己去控制台改，千万别删**（镜像只收存储费，比重新下 18 GB 权重便宜太多）。

```bash
X=~/.pi/agent/skills/xgc-api/scripts/xgc.py
python3 "$X" deploy --gpu "NVIDIA GeForce RTX 4090 D" --count 1 \
    --image d1dd208b-6ebd-4028-a766-1b9e2fa9b301 --image-type private \
    --name flashvsr --dry-run      # 先看 payload，确认后再 --yes
# 运行中后直接开
#   https://<实例ID>-7860.container.x-gpu.com
```

## 镜像里有什么

| 项 | 位置 | 说明 |
|---|---|---|
| miniconda3 / python 3.10.14 | `/root/miniconda3` | base 环境直接用，没另建 venv |
| torch 2.7.1+cu126 + torchvision 0.22.1 | site-packages | 驱动 610 起可用 |
| 官方 FlashVSR+ | `/opt/FlashVSR_Plus` | `pip install -e . --no-deps` 已装，`python -m inference.run_stream` 可用 |
| 权重 turbo | `ckpts/turbo` | dit 5.68 G + lq_proj_in 1.15 G |
| 权重 base | `ckpts/base` | dit 5.64 G + lq_proj_in 1.15 G |
| 权重 v1 | `ckpts/v1` | dit 5.68 G + lq_proj_in 0.58 G（由 ModelScope 的 ckpt 转 safetensors） |
| TCDecoder / Wan2.1 VAE | `ckpts/` | 0.19 G / 0.51 G（三档共用） |
| WebUI | `/root/app2` | Flask 占 7860；产物/上传在 `/root/fvsr-data` |
| flashvsr-attn（Triton） | site-packages | `--attn-kernel triton_block_sparse`，**必需**，见下方坑 1 |
| 开机自启 | `/scripts/start.d/flashvsr.sh` | 已 `export FVSR_ATTN_KERNEL=triton_block_sparse` |
| 补权重脚本 | `/root/fetch_ckpts.sh` + `/root/dl_parallel.py` | 换机器/重装后补齐 |
| 说明 | `/root/fvsr-image-README.txt` | 容器里直接看 |

WebUI 的 `/api/hardware` 会**按 `ckpts/<mode>/dit.safetensors` 是否在盘**决定暴露哪些档位，
所以只装了 v1 的旧镜像里前端只有 v1，补完权重刷新页面就多出 turbo/base，不用改代码。

## 下载权重：单连接是坑，必须并行

2026-11 在浙江一区实测（同一台 4090D，同一时刻）：

| 方式 | 速度 | 5.68 GB 要多久 |
|---|---|---|
| HuggingFace 单连接（= `hf download` 默认） | 0.13~0.17 MB/s | **~13 小时** |
| hf-mirror 单连接 | 0.15 MB/s | ~11 小时 |
| hf-mirror 8 连接 | 0.00 MB/s（直接 429） | 永远 |
| huggingface.co 8 连接 | 1.18 MB/s | 80 分钟 |
| huggingface.co 24 连接 | 2.98 MB/s | 32 分钟 |
| **huggingface.co 48 连接** | **8.9 MB/s** | **12 分钟** |

所以补齐权重、或者换机器重装，用 [`dl_parallel.py`](dl_parallel.py)：

```bash
bash deploy/xgc/fetch_ckpts.sh          # 缺什么补什么，内部就是 48 段并行
bash deploy/xgc/fetch_ckpts.sh turbo    # 只补 turbo
N=64 bash deploy/xgc/fetch_ckpts.sh     # 再加并发
```

它就是 `hf_transfer` 的思路（并行 Range），但纯 curl 分段、失败自动换 hf-mirror、断点续传自己做。
**v1 权重还有更快来源**：ModelScope `AI-ModelScope/FlashVSR`（本机房 10~80 MB/s，2 分钟下完），
只是要把 `diffusion_pytorch_model_streaming_dmd.safetensors` 命名为 `dit.safetensors`、
把 `LQ_proj_in.ckpt` 转成 `lq_proj_in.safetensors`（`model_io` 按后缀选加载器，直接改名会走错分支）。

## 关键坑（都是真踩过的）

1. **SDPA 后端在大尺寸是直接报错**，不是慢：
   `SDPA block-mask fallback would allocate too many mask entries; install block_sparse_attn or select a shorter sequence`
   256×384 能跑，1280×768 就炸。**必须用 triton 后端**（`--attn-kernel triton_block_sparse`）。
   Triton 后端**不需要 flash-attn**：`pip install -e operators/flashvsr_attn --no-build-isolation --no-deps`
   就够（flash-attn 的 wheel 从 GitHub release 拖不动也不影响出片）。
2. **`saveimage` 前必须先关机**：按量实例直接调会回 `code=1000 请关机后执行储存镜像操作`。
   顺序：`shutdown --release-gpu` → 等 status=`shutdown`（关机备份 3~4 分钟）→ `saveimage`
   → 等镜像 `available` → `destroy`。关机后 GPU 停止计费，只剩磁盘 ¥0.00003/GB/h。
3. **存完镜像别急着信**：第一版镜像（`08da873c`）存之前没验证过重启，用户开出来容器一直不健康
   （SSH banner 超时、7860/8888 全 502），reboot 一次才恢复。存完镜像**至少开机验一次自启**。
4. **`saveimage` 的 `name` 不生效**，镜像名落成实例 id 的 UUID；平台还会额外生成一个
   「临时镜像 xxxxx」，内容是同一份快照（可以在控制台删掉省空间，API 不给删）。
5. 系统盘扩容别乱填：`--disk-size` 传 20 G 也报「扩容系统盘大小超过最大值」，默认 100 G 够用。
6. 装 torch 认准**华为云 pypi**（本机房 10 MB/s；阿里云 0.16 MB/s、清华时快时卡）。

## 实测记录

- **完整版镜像这台**：turbo 推理 `256×384 / 8 帧` → `✓ Inference complete`；此前 v1 在
  `1280×768 / 8 帧` 任务总耗时 24 s。WebUI 上传 → 预检 → 任务 → 产物 → `/api/outputs` 全通，
  公网 `https://<id>-7860.container.x-gpu.com` 返回 200。
- **花费**：第一次做 v1 镜像 ¥1.80（85 分钟）；这次补 turbo+base + 存完整版镜像 ¥1.10
  （41 分钟，其中 18 GB 权重下载约 25 分钟）。
