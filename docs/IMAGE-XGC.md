# 仙宫云镜像（直接用，别再装环境）

**镜像 id：`08da873c-f3d5-4805-a649-53e6d753967e`** · 26.9 GB · 2026-11-02 制作 · status `available`

> 平台把镜像名存成了 UUID（`saveimage` 的 `name` 参数没生效），控制台里显示的名字就是那串 id，
> introduction 是「实例flashvsr-imgbuild储存的镜像」。**要改名请自己在控制台改**，别删它。

```bash
X=~/.pi/agent/skills/xgc-api/scripts/xgc.py
python3 "$X" deploy --gpu "NVIDIA GeForce RTX 4090 D" --count 1 \
    --image 08da873c-f3d5-4805-a649-53e6d753967e --image-type private \
    --name flashvsr --dry-run      # 先看 payload，再 --yes
# 开机后等 status=running，直接访问
#   https://<实例ID>-7860.container.x-gpu.com
```

## 镜像里有什么

| 项 | 位置 | 说明 |
|---|---|---|
| miniconda3 / python 3.10.14 | `/root/miniconda3` | base 环境直接用，没另建 venv |
| torch 2.7.1+cu126 + torchvision 0.22.1 | site-packages | 驱动 610 起可用 |
| 官方 FlashVSR+ | `/opt/FlashVSR_Plus` | `pip install -e . --no-deps` 已装，`python -m inference.run_stream` 可用 |
| 权重 v1 | `/opt/FlashVSR_Plus/ckpts/v1` | dit 5.7 G + lq_proj_in 0.57 G |
| TCDecoder / Wan2.1 VAE | `ckpts/` | 0.18 G / 0.49 G |
| WebUI | `/root/app2` | Flask，占 7860；产物在 `/root/fvsr-data` |
| flashvsr-attn（Triton） | site-packages | `--attn-kernel triton_block_sparse` 实测可用 |
| 开机自启 | `/scripts/start.d/flashvsr.sh` | 已 `export FVSR_ATTN_KERNEL=triton_block_sparse` |
| 补权重脚本 | `/root/fetch_ckpts.sh` | 拉 turbo/base（网好时跑） |
| 说明 | `/root/fvsr-image-README.txt` | 同上内容，容器里直接看 |

## 只装了 v1，turbo / base 要补

本机房（浙江一区）到 HuggingFace **只有 0.3~0.9 MB/s**（文件走 Xet 存储，`hf-mirror.com`
302 到 `cas-bridge.xethub.hf.co`，慢到 6.4 G 要几小时，且很快 429），所以 turbo/base 没拉。

**v1 的权重是从 ModelScope 拿的**（`AI-ModelScope/FlashVSR`，本机房 10~80 MB/s）：
`diffusion_pytorch_model_streaming_dmd.safetensors` → `ckpts/v1/dit.safetensors`，
`LQ_proj_in.ckpt` → 转成 `ckpts/v1/lq_proj_in.safetensors`（`model_io` 按后缀选加载器，
直接改名 ckpt 会走错分支）。

补 turbo/base：

```bash
bash /root/fetch_ckpts.sh        # turbo（6.4 G）
bash /root/fetch_ckpts.sh all    # turbo + base
```

WebUI 的 `/api/hardware` 现在会**按 `ckpts/<mode>/dit.safetensors` 是否在盘**决定暴露哪些档位，
所以补完刷新页面就多出 turbo/base，不用改代码。

## 关键坑（这次踩到的）

1. **SDPA 后端在大尺寸直接报错**，不是慢：
   `SDPA block-mask fallback would allocate too many mask entries; install block_sparse_attn or select a shorter sequence`
   256×384 能跑，1280×768 就炸。**必须用 triton 后端**（`--attn-kernel triton_block_sparse`）。
   好消息：Triton 后端**不需要 flash-attn**，`pip install -e operators/flashvsr_attn --no-build-isolation --no-deps`
   就够；flash-attn 的 wheel 从 GitHub release 拖不动（直连/RunPod 代理都超时）也不影响出片。
   所以 start.d 里钉了 `FVSR_ATTN_KERNEL=triton_block_sparse`。
2. **`saveimage` 前必须先关机**：按量实例调 `/open/instance/saveimage` 会回
   `code=1000 请关机后执行储存镜像操作`。正确顺序：
   `shutdown --release-gpu`（GPU 停止计费，只剩磁盘 ¥0.00003/GB/h）→ 等 status=`shutdown`
   （关机备份约 3~4 分钟）→ `saveimage` → 等镜像 `available` → `destroy`。
3. **`saveimage` 的 `name` 不生效**，镜像名落成实例 id 的 UUID，description 也没进名字。
   存完自己看一眼 `xgc.py images` 认一下 id。
4. 系统盘扩容别乱填：`--disk-size` 传 20 G 也报「扩容系统盘大小超过最大值」，默认 100 G 够用。
5. 装 torch 认准**华为云 pypi**（本机房 10 MB/s；阿里云 0.16 MB/s、清华时快时卡）。
   `pip install torch==2.7.1 torchvision==0.22.1` 大约 13 分钟（含 2.5 G nvidia 轮子）。

## 实测评测（本次）

- 640×360 → 1280×768，8 帧，v1 + triton：**任务总耗时 24 s**（不含 1~2 min 冷启动）
- WebUI：上传 → 预检 → 任务 → 产物 → `/api/outputs` 全通；公网 `https://<id>-7860.container.x-gpu.com` 200
- 花费：整场（装环境 + 拉 7 G 权重 + 两次实测 + 存镜像）**¥1.80**，85 分钟计费窗
