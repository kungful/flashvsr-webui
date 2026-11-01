# FlashVSR WebUI

给 [FlashVSR+](https://github.com/OpenImagingLab/FlashVSR_Plus)（CVPR 2026 系的一步扩散流式视频超分）
做的一套**网页界面 + 部署脚手架**。官方代码不 vendored 进来，由 `scripts/bootstrap.sh` 按固定 commit 克隆，
本仓库只维护 UI、引擎适配层和部署脚本。

```
官方 FlashVSR+ (Apache-2.0, 深度学习部分)
        ▲  --input/--height/--width/--mode/--temporal-compression ...
        │
本仓库  WebUI(Flask + 单页) ──► 任务/进度/日志/显存预检/对比播放/下载传送门
        │
        └──► 部署脚手架：裸机 systemd / Docker / 仙宫云一键脚本
```

## 特性

- **真推理**：直接驱动官方 `python -m inference.run_stream`，不是 demo
- **显存/内存预检**：选完参数立刻算"会不会爆"，超了就禁用按钮并给降档建议（可用「允许超大面积」强跑）
- **实时状态**：右上角 GPU 利用率/显存/温度 + CPU/内存，卡住时前端会明确报"后端连不上"
- **安全**：同一时刻只允许一个推理任务、残留进程自动清理、无输出 900s 判卡死、输出永不小于原片
- **尺寸档位**：只选"几百万像素"，长宽比永远跟素材一致（竖屏就出竖屏），并按 128 对齐
- **下载传送门**：`/fvdl` 原生 Range/206 断点续传；页面内「⚡ 极速」8 路并发；可删除、可清空
- **可迁移**：一套代码跑裸机 / Docker / 仙宫云，路径与解释器全靠环境变量

## 快速开始

```bash
git clone <this repo> && cd flashvsr-webui

# 1) 只看 UI（不需要 GPU、不用下载权重）
make dev            # → http://127.0.0.1:7861  （mock 引擎）

# 2) 真推理
./scripts/bootstrap.sh                 # 克隆官方代码 + venv + torch + 依赖
                                       # 权重按上游 README 放到 third_party/FlashVSR_Plus/ckpts
make real           # → http://127.0.0.1:7860

# 3) 冒烟测试（起服务 → 上传 → 跑一次 → 检查产物）
make smoke
```

## 部署

| 场景 | 命令 | 说明 |
|---|---|---|
| 任意 Linux + GPU | `deploy/systemd/flashvsr-webui.service` | 最通用，暴露 7860 |
| Docker / RunPod / Vast | `make docker-up` | 权重挂载，别打进镜像 |
| 仙宫云 XGC | `./deploy/xgc/deploy_xgc.sh` | 一键：开实例→装→自检 |

细节见 [`docs/PROVIDERS.md`](docs/PROVIDERS.md)。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `FVSR_ENGINE` | `mock` | `real` 走官方模型，`mock` 只演 UI |
| `FVSR_PORT` | `7861` | 监听端口（云上一般用 7860） |
| `FVSR_DATA` | `./data` | 产物/上传/校准数据，**必须独立于代码目录** |
| `FVSR_REPO` | `./third_party/FlashVSR_Plus` | 官方代码位置 |
| `FVSR_PY` | venv 或 sys.executable | 跑推理的解释器 |
| `FVSR_PREFIX` | 空 | 部署在子路径（如 `/fv`）时用 |
| `FVSR_UPSTREAM` | 空 | 官方 Gradio UI 地址，反代到 `/g` |
| `FVSR_GPU_MB` / `FVSR_RAM_GB` | 24564 / 56 | 预检用的硬件上限 |
| `FVSR_SAFE_MP` / `FVSR_STALL_TIMEOUT` | — | 面积安全线 / 卡死判定秒数 |

## 接口

```
GET    /api/hardware         引擎、GPU、ffmpeg、可用 mode
POST   /api/upload           上传并 ffprobe（multipart: file）
POST   /api/estimate         显存/内存预检 {file_id, scale|target_mp, low_vram, decode_every...}
POST   /api/job              建任务（预检会直接 409 拒绝超限）
GET    /api/job/<id>         进度 / 阶段 / 日志 / 峰值显存
GET    /api/outputs          产物列表
DELETE /api/outputs          清空产物
DELETE /api/files/<kind>/<n> 删单个产物（outputs|uploads）
GET    /api/gpu | /api/sys   实时显卡 / CPU 内存
GET    /fvdl/<kind>/<name>   直连下载（Range，?dl=1 附件）
GET    /g                    反代到官方 Gradio UI
```

## 从零重建

镜像/实例全丢时看 [`docs/REBUILD.md`](docs/REBUILD.md)：装环境 → 拉权重（hf-mirror）→ 起服务 → **存镜像**。

## 已知限制（务必读）

- 宽高必须 **128 的倍数**，所以 1080p 的 2× 是 `3840×2176`
- 24 GB 卡的建议甜点是 **3.6 MP**（1536×2560 / 2560×1536）
- 冷启动加载权重约 **1~2 分钟**（在计费窗口里）
- 长片 CPU 跑满正常：x264 编码 + 色彩校正 + 时间合并在 CPU 上
- 完整踩坑记录：[`docs/PITFALLS.md`](docs/PITFALLS.md)（建议换机器前先读）

## 许可

本仓库代码 Apache-2.0。上游 [FlashVSR_Plus](https://github.com/OpenImagingLab/FlashVSR_Plus)
同样是 Apache-2.0，其 LICENSE 与版权归原作者所有；本项目只提供界面与部署层，不分发其权重。
