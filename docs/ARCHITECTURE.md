# 架构

```
browser ──► Flask(app.py) ──┬─ /                 我的 UI（单页，无构建）
                            ├─ /api/*            任务提交 / 轮询 / 预检 / GPU 与系统状态
                            ├─ /fvdl/<kind>/<f>  Range 直连下载（send_from_directory conditional）
                            └─ /g, /assets/*     反代到 7861 上的官方 Gradio UI

job 线程 ──► subprocess ──► python -m inference.run_stream   （真引擎）
                        └─ 逐行解析 stdout → 进度 / 阶段 / 日志
```

## 引擎适配层

`app.py` 顶部的 `_guess_repo()` / `_guess_py()` 决定用哪份官方代码和哪个解释器：

| 变量 | 作用 | 默认 |
|---|---|---|
| `FVSR_ENGINE` | `real` / `mock` | mock |
| `FVSR_REPO` | 官方 FlashVSR_Plus 目录 | `./third_party/FlashVSR_Plus` |
| `FVSR_PY` | 跑推理的解释器 | `./.venv/bin/python` 或 `sys.executable` |
| `FVSR_DATA` | 产物/上传/校准数据 | `./data` |
| `FVSR_PORT` / `FVSR_PREFIX` | 监听端口 / 子路径部署 | 7861 / 空 |
| `FVSR_UPSTREAM` | 老 Gradio UI 地址（挂在 /g） | 空 |
| `FVSR_GPU_MB` / `FVSR_RAM_GB` | 预检用的硬件上限 | 24564 / 56 |
| `FVSR_STALL_TIMEOUT` | 无输出多久判卡死（秒） | 900 |

加新模型 = 在 `MODELS` 里加一条（权重 GB、每 MP 显存系数），再写一个 `_xxx_infer()`
负责起进程、解析日志、调 `bump()` 报进度。UI 与预检不用改。

## 真推理的数据流

```
上传 → probe() ffprobe 元数据
     → target_size(): 目标像素档位/宽高 → 128 对齐 → 不小于原片
     → estimate_resources(): 需求显存/内存 + 判定（ok / warn / block）
     → Popen(run_stream) → stdout 逐行 → 里程碑+bump()
     → 产物落 FVSR_DATA/outputs/ → calib.jsonl 记录实测峰值
     → 前端：对比滑块（原片 vs 产物）、统计、下载传送门
```

## 进度口径

上游只在两种情况下输出进度：`Decode segment a/b`（需要 `--decode-every`）或阶段标志行。
所以进度 = **里程碑阶梯 + ticker 缓爬 + 只增不减**：

| 里程碑 | 进度 |
|---|---|
| Loading model weights | 5% |
| Model loading complete | 42% |
| Running inference | 46% |
| DiT stream inference complete | 84% |
| TinyDecoder complete | 89% |
| Color correction complete | 92% |
| Inference complete / Video saved | 96% / 99% |

## 为什么默认 7860

很多 GPU 云（仙宫云、RunPod 的 http 端口、AutoDL 的 6006…）只放行镜像里声明过的端口。
本应用默认占最常被放行的那个，官方 Gradio 退到 7861 由 `/g` 反代。
