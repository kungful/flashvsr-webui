# 踩过的坑（都是真实发生、且都修了）

按"会不会再犯"排序。换服务器商/换镜像时最先撞上的就在最上面。

## 1. 宽高必须是 128 的倍数，否则模型要么拒绝要么裁剪

`inference.run_stream` 要求 `--height/--width` 都能被 128 整除。1080、720 都不行。

- 就近取整会**变小**：`1440 → 1408`，所以 2× 可能变成 1.78×
- 现在的做法：倍数模式一律 **向上取整**（`ceil128`），并保证输出 **永不小于原片**
- 曾经更离谱的版本：我在提交后加了一段"面积超过 1.0 MP 就等比缩小"的保护，把
  `720×1280 @2×` 压成了 `768×1152`（比原片还小）。**已彻底删除**——尺寸只由
  `target_size()` 决定，预检负责拦，不做暗中缩放

## 2. 显存：torch 会 reserve 到接近满，别拿它当"需求"

实测 1280×768（0.98 MP）跑 turbo，`nvidia-smi` 显示 23.4/24 GB。这不代表它需要 23 GB
（上游 README 三档实测：2.36 MP≈12 GB、3.6 MP≈15 GB、8.36 MP≈23.4 GB）。

- 估算用**需求公式**（`vram ≈ 1.6 + 2.0×输出MP`，未开省显存再加权重 6.4 GB）
- `reserved` 峰值只作为"参考"显示，不参与判定
- 用错这个数字会导致两种灾难：白白降档（见坑 1），或者以为很安全然后卡死（见坑 3）

## 3. 24 GB 卡上两个推理进程 = 整机卡死

真实发生过两次：机器所有端口 502、SSH 连 banner 都交换不了，只能 `shutdown → boot`。

现在的防线：

1. `REAL_LOCK`：同一时间只允许一个真推理任务
2. 启动前 `pkill -f inference[.]run_stream` 清残留
3. 900 秒没有任何 stdout → 杀掉子进程并标失败
4. 预检拦截 >95% 需求面积；`允许超大面积` 可强跑（显示"已强制允许"）

## 4. 产物目录千万别放代码目录里

我把 `outputs/` 放在项目根，每次部署 `rm -rf` 旧目录再 `cp -a outputs` 回来 →
用户删掉的文件**又出现了**，被投诉"删除是摆设"。

现在产物固定放 `FVSR_DATA`（默认 `./data`，XGC 上是 `/root/fvsr-data`），代码目录里
一个产物都不留。

## 5. 平台反代只放行镜像里声明过的端口

仙宫云的 `https://<id>-7860.container.x-gpu.com` 只认 7860/8888，**80 和 7861 一律 403**
（不是 502，是 nginx 直接拒绝）。所以：

- 本应用必须占住 **7860**
- 官方那个 Gradio UI 得让路：`server_port=7861, root_path="/g"`，再由本应用反代到 `/g`
- 反代要处理 Gradio HTML 里的**相对路径**（`./assets/...`）：`/g` 必须 302 到 `/g/`，
  并且 `404` 兜底把 `/assets`、`/queue`、`/file=`、`/theme.css` 都转给上游

## 6. 前端改了但浏览器还吃缓存

Flask 默认不发 `Cache-Control`，Chrome 会启发式缓存 index.html —— 改完前端用户刷新还是旧的，
会以为"你没改"。现在 HTML/JSON 一律 `no-store`。

## 7. 传参名和上游对不上，UI 看着能调其实没用

上游 `run_stream.py` **没有 `--dtype`**（bf16 由 preset 写死），也没有 window/tile/sparse。
真引擎模式下这些开关必须**隐藏**，否则就是假开关。真实可调的只有：

`--mode turbo|base|v1`、`--height/--width`、`--temporal-compression 4|2|1`、
`--max-frames`、`--seed`、`--cf`、`--attn-kernel`、`--decode-every`、`--cache-offload-mode`、
`--video-preset`

## 8. 时间压缩会误导人

`--temporal-compression` 是"一个 VAE latent 装几个真实帧"：`4`（默认，最快）、
`2`、`1`（最慢，4 倍开销，大运动素材才用）。它不是"压缩率越高越糊"那种直觉，
UI 上必须写清"几帧/步"。

## 9. 长片吃 CPU 不是 bug

x264 编码、色彩校正、gaussian 时间合并都在 CPU 上；594 帧的片子把 14 核跑满是正常的。
应对：`--video-preset ultrafast`、`--decode-every 8/16`。右上角现在直接显示 CPU/内存占用。

## 10. 冷启动要 1~2 分钟

ckpt 在网络盘上（`/.xgcos/.links/...`），第一次加载 5.3 GB 的 dit + 1.1 GB 的 lq_proj_in
要 110 秒左右，triton 还要 autotune。**这段在计费窗口里**，所以：能一次开机跑一批就别反复开关机。

## 11. SDPA 后端在大尺寸是“直接报错”，不是慢

`--attn-kernel auto` 在没装 BSA / flash-attn 的机器上会落到 SDPA，小尺寸能跑，一旦输出到
1280×768 这类尺寸就抛：

```
RuntimeError: SDPA block-mask fallback would allocate too many mask entries;
install block_sparse_attn or select a shorter sequence
```

这不是显存不够，也不是参数错，就是 mask 太大。解：

```bash
pip install -e operators/flashvsr_attn --no-build-isolation --no-deps   # 纯 Triton，秒装
python -m inference.run_stream ... --attn-kernel triton_block_sparse
```

Triton 后端**不需要 flash-attn**（官方说“depends on FlashAttention utilities”，实测不用也能跑
1280×768/2048 级别）。WebUI 侧用 `FVSR_ATTN_KERNEL=triton_block_sparse` 把默认后端钉住。

## 12. 拉权重：单连接 13 小时，48 连接 12 分钟

国内机房实测（同一台 4090D）：`hf download` 默认单连接只有 **0.13~0.17 MB/s**（文件走
HuggingFace 的 Xet 存储，hf-mirror 也是同一套后端），6.4 GB 要 **13 小时**；hf-mirror 走
多连接直接 429。改成 48 个并行 Range 请求后：

| 方式 | 速度 | 5.68 GB |
|---|---|---|
| 单连接（`hf download` 默认） | 0.13~0.17 MB/s | ~13 h |
| huggingface.co 8 连接 | 1.18 MB/s | 80 min |
| huggingface.co 24 连接 | 2.98 MB/s | 32 min |
| **huggingface.co 48 连接** | **8.9 MB/s** | **12 min** |

用 `deploy/xgc/dl_parallel.py`（或 `fetch_ckpts.sh`，内部就是它）：失败自动换源、断点续传、
完成后校验字节数。下载前先想清楚**这钱该不该花**：GPU 实例按秒计费，拿它挂着下 13 小时的
权重 ≈ ¥20；并行下 12 分钟只要 ¥0.7。v1 权重还有 ModelScope 这条 10~80 MB/s 的快路。

## 13. 长片 + `decode-every auto` = 内存被吃光，进程被 SIGKILL（退出码 -9）

现象：UI 只显示「未知错误」，后端 `RuntimeError: run_stream 退出码 -9`。跟模型无关，跟 turbo/base 也无关。

铁证在 cgroup 里，不是靠猜：

```bash
cat /sys/fs/cgroup/memory.events
# oom_kill 3     ← 进程被内核 OOM killer 干掉了几次
# max 33417      ← 摸了多少次 56 GB 上限
```

`-9` = SIGKILL，模型根本没机会抛错，所以 WebUI 只能拿到一个退出码。

**根因**：`--decode-every` 默认 `auto`，上游会把**整个视频解码进内存**。720×1280 / 594 帧光原始帧
（转 float32）就 6.5 GB，再叠加输出缓冲、色彩校正、时间合并的中间张量，56 GB 内存一头撞死。
**显存反而是富裕的**（3.9 MP turbo 峰值只 11.9 GB / 24 GB）。

实测对照（同一段 594 帧 720×1280，turbo → 1536×2560）：

| 参数 | 结果 |
|---|---|
| `decode-every auto` | **72 s 被 SIGKILL**，内存冲到上限 |
| `decode-every 28` | 内存稳态 **35~40 GB**，显存 11.9 GB，正常分段出片 |

**已经做进 WebUI**（不用记参数）：

- 素材超过 120 帧时**自动**加 `--decode-every 28`（`FVSR_AUTO_DECODE_EVERY` / `FVSR_AUTO_DECODE_MIN_FRAMES` 可调），
  日志里会写「⚠ 素材 N 帧，自动分段解码」
- `-9` / `137` 会被翻译成「内存被 OOM killer ...」加具体建议，不再是「未知错误」
- 内存吃紧的手动填 `12`；**长片不要开省显存**（它把显存压力搬到内存，反而更容易被 OOM 杀）

## 14. base + 8 MP 不是「慢」，是直接把整机拖死（48G 卡也不行）

现象：进度条卡在 84%，前端报「后端连不上（31 次）」，SSH 也 banner 超时，GPU/CPU 双双 100%。

上游自己算出来的数是铁证（日志那一行）：

```
✓ KV cache policy: auto; offload 30 layers;
  estimated_no_offload_peak=59843MiB exceeds available VRAM (43683MiB)
```

**base + 2176×3840（8.36 MP）需要 59.8 GB 显存**，48G 卡可用只有 43.6 GB。于是它把 30 层
KV cache offload 到内存硬撑 —— 结果不是慢，是 GPU 满载 + CPU 满载 + 内存带宽打满，
Flask 和 sshd 都抢不到 CPU，整机看起来「死」了（重启才恢复）。

对照：同一个 8.36 MP，turbo 只要 ~23.4 GB（hybrid-b25 省 45% 峰值显存），48G 卡轻松跑。

所以预算要分开算（预检已按这个改，`mp_coef` 从 2.0 拆成 turbo 2.0 / base 7.0）：

| 模型 | 8.36 MP | 3.93 MP |
|---|---|---|
| turbo | 23.7 GB = 49% ✅ | 15.3 GB = 32% ✅ |
| base | **65.5 GB = 136% ❌ 会卡死整机** | 35.9 GB = 75% ⚠️ 能用但没余量 |

**结论**：base 想上 4K 得 80 GB 级别的卡；48G 卡上 base 就到 4~5 MP，8 MP 交给 turbo。
