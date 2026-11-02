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
