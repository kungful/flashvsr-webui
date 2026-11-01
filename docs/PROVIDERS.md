# 换个服务器商怎么复用

一套代码，三种部署形态。**镜像里只要有 CUDA 驱动 + python3 + ffmpeg 就能跑。**

## A. 裸机 / 有 root 的云主机（最通用）

```bash
git clone <this repo> /opt/flashvsr-webui && cd /opt/flashvsr-webui
./scripts/bootstrap.sh                    # 克隆官方代码 + venv + 依赖 + 权重
sudo cp deploy/systemd/flashvsr-webui.service /etc/systemd/system/
sudo systemctl enable --now flashvsr-webui
```

对外暴露 **7860**。各家控制台都在这里填端口：RunPod 的 HTTP Port、Vast.ai 的 `-p 7860:7860`、
AutoDL 的自定义服务、阿里云的安全组。

## B. Docker（RunPod / Vast / 自建 K8s）

```bash
docker build -f deploy/docker/Dockerfile -t flashvsr-webui .
docker run --gpus all -p 7860:7860 \
  -v $PWD/data:/data \
  -v /host/ckpts:/opt/flashvsr-webui/third_party/FlashVSR_Plus/ckpts \
  flashvsr-webui
```

镜像里不放权重（6.4 GB/模式），用挂载。没有 GPU 就把 `FVSR_ENGINE=mock`，UI 照样能看。

## C. 仙宫云（XGC）—— 本项目最初的落地方式

```bash
IMAGE=<镜像id> ./deploy/xgc/deploy_xgc.sh
```

它会：开实例 → 等 running → 传代码 → 装 flask → 写 `/scripts/start.d/` 自启 →
把镜像自带的 Gradio 挪到 7861 并挂到 `/g` → 公网自检。

注意 XGC 特有的三件事（详见 `docs/PITFALLS.md`）：

1. 反代只放行镜像里声明过的端口（7860/8888），所以本应用必须占 7860
2. 产物必须放 `FVSR_DATA`（如 `/root/fvsr-data`），不能放代码目录，否则重新部署会把删掉的文件"复活"
3. 按秒计费，只在 `运行中` 计费，但**用完必须销毁**，余额归零会被自动销毁

## 显存对不同卡的换算

估算是线性的：`需求显存 ≈ 1.6 + 2.0×输出MP`（+ 6.4 GB 权重，若不开省显存）。

| 卡 | 建议上限 | 说明 |
|---|---|---|
| 16 GB (4080 / A4000) | ≈ 2.4 MP | 开省显存 + `decode-every 8` |
| 24 GB (4090 / 4090D / A10G) | ≈ 3.6 MP | 本项目实测甜点：1536×2560 |
| 48 GB (A6000 / L40S) | ≈ 8 MP | 可上 4K 档 |
| 80 GB (A100/H100) | 16 MP+ | 一般用不到，编码先成瓶颈 |

换卡请改 `FVSR_GPU_MB`，预检和拦截线会跟着变。
