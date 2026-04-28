# 远程运行实验指南

## 可用机器

| 主机名 | 说明 |
|--------|------|
| nnmc60 | 8×L40S (46GB each) fw60 |
| nnmc61 | 8×L40S (46GB each) fw61 |
| nnmc62 | 8×L40S (46GB each) fw62 |
| nnmc63 | 8×L40S (46GB each) fw63 |
| nnmc64 | 8×L40S (46GB each) fw64 |
| nnmc65 | 8×L40S (46GB each) fw65 — ❌ 不可用 |
| nnmc71 | 8×H100 NVL (95GB each) fw71 — ❌ 不可用 |
| nnmc72 | 8×H100 NVL (95GB each) fw72 — ❌ 不可用 |
| juyi-benchmark | Google Cloud, us-central1-b, 内网 10.128.0.41, 外网 136.113.77.135, 用 `gcloud compute ssh juyi-benchmark --zone=us-central1-b` 连接 |
| ak-a100-80 | Google Cloud, us-east5-a, A100 80GB, 内网 10.202.0.19, 外网 34.186.253.190, 用 `gcloud compute ssh ak-a100-80 --zone=us-east5-a` 连接 |
| changdi-4-a100 | Google Cloud, us-central1-c, A100, 内网 10.128.15.196, 外网 35.226.225.203, 用 `gcloud compute ssh changdi-4-a100 --zone=us-central1-c` 连接 |
| nan-a100 | Google Cloud, us-central1-c, A100, 内网 10.128.15.239, 外网 34.30.51.212, 用 `gcloud compute ssh nan-a100 --zone=us-central1-c` 连接 |
| juyi-finetune | Google Cloud, us-east5-a, 4×A100 80GB, 内网 10.202.0.2, 外网 34.162.110.172, 用 `gcloud compute ssh juyi-finetune --zone=us-east5-a` 连接 |
| juyi-videorl | Google Cloud, us-east5-a, 8×A100 80GB, 内网 10.202.0.50, 外网 34.162.88.154, 用 `gcloud compute ssh juyi-videorl --zone=us-east5-a` 连接 |

## 重要：共享目录

`/shared/user60/worldmodel/` 是所有机器共享的，**不需要 SSH 来读写文件**。本地直接读写即可。SSH 只用于在远程机器上**执行命令**（跑脚本、查 GPU 状态等）。

## 基本操作

```bash
# 检查 GPU 使用情况
ssh nnmc71 "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader"

# 检查谁在跑
ssh nnmc71 "nvidia-smi"
```

## 跑实验的标准流程

### 1. 确认 GPU 空闲

```bash
ssh nnmc71 "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader"
```

### 2. 在远程机器上跑脚本

SSH 过去执行，需要先 cd 到项目目录、激活 conda 环境：


## 端口转发

juyi-benchmark 没有共享目录，需要通过 SSH 远程端口转发来暴露本地服务：

```bash
# 把 juyi-benchmark:5001 转发到本地 127.0.0.1:5001
# 这样外部可以通过 http://136.113.77.135:5001 访问本地 Flask 服务
gcloud compute ssh juyi-benchmark --zone=us-central1-b -- \
  -N -R 5001:127.0.0.1:5001 \
  -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes
```

前提：本地 Flask 服务需要在 `127.0.0.1:5001` 运行。juyi-benchmark 上的 `/etc/ssh/sshd_config` 需要 `GatewayPorts yes` 才能让外网访问转发端口。

## 注意事项

- 开始长任务前，更新 `docs/machinedoing.md` 记录哪台机器在跑什么
- GPU 4-7 经常被占用，先检查再跑
- VLM 服务跑在 `http://10.145.87.71:29672/v1`（Qwen3.5-27B, 4fps），port 29671 已 deprecated
- 结果输出在 `data/videos/` 下，本地直接可以看

