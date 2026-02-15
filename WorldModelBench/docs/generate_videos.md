# WorldModelBench 视频生成脚本

## 运行命令

```bash
cd "$(git rev-parse --show-toplevel)"
conda activate wan
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 WorldModelBench/generate_videos.py
```

## 配置

| 参数 | 值 |
|---|---|
| 模型 | Wan2.2-TI2V-5B |
| 任务 | TI2V（文本+图像生成视频） |
| GPU | 2× H100 NVL 93GB（FSDP + 序列并行） |
| 分辨率 | 704×1280 |
| 帧数 | 81 |
| 采样步数 | 50（unipc solver） |
| guide_scale | 5.0 |
| sample_fps | 24 |
| seed | 42 |

## 输入输出

- **输入**: `worldmodelbench.json` 中 350 个实例，每个包含 `first_frame`（图片）+ `text_instruction`（文本提示）
- **输出**: `outputs/<stem>.mp4`，文件名与输入图片 stem 一致
- **数量**: 350 个 I2V 视频

## 特性

- 模型只加载一次，循环生成 350 个视频
- 断点续跑：已存在的 `.mp4` 自动跳过

## 生成结果

- **生成数量**: 350/350
- **总耗时**: ~8.5h（2026-02-14 12:25 → 20:49）
- **单视频耗时**: ~1.5 min（50 steps × 1.44s/step + VAE decode）
- **GPU**: 0, 1

## 评测结果（CoT）

| 维度 | 分项 | 分数 |
|---|---|---|
| Instruction Following | Overall | 2.23 / 3 |
| Physical Laws | Newton | 1.00 |
| | Mass | 0.81 |
| | Fluid | 0.99 |
| | Penetration | 0.87 |
| | Gravity | 1.00 |
| | Overall | 4.67 / 5 |
| Common Sense | Framewise | 0.89 |
| | Temporal | 0.87 |
| | Overall | 1.76 / 2 |
| **Total** | | **8.65 / 10** |

## 踩坑记录

1. **shell 脚本中 `conda activate` 失败**: 非交互式 shell 需先 `source /home/user1/miniconda3/etc/profile.d/conda.sh`
2. **评测环境缺依赖**: `vila` env 需安装 `rich` 和 `mmengine`，且 numpy 必须保持 `1.26.4`（`pip install rich mmengine numpy==1.26.4`）
3. **VILA import `ps3` 报错**: `builder.py` 无条件 import `ps3_encoder`，但 judge 模型不需要。已改为 `try/except` lazy import
4. **评测 GPU OOM**: 评测默认用 GPU 0，如果被占用会 OOM。必须用 `CUDA_VISIBLE_DEVICES` 指定空闲 GPU
