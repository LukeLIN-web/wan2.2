# WorldModelBench 视频生成脚本

## 运行命令

```bash
cd /shared/user72/workspace/juyi/Wan2.2
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
