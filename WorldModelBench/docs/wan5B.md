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

### 与论文 Model Judger 排名对比

| 排名 | 模型 | 类型 | Instr. | Frame | Temp. | Newton | Mass | Fluid | Penetr. | Grav. | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | KLING | closed | 2.32 | 0.99 | 0.97 | 1.00 | 0.90 | 1.00 | 0.93 | 0.99 | 9.10 |
| 2 | Minimax | closed | 2.28 | 0.99 | 0.93 | 1.00 | 0.86 | 0.99 | 0.88 | 0.99 | 8.92 |
| 3 | Mochi-official | closed | 2.00 | 0.97 | 0.89 | 1.00 | 0.88 | 1.00 | 0.93 | 0.99 | 8.66 |
| **4** | **Wan2.2-TI2V-5B** | **open** | **2.23** | **0.89** | **0.87** | **1.00** | **0.81** | **0.99** | **0.87** | **1.00** | **8.65** |
| 5 | Runway | closed | 2.17 | 0.99 | 0.87 | 1.00 | 0.77 | 0.98 | 0.89 | 0.96 | 8.64 |
| 6 | Luma | closed | 1.98 | 0.96 | 0.81 | 1.00 | 0.70 | 0.98 | 0.87 | 0.95 | 8.24 |
| 7 | OpenSoraPlan-T2V | open | 1.72 | 0.83 | 0.85 | 1.00 | 0.77 | 0.99 | 0.91 | 0.98 | 8.04 |
| 8 | Mochi | open | 2.06 | 0.78 | 0.68 | 0.99 | 0.63 | 0.99 | 0.79 | 0.98 | 7.91 |
| 9 | CogVideoX-T2V | open | 2.03 | 0.75 | 0.60 | 0.99 | 0.58 | 0.99 | 0.73 | 0.98 | 7.65 |

### 分析

**总分**: 8.65，全场第 4、**开源第 1**，仅次 KLING/Minimax/Mochi-official，与 Mochi-official 仅差 0.01，超越 Runway (closed)。比此前最佳开源 OpenSoraPlan-T2V (8.04) 高 **+0.61**。

**强项**:
- Newton 1.00、Gravity 1.00 — 并列全场最高
- Instruction Following 2.23 — 开源最高，超越 Mochi-official (2.00)、Runway (2.17)
- Fluid 0.99 — 持平开源最佳

**弱项**:
- Penetration 0.87 — 低于 OpenSoraPlan (0.91)、闭源模型 (0.88-0.93)
- Common Sense Frame 0.89 — 明显低于闭源 (0.96-0.99)，但仍为开源最高

**注意**: 论文中 Physics 子维度名称有差异（Human 表用 "Mass"，Model Judger 表用 "Deform."），此处按 wan5B 评测输出使用 "Mass"。

## 踩坑记录

1. **shell 脚本中 `conda activate` 失败**: 非交互式 shell 需先 `source /home/user1/miniconda3/etc/profile.d/conda.sh`
2. **评测环境缺依赖**: `vila` env 需安装 `rich` 和 `mmengine`，且 numpy 必须保持 `1.26.4`（`pip install rich mmengine numpy==1.26.4`）
3. **VILA import `ps3` 报错**: `builder.py` 无条件 import `ps3_encoder`，但 judge 模型不需要。已改为 `try/except` lazy import
4. **评测 GPU OOM**: 评测默认用 GPU 0，如果被占用会 OOM。必须用 `CUDA_VISIBLE_DEVICES` 指定空闲 GPU
