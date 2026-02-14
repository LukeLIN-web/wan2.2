# 用 Wan2.2 推理 VIDEOPHY2 Benchmark

## 1. VIDEOPHY2 是什么

VIDEOPHY2 是一个评估 T2V（Text-to-Video）模型**物理常识**能力的 benchmark。包含 200 个动作类别、600 条唯一 prompt，评估三个维度：

| 维度 | 说明 | 评分 |
|------|------|------|
| **SA (Semantic Adherence)** | 视频是否匹配文本描述 | 1-5 分 |
| **PC (Physical Commonsense)** | 视频是否遵循物理定律 | 1-5 分 |
| **Rule (Physical Rule)** | 视频是否遵循特定物理规则 | 0/1/2 |

**Joint Score** = SA≥4 且 PC≥4 的比例（当前 SOTA Wan2.1-14B 只有 32.6%）。

## 2. 整体流程

```
VIDEOPHY2 prompts → Wan2.2 生成视频 → 整理成 CSV → VIDEOPHY2 AutoEval 评分
```

### 三步走：
1. **下载 VIDEOPHY2 测试集 prompts**（从 HuggingFace）— 已完成
2. **用 Wan2.2 批量生成视频**（5B 或 14B）
3. **下载 AutoEval 模型并评分**

## 3. Step 1: 获取 VIDEOPHY2 测试 Prompts（已完成）

```bash
conda activate wan
python download_videophy2.py
```

数据已下载到 `./videophy_data/`。

### videophy2_test.csv vs videophy2_prompts.csv

两个文件都在 `./videophy_data/` 下：

| | `videophy2_test.csv` | `videophy2_prompts.csv` |
|---|---|---|
| **行数** | 3397 | **600** |
| **含义** | 原始测试集，每行 = 1个prompt × 1个模型的评测结果 | **去重后的唯一 prompts，用于生成** |
| **重复** | 同一 caption 出现多次（7个模型各一条） | 每个 caption 只出现一次 |
| **模型** | 包含 hunyuan, ray2, wan, cogvideo, cosmos, videocrafter, sora 7个模型的结果 | 无模型信息 |
| **评分列** | 有 sa, pc, joint（human eval 分数） | 无评分列 |
| **用途** | 参考其他模型的表现 | **直接喂给 Wan2.2 生成视频** |

**`videophy2_test.csv` 列说明：**
- `caption` — 原始 prompt（用于生成视频）
- `upsampled_caption` — 扩展后的详细 prompt（平均 138 tokens，可选用）
- `video_url` — 该模型生成的视频 URL
- `sa`, `pc`, `joint` — human eval 分数
- `action` — 动作类别（197个，如 yoyo, bulldozing, folding clothes）
- `is_hard` — 是否属于 hard subset
- `model_name` — 生成该视频的模型名
- `physics_rules_*` — 物理规则评估详情
- `metadata_rules` — 规则对应的物理定律分类

**`videophy2_prompts.csv` 列说明：**
- `id` — 序号 0-599
- `caption` — 用于 T2V 生成的 prompt
- `upsampled_caption` — 扩展版 prompt（可选用，更详细）
- `action` — 动作类别
- `is_hard` — 是否 hard subset（600 中有 180 条）
- `category` — 大类（Sports and Physical Activities / Object Interactions）

**简单说：用 `videophy2_prompts.csv` 的 600 条 caption 生成视频就行。**

## 4. Step 2: 批量生成视频

两个模型可选：

### 模型对比

| | **TI2V-5B** | **T2V-A14B** |
|---|---|---|
| 脚本 | `run_videophy.py` | `run_videophy_14b.py` |
| Checkpoint | `./Wan2.2-TI2V-5B` | `./Wan2.2-T2V-A14B` |
| 输出目录 | `./videophy_outputs/` | `./videophy_outputs_14b/` |
| 分辨率 | `1280*704` | `1280*720` |
| 帧数 | 121 | 81 |
| 采样步数 | 50 | 40 |
| CFG scale | 5.0 | (3.0, 4.0) 双阶段 |
| Shift | 5.0 | 12.0 |
| FPS | 24 | 16 |
| VAE | Wan2.2 (16×16×4) | Wan2.1 (4×8×8) |
| 架构 | 单模型 | 双模型 (high/low noise) |
| 显存 | ~20-30GB | ~40-60GB |

### 方案 A: TI2V-5B（`run_videophy.py`）

```bash
conda activate wan
cd /shared/user72/workspace/juyi/Wan2.2

# 双 GPU 并行（单进程，内部多线程）
python run_videophy.py --gpus 0,1
```

**`run_videophy.py` 代码逻辑：**

1. 从 `videophy2_prompts.csv` 加载 600 条 prompt
2. `--start/--end` 切片选取范围，扫描 `videophy_outputs/` 跳过已有视频，得到 `to_generate` 列表
3. 按 `--gpus` 参数（默认 `0,1`）在每个 GPU 上各加载一份 TI2V-5B pipeline
4. **Round-robin 分配**：遍历 `to_generate`，按 `idx % num_gpus` 交替分给各 GPU。即 GPU 0 拿第 0、2、4... 项，GPU 1 拿第 1、3、5... 项
5. 每个 GPU 启动一个 `threading.Thread`，各自独立循环生成自己的 chunk
6. 主线程 `join()` 等待所有线程完成

### 方案 B: T2V-A14B（`run_videophy_14b.py`）

Leaderboard 上的 Wan2.1-14B 就是这个架构，结果更可比。

```bash
conda activate wan
cd /shared/user72/workspace/juyi/Wan2.2

# GPU 0 和 1 各跑一半
CUDA_VISIBLE_DEVICES=0 python run_videophy_14b.py --start 0 --end 300 &
CUDA_VISIBLE_DEVICES=1 python run_videophy_14b.py --start 300 --end 600 &
wait
```

### 通用说明

单 GPU 先测几条：
```bash
CUDA_VISIBLE_DEVICES=1 python run_videophy.py --start 0 --end 5
CUDA_VISIBLE_DEVICES=1 python run_videophy_14b.py --start 0 --end 5
```

脚本特点：
- 模型只加载一次，循环生成所有 prompt
- 自动跳过已有视频，支持断点续跑
- `--start/--end` 分片，方便多 GPU 并行
- 文件名格式 `0001_caption_text.mp4`

## 5. Step 3: 评估生成的视频

### 下载 AutoEval 模型

```bash
cd /shared/user72/workspace/juyi/Wan2.2/videophy/
git lfs install
git clone https://huggingface.co/videophysics/videophy_2_auto
```

### 准备评估 CSV

生成视频后，需要创建符合格式的 CSV 文件：

```python
"""prepare_eval_csv.py - 准备评估用 CSV"""
import os
import csv

OUTPUT_DIR = "./videophy_outputs"
PROMPT_CSV = "./videophy_data/videophy2_prompts.csv"

# 读取 prompts
prompts = []
with open(PROMPT_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        prompts.append(row)

# SA + PC 评估用 CSV
with open("eval_sa_pc.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["caption", "videopath"])
    for row in prompts:
        pid = int(row["id"])
        caption = row["caption"]
        clean = caption.replace(" ", "_").replace("/", "_").replace('"', "")
        clean = "".join(c for c in clean if c.isalnum() or c in "_-.,")[:80]
        video_path = os.path.join(OUTPUT_DIR, f"{pid:04d}_{clean}.mp4")
        if os.path.exists(video_path):
            writer.writerow([caption, video_path])
```

### 运行评估

```bash
cd /shared/user72/workspace/juyi/Wan2.2/videophy/VIDEOPHY2

# 语义匹配度评估
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --input_csv ../../eval_sa_pc.csv \
    --checkpoint ../videophy_2_auto \
    --output_csv ../../eval_output_sa.csv \
    --task sa

# 物理常识评估
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --input_csv ../../eval_sa_pc.csv \
    --checkpoint ../videophy_2_auto \
    --output_csv ../../eval_output_pc.csv \
    --task pc
```

### 计算 Joint Score

```python
"""compute_score.py - 计算 VIDEOPHY2 Joint Score"""
import pandas as pd

sa = pd.read_csv("eval_output_sa.csv")
pc = pd.read_csv("eval_output_pc.csv")

# 合并
merged = sa.merge(pc, on="videopath", suffixes=("_sa", "_pc"))

# Joint Score: SA>=4 AND PC>=4 的比例
joint = ((merged["score_sa"] >= 4) & (merged["score_pc"] >= 4)).mean() * 100
print(f"Joint Score (SA>=4 & PC>=4): {joint:.1f}%")
print(f"Mean SA: {merged['score_sa'].mean():.2f}")
print(f"Mean PC: {merged['score_pc'].mean():.2f}")
print(f"Total videos evaluated: {len(merged)}")
```

## 6. 实测结果与对比

### Baseline: Wan2.2-TI2V-5B 直接生成（2026-02-08）

| 指标 | All (600) | Hard (180) | Easy (420) |
|------|-----------|------------|------------|
| Mean SA | 3.18 | — | — |
| Mean PC | 3.61 | — | — |
| **Joint Score** | **26.3%** | **8.3%** | **34.0%** |

### PhyT2V: Wan2.2-TI2V-5B + Qwen3-VL prompt 优化（2026-02-13）

使用 PhyT2V pipeline（物理定律提取 → round1 视频 caption → mismatch 分析 → 增强 prompt → round2 生成），仅评估了 360/600 个视频（GPU 4/5 OOM 导致 240 个未生成）。

| 指标 | PhyT2V (360) | Baseline (360, 同子集) | Delta |
|------|-------------|----------------------|-------|
| Mean SA | 3.106 | 3.142 | **-0.036** |
| Mean PC | 3.631 | 3.642 | **-0.011** |
| **Joint** | **22.8%** | **24.2%** | **-1.4%** |

Hard/Easy 分布（PhyT2V, 360 个视频中 111 hard / 249 easy）：

| 子集 | SA | PC | Joint |
|------|-----|-----|-------|
| Hard (111) | 2.676 | 3.360 | **2.7%** |
| Easy (249) | 3.297 | 3.751 | **31.7%** |

逐样本变化（同 360 个 prompt 对比）：

| | SA | PC | Joint |
|---|---|---|---|
| 改善 | 40 | 62 | +22 |
| 不变 | 267 | 234 | — |
| 退化 | 53 | 64 | -27 |

**结论：PhyT2V prompt 优化未带来提升，反而略有退化。** 可能原因：
1. **仅 1 轮迭代**：PhyT2V 论文中 CogVideoX-5B 在 Round 4 才有显著提升（SA: 0.48→0.59），Round 1→2 改进有限
2. **TI2V 当 T2V 用**：round2 生成用了 TI2V-5B 但 `img=None`（无图像输入），可能不如直接 T2V 效果
3. **增强 prompt 可能过长/过细**：Qwen3-VL 生成的增强 prompt 可能引入噪声，SA 退化多于改善（53 退化 vs 40 改善）
4. **缺失 240 个样本**：未评估的样本可能影响整体趋势

### VIDEOPHY2 Leaderboard 对比（Human Evaluation）

| Model | Joint Score (All) | Joint Score (Hard) |
|-------|-------------------|-------------------|
| **Wan2.1-T2V-14B** | **32.6%** | **21.9%** |
| **Wan2.2-TI2V-5B (ours, Baseline)** | **26.3%** | **8.3%** |
| Wan2.2-TI2V-5B + PhyT2V (ours) | 22.8%* | 2.7%* |
| CogVideoX-5B | 25.0% | 0% |
| Cosmos-Diffusion-7B | 24.1% | 10.9% |
| OpenAI Sora | 23.3% | 5.3% |

*仅 360/600 个视频

- Baseline 已超过 CogVideoX-5B 和 Sora，但低于 14B 版本
- PhyT2V 单轮优化未改善结果，需要多轮迭代或调整策略

### Wan2.2-TI2V-5B VideoCon-Physics 结果（2026-02-13）

评估模型：`videocon-physics`（entailment 概率 0~1），评估 600 个视频。

| 指标 | All (600) | Hard (180) | Easy (420) |
|------|-----------|------------|------------|
| Mean SA | 0.3888 | 0.2817 | 0.4347 |
| Mean PC | 0.1840 | 0.1406 | 0.2026 |
| Combined | 0.2864 | 0.2112 | 0.3187 |

按类别：

| 类别 | 数量 | SA | PC |
|------|------|-----|-----|
| Sports and Physical Activities | 422 | 0.3834 | 0.1675 |
| Object Interactions | 159 | 0.4047 | 0.2278 |

### VideoCon-Physics 对比（同评估模型，0~1）

PhyT2V 论文 Round 1 = 原始模型直接生成，Round 4 = PhyT2V 迭代优化后。

| Model | SA | PC | 备注 |
|-------|-----|-----|------|
| **Wan2.2-TI2V-5B (ours)** | **0.39** | **0.18** | Round 1，无 PhyT2V |
| CogVideoX-5B | 0.48 | 0.26 | Round 1 |
| CogVideoX-5B + PhyT2V | 0.59 | 0.42 | Round 4 |
| VideoCrafter | 0.24 | 0.15 | Round 1 |
| VideoCrafter + PhyT2V | 0.49 | 0.33 | Round 4 |
| OpenSora | 0.29 | 0.17 | Round 1 |
| OpenSora + PhyT2V | 0.47 | 0.31 | Round 4 |
| CogVideoX-2B | 0.22 | 0.13 | Round 1 |
| CogVideoX-2B + PhyT2V | 0.42 | 0.29 | Round 4 |

- Wan2.2-TI2V-5B 的 PC=0.18 与 OpenSora Round 1（0.17）持平，但 SA=0.39 低于 CogVideoX-5B（0.48）
- PhyT2V 迭代优化能显著提升 SA 和 PC（CogVideoX-5B: SA +0.11, PC +0.16）
- Wan2.2 若加上 PhyT2V 优化，预期可达 SA~0.50, PC~0.30 左右

### 失败原因分析

#### 失分分布

| 失败类型 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| SA<4 且 PC<4 | 229 | 38.2% | 语义和物理都不行 |
| 仅 SA<4（PC>=4）| 181 | 30.2% | **语义不匹配是主要瓶颈** |
| 仅 PC<4（SA>=4）| 32 | 5.3% | 物理常识单独失分较少 |
| Joint 通过 | 158 | 26.3% | — |

**核心发现：SA（语义匹配）是最大瓶颈**，68.3% 的 prompt SA<4，而 PC<4 只有 43.5%。模型经常生成视觉合理但语义不匹配的视频。

#### Hard vs Easy

| 子集 | SA 失败率 | PC 失败率 | Joint |
|------|-----------|-----------|-------|
| Easy (420) | 59.5% | 38.6% | 34.0% |
| Hard (180) | 88.9% | 55.0% | 8.3% |

Hard subset 的 SA 失败率高达 88.9%，说明模型难以理解和生成复杂/罕见场景。

#### 按类别

- **Object Interactions**：Joint=31.4%，SA=3.24，PC=3.75（相对较好）
- **Sports and Physical Activities**：Joint=24.2%，SA=3.16，PC=3.56（较差）

#### 最差的动作类别（Joint=0%，共 108 个 action）

典型失败模式：

1. **球类/拍类运动**：badminton (SA=2.0)、volleyball (SA=2.3)、tennis (SA=2.5)、squash (SA=2.3)、polo (SA=2.3)、field hockey (SA=2.3)、ping pong、kickball、cricket — 模型无法准确生成运动场景细节（球拍、球网、击球动作）

2. **投掷类运动**：javelin throw、discus throw、throwing axe、throwing knife、drop kicking — 需要精确的抛物线轨迹和物体旋转，模型难以捕捉

3. **罕见/专业活动**：pole vault、bobsledding、parasailing、spinning poi、nunchucks、pizzatossing、tightrope walking — 训练数据中此类场景稀少

4. **碰撞/连锁反应**：`something colliding with something`、`poking a stack so it collapses` — 需要多物体交互的物理推理

5. **精细操作**：playing darts、billiards、threading needle、opening bottle — 需要手部精细动作和小物体交互

#### 最好的动作类别（Joint=100%，共 20 个 action）

成功模式都是**简单、常见的日常场景**：
- wading through water/mud、walking through snow（步行类）
- smoking、mopping floor、clay pottery making（日常动作）
- poking a hole、burying something、using a paint roller（简单物理交互）
- tying bow tie、tying knot、twisting something（手部常规操作）

#### SA<=2 的极差案例（102 个）

高频出现：球类运动（basketball、baseball、badminton × 3）、投掷运动（javelin × 3、discus × 2）、drop kicking × 3、playing polo × 2。这些 prompt 通常要求特定的运动姿势、球的轨迹、或器材细节，模型倾向于生成模糊或错误的运动场景。

#### 总结

| 失败原因 | 影响程度 | 说明 |
|----------|----------|------|
| 语义理解不足（SA） | **最高** | 模型不理解 prompt 中的具体动作、器材、运动规则 |
| 运动/稀有场景 | 高 | 球类运动、专业体育几乎全军覆没 |
| 精细物体交互 | 中 | 小物体、精细手部动作难以生成 |
| 物理常识（PC） | 较低 | 仅 5.3% 的 prompt 是纯物理失分 |

## 7. 注意事项

1. **视频格式**: Wan2.2 输出 MP4，VIDEOPHY2 评估器也接受 MP4，无需转换
2. **分辨率**: TI2V-5B 只支持 `1280*704` / `704*1280`；T2V-A14B 支持 `1280*720` / `720*1280` / `832*480` / `480*832`
3. **Prompt 长度**: VIDEOPHY2 的 prompt 平均 138 tokens，Wan2.2 的 T5 encoder 支持 512 tokens，不会截断
4. **评估模型依赖**: VIDEOPHY2 AutoEval 基于 mPLUG-Owl-Video，需要 `peft`、`transformers` 和 `llama tokenizer`
5. **批量生成时间**: 600 个 prompt × 每个约 2-5 分钟 ≈ 20-50 小时（单 GPU），两路并行约 10-25 小时
6. **可以先跑一小批**: `python run_videophy.py --start 0 --end 5` 先测试流程
