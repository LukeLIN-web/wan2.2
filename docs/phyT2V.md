# PhyT2V with Wan 2.2 + Qwen3-VL

用 Wan 2.2 替换 CogVideoX-5B，用 Qwen3-VL 统一替换 GPT-4 + Tarsier-34B（一个模型搞定 LLM 推理 + 视频理解）。

## 原版 vs 本方案

| 组件 | 原版 PhyT2V | 本方案 |
|------|------------|--------|
| 视频生成 | CogVideoX-5B | Wan 2.2 TI2V-5B / T2V-A14B |
| LLM 推理 + 视频描述 | GPT-4 + Tarsier-34B (两个模型) | **Qwen3-VL-8B-Instruct (一个模型)** |
| 视频评估 | videocon-physics | VIDEOPHY2 AutoEval (1-5 分) |

**transformers 版本要求**：
- Qwen3-VL: `pip install git+https://github.com/huggingface/transformers`（需 5.2.0.dev0+）
- VIDEOPHY2 AutoEval (mplug-owl): 需要 `transformers==4.44.0`，与 Qwen3-VL 冲突
- 解决方案：Phase 1-4 用新版 transformers 跑 Qwen3-VL + Wan，Phase 5 评分切回旧版单独跑

## Pipeline

```
用户 prompt
    │
    ├─ [Qwen3-VL] 提取 main_object + physical_law    (纯文本推理)
    ├─ [Wan 2.2] 生成 round 1 视频
    │
    └─ for i in 1..N:
         ├─ [Qwen3-VL] 视频描述                       (视频理解)
         ├─ [Qwen3-VL] mismatch 分析                  (纯文本推理)
         ├─ [Qwen3-VL] prompt 增强                    (纯文本推理)
         └─ [Wan 2.2] 生成 round i+1 视频
```

## 代码结构

所有代码在 `myphyt2v/` 下：

| 文件 | 作用 |
|------|------|
| `qwen_reasoner.py` | `Qwen3VLReasoner` — 统一的文本推理 + 视频理解，用 `AutoModelForImageTextToText` 加载 |
| `prompts.py` | Prompt 模板 (physical_law / mismatch / enhanced) + 对应调用函数 |
| `wan_gen.py` | Wan 2.2 视频生成封装 (`make_pipeline` / `generate_video` / `save`) |
| `scoring.py` | videocon-physics SA & PC 评分（单 prompt 模式用） |
| `run.py` | 单 prompt 迭代优化 pipeline (交互式调试用) |
| `run_batch.py` | **VIDEOPHY2 benchmark 批量 pipeline，5 phase 可断点续跑** |

## 环境

```bash
conda activate wan
pip install qwen-vl-utils
pip install git+https://github.com/huggingface/transformers  # Qwen3-VL 需要
```

## 模型

```bash
# Wan 2.2
ls ./models/Wan2.2-TI2V-5B/
ls ./models/Wan2.2-T2V-A14B/

# Qwen3-VL (推荐 8B-Instruct)
ls ./models/Qwen3-VL-8B-Instruct/

# VIDEOPHY2 AutoEval
ls ./videophy/videophy_2_auto/
```

## VIDEOPHY2 Benchmark 批量运行

### run_batch.py 分 5 phase

| Phase | 内容 | 模型 | checkpoint | 耗时估算 (600条) |
|-------|------|------|------------|-----------------|
| 1 | 提取物理定律 | Qwen3-VL 文本 | `phys_laws.json` | ~2h |
| 2 | 描述 round1 视频 | Qwen3-VL 视频 | `captions.json` | ~2h |
| 3 | mismatch + prompt 增强 | Qwen3-VL 文本 | `enhanced.json` | ~50min |
| 4 | 多 GPU 生成 round2 视频 | Wan TI2V-5B | `round2_videos/` | ~6h (5 GPU) |
| 5 | 生成 eval CSV | 无 | `eval/eval_sa_pc.csv` | 即时 |

每个 phase 有 JSON checkpoint，每 10 条自动保存，可断点续跑。

### 运行命令

```bash
# 全量 600 条
python -m myphyt2v.run_batch \
    --qwen_model ./models/Qwen3-VL-8B-Instruct \
    --qwen_device cuda:0 \
    --wan_ckpt ./models/Wan2.2-TI2V-5B \
    --wan_gpus 1,2,3,4,5 \
    --round1_dir ./videophy_outputs \
    --output_dir ./phyt2v_outputs

# 后台运行
nohup python -m myphyt2v.run_batch \
    --qwen_model ./models/Qwen3-VL-8B-Instruct \
    --qwen_device cuda:0 \
    --wan_ckpt ./models/Wan2.2-TI2V-5B \
    --wan_gpus 1,2,3,4,5 \
    --round1_dir ./videophy_outputs \
    --output_dir ./phyt2v_outputs \
    > phyt2v_run.log 2>&1 &

# 从 phase 4 恢复 (跳过 Qwen3-VL 阶段)
python -m myphyt2v.run_batch --phase 4 ...

# 只跑部分数据
python -m myphyt2v.run_batch --start 0 --end 50 ...
```

### 评分（需切换 transformers 版本）

Phase 1-4 完成后，eval CSV 在 `phyt2v_outputs/eval/eval_sa_pc.csv`。

```bash
# 切回旧版 transformers
pip install transformers==4.44.0

# SA 评分
cd videophy/VIDEOPHY2
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --input_csv ../../phyt2v_outputs/eval/eval_sa_pc.csv \
    --checkpoint ../videophy_2_auto \
    --output_csv ../../phyt2v_outputs/eval/eval_output_sa.csv \
    --task sa

# PC 评分
CUDA_VISIBLE_DEVICES=1 python inference.py \
    --input_csv ../../phyt2v_outputs/eval/eval_sa_pc.csv \
    --checkpoint ../videophy_2_auto \
    --output_csv ../../phyt2v_outputs/eval/eval_output_pc.csv \
    --task pc
```

### 计算 Joint Score

```python
import pandas as pd
sa = pd.read_csv("phyt2v_outputs/eval/eval_output_sa.csv")
pc = pd.read_csv("phyt2v_outputs/eval/eval_output_pc.csv")
merged = sa.merge(pc, on="videopath", suffixes=("_sa", "_pc"))
joint = ((merged["score_sa"] >= 4) & (merged["score_pc"] >= 4)).mean() * 100
print(f"Joint Score: {joint:.1f}%")
print(f"Mean SA: {merged['score_sa'].mean():.2f}")
print(f"Mean PC: {merged['score_pc'].mean():.2f}")
```

## GPU 分配

| GPU | 用途 |
|-----|------|
| GPU 0 | Qwen3-VL-8B-Instruct (~18G) |
| GPU 1-5 | Wan TI2V-5B 多卡并行 (offload_model=True, ~20-30G/卡) |

## 两套评分体系

| | PhyT2V 论文用 (videocon-physics) | VIDEOPHY2 AutoEval |
|---|---|---|
| 分数范围 | **0~1** (entailment 概率) | **1~5** (整数评分) |
| 评测数据集 | VideoPhy (原版) | VIDEOPHY2 (600 条) |
| Joint 定义 | — | SA≥4 且 PC≥4 的比例 |
| checkpoint | `videophysyics/videophy/videocon_physics/` | `videophy/videophy_2_auto/` |
| 评估脚本 | `myphyt2v/eval_videocon.py` | `videophy/VIDEOPHY2/inference.py` |

### PhyT2V 论文结果 (videocon-physics, 0~1)

| Model | Round 1 PC/SA | Round 4 PC/SA |
|-------|---------------|---------------|
| CogVideoX-5B | 0.26 / 0.48 | 0.42 / 0.59 |
| CogVideoX-2B | 0.13 / 0.22 | 0.29 / 0.42 |
| OpenSora | 0.17 / 0.29 | 0.31 / 0.47 |
| VideoCrafter | 0.15 / 0.24 | 0.33 / 0.49 |

### VIDEOPHY2 AutoEval 结果 (1~5)

| 模型 | Mean SA | Mean PC | Joint (All) | Joint (Hard) |
|------|---------|---------|-------------|--------------|
| Wan2.2-TI2V-5B (baseline, 无 PhyT2V) | 3.18 | 3.61 | 26.3% | 8.3% |
| **Wan2.2-TI2V-5B + PhyT2V (本方案)** | 待测 | 待测 | 待测 | 待测 |
| CogVideoX-5B (leaderboard, human eval) | — | — | 25.0% | 0% |
| Wan2.1-T2V-14B (leaderboard, human eval) | — | — | 32.6% | 21.9% |

### VideoCon-Physics 评估 (0~1, 与论文对比用)

| 模型 | SA | PC | SA (Easy) | PC (Easy) | SA (Hard) | PC (Hard) |
|------|------|------|-----------|-----------|-----------|-----------|
| Wan2.2-TI2V-5B (baseline) | 0.3888 | 0.1840 | 0.4347 | 0.2026 | 0.2817 | 0.1406 |
| **Wan2.2-TI2V-5B + PhyT2V** | 待测 | 待测 | — | — | — | — |
| CogVideoX-5B (论文 Round 1) | 0.48 | 0.26 | — | — | — | — |
| CogVideoX-5B (论文 Round 4) | 0.59 | 0.42 | — | — | — | — |

checkpoint 已下载到 `videophysyics/videophy/videocon_physics/`。

**注意**: 需要 `transformers==4.44.0`，和 Qwen3-VL 冲突，用 `conda activate videophy` 环境跑。

```bash
# 评估 baseline (round1 视频)
python -m myphyt2v.eval_videocon \
    --video_dir ./videophy_outputs \
    --prompt_csv ./videophy_data/videophy2_prompts.csv \
    --checkpoint ./videophysyics/videophy/videocon_physics \
    --output_dir ./eval_videocon_baseline \
    --gpu 0

# 评估 PhyT2V round2 视频
python -m myphyt2v.eval_videocon \
    --video_dir ./phyt2v_outputs/round2_videos \
    --prompt_csv ./videophy_data/videophy2_prompts.csv \
    --checkpoint ./videophysyics/videophy/videocon_physics \
    --output_dir ./eval_videocon_phyt2v \
    --gpu 0
```

输出 `summary.json` 包含 mean_sa, mean_pc (0~1)，可直接与论文 Table 1 对比。

## 注意事项

- **分辨率**: TI2V-5B 只支持 `704*1280` / `1280*704`
- **Round1 视频复用**: `--round1_dir` 指向已有的 baseline 视频目录（`videophy_outputs/`），不重复生成
- **Qwen3-VL 加载**: 使用 `AutoModelForImageTextToText` + `AutoProcessor`，兼容 dense 和 MoE 模型
- **视频帧率**: Wan fps 由 `cfg.sample_fps` 决定 (TI2V-5B = 24fps)
- **Phase 4 多进程**: 用 `torch.multiprocessing` spawn 方式，每个 GPU 独立进程
