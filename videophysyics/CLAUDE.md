# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **VideoPhy / VideoPhy-2 benchmark evaluation** workspace. It generates videos from text prompts using Wan2.2 models, then evaluates them for semantic adherence (SA) and physical commonsense (PC) using auto-evaluator models. The goal is to benchmark how well video generation models follow physical laws.

## Environment

Always `conda activate wan` before running any command.

## Commands

### 1. Download benchmark prompts

```
python download_videophy2.py
```
Downloads `videophysics/videophy2_test` from HuggingFace into `./videophy_data/`.

### 2. Generate videos (TI2V-5B, text+image-to-video)

```
python run_videophy.py --gpus 0,1 --ckpt_dir ./Wan2.2-TI2V-5B --start 0 --end 100
```
Uses threading for multi-GPU parallelism. Loads one `WanTI2V` pipeline per GPU. Outputs to `./videophy_outputs/`. Skips already-generated files automatically.

### 3. Generate videos (T2V-A14B, text-to-video)

```
python run_videophy_t2v.py --gpus 2,3,4,5 --ckpt_dir ./Wan2.2-T2V-A14B --start 0 --end 100
```
Uses `torch.multiprocessing` (spawn) for multi-GPU parallelism. Outputs to `./videophy_outputs_t2v/`.

### 4. Auto-evaluate with VideoPhy-2-AutoEval

```
CUDA_VISIBLE_DEVICES=0 python videophy/VIDEOPHY2/inference.py \
  --input_csv <csv_with_videopath_and_caption> \
  --checkpoint videophy/videophy_2_auto \
  --output_csv output.csv \
  --task sa   # or pc, or rule
```
Tasks: `sa` (semantic adherence, 1-5), `pc` (physical commonsense, 1-5), `rule` (physical rule grounding, 0/1/2).

### 5. Auto-evaluate with VideoCon-Physics

需要 `videophy` 环境（transformers==4.44.0）。

```bash
conda activate videophy
cd /shared/user72/workspace/juyi/Wan2.2

python -m myphyt2v.eval_videocon \
  --video_dir ./videophysyics/videophy_outputs \
  --prompt_csv ./videophysyics/videophy_data/videophy2_prompts.csv \
  --checkpoint ./videophysyics/videophy/videocon_physics \
  --output_dir ./eval_videocon_baseline \
  --gpu 0
```

参数说明：
- `--video_dir` — 生成的视频目录
- `--output_dir` — 输出目录，会生成 `sa_input.csv`, `sa_output.csv`, `pc_input.csv`, `pc_output.csv`, `summary.json`
- `--gpu` — 使用的 GPU 编号
- `--start/--end` — 可选，只评估部分 prompt
- 已有输出文件会自动跳过，删除 `sa_output.csv`/`pc_output.csv` 可重跑

输出为 entailment 概率（0~1），非 1-5 分制。

## Data Layout

- `videophy_data/videophy2_prompts.csv` — benchmark prompts (columns: `id`, `caption`, `action`, `is_hard`, `category`, `upsampled_caption`)
- `videophy_outputs/` — generated TI2V-5B videos (600 .mp4 files)
- `videophy_outputs_t2v/` — generated T2V-A14B videos

## Auto-Evaluator Model Weights (local)

- `videophy/videophy_2_auto/` — VideoPhy-2 auto-rater (mPLUG-Owl based, multi-task: SA/PC/rule)
- `videophy/videocon_physics/` — VideoCon-Physics (VideoPhy-1 auto-rater, entailment-based Yes/No scoring)

Both are mPLUG-Owl-Video fine-tunes using LLaMA tokenizer + custom video processor.

## Key Differences Between Generation Scripts

| | `run_videophy.py` (TI2V) | `run_videophy_t2v.py` (T2V) |
|---|---|---|
| Task config | `ti2v-5B` | `t2v-A14B` |
| Default size | `1280*704` | `1280*720` |
| Default frames | 121 | 81 |
| Parallelism | `threading.Thread` | `torch.multiprocessing` (spawn) |
| Pipeline class | `wan.WanTI2V` | `wan.WanT2V` |

## Joint Score Metric

The benchmark's headline metric is the fraction of instances where **both** SA >= 4 **and** PC >= 4 (for VideoPhy-2) or SA=1 and PC=1 (for VideoPhy-1 binary).
