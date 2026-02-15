#!/bin/bash
source /home/user1/miniconda3/etc/profile.d/conda.sh
conda activate wan
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 WorldModelBench/generate_videos.py

# Evaluation
conda activate vila
cd WorldModelBench
CUDA_VISIBLE_DEVICES=0 python evaluation.py \
  --model_name "Wan2.2-TI2V-5B" \
  --video_dir outputs \
  --judge vila-ewm-qwen2-1.5b \
  --cot
