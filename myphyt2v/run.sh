#!/bin/bash
# PhyT2V with Wan 2.2 + Qwen3-VL
# GPU 0: Qwen3-VL (LLM + video understanding)
# GPU 1: Wan 2.2 (video generation, offload_model=True)

conda activate wan

# --- 单 prompt 迭代优化 (TI2V-5B) ---
python -m myphyt2v.run \
    --qwen_model ./models/Qwen3-VL-8B-Instruct \
    --wan_task ti2v-5B --wan_ckpt ./models/Wan2.2-TI2V-5B --size 1280*704 \
    --prompt "a rubber ball hits the ground and bounces up" \
    --rounds 2 --qwen_device cuda:0 --wan_device 1

# --- 单 prompt (T2V-A14B) ---
# python -m myphyt2v.run \
#     --qwen_model ./models/Qwen3-VL-8B-Instruct \
#     --wan_task t2v-A14B --wan_ckpt ./models/Wan2.2-T2V-A14B --size 1280*720 \
#     --prompt "a glass of water is poured onto a table" \
#     --rounds 2 --qwen_device cuda:0 --wan_device 1

# --- VIDEOPHY2 Benchmark 批量运行 ---
# python -m myphyt2v.run_batch \
#     --prompt_csv ./videophy_data/videophy2_prompts.csv \
#     --round1_dir ./videophy_outputs \
#     --output_dir ./phyt2v_outputs \
#     --qwen_model ./models/Qwen3-VL-8B-Instruct \
#     --qwen_device cuda:0 \
#     --wan_ckpt ./models/Wan2.2-TI2V-5B \
#     --wan_gpus 1,2,3,4,5 \
#     --eval_gpu 0
