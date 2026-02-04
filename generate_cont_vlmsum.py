# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Continuous Video Generation Script
# 连续视频生成脚本：从 video1 生成 video2, 从 video2 生成 video3, ..., 直到生成 video12

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random
import torch
import torch.distributed as dist

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video
from concat_videos import concat_videos


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Continuous video generation: video1 -> video2 -> ... -> video12"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to initial input video (video1).")
    parser.add_argument(
        "--num_videos",
        type=int,
        default=12,
        help="Total number of videos to generate (default: 12).")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./continuous_outputs",
        help="Directory to save generated videos.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        required=True,
        help="Path to the checkpoint directory.")
    parser.add_argument(
        "--size",
        type=str,
        default="704*1280",
        choices=list(SIZE_CONFIGS.keys()),
        help="The area (width*height) of the generated video.")
    parser.add_argument(
        "--frame_num",
        type=int,
        default=121,
        help="Number of frames to generate per video (should be 4n+1).")
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=40,
        help="Sampling steps.")
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=5.0,
        help="Sampling shift factor.")
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Classifier free guidance scale.")
    parser.add_argument(
        "--sample_solver",
        type=str,
        default='unipc',
        choices=['unipc', 'dpm++'],
        help="The solver used to sample.")
    parser.add_argument(
        "--base_seed",
        type=int,
        default=-1,
        help="Base random seed. Each video will use base_seed + video_index.")
    parser.add_argument(
        "--offload_model",
        action="store_true",
        default=True,
        help="Whether to offload the model to CPU after generation.")
    parser.add_argument(
        "--vlm_device",
        type=str,
        default=None,
        help="Device for VLM model (e.g., 'cuda:1').")
    parser.add_argument(
        "--custom_prompt",
        type=str,
        default="continue video, car always go ahead, ",
        help="Custom prompt prefix to prepend to VLM-generated description.")
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.")
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.")
    parser.add_argument(
        "--convert_model_dtype",
        action="store_true",
        default=False,
        help="Whether to convert model parameters dtype.")
    
    args = parser.parse_args()
    
    # Validate
    assert os.path.exists(args.video), f"Input video not found: {args.video}"
    assert args.ckpt_dir is not None, "Please specify the checkpoint directory."
    assert args.num_videos >= 1, "num_videos must be at least 1"
    
    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)
    
    return args


def _init_logging(rank=0):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def generate_continuous(args):
    """
    连续生成视频的主函数
    
    流程:
    1. 读取 video1, 提取最后一帧 + VLM分析 -> 生成 video2
    2. 读取 video2, 提取最后一帧 + VLM分析 -> 生成 video3
    ...
    12. 读取 video11, 提取最后一帧 + VLM分析 -> 生成 video12
    """
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    
    _init_logging(rank)
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), \
            "t5_fsdp and dit_fsdp are not supported in non-distributed environments."
    
    # Create output directory with timestamp and input video name
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Extract first 10 characters of input video filename (without extension)
    input_video_name = os.path.splitext(os.path.basename(args.video))[0][:10]
    run_output_dir = os.path.join(args.output_dir, f"{input_video_name}_{run_timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    
    # Get config for v2v-5B (uses ti2v config)
    cfg = WAN_CONFIGS["v2v-5B"]
    
    logging.info("=" * 60)
    logging.info("Continuous Video Generation")
    logging.info(f"Input video: {args.video}")
    logging.info(f"Number of videos to generate: {args.num_videos}")
    logging.info(f"Output directory: {run_output_dir}")
    logging.info(f"Base seed: {args.base_seed}")
    logging.info("=" * 60)
    
    # Initialize WanV2V5B pipeline (only once)
    logging.info("Creating WanV2V5B pipeline (5B model)...")
    from wan.wan22video2video5B import WanV2V5B
    wan_v2v = WanV2V5B(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        vlm_device=args.vlm_device,
    )
    
    # Start continuous generation
    current_video_path = args.video
    generated_video_paths = []
    generated_prompts = []
    
    for i in range(args.num_videos):
        video_index = i + 1
        logging.info("=" * 60)
        logging.info(f"Generating video {video_index}/{args.num_videos}")
        logging.info(f"Input: {current_video_path}")
        
        # Calculate seed for this video
        current_seed = args.base_seed + i
        logging.info(f"Using seed: {current_seed}")
        
        # Generate video
        video, prompt = wan_v2v.v2v(
            video_path=current_video_path,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=current_seed,
            offload_model=args.offload_model,
            custom_prompt=args.custom_prompt,
        )
        
        logging.info(f"VLM generated prompt: {prompt[:100]}...")
        generated_prompts.append(prompt)
        
        # Save the generated video
        if rank == 0:
            formatted_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"video_{video_index:02d}_{formatted_time}.mp4"
            output_path = os.path.join(run_output_dir, output_filename)
            
            logging.info(f"Saving video to: {output_path}")
            save_video(
                tensor=video[None],
                save_file=output_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1))
            
            generated_video_paths.append(output_path)
            
            # Update current_video_path for next iteration
            current_video_path = output_path
            
            logging.info(f"Video {video_index} saved successfully!")
        
        del video
        torch.cuda.empty_cache()
    
    # Summary
    if rank == 0:
        logging.info("=" * 60)
        logging.info("GENERATION COMPLETE!")
        logging.info("=" * 60)
        logging.info(f"Total videos generated: {len(generated_video_paths)}")
        logging.info("Generated videos:")
        for idx, path in enumerate(generated_video_paths, 1):
            logging.info(f"  {idx}. {path}")
        
        # Save prompts to a log file
        prompts_log_path = os.path.join(run_output_dir, f"{input_video_name}_prompts_log.txt")
        with open(prompts_log_path, 'w', encoding='utf-8') as f:
            f.write("Continuous Video Generation - Prompts Log\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Input video: {args.video}\n")
            f.write(f"Base seed: {args.base_seed}\n\n")
            for idx, (path, prompt) in enumerate(zip(generated_video_paths, generated_prompts), 1):
                f.write(f"Video {idx}: {path}\n")
                f.write(f"Prompt: {prompt}\n")
                f.write("-" * 40 + "\n\n")
        logging.info(f"Prompts saved to: {prompts_log_path}")
    
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    
    logging.info("Finished!")
    return generated_video_paths


if __name__ == "__main__":
    args = _parse_args()
    generate_continuous(args)
