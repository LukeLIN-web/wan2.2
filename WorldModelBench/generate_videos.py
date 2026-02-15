"""Batch TI2V generation for WorldModelBench using Wan2.2-TI2V-5B (2 GPUs).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 WorldModelBench/generate_videos.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

# Add parent dir to path so we can import wan
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video

BENCH_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BENCH_DIR / "outputs"
CKPT_DIR = BENCH_DIR.parent / "models" / "Wan2.2-TI2V-5B"
SIZE = "704*1280"


def main():
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.ERROR,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load benchmark data
    with open(BENCH_DIR / "worldmodelbench.json") as f:
        instances = json.load(f)
    logging.info(f"Loaded {len(instances)} instances")

    # Filter out already generated
    todo = []
    for inst in instances:
        stem = Path(inst["first_frame"]).stem
        out_path = OUTPUT_DIR / f"{stem}.mp4"
        if out_path.exists():
            continue
        todo.append(inst)
    logging.info(f"Skipping {len(instances) - len(todo)} existing, {len(todo)} to generate")

    if not todo:
        logging.info("All videos already generated.")
        return

    # Init model once
    cfg = WAN_CONFIGS["ti2v-5B"]
    logging.info("Loading WanTI2V pipeline...")
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir=str(CKPT_DIR),
        device_id=local_rank,
        rank=rank,
        t5_fsdp=(world_size > 1),
        dit_fsdp=(world_size > 1),
        use_sp=(world_size > 1),
        t5_cpu=False,
        convert_model_dtype=False,
    )
    logging.info("Model loaded.")

    for i, inst in enumerate(todo):
        stem = Path(inst["first_frame"]).stem
        out_path = OUTPUT_DIR / f"{stem}.mp4"
        img_path = BENCH_DIR / inst["first_frame"]
        prompt = inst["text_instruction"]

        logging.info(f"[{i+1}/{len(todo)}] {stem}")

        img = Image.open(img_path).convert("RGB")
        video = wan_ti2v.generate(
            prompt,
            img=img,
            size=SIZE_CONFIGS[SIZE],
            max_area=MAX_AREA_CONFIGS[SIZE],
            frame_num=81,
            shift=cfg.sample_shift,
            sample_solver="unipc",
            sampling_steps=cfg.sample_steps,
            guide_scale=cfg.sample_guide_scale,
            seed=42,
            offload_model=False,
        )

        if rank == 0:
            save_video(
                tensor=video[None],
                save_file=str(out_path),
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            logging.info(f"  Saved {out_path.name}")

        del video
        torch.cuda.empty_cache()

    logging.info("Done.")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
