#!/usr/bin/env python
"""
批量生成 VIDEOPHY2 benchmark 视频 (Wan2.2-TI2V-5B)
自动检测可用 GPU 数量，多 GPU 并行生成。
"""
import argparse
import csv
import logging
import os
import sys
import time
import threading

import torch

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video


def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate VIDEOPHY2 videos")
    parser.add_argument("--prompt_csv", type=str, default="./videophy_data/videophy2_prompts.csv")
    parser.add_argument("--output_dir", type=str, default="./videophy_outputs")
    parser.add_argument("--ckpt_dir", type=str, default="./Wan2.2-TI2V-5B")
    parser.add_argument("--size", type=str, default="1280*704")
    parser.add_argument("--frame_num", type=int, default=121)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive), -1 for all")
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--offload_model", action="store_true", default=True)
    parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated GPU IDs to use, e.g. 0,1")
    return parser.parse_args()


def load_prompts(csv_path):
    prompts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompts.append({
                "id": int(row["id"]),
                "caption": row["caption"],
                "action": row["action"],
                "is_hard": int(row["is_hard"]),
            })
    return prompts


def safe_filename(prompt_id, caption, max_len=80):
    clean = caption.replace(" ", "_").replace("/", "_").replace('"', "")
    clean = "".join(c for c in clean if c.isalnum() or c in "_-.,")
    return f"{prompt_id:04d}_{clean[:max_len]}.mp4"


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Load prompts
    prompts = load_prompts(args.prompt_csv)
    total = len(prompts)
    start = args.start
    end = args.end if args.end > 0 else total
    prompts = prompts[start:end]
    logging.info(f"Loaded {total} prompts, processing [{start}:{end}] = {len(prompts)} prompts")

    # Check which already exist
    skip_count = 0
    to_generate = []
    for p in prompts:
        save_path = os.path.join(args.output_dir, safe_filename(p["id"], p["caption"]))
        if os.path.exists(save_path):
            skip_count += 1
        else:
            to_generate.append(p)
    logging.info(f"Skip {skip_count} existing, {len(to_generate)} to generate")

    if not to_generate:
        logging.info("Nothing to generate. Done.")
        return

    task = "ti2v-5B"
    cfg = WAN_CONFIGS[task]
    size = SIZE_CONFIGS[args.size]
    max_area = MAX_AREA_CONFIGS[args.size]
    sample_shift = cfg.sample_shift
    sample_guide_scale = cfg.sample_guide_scale

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    logging.info(f"Using {num_gpus} GPU(s): {gpu_ids}")

    # Load one pipeline per GPU
    pipelines = {}
    for gpu_id in gpu_ids:
        logging.info(f"Loading TI2V-5B on GPU {gpu_id} ...")
        t_load = time.time()
        p = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=gpu_id,
            rank=0,
            t5_cpu=args.t5_cpu,
        )
        pipelines[gpu_id] = p
        logging.info(f"GPU {gpu_id} model loaded in {time.time() - t_load:.1f}s")

    # Split prompts across GPUs
    chunks = {gid: [] for gid in gpu_ids}
    for idx, p in enumerate(to_generate):
        chunks[gpu_ids[idx % num_gpus]].append(p)

    def worker(gpu_id, pipeline, chunk):
        for i, p in enumerate(chunk):
            save_path = os.path.join(args.output_dir, safe_filename(p["id"], p["caption"]))
            if os.path.exists(save_path):
                continue
            hard_tag = " [HARD]" if p["is_hard"] else ""
            logging.info(f"[GPU{gpu_id}] [{i+1}/{len(chunk)}] id={p['id']} action={p['action']}{hard_tag}")
            logging.info(f"[GPU{gpu_id}]   prompt: {p['caption'][:120]}")

            t_start = time.time()
            video = pipeline.generate(
                input_prompt=p["caption"],
                img=None,
                size=size,
                max_area=max_area,
                frame_num=args.frame_num,
                shift=sample_shift,
                sample_solver="unipc",
                sampling_steps=args.sample_steps,
                guide_scale=sample_guide_scale,
                seed=args.base_seed,
                offload_model=args.offload_model,
            )

            save_video(
                tensor=video[None],
                save_file=save_path,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
            del video
            torch.cuda.empty_cache()

            elapsed = time.time() - t_start
            logging.info(f"[GPU{gpu_id}]   saved: {save_path} ({elapsed:.1f}s)")

    # Launch threads
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(target=worker, args=(gpu_id, pipelines[gpu_id], chunks[gpu_id]))
        t.start()
        threads.append(t)
        logging.info(f"Started GPU {gpu_id} thread with {len(chunks[gpu_id])} prompts")

    for t in threads:
        t.join()

    logging.info(f"All done! Generated {len(to_generate)} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
