#!/usr/bin/env python
"""
批量生成 VIDEOPHY2 benchmark 视频 (Wan2.2-T2V-A14B)
多 GPU 并行生成 (multiprocessing)。
"""
import argparse
import csv
import logging
import os
import sys
import time
import torch.multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description="Batch generate VIDEOPHY2 videos (T2V-A14B)")
    parser.add_argument("--prompt_csv", type=str, default="./videophy_data/videophy2_prompts.csv")
    parser.add_argument("--output_dir", type=str, default="./videophy_outputs_t2v")
    parser.add_argument("--ckpt_dir", type=str, default="./Wan2.2-T2V-A14B")
    parser.add_argument("--size", type=str, default="1280*720")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive), -1 for all")
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--offload_model", action="store_true", default=True)
    parser.add_argument("--gpus", type=str, default="2,3,4,5", help="Comma-separated GPU IDs")
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


def worker(gpu_id, chunk, args):
    import torch
    import wan
    from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    task = "t2v-A14B"
    cfg = WAN_CONFIGS[task]
    size = SIZE_CONFIGS[args.size]

    logging.info(f"[GPU{gpu_id}] Loading T2V-A14B ...")
    t_load = time.time()
    pipeline = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=gpu_id,
        rank=0,
        t5_cpu=args.t5_cpu,
    )
    logging.info(f"[GPU{gpu_id}] Model loaded in {time.time() - t_load:.1f}s")

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
            size=size,
            frame_num=args.frame_num,
            shift=cfg.sample_shift,
            sample_solver="unipc",
            sampling_steps=args.sample_steps,
            guide_scale=cfg.sample_guide_scale,
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


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(stream=sys.stdout)],
    )

    os.makedirs(args.output_dir, exist_ok=True)

    prompts = load_prompts(args.prompt_csv)
    total = len(prompts)
    start = args.start
    end = args.end if args.end > 0 else total
    prompts = prompts[start:end]
    logging.info(f"Loaded {total} prompts, processing [{start}:{end}] = {len(prompts)} prompts")

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

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    num_gpus = len(gpu_ids)
    logging.info(f"Using {num_gpus} GPU(s): {gpu_ids}")

    # Split prompts across GPUs
    chunks = {gid: [] for gid in gpu_ids}
    for idx, p in enumerate(to_generate):
        chunks[gpu_ids[idx % num_gpus]].append(p)

    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in gpu_ids:
        p = mp.Process(target=worker, args=(gpu_id, chunks[gpu_id], args))
        p.start()
        processes.append(p)
        logging.info(f"Started GPU {gpu_id} process with {len(chunks[gpu_id])} prompts")

    for p in processes:
        p.join()

    logging.info(f"All done! Generated {len(to_generate)} videos in {args.output_dir}")


if __name__ == "__main__":
    main()
