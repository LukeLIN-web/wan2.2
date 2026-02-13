"""
PhyT2V with Wan 2.2 + Qwen3-VL

Pipeline:
  1. Extract physical law & main object (Qwen3-VL text)
  2. Generate round 1 video (Wan 2.2)
  3. Iterate N rounds:
     - Caption video (Qwen3-VL video understanding)
     - Analyze mismatch (Qwen3-VL text)
     - SA & PC scoring (videocon-physics)
     - Enhance prompt (Qwen3-VL text)
     - Generate next video (Wan 2.2)
"""

import argparse
import json
import logging
import os
import sys

import torch
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from myphyt2v.qwen_reasoner import Qwen3VLReasoner
from myphyt2v.prompts import extract_physical_law, get_mismatch, get_enhanced_prompt
from myphyt2v.wan_gen import make_pipeline, generate_video, save
from myphyt2v.scoring import sa_score, pc_score

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="PhyT2V: Wan 2.2 + Qwen3-VL")
    p.add_argument("--prompt", type=str, required=True, help="Input text prompt")
    p.add_argument("--rounds", type=int, default=2, help="Number of refinement rounds")
    # Qwen3-VL
    p.add_argument("--qwen_model", type=str, default="./models/Qwen3-VL-8B-Instruct")
    p.add_argument("--qwen_device", type=str, default="cuda:0")
    # Wan 2.2
    p.add_argument("--wan_task", type=str, default="ti2v-5B", choices=["ti2v-5B", "t2v-A14B"])
    p.add_argument("--wan_ckpt", type=str, default="./models/Wan2.2-TI2V-5B")
    p.add_argument("--wan_device", type=int, default=1)
    p.add_argument("--size", type=str, default="1280*704")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    # Scoring
    p.add_argument("--videocon_gpu", type=int, default=0, help="GPU id for videocon scoring")
    # Output
    p.add_argument("--output_dir", type=str, default="./myphyt2v_outputs")
    return p.parse_args()


def main():
    args = parse_args()
    video_dir = os.path.join(args.output_dir, "videos")
    eval_dir = os.path.join(args.output_dir, "eval_csv")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # --- Init models ---
    log.info("Loading Qwen3-VL: %s on %s", args.qwen_model, args.qwen_device)
    reasoner = Qwen3VLReasoner(model_path=args.qwen_model, device=args.qwen_device)

    log.info("Loading Wan 2.2: %s from %s on GPU %d", args.wan_task, args.wan_ckpt, args.wan_device)
    pipeline, cfg = make_pipeline(args.wan_task, args.wan_ckpt, args.wan_device)

    # --- Data tracking ---
    data = {"prompt_1": args.prompt}

    # --- Step 1: Physical law extraction ---
    log.info("=" * 60)
    log.info("Step 1: Extracting physical law & main object")
    main_objects, physical_law = extract_physical_law(reasoner, args.prompt)
    data["main_object"] = main_objects
    data["physical_law"] = physical_law
    log.info("  main_object: %s", main_objects)
    log.info("  physical_law: %s", physical_law[:200])

    # --- Step 2: Round 1 video generation ---
    log.info("=" * 60)
    log.info("Step 2: Generating round 1 video")
    video = generate_video(pipeline, cfg, args.prompt, args.size, args.frame_num, args.seed)
    vid_path_1 = os.path.join(video_dir, "output1.mp4")
    save(video, vid_path_1, cfg)
    del video
    torch.cuda.empty_cache()
    log.info("  saved: %s", vid_path_1)

    # --- Step 3: Iterative refinement ---
    current_prompt = args.prompt
    for i in range(1, args.rounds + 1):
        log.info("=" * 60)
        log.info("Round %d / %d", i, args.rounds)
        vid_path = os.path.join(video_dir, f"output{i}.mp4")

        # 3a: Video captioning
        log.info("  [caption] Qwen3-VL video understanding ...")
        caption = reasoner.caption_video(
            vid_path,
            f"This is a video of {main_objects}. Describe the motion, deformation, "
            "and what physics laws it obeys or disobeys in detail.",
        )
        data[f"caption_{i}"] = caption
        log.info("  caption: %s", caption[:200])

        # 3b: Mismatch analysis
        log.info("  [mismatch] Qwen3-VL text reasoning ...")
        mismatch = get_mismatch(reasoner, current_prompt, caption)
        data[f"mismatch_{i}"] = mismatch
        log.info("  mismatch: %s", mismatch[:200])

        # 3c: SA & PC scoring
        log.info("  [scoring] videocon-physics ...")
        sa = sa_score(vid_path, current_prompt, eval_dir, args.videocon_gpu)
        pc = pc_score(vid_path, eval_dir, args.videocon_gpu)
        score = sa * 0.5 + pc * 0.5
        data[f"sa_{i}"] = sa
        data[f"pc_{i}"] = pc
        log.info("  SA=%.4f  PC=%.4f  combined=%.4f", sa, pc, score)

        # 3d: Prompt enhancement
        log.info("  [enhance] Qwen3-VL text reasoning ...")
        refined = get_enhanced_prompt(
            reasoner, current_prompt, physical_law, mismatch, str(score)
        )
        data[f"prompt_{i + 1}"] = refined
        log.info("  refined prompt: %s", refined[:200])
        current_prompt = refined

        # 3e: Generate next video
        log.info("  [generate] Wan 2.2 round %d ...", i + 1)
        video = generate_video(pipeline, cfg, refined, args.size, args.frame_num, args.seed)
        vid_path_next = os.path.join(video_dir, f"output{i + 1}.mp4")
        save(video, vid_path_next, cfg)
        del video
        torch.cuda.empty_cache()
        log.info("  saved: %s", vid_path_next)

    # --- Save tracking data ---
    df = pd.DataFrame([data])
    csv_path = os.path.join(args.output_dir, "data_df.csv")
    df.to_csv(csv_path, index_label="index")
    log.info("=" * 60)
    log.info("Done! Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
