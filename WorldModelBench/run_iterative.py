#!/usr/bin/env python
"""
Dynamic-steps iterative refinement on WorldModelBench (I2V).

Applies Phyrefine's caption→mismatch→enhance→regenerate loop to WorldModelBench,
with linearly ramping denoising steps and per-round VILA evaluation.

Usage:
    cd Wan2.2 && conda activate wan
    python WorldModelBench/run_iterative.py \
        --rounds 2 --min_steps 20 --max_steps 50 \
        --wan_gpus 1,2,3,4,5 --qwen_device cuda:0 \
        --eval_gpu 0 --judge_path ./models/vila-ewm-qwen2-1.5b
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp

BENCH_DIR = Path(__file__).resolve().parent
WAN_ROOT = BENCH_DIR.parent  # Wan2.2/

# Add Phyrefine to path for Qwen3-VL modules
sys.path.insert(0, str(WAN_ROOT / "Phyrefine"))
from myphyt2v.qwen_reasoner import Qwen3VLReasoner
from myphyt2v.prompts import (
    PHYSICAL_LAW_SYSTEM, PHYSICAL_LAW_USER,
    get_mismatch, get_enhanced_prompt,
)
from myphyt2v.utils import load_json, save_json, extract_json_from_response

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────

def get_steps_for_round(current_round, total_rounds, min_steps=20, max_steps=50):
    """Linear ramp from min_steps (round 1) to max_steps (final round)."""
    if total_rounds <= 1:
        return max_steps
    ratio = (max(1, min(current_round, total_rounds)) - 1) / (total_rounds - 1)
    return int(min_steps + ratio * (max_steps - min_steps))


def stem_of(inst):
    """Video filename stem matching WorldModelBench convention."""
    return Path(inst["first_frame"]).stem


def video_path_for(inst, video_dir):
    return os.path.join(video_dir, f"{stem_of(inst)}.mp4")


def load_instances(limit=0):
    """Load worldmodelbench.json, optionally limited to first N."""
    with open(BENCH_DIR / "worldmodelbench.json") as f:
        instances = json.load(f)
    if limit > 0:
        instances = instances[:limit]
    return instances


# ─── Phase: Extract physical laws (one-time) ─────────────────────────────

def phase_physical_laws(reasoner, instances, checkpoint_path):
    log.info("=" * 60)
    log.info("Extracting physical laws for %d instances", len(instances))
    results = load_json(checkpoint_path)

    for i, inst in enumerate(instances):
        key = stem_of(inst)
        if key in results:
            continue
        prompt = inst["text_instruction"]
        log.info("[%d/%d] %s: %s", i + 1, len(instances), key, prompt[:80])
        try:
            user = PHYSICAL_LAW_USER.replace("<input_prompt>", prompt)
            resp = reasoner.chat(PHYSICAL_LAW_SYSTEM, user)
            parsed = extract_json_from_response(resp)
            if parsed and "main_object" in parsed and "physical_law" in parsed:
                results[key] = {
                    "main_object": parsed["main_object"],
                    "physical_law": parsed["physical_law"],
                }
            else:
                log.warning("  Failed to parse JSON, saving raw")
                results[key] = {
                    "main_object": prompt.split()[0],
                    "physical_law": resp[:500],
                    "parse_failed": True,
                }
        except Exception as e:
            log.error("  Error: %s", e)
            results[key] = {"main_object": "", "physical_law": "", "error": str(e)}

        if (i + 1) % 10 == 0:
            save_json(results, checkpoint_path)

    save_json(results, checkpoint_path)
    log.info("Physical laws done: %d results", len(results))
    return results


# ─── Phase: Caption videos ───────────────────────────────────────────────

def phase_caption(reasoner, instances, phys_laws, video_dir, checkpoint_path):
    log.info("Captioning videos from %s", video_dir)
    results = load_json(checkpoint_path)

    for i, inst in enumerate(instances):
        key = stem_of(inst)
        if key in results:
            continue

        vid_path = video_path_for(inst, video_dir)
        if not os.path.exists(vid_path):
            log.warning("  Video not found: %s", vid_path)
            results[key] = {"caption": "", "error": "video_not_found"}
            continue

        main_obj = phys_laws.get(key, {}).get("main_object", "the object")
        instruction = (
            f"This is a video of {main_obj}. Describe the motion, deformation, "
            "and what physics laws it obeys or disobeys in detail."
        )
        log.info("[%d/%d] %s captioning...", i + 1, len(instances), key)
        try:
            caption = reasoner.caption_video(vid_path, instruction)
            results[key] = {"caption": caption}
        except Exception as e:
            log.error("  Error: %s", e)
            results[key] = {"caption": "", "error": str(e)}

        if (i + 1) % 10 == 0:
            save_json(results, checkpoint_path)

    save_json(results, checkpoint_path)
    log.info("Captioning done: %d results", len(results))
    return results


# ─── Phase: Mismatch + enhanced prompts ──────────────────────────────────

def phase_enhance(reasoner, instances, phys_laws, captions, checkpoint_path):
    log.info("Mismatch analysis + prompt enhancement")
    results = load_json(checkpoint_path)

    for i, inst in enumerate(instances):
        key = stem_of(inst)
        if key in results:
            continue

        video_caption = captions.get(key, {}).get("caption", "")
        if not video_caption:
            results[key] = {
                "mismatch": "",
                "enhanced_prompt": inst["text_instruction"],
                "error": "no_caption",
            }
            continue

        physical_law = phys_laws.get(key, {}).get("physical_law", "")
        prompt = inst["text_instruction"]

        log.info("[%d/%d] %s mismatch + enhance...", i + 1, len(instances), key)
        try:
            mismatch = get_mismatch(reasoner, prompt, video_caption)
            enhanced = get_enhanced_prompt(
                reasoner, prompt, physical_law, mismatch, score=0.3,
            )
            results[key] = {
                "mismatch": mismatch[:500],
                "enhanced_prompt": enhanced,
            }
        except Exception as e:
            log.error("  Error: %s", e)
            results[key] = {
                "mismatch": "",
                "enhanced_prompt": prompt,
                "error": str(e),
            }

        if (i + 1) % 10 == 0:
            save_json(results, checkpoint_path)

    save_json(results, checkpoint_path)
    log.info("Enhancement done: %d results", len(results))
    return results


# ─── Phase: Generate videos (multi-GPU, I2V) ─────────────────────────────

def wan_worker(gpu_id, chunk, args, enhanced_data, video_dir, num_inference_steps):
    """Worker process: I2V generation on a single GPU."""
    wan_root = str(WAN_ROOT)
    if wan_root not in sys.path:
        sys.path.insert(0, wan_root)

    import wan
    from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video
    from PIL import Image

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    cfg = WAN_CONFIGS["ti2v-5B"]
    size = SIZE_CONFIGS[args.size]
    max_area = MAX_AREA_CONFIGS[args.size]

    log.info("[GPU%d] Loading TI2V-5B ...", gpu_id)
    t_load = time.time()
    pipeline = wan.WanTI2V(
        config=cfg, checkpoint_dir=args.wan_ckpt,
        device_id=gpu_id, rank=0, t5_cpu=True,
    )
    log.info("[GPU%d] Model loaded in %.1fs", gpu_id, time.time() - t_load)

    for i, inst in enumerate(chunk):
        key = stem_of(inst)
        save_path = video_path_for(inst, video_dir)

        if os.path.exists(save_path):
            continue

        enhanced_prompt = inst["text_instruction"]
        if enhanced_data:
            enhanced_prompt = enhanced_data.get(key, {}).get(
                "enhanced_prompt", enhanced_prompt
            )

        img_path = str(BENCH_DIR / inst["first_frame"])
        img = Image.open(img_path).convert("RGB")

        log.info("[GPU%d] [%d/%d] %s", gpu_id, i + 1, len(chunk), key)
        log.info("[GPU%d]   prompt: %s", gpu_id, enhanced_prompt[:120])

        t_start = time.time()
        video = pipeline.generate(
            input_prompt=enhanced_prompt,
            img=img,
            size=size,
            max_area=max_area,
            frame_num=args.frame_num,
            shift=cfg.sample_shift,
            sample_solver="unipc",
            sampling_steps=num_inference_steps,
            guide_scale=cfg.sample_guide_scale,
            seed=args.seed,
            offload_model=True,
        )
        save_video(
            tensor=video[None], save_file=save_path,
            fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
        )
        del video, img
        torch.cuda.empty_cache()
        log.info("[GPU%d]   saved: %s (%.1fs)", gpu_id, save_path, time.time() - t_start)


def phase_generate(instances, enhanced_data, video_dir, args, num_inference_steps):
    log.info("Generating I2V videos to %s (%d steps)", video_dir, num_inference_steps)
    os.makedirs(video_dir, exist_ok=True)

    to_generate = [
        inst for inst in instances
        if not os.path.exists(video_path_for(inst, video_dir))
    ]
    log.info("Skip %d existing, %d to generate",
             len(instances) - len(to_generate), len(to_generate))

    if not to_generate:
        log.info("All videos exist. Skipping.")
        return

    gpu_ids = [int(x) for x in args.wan_gpus.split(",")]
    chunks = {gid: [] for gid in gpu_ids}
    for idx, inst in enumerate(to_generate):
        chunks[gpu_ids[idx % len(gpu_ids)]].append(inst)

    processes = []
    for gpu_id in gpu_ids:
        if not chunks[gpu_id]:
            continue
        proc = mp.Process(
            target=wan_worker,
            args=(gpu_id, chunks[gpu_id], args, enhanced_data, video_dir,
                  num_inference_steps),
        )
        proc.start()
        processes.append((gpu_id, proc))
        log.info("Started GPU %d with %d instances", gpu_id, len(chunks[gpu_id]))

    for gpu_id, proc in processes:
        proc.join()
        if proc.exitcode != 0:
            log.error("GPU %d worker failed with exit code %d", gpu_id, proc.exitcode)

    log.info("Generation done!")


# ─── Phase: Evaluate (VILA subprocess) ───────────────────────────────────

def phase_eval(video_dir, args, round_num):
    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    save_name = os.path.abspath(os.path.join(eval_dir, f"eval_round{round_num}"))
    result_file = f"{save_name}_cot.json" if args.cot else f"{save_name}.json"

    if os.path.exists(result_file):
        log.info("Round %d eval already exists, loading...", round_num)
        _print_eval_summary(result_file, round_num)
        return

    log.info("=" * 60)
    log.info("VILA evaluation for round %d on %s", round_num, video_dir)

    judge_path = os.path.abspath(args.judge_path)
    video_dir_abs = os.path.abspath(video_dir)
    cot_flag = "--cot" if args.cot else ""

    # evaluation.py loads ./worldmodelbench.json from CWD → must run from BENCH_DIR
    cmd = (
        f"CUDA_VISIBLE_DEVICES={args.eval_gpu} "
        f"conda run --no-banner -n vila python evaluation.py "
        f"--judge {judge_path} "
        f"--video_dir {video_dir_abs} "
        f"--model_name Wan2.2-TI2V-5B-round{round_num} "
        f"--save_name {save_name} {cot_flag}"
    )
    log.info("Running: %s", cmd)
    ret = subprocess.run(cmd, shell=True, cwd=str(BENCH_DIR))
    if ret.returncode != 0:
        log.error("Evaluation failed with exit code %d", ret.returncode)
        return

    _print_eval_summary(result_file, round_num)


def _print_eval_summary(result_file, round_num):
    """Parse eval JSON and log score summary."""
    if not os.path.exists(result_file):
        return
    import numpy as np
    with open(result_file) as f:
        results = json.load(f)
    accs = results.get("accs", {})
    num_vids = len(results.get("preds", {}))

    log.info("─── Round %d Eval Summary (%d videos) ───", round_num, num_vids)
    total = 0.0

    if "instruction" in accs:
        score = float(np.mean(accs["instruction"]))
        log.info("  Instruction Following: %.2f / 3", score)
        total += score

    if "physical_laws" in accs:
        pl = accs["physical_laws"]
        names = ["Newton", "Mass", "Fluid", "Penetration", "Gravity"]
        pl_total = 0.0
        for j, name in enumerate(names):
            sub = float(np.mean([pl[k] for k in range(j, len(pl), 5)]))
            log.info("  Physics / %-12s: %.2f", name, sub)
            pl_total += sub
        log.info("  Physics Overall: %.2f / 5", pl_total)
        total += pl_total

    if "common_sense" in accs:
        cs = accs["common_sense"]
        cs_names = ["Framewise", "Temporal"]
        cs_total = 0.0
        for j, name in enumerate(cs_names):
            sub = float(np.mean([cs[k] for k in range(j, len(cs), 2)]))
            log.info("  Common Sense / %-8s: %.2f", name, sub)
            cs_total += sub
        log.info("  Common Sense Overall: %.2f / 2", cs_total)
        total += cs_total

    log.info("  TOTAL: %.2f / 10", total)


# ─── Main ─────────────────────────────────────────────────────────────────

def _gpu_overlap(qwen_device, wan_gpus):
    qwen_id = int(qwen_device.replace("cuda:", ""))
    wan_ids = [int(x) for x in wan_gpus.split(",")]
    return qwen_id in wan_ids


def parse_args():
    p = argparse.ArgumentParser(description="Dynamic-steps iterative refinement on WorldModelBench")
    # Rounds
    p.add_argument("--rounds", type=int, default=2, help="Refinement rounds (total videos = rounds+1)")
    p.add_argument("--min_steps", type=int, default=20, help="Min denoising steps (round 1)")
    p.add_argument("--max_steps", type=int, default=50, help="Max denoising steps (final round)")
    p.add_argument("--limit", type=int, default=0, help="Limit to first N instances (0 = all)")
    # Directories
    p.add_argument("--output_dir", default=str(BENCH_DIR / "iterative_outputs"))
    # Qwen3-VL
    p.add_argument("--qwen_model", default=str(WAN_ROOT / "models" / "Qwen3-VL-8B-Instruct"))
    p.add_argument("--qwen_device", default="cuda:0")
    # Wan 2.2
    p.add_argument("--wan_ckpt", default=str(WAN_ROOT / "models" / "Wan2.2-TI2V-5B"))
    p.add_argument("--wan_gpus", default="1,2,3,4,5")
    p.add_argument("--size", default="704*1280")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    # Eval
    p.add_argument("--judge_path", default=str(WAN_ROOT / "models" / "vila-ewm-qwen2-1.5b"))
    p.add_argument("--eval_gpu", type=int, default=0)
    p.add_argument("--cot", action="store_true", default=True, help="Chain-of-thought eval")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mp.set_start_method("spawn", force=True)

    instances = load_instances(args.limit)
    log.info("Loaded %d instances (limit=%d)", len(instances), args.limit)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    shared_gpu = _gpu_overlap(args.qwen_device, args.wan_gpus)
    if shared_gpu:
        log.info("Qwen and Wan share GPU — will unload/reload around generation")

    # Load Qwen3-VL
    log.info("Loading Qwen3-VL: %s on %s", args.qwen_model, args.qwen_device)
    reasoner = Qwen3VLReasoner(model_path=args.qwen_model, device=args.qwen_device)

    # One-time: extract physical laws
    phys_path = os.path.join(ckpt_dir, "phys_laws.json")
    phys_laws = phase_physical_laws(reasoner, instances, phys_path)

    total_rounds = args.rounds + 1

    for r in range(1, total_rounds + 1):
        log.info("=" * 60)
        log.info("===== ROUND %d / %d =====", r, total_rounds)

        # ── Generate ──
        video_dir = os.path.join(args.output_dir, f"round{r}_videos")
        steps = get_steps_for_round(r, total_rounds, args.min_steps, args.max_steps)
        enhanced_data = None
        if r > 1:
            enhanced_data = load_json(os.path.join(ckpt_dir, f"enhanced_r{r - 1}.json"))
        log.info("Generating round %d with %d steps", r, steps)

        if shared_gpu:
            log.info("Unloading Qwen3-VL for generation")
            reasoner.unload()
            reasoner = None
            torch.cuda.empty_cache()

        phase_generate(instances, enhanced_data, video_dir, args, steps)

        if shared_gpu and r < total_rounds:
            log.info("Reloading Qwen3-VL")
            reasoner = Qwen3VLReasoner(model_path=args.qwen_model, device=args.qwen_device)

        # ── Evaluate ──
        phase_eval(video_dir, args, round_num=r)

        # ── Caption + Enhance (if not final round) ──
        if r < total_rounds:
            if reasoner is None:
                reasoner = Qwen3VLReasoner(model_path=args.qwen_model, device=args.qwen_device)

            cap_path = os.path.join(ckpt_dir, f"captions_r{r}.json")
            captions = phase_caption(reasoner, instances, phys_laws, video_dir, cap_path)

            enh_path = os.path.join(ckpt_dir, f"enhanced_r{r}.json")
            phase_enhance(reasoner, instances, phys_laws, captions, enh_path)

    # Cleanup
    if reasoner is not None:
        reasoner.unload()
    torch.cuda.empty_cache()

    log.info("All done! Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
