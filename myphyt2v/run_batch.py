#!/usr/bin/env python
"""
PhyT2V batch pipeline for VIDEOPHY2 benchmark.

Reuses existing round1 videos from videophy_outputs/, refines prompts
with Qwen3-VL, generates round2 videos, evaluates with VIDEOPHY2 AutoEval.

Phases (resumable via JSON checkpoints):
  1. Extract physical laws (Qwen3-VL text)
  2. Caption round1 videos (Qwen3-VL video understanding)
  3. Mismatch + enhanced prompts (Qwen3-VL text)
  4. Generate round2 videos (Wan, multi-GPU)
  5. Prepare eval CSV + run VIDEOPHY2 AutoEval
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from myphyt2v.qwen_reasoner import Qwen3VLReasoner
from myphyt2v.prompts import (
    PHYSICAL_LAW_SYSTEM, PHYSICAL_LAW_USER,
    MISMATCH_SYSTEM, MISMATCH_USER,
    ENHANCED_SYSTEM, ENHANCED_USER,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="PhyT2V batch pipeline for VIDEOPHY2")
    p.add_argument("--prompt_csv", default="./videophy_data/videophy2_prompts.csv")
    p.add_argument("--round1_dir", default="./videophy_outputs",
                    help="Directory with existing round1 videos")
    p.add_argument("--output_dir", default="./phyt2v_outputs",
                    help="Directory for round2 videos + intermediate JSONs")
    # Qwen3-VL
    p.add_argument("--qwen_model", default="./models/Qwen3-VL-8B-Instruct")
    p.add_argument("--qwen_device", default="cuda:0")
    # Wan 2.2
    p.add_argument("--wan_ckpt", default="./models/Wan2.2-TI2V-5B")
    p.add_argument("--wan_gpus", default="1,2,3,4,5", help="GPUs for Wan generation")
    p.add_argument("--size", default="1280*704")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    # Control
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--phase", type=int, default=0,
                    help="Start from phase (0=all, 1-5=specific phase)")
    # Eval
    p.add_argument("--eval_gpu", type=int, default=0, help="GPU for VIDEOPHY2 AutoEval")
    return p.parse_args()


def load_prompts(csv_path, start=0, end=-1):
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
    total = len(prompts)
    end = end if end > 0 else total
    return prompts[start:end], total


def safe_filename(prompt_id, caption, max_len=80):
    clean = caption.replace(" ", "_").replace("/", "_").replace('"', "")
    clean = "".join(c for c in clean if c.isalnum() or c in "_-.,")
    return f"{prompt_id:04d}_{clean[:max_len]}.mp4"


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json_from_response(resp):
    """Extract JSON object from Qwen3-VL response (may contain thinking prefix)."""
    for i in range(len(resp)):
        if resp[i] == "{":
            # Find matching closing brace
            depth = 0
            for j in range(i, len(resp)):
                if resp[j] == "{":
                    depth += 1
                elif resp[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(resp[i:j+1], strict=False)
                        except json.JSONDecodeError:
                            break
    return None


# ─── Phase 1: Extract physical laws ───────────────────────────────────────────

def phase1_physical_laws(reasoner, prompts, checkpoint_path):
    log.info("=" * 60)
    log.info("Phase 1: Extracting physical laws for %d prompts", len(prompts))
    results = load_json(checkpoint_path)

    for i, p in enumerate(prompts):
        key = str(p["id"])
        if key in results:
            continue
        log.info("[%d/%d] id=%d: %s", i+1, len(prompts), p["id"], p["caption"][:80])
        user = PHYSICAL_LAW_USER.replace("<input_prompt>", p["caption"])
        try:
            resp = reasoner.chat(PHYSICAL_LAW_SYSTEM, user)
            parsed = extract_json_from_response(resp)
            if parsed and "main_object" in parsed and "physical_law" in parsed:
                results[key] = {
                    "main_object": parsed["main_object"],
                    "physical_law": parsed["physical_law"],
                }
            else:
                log.warning("  Failed to parse JSON, saving raw response")
                results[key] = {
                    "main_object": p["caption"].split()[0],
                    "physical_law": resp[:500],
                    "parse_failed": True,
                }
        except Exception as e:
            log.error("  Error: %s", e)
            results[key] = {"main_object": "", "physical_law": "", "error": str(e)}

        if (i + 1) % 10 == 0:
            save_json(results, checkpoint_path)
            log.info("  [checkpoint saved: %d/%d]", len(results), len(prompts))

    save_json(results, checkpoint_path)
    log.info("Phase 1 done: %d results", len(results))
    return results


# ─── Phase 2: Caption round1 videos ──────────────────────────────────────────

def phase2_captions(reasoner, prompts, phys_laws, round1_dir, checkpoint_path):
    log.info("=" * 60)
    log.info("Phase 2: Captioning %d round1 videos", len(prompts))
    results = load_json(checkpoint_path)

    for i, p in enumerate(prompts):
        key = str(p["id"])
        if key in results:
            continue

        vid_path = os.path.join(round1_dir, safe_filename(p["id"], p["caption"]))
        if not os.path.exists(vid_path):
            log.warning("  Video not found: %s", vid_path)
            results[key] = {"caption": "", "error": "video_not_found"}
            continue

        main_obj = phys_laws.get(key, {}).get("main_object", "the object")
        instruction = (
            f"This is a video of {main_obj}. Describe the motion, deformation, "
            "and what physics laws it obeys or disobeys in detail."
        )
        log.info("[%d/%d] id=%d captioning...", i+1, len(prompts), p["id"])
        try:
            caption = reasoner.caption_video(vid_path, instruction)
            results[key] = {"caption": caption}
        except Exception as e:
            log.error("  Error: %s", e)
            results[key] = {"caption": "", "error": str(e)}

        if (i + 1) % 10 == 0:
            save_json(results, checkpoint_path)
            log.info("  [checkpoint saved: %d/%d]", len(results), len(prompts))

    save_json(results, checkpoint_path)
    log.info("Phase 2 done: %d results", len(results))
    return results


# ─── Phase 3: Mismatch + enhanced prompts ────────────────────────────────────

def phase3_enhance(reasoner, prompts, phys_laws, captions, checkpoint_path):
    log.info("=" * 60)
    log.info("Phase 3: Mismatch analysis + prompt enhancement for %d prompts", len(prompts))
    results = load_json(checkpoint_path)

    for i, p in enumerate(prompts):
        key = str(p["id"])
        if key in results:
            continue

        video_caption = captions.get(key, {}).get("caption", "")
        if not video_caption:
            results[key] = {"mismatch": "", "enhanced_prompt": p["caption"], "error": "no_caption"}
            continue

        phys = phys_laws.get(key, {})
        physical_law = phys.get("physical_law", "")

        log.info("[%d/%d] id=%d mismatch + enhance...", i+1, len(prompts), p["id"])
        try:
            # Mismatch analysis
            mismatch_user = (MISMATCH_USER
                .replace("<input_prompt>", p["caption"])
                .replace("<video_caption>", video_caption))
            mismatch = reasoner.chat(MISMATCH_SYSTEM, mismatch_user)
            mismatch = mismatch.replace("Mismatch:", "").strip()

            # Enhanced prompt (use 0.3 as default score for round1 - indicating needs improvement)
            enhance_user = (ENHANCED_USER
                .replace("<physical_law>", physical_law)
                .replace("<mismatch>", mismatch)
                .replace("<input_prompt>", p["caption"])
                .replace("<score>", "0.3"))
            enhanced = reasoner.chat(ENHANCED_SYSTEM, enhance_user)
            enhanced = enhanced.replace("Enhanced prompt:", "").strip()

            results[key] = {
                "mismatch": mismatch[:500],
                "enhanced_prompt": enhanced,
            }
        except Exception as e:
            log.error("  Error: %s", e)
            results[key] = {"mismatch": "", "enhanced_prompt": p["caption"], "error": str(e)}

        if (i + 1) % 10 == 0:
            save_json(results, checkpoint_path)
            log.info("  [checkpoint saved: %d/%d]", len(results), len(prompts))

    save_json(results, checkpoint_path)
    log.info("Phase 3 done: %d results", len(results))
    return results


# ─── Phase 4: Generate round2 videos (multi-GPU) ─────────────────────────────

def wan_worker(gpu_id, chunk, args, enhanced_data):
    """Worker process for Wan video generation on a single GPU."""
    import wan
    from wan.configs import SIZE_CONFIGS, MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import save_video

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    task = "ti2v-5B"
    cfg = WAN_CONFIGS[task]
    size = SIZE_CONFIGS[args.size]
    max_area = MAX_AREA_CONFIGS[args.size]

    log.info("[GPU%d] Loading TI2V-5B ...", gpu_id)
    t_load = time.time()
    pipeline = wan.WanTI2V(
        config=cfg, checkpoint_dir=args.wan_ckpt,
        device_id=gpu_id, rank=0, t5_cpu=True,
    )
    log.info("[GPU%d] Model loaded in %.1fs", gpu_id, time.time() - t_load)

    video_dir = os.path.join(args.output_dir, "round2_videos")
    for i, p in enumerate(chunk):
        key = str(p["id"])
        enhanced_prompt = enhanced_data.get(key, {}).get("enhanced_prompt", p["caption"])
        save_path = os.path.join(video_dir, safe_filename(p["id"], p["caption"]))

        if os.path.exists(save_path):
            continue

        log.info("[GPU%d] [%d/%d] id=%d", gpu_id, i+1, len(chunk), p["id"])
        log.info("[GPU%d]   prompt: %s", gpu_id, enhanced_prompt[:120])

        t_start = time.time()
        video = pipeline.generate(
            input_prompt=enhanced_prompt,
            img=None,
            size=size,
            max_area=max_area,
            frame_num=args.frame_num,
            shift=cfg.sample_shift,
            sample_solver="unipc",
            sampling_steps=50,
            guide_scale=cfg.sample_guide_scale,
            seed=args.seed,
            offload_model=True,
        )
        save_video(
            tensor=video[None], save_file=save_path,
            fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1),
        )
        del video
        torch.cuda.empty_cache()
        log.info("[GPU%d]   saved: %s (%.1fs)", gpu_id, save_path, time.time() - t_start)


def phase4_generate(prompts, enhanced_data, args):
    log.info("=" * 60)
    log.info("Phase 4: Generating round2 videos for %d prompts", len(prompts))

    video_dir = os.path.join(args.output_dir, "round2_videos")
    os.makedirs(video_dir, exist_ok=True)

    # Filter out already generated
    to_generate = []
    for p in prompts:
        save_path = os.path.join(video_dir, safe_filename(p["id"], p["caption"]))
        if not os.path.exists(save_path):
            to_generate.append(p)
    log.info("Skip %d existing, %d to generate", len(prompts) - len(to_generate), len(to_generate))

    if not to_generate:
        log.info("All round2 videos exist. Skipping.")
        return

    gpu_ids = [int(x) for x in args.wan_gpus.split(",")]
    num_gpus = len(gpu_ids)
    log.info("Using %d GPU(s): %s", num_gpus, gpu_ids)

    # Split across GPUs
    chunks = {gid: [] for gid in gpu_ids}
    for idx, p in enumerate(to_generate):
        chunks[gpu_ids[idx % num_gpus]].append(p)

    mp.set_start_method("spawn", force=True)
    processes = []
    for gpu_id in gpu_ids:
        proc = mp.Process(target=wan_worker, args=(gpu_id, chunks[gpu_id], args, enhanced_data))
        proc.start()
        processes.append(proc)
        log.info("Started GPU %d with %d prompts", gpu_id, len(chunks[gpu_id]))

    for proc in processes:
        proc.join()

    log.info("Phase 4 done!")


# ─── Phase 5: Evaluate with VIDEOPHY2 AutoEval ───────────────────────────────

def phase5_eval(prompts, args):
    log.info("=" * 60)
    log.info("Phase 5: VIDEOPHY2 AutoEval")

    video_dir = os.path.join(args.output_dir, "round2_videos")
    eval_dir = os.path.join(args.output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Prepare eval CSV
    eval_csv = os.path.join(eval_dir, "eval_sa_pc.csv")
    found = 0
    with open(eval_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["caption", "videopath"])
        for p in prompts:
            vid_path = os.path.join(video_dir, safe_filename(p["id"], p["caption"]))
            if os.path.exists(vid_path):
                abs_path = os.path.abspath(vid_path)
                writer.writerow([p["caption"], abs_path])
                found += 1
    log.info("Eval CSV: %d/%d videos found", found, len(prompts))

    if found == 0:
        log.error("No videos found for evaluation!")
        return

    log.info("Eval CSV ready: %s", eval_csv)
    log.info("To run VIDEOPHY2 AutoEval (requires transformers==4.44.0):")
    log.info("  cd videophy/VIDEOPHY2")
    log.info("  CUDA_VISIBLE_DEVICES=%d python inference.py --input_csv %s --checkpoint ../videophy_2_auto --output_csv %s --task sa",
             args.eval_gpu, os.path.abspath(eval_csv), os.path.join(eval_dir, "eval_output_sa.csv"))
    log.info("  CUDA_VISIBLE_DEVICES=%d python inference.py --input_csv %s --checkpoint ../videophy_2_auto --output_csv %s --task pc",
             args.eval_gpu, os.path.abspath(eval_csv), os.path.join(eval_dir, "eval_output_pc.csv"))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompts, total = load_prompts(args.prompt_csv, args.start, args.end)
    log.info("Loaded %d prompts (total %d, range [%d:%d])",
             len(prompts), total, args.start, args.end if args.end > 0 else total)

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Phase 1-3: Qwen3-VL ──
    if args.phase <= 3:
        reasoner = None
        if args.phase <= 1:
            log.info("Loading Qwen3-VL: %s on %s", args.qwen_model, args.qwen_device)
            reasoner = Qwen3VLReasoner(model_path=args.qwen_model, device=args.qwen_device)

        # Phase 1
        phys_path = os.path.join(ckpt_dir, "phys_laws.json")
        if args.phase <= 1:
            phys_laws = phase1_physical_laws(reasoner, prompts, phys_path)
        else:
            phys_laws = load_json(phys_path)

        # Phase 2
        captions_path = os.path.join(ckpt_dir, "captions.json")
        if args.phase <= 2:
            if reasoner is None:
                log.info("Loading Qwen3-VL: %s on %s", args.qwen_model, args.qwen_device)
                reasoner = Qwen3VLReasoner(model_path=args.qwen_model, device=args.qwen_device)
            captions = phase2_captions(reasoner, prompts, phys_laws, args.round1_dir, captions_path)
        else:
            captions = load_json(captions_path)

        # Phase 3
        enhanced_path = os.path.join(ckpt_dir, "enhanced.json")
        if args.phase <= 3:
            if reasoner is None:
                log.info("Loading Qwen3-VL: %s on %s", args.qwen_model, args.qwen_device)
                reasoner = Qwen3VLReasoner(model_path=args.qwen_model, device=args.qwen_device)
            enhanced_data = phase3_enhance(reasoner, prompts, phys_laws, captions, enhanced_path)
        else:
            enhanced_data = load_json(enhanced_path)

        # Free Qwen3-VL
        if reasoner is not None:
            reasoner.unload()
            del reasoner
            torch.cuda.empty_cache()
    else:
        phys_laws = load_json(os.path.join(ckpt_dir, "phys_laws.json"))
        captions = load_json(os.path.join(ckpt_dir, "captions.json"))
        enhanced_data = load_json(os.path.join(ckpt_dir, "enhanced.json"))

    # ── Phase 4: Wan generation ──
    if args.phase <= 4:
        phase4_generate(prompts, enhanced_data, args)

    # ── Phase 5: Prepare eval CSV (scoring done separately) ──
    if args.phase <= 5:
        phase5_eval(prompts, args)
        log.info("Phase 1-4 complete. Run eval separately with transformers==4.44.0")


if __name__ == "__main__":
    main()
