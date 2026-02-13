#!/usr/bin/env python
"""
用 VideoCon-Physics 评估器计算 SA & PC 分数 (0~1)。
这是 PhyT2V 论文用的评分方式，输出 entailment 概率。

用法：
    # 先准备好视频目录和 prompt CSV
    # 需要 transformers==4.44.0 环境

    # 评估 baseline (round1)
    python -m myphyt2v.eval_videocon \
        --video_dir ./videophy_outputs \
        --prompt_csv ./videophy_data/videophy2_prompts.csv \
        --checkpoint ./videophysyics/videophy/videocon_physics \
        --output_dir ./eval_videocon_baseline \
        --gpu 0

    # 评估 PhyT2V round2
    python -m myphyt2v.eval_videocon \
        --video_dir ./phyt2v_outputs/round2_videos \
        --prompt_csv ./videophy_data/videophy2_prompts.csv \
        --checkpoint ./videophysyics/videophy/videocon_physics \
        --output_dir ./eval_videocon_phyt2v \
        --gpu 0
"""

import argparse
import csv
import os
import sys
import subprocess

import pandas as pd


def safe_filename(prompt_id, caption, max_len=80):
    clean = caption.replace(" ", "_").replace("/", "_").replace('"', "")
    clean = "".join(c for c in clean if c.isalnum() or c in "_-.,")
    return f"{prompt_id:04d}_{clean[:max_len]}.mp4"


SA_TEMPLATE = (
    'The following is a conversation between a curious human and AI assistant. '
    'The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n'
    'Human: <|video|>\n'
    'Human: Does this video entail the description: "{caption}"?\n'
    'AI: '
)

PC_TEMPLATE = (
    'The following is a conversation between a curious human and AI assistant. '
    'The assistant gives helpful, detailed, and polite answers to the user\'s questions.\n'
    'Human: <|video|>\n'
    'Human: Does this video follow the physical laws?\n'
    'AI: '
)


def parse_args():
    p = argparse.ArgumentParser(description="VideoCon-Physics SA & PC evaluation")
    p.add_argument("--video_dir", required=True, help="Directory with generated videos")
    p.add_argument("--prompt_csv", default="./videophy_data/videophy2_prompts.csv")
    p.add_argument("--checkpoint", default="./videophysyics/videophy/videocon_physics")
    p.add_argument("--output_dir", required=True, help="Output directory for eval results")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=16)
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
    return prompts[start:end]


def prepare_csv(prompts, video_dir, output_csv, template):
    """Prepare input CSV for entailment_inference.py (no header, columns: videopath, caption)."""
    found = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["videopath", "caption"])
        for p in prompts:
            vid_path = os.path.join(video_dir, safe_filename(p["id"], p["caption"]))
            if not os.path.exists(vid_path):
                continue
            abs_path = os.path.abspath(vid_path)
            if "{caption}" in template:
                caption_text = template.format(caption=p["caption"])
            else:
                caption_text = template
            writer.writerow([abs_path, caption_text])
            found += 1
    return found


def run_entailment(input_csv, output_csv, checkpoint, gpu, batch_size):
    """Run entailment_inference.py."""
    script = os.path.join(
        os.path.dirname(__file__), "..",
        "videophysyics", "videophy", "videocon", "training", "pipeline_video",
        "entailment_inference.py"
    )
    cwd = os.path.join(
        os.path.dirname(__file__), "..",
        "videophysyics", "videophy", "videocon", "training", "pipeline_video"
    )
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} python {os.path.abspath(script)} "
        f"--input_csv {os.path.abspath(input_csv)} "
        f"--output_csv {os.path.abspath(output_csv)} "
        f"--checkpoint {os.path.abspath(checkpoint)} "
        f"--batch_size {batch_size}"
    )
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=os.path.abspath(cwd))


def compute_scores(sa_csv, pc_csv, prompts):
    """Read output CSVs and compute mean scores."""
    # Output CSV has no header: videopath, caption, score
    sa_df = pd.read_csv(sa_csv, header=None, names=["videopath", "caption", "score"])
    pc_df = pd.read_csv(pc_csv, header=None, names=["videopath", "caption", "score"])

    print("=" * 60)
    print(f"VideoCon-Physics Results ({len(sa_df)} videos)")
    print(f"  Mean SA: {sa_df['score'].mean():.4f}")
    print(f"  Mean PC: {pc_df['score'].mean():.4f}")
    print(f"  Combined (0.5*SA + 0.5*PC): {(sa_df['score'].mean() + pc_df['score'].mean()) / 2:.4f}")

    # Hard/Easy breakdown
    id_to_hard = {p["id"]: p["is_hard"] for p in prompts}
    sa_df["pid"] = sa_df["videopath"].apply(
        lambda x: int(os.path.basename(x).split("_")[0]))
    pc_df["pid"] = pc_df["videopath"].apply(
        lambda x: int(os.path.basename(x).split("_")[0]))
    sa_df["is_hard"] = sa_df["pid"].map(id_to_hard)
    pc_df["is_hard"] = pc_df["pid"].map(id_to_hard)

    for subset, label in [(0, "Easy"), (1, "Hard")]:
        sa_sub = sa_df[sa_df["is_hard"] == subset]
        pc_sub = pc_df[pc_df["is_hard"] == subset]
        if len(sa_sub) > 0:
            print(f"  {label} ({len(sa_sub)}): SA={sa_sub['score'].mean():.4f}  PC={pc_sub['score'].mean():.4f}")

    # Save summary
    summary = {
        "total": len(sa_df),
        "mean_sa": sa_df["score"].mean(),
        "mean_pc": pc_df["score"].mean(),
    }
    return summary


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prompts = load_prompts(args.prompt_csv, args.start, args.end)
    print(f"Loaded {len(prompts)} prompts")

    # Prepare SA input CSV
    sa_input = os.path.join(args.output_dir, "sa_input.csv")
    sa_output = os.path.join(args.output_dir, "sa_output.csv")
    n_sa = prepare_csv(prompts, args.video_dir, sa_input, SA_TEMPLATE)
    print(f"SA: {n_sa} videos found")

    # Prepare PC input CSV
    pc_input = os.path.join(args.output_dir, "pc_input.csv")
    pc_output = os.path.join(args.output_dir, "pc_output.csv")
    n_pc = prepare_csv(prompts, args.video_dir, pc_input, PC_TEMPLATE)
    print(f"PC: {n_pc} videos found")

    if n_sa == 0:
        print("No videos found! Check --video_dir")
        return

    # Run SA evaluation
    if not os.path.exists(sa_output):
        print("\n--- Running SA evaluation ---")
        run_entailment(sa_input, sa_output, args.checkpoint, args.gpu, args.batch_size)
    else:
        print(f"SA output exists: {sa_output}, skipping")

    # Run PC evaluation
    if not os.path.exists(pc_output):
        print("\n--- Running PC evaluation ---")
        run_entailment(pc_input, pc_output, args.checkpoint, args.gpu, args.batch_size)
    else:
        print(f"PC output exists: {pc_output}, skipping")

    # Compute scores
    if os.path.exists(sa_output) and os.path.exists(pc_output):
        summary = compute_scores(sa_output, pc_output, prompts)
        import json
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
