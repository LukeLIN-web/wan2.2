"""Videocon-physics SA & PC scoring (calls original entailment_inference.py)."""

import csv
import os
import subprocess
import tempfile

VIDEOCON_SCRIPT = "videophy/videocon/training/pipeline_video/entailment_inference.py"
VIDEOCON_CKPT = "videophy/videocon_physics"


def _run_videocon(input_csv, output_csv, gpu_id=0):
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {VIDEOCON_SCRIPT} "
        f"--input_csv {input_csv} --output_csv {output_csv} "
        f"--checkpoint {VIDEOCON_CKPT}"
    )
    subprocess.run(cmd, shell=True, check=True, cwd=os.path.join(os.path.dirname(__file__), ".."))


def sa_score(video_path, prompt, eval_dir, gpu_id=0):
    """Semantic Alignment score: does video entail the prompt?"""
    caption = (
        "The following is a conversation between a curious human and AI assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "Human: <|video|>\n"
        f"Human: Does this video entail the description: {prompt}?\n"
        "AI: "
    )
    in_csv = os.path.join(eval_dir, "sa_input.csv")
    out_csv = os.path.join(eval_dir, "sa_output.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    with open(in_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["videopath", "caption"])
        writer.writerow([video_path, caption])
    _run_videocon(in_csv, out_csv, gpu_id)
    with open(out_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            return float(row[-1])
    return 0.0


def pc_score(video_path, eval_dir, gpu_id=0):
    """Physical Correctness score: does video follow physical laws?"""
    caption = (
        "The following is a conversation between a curious human and AI assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
        "Human: <|video|>\n"
        "Human: Does this video follow the physical laws?\n"
        "AI: "
    )
    in_csv = os.path.join(eval_dir, "pc_input.csv")
    out_csv = os.path.join(eval_dir, "pc_output.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    with open(in_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["videopath", "caption"])
        writer.writerow([video_path, caption])
    _run_videocon(in_csv, out_csv, gpu_id)
    with open(out_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            return float(row[-1])
    return 0.0
