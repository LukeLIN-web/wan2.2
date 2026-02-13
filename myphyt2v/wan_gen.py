"""Wan 2.2 video generation wrapper."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wan
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video


def make_pipeline(task="ti2v-5B", ckpt_dir="./models/Wan2.2-TI2V-5B", device_id=0):
    cfg = WAN_CONFIGS[task]
    if "t2v" in task and "ti2v" not in task:
        pipeline = wan.WanT2V(
            config=cfg, checkpoint_dir=ckpt_dir, device_id=device_id, rank=0, t5_cpu=True
        )
    else:
        pipeline = wan.WanTI2V(
            config=cfg, checkpoint_dir=ckpt_dir, device_id=device_id, rank=0, t5_cpu=True
        )
    return pipeline, cfg


def generate_video(
    pipeline,
    cfg,
    prompt,
    size="1280*704",
    frame_num=81,
    seed=42,
    sampling_steps=50,
):
    video = pipeline.generate(
        input_prompt=prompt,
        size=SIZE_CONFIGS[size],
        frame_num=frame_num,
        shift=cfg.sample_shift,
        sample_solver="unipc",
        sampling_steps=sampling_steps,
        guide_scale=cfg.sample_guide_scale,
        seed=seed,
        offload_model=True,
    )
    return video


def save(video, path, cfg):
    save_video(
        tensor=video[None],
        save_file=path,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
