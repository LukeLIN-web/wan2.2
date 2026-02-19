# WorldModelBench

[**🌐 Homepage**](https://worldmodelbench-team.github.io/) | [**🏆 Leaderboard**](https://worldmodelbench-team.github.io/#leaderboard) | [**📖 WorldModelBench arXiv**](https://arxiv.org/pdf/2502.20694)

This repo contains the evaluation instructions for the paper "[WorldModelBench: Judging Video Generation Models As World Models](https://arxiv.org/pdf/2502.20694)".

## 🔔News
- **🔥[2025-07-28]: We have moved the data from huggingface to github for easier documentation.**
- **🔥[2025-05-21]: WorldModelBench has been accepted as an oral paper in the CVPR 2025 WorldModelBench workshop. 😆**
- **🔥[2025-03-02]: Our [WorldModelBench](https://worldmodelbench-team.github.io/) is now available. We look forward to your participation! 😆**

## Introduction

### WorldModelBench

WorldModelBencha is a benchmark designed to evaluate the **world modeling capabilities** of video generation models across **7** application-driven domains (spanning from Robotics, Driving, Industry, Human Activities, Gaming, Animation, and Natural) and **56** subdomains. Each domain features 50 carefully curated prompts, comprising a text description and an initial video frame, tailored for video generation. We provide a **human-aligned** VLM (Vision-Language Model) based judger to automatically evaluate model-generated videos on **Instruction Following**, **Common Sense**, and **Physical Adherence**.

![Alt text](worldmodelbench.png)

## Evaluation

🎯 Please refer to the following instructions to evaluate with WorldModelBench:
- **Environment Setup**: Clone and install VILA by following the instructions in [VILA Installation Guide](https://github.com/NVlabs/VILA?tab=readme-ov-file#installation).
- **Data&Model Preparation**: Download the [judge](https://huggingface.co/Efficient-Large-Model/vila-ewm-qwen2-1.5b).
```
└── worldmodelbench
    └── images (first frames of videos)
    └── evaluation.py (evaluation script)
    └── worldmodelbench.json (test set)
    ...
```
The ```worldmodelbench.json``` has a list of dict containing instances for video generation.
```
[
  {
          "domain": "autonomous vehicle",
          "subdomain": "Stopping",
          "text_first_frame": "The autonomous vehicle approaches a traffic light on a bridge surrounded by tall buildings. Construction barriers line the sides of the bridge with a yellow traffic light visible ahead.",
          "text_instruction": "The autonomous vehicle stops at the traffic light on the bridge.",
          "first_frame": "images/69620089860948e38a4921dd4869d24f.jpg"
      }
...
]
```
- **Video Generation**: There are 350 test instances in `worldmodelbench.json`. For *each instance*, generate a video using either T2V or I2V mode:
  - **Text-to-Video (T2V)**: Concatenate the scene description and instruction as the prompt:
    ```python
    prompt = instance["text_first_frame"] + " " + instance["text_instruction"]
    ```
  - **Image-to-Video (I2V)**: Use the first frame as the image input and the instruction as the text prompt:
    ```python
    image = instance["first_frame"]   # e.g. "images/xxxx.jpg"
    prompt = instance["text_instruction"]
    ```

  **Note**: Save the video using the **same name** as `instance["first_frame"]`, replacing `.jpg` with `.mp4` (e.g. `images/xxxx.jpg` → `xxxx.mp4`).

  **Batch generation with Wan2.2** (TI2V-5B, 2 GPUs):
  ```bash
  conda activate wan
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 WorldModelBench/generate_videos.py
  ```
  This script loads `worldmodelbench.json`, skips already-generated videos, and saves outputs to `WorldModelBench/outputs/`.

- **Evaluation**: Install VILA environment, then run:
  ```bash
  conda activate vila
  python evaluation.py --model_name TESTED_MODEL --video_dir GENERATED_VIDEOS --judge PATH_TO_JUDGE --cot
  ```
  Multi-GPU parallel evaluation (8 GPUs):
  ```bash
  ./eval_v2.sh outputs 350 Wan2.2-TI2V-5B
  ```

## Scoring

The VILA judge (`vila-ewm-qwen2-1.5b`) evaluates each video on **three axes**:

| Axis | Questions per Video | Scoring Method |
|---|---|---|
| **Instruction Following** | 1 (uses `text_instruction`) | 0–3 scale: 0 = no match, 1 = wrong object/action, 2 = tendency toward goal, 3 = precise |
| **Physical Laws** | 5 (Newton's Law, Conservation of Mass, Fluid Law, Non-penetration, Gravity) | Binary per question: judge answers "Yes/No" to whether a violation exists; "No" = pass (1 point) |
| **Common Sense** | 2 (Aesthetics, Temporal Consistency) | Binary per question: judge answers "Yes/No" to whether an issue exists; "No" = pass (1 point) |

- **Instruction Following** score: 0–3 (single question, parsed from `Score: [N]` in judge output)
- **Physical Laws** score: 0–5 (sum of 5 binary sub-scores)
- **Common Sense** score: 0–2 (sum of 2 binary sub-scores)
- **Total score** = Instruction Following + Physical Laws + Common Sense (max 10 per video)

Use `--cot` to enable chain-of-thought reasoning for more detailed judge outputs (saved as `*_cot.json`).

The answers and explanations for the test set questions are withheld. You can submit your results to worldmodelbench.team@gmail.com to be considered for the leaderboard.

## Disclaimers
The guidelines for the annotators emphasized strict compliance with copyright and licensing rules from the initial data source, specifically avoiding materials from websites that forbid copying and redistribution. 
Should you encounter any data samples potentially breaching the copyright or licensing regulations of any site, we encourage you to [contact](#contact) us. Upon verification, such samples will be promptly removed.

## Contact
- Dacheng Li: dacheng177@berkeley.edu
- Yunhao Fang: yuf026@ucsd.edu
- Song Han: songhan@mit.edu
- Yao Lu: jasonlu@nvidia.com

## Citation

**BibTeX:**
```bibtex
@article{Li2025WorldModelBench,
  title={WorldModelBench: Judging Video Generation Models As World Models},
  author={Dacheng Li and Yunhao Fang and Yukang Chen and Shuo Yang and Shiyi Cao and Justin Wong and Michael Luo and Xiaolong Wang and Hongxu Yin and Joseph E. Gonzalez and Ion Stoica and Song Han and Yao Lu},
  year={2025},
}
```
This website is adapted from [MMMU](https://github.com/MMMU-Benchmark/MMMU).
