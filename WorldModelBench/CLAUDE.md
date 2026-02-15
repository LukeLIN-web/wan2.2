# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

WorldModelBench evaluates video generation models as world models. It uses a VILA-based VLM judge (`vila-ewm-qwen2-1.5b`) to score generated videos on three axes: Instruction Following, Physical Laws adherence, and Common Sense.

This repo lives inside the Wan2.2 project and is used to benchmark its video generation outputs.

## Running Evaluation

```
conda activate vila
python evaluation.py --model_name TESTED_MODEL --video_dir GENERATED_VIDEOS --judge PATH_TO_JUDGE
```

Key flags:
- `--cot` — Enable chain-of-thought reasoning (saves to `*_cot.json`)
- `--save_name` — Custom output filename prefix (default: `worldmodelbench_results`)
- `--no-save` — Skip saving results to disk

Resumes automatically: if the output JSON already exists, it loads cached results instead of re-evaluating.

## Video Generation → Evaluation Workflow

1. Load `worldmodelbench.json` (350 instances across 7 domains, 56 subdomains)
2. Generate videos using Wan2.2 (or any model):
   - **T2V**: prompt = `text_first_frame + " " + text_instruction`
   - **I2V**: image = `instance["first_frame"]`, text = `instance["text_instruction"]`
3. Save videos as `.mp4` files named identically to `first_frame` (replacing `.jpg` → `.mp4`)
4. Run `evaluation.py` pointing `--video_dir` at the generated videos

## Architecture

Single-file design in `evaluation.py`:

- **`EvaluationConfig`** — Prompt templates and question pools for each evaluation type
- **`WorldModelEvaluator`** — Loads VILA judge via `llava.load()`, runs per-video evaluation
- **`ResultsPrinter`** — Rich-based formatted output

### Scoring Logic

| Evaluation Type | Questions | Scoring |
|---|---|---|
| `instruction` | 1 per video (uses `text_instruction`) | 0–3 scale parsed from judge output |
| `physical_laws` | 5 fixed questions (Newton, mass, fluid, penetration, gravity) | Binary (yes/no → "no" = pass) |
| `common_sense` | 2 fixed questions (aesthetics, temporal consistency) | Binary (yes/no → "no" = pass) |

Final score = sum of sub-category scores. Results submitted to `worldmodelbench.team@gmail.com` for leaderboard.

## Dependencies

- **VILA** (`llava` package) — Bundled in `VILA/` directory. Install per [VILA Installation Guide](https://github.com/NVlabs/VILA).
- `mmengine` (load/dump JSON), `rich` (console output), `numpy`, `tqdm`

## Data

- `worldmodelbench.json` — 350 test instances, each with `domain`, `subdomain`, `text_first_frame`, `text_instruction`, `first_frame`
- `images/` — 350 first-frame JPGs referenced by `first_frame` field
