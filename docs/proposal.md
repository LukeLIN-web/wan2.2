# Physics-Aware Video Generation via Iterative VLM Feedback

## Problem

State-of-the-art text-to-video models generate visually impressive videos but frequently violate basic physical laws — objects float, collisions lack reaction forces, fluids defy gravity. On the VIDEOPHY2 benchmark (600 physics-critical prompts), even strong models score poorly:

| Model | Joint Score (SA>=4 & PC>=4) | Hard Subset |
|-------|:---------------------------:|:-----------:|
| CogVideoX-5B | 25.0% | 0% |
| Wan2.1-T2V-14B | 32.6% | 21.9% |

The core issue: T2V models are trained on visual quality, not physical correctness. Simply scaling up model parameters does not solve physics violations.

## Approach: PhyT2V with Wan 2.2 + Qwen3-VL

We adapt the PhyT2V framework (ICLR 2025) — which uses VLM feedback to iteratively refine physics in generated videos — and make two key upgrades:

**1. Stronger Video Backbone: Wan 2.2 (replaces CogVideoX-5B)**

Wan 2.2 is a significantly stronger base model with higher visual fidelity, providing a better starting point for physics refinement.

**2. Unified VLM: Single Qwen3-VL-8B (replaces GPT-4 + Tarsier-34B)**

The original PhyT2V requires two separate models with distinct roles:
- **GPT-4** (text-only LLM, closed-source API): the "reasoning brain" — extracts physical laws from the text prompt (e.g., gravity, elastic collision), compares Tarsier's video caption against the original prompt to identify physics violations, and rewrites the prompt to fix mismatches guided by physics scores. GPT-4 never sees the video directly.
- **Tarsier-34B** (video-language model, 34B params): the "video eye" — watches the generated video (8 sampled frames) and produces a detailed text description of object motion, deformation, and physics behavior, which GPT-4 then reasons over.

We replace both with a single **Qwen3-VL-8B-Instruct** that handles all tasks: physics law extraction, video captioning, mismatch detection, and prompt refinement — since Qwen3-VL is natively multimodal, it can both see the video and reason about physics in one model.

### Pipeline

```
Text Prompt
    │
    ├─ [Qwen3-VL] Extract physical laws relevant to the scene
    ├─ [Wan 2.2]   Generate Round 1 video
    │
    └─ Iterative refinement (N rounds):
         ├─ [Qwen3-VL] Describe what actually happens in the video
         ├─ [Qwen3-VL] Identify physics mismatches (prompt vs video)
         ├─ [Qwen3-VL] Generate enhanced prompt with physics cues
         └─ [Wan 2.2]   Generate improved video
```

### Why This Matters

| | Original PhyT2V | Ours |
|---|---|---|
| Video model | CogVideoX-5B (open-source, weaker) | **Wan 2.2** (stronger baseline) |
| VLM reasoning | GPT-4 (closed-source, API cost) | **Qwen3-VL-8B** (open, local) |
| Video understanding | Tarsier-34B (34B params) | **Qwen3-VL-8B** (same model, 8B) |
| Total VLM models | 2 | **1** |
| API dependency | Yes (GPT-4) | **No** (fully local) |

## Preliminary Results

### VideoCon-Physics Evaluation (0-1, same metric as PhyT2V paper)

| Model | SA | PC |
|-------|:----:|:----:|
| CogVideoX-5B, Round 1 (paper) | 0.48 | 0.26 |
| CogVideoX-5B, Round 4 + PhyT2V (paper) | 0.59 | 0.42 |
| **Wan2.2-TI2V-5B, Round 1 (ours)** | **0.39** | **0.18** |
| **Wan2.2-TI2V-5B + PhyT2V (ours)** | running | running |

> Note: TI2V-5B baseline is lower than CogVideoX-5B on VideoCon-Physics because TI2V is primarily an image-conditioned model. We expect the PhyT2V feedback loop to provide larger relative gains. Full results on 600 prompts are in progress.

### VIDEOPHY2 AutoEval (1-5 scale)

| Model | Mean SA | Mean PC | Joint | Joint (Hard) |
|-------|:-------:|:-------:|:-----:|:------------:|
| Wan2.2-TI2V-5B (baseline) | 3.18 | 3.61 | 26.3% | 8.3% |
| Wan2.2-TI2V-5B + PhyT2V | pending | pending | pending | pending |

## Expected Contributions

1. **First reproduction of PhyT2V on Wan 2.2** — demonstrating the framework generalizes beyond CogVideoX
2. **Unified VLM pipeline** — single 8B model replaces GPT-4 + Tarsier-34B, eliminating API costs and reducing deployment complexity
3. **VIDEOPHY2 benchmark results** — comprehensive evaluation on 600 physics-critical prompts with SA/PC/Joint metrics
4. **Extensible to Wan 2.2 T2V-A14B** — the MoE 14B model should provide even stronger baselines

## Next Steps

| Step | Status | ETA |
|------|--------|-----|
| Baseline evaluation (VideoCon-Physics) | Done | — |
| PhyT2V Round 2 generation (600 prompts) | Running | ~8h |
| PhyT2V Round 2 evaluation | Pending | +1h |
| Upgrade to T2V-A14B backbone | Planned | +1 day |
| Multi-round iteration (Round 4) | Planned | +2 days |

## Resource Requirements

- 6x 80GB GPUs (1 for Qwen3-VL, 5 for parallel Wan generation)
- Models: Wan2.2-TI2V-5B (local), Qwen3-VL-8B-Instruct (local), VideoCon-Physics evaluator (local)
- No external API calls required
