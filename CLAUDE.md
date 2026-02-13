# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Apply KISS, YAGNI, and SOLID principles.

always firstly `conda activate wan` before run any command.

don't write redundant code in markdown, keep it concise.

Don't delete anything in `docs/`.

## Model Weights

All model weights are stored at `./models/`:

- **`models/Wan2.2-T2V-A14B/`** — T2V 14B (MoE, high/low noise experts, Wan2.1 VAE)
- **`models/Wan2.2-I2V-A14B/`** — I2V 14B (MoE, high/low noise experts, Wan2.1 VAE)
- **`models/Wan2.2-TI2V-5B/`** — TI2V 5B (single model, Wan2.2 VAE)

Each directory contains its own T5 weights (`models_t5_umt5-xxl-enc-bf16.pth`) and VAE checkpoint. Use `--ckpt_dir ./models/Wan2.2-<task>` to specify.


## Architecture


### Configuration System (`wan/configs/`)

Three-layer config structure using `EasyDict`:

1. **`shared_config.py`** — Base template: T5 model (`umt5_xxl`), bfloat16 dtype, 1000 timesteps, negative prompt (Chinese aesthetic guidelines)
2. **Task-specific configs** (`wan_t2v_A14B.py`, `wan_i2v_A14B.py`, `wan_ti2v_5B.py`, etc.) — Model architecture (dim, heads, layers), VAE checkpoint/stride, sampling parameters
3. **`configs/__init__.py`** — Runtime resolution: `WAN_CONFIGS` dict maps task names to config objects, `SUPPORTED_SIZES` defines valid resolutions per task. V2V configs are deep copies of their base configs (v2v-14B = i2v-A14B, v2v-5B = ti2v-5B).

### Model Components (`wan/modules/`)

- **`model.py`** — `WanModel`: Diffusion Transformer (DiT) backbone with RoPE, RMS norm, flash attention, MoE gating. `from_pretrained()` loads checkpoints.
- **`vae2_1.py`** — Wan2.1 VAE (4×8×8 compression), used by A14B models
- **`vae2_2.py`** — Wan2.2 VAE (16×16×4 compression), used by TI2V-5B (more memory efficient)
- **`t5.py`** — T5 text encoder (`umt5-xxl`, 512 token length, bfloat16)
- **`attention.py`** — Flash Attention 2/3 implementations
- **`s2v/`** — Speech-to-video modules (audio encoder, motion control)
- **`animate/`** — Character animation (CLIP, pose detection, retargeting)

### Distributed Inference (`wan/distributed/`)

- **`fsdp.py`** — FSDP model sharding across GPUs (flags: `--dit_fsdp`, `--t5_fsdp`)
- **`sequence_parallel.py`** + **`ulysses.py`** — Ulysses sequence parallelism splitting sequence dim across devices (`--ulysses_size N`)

### Sampling (`wan/utils/`)

- **`fm_solvers.py`** — DPM++ flow matching solver
- **`fm_solvers_unipc.py`** — UniPC solver (select via `--sample_solver`)
- **`prompt_extend.py`** — Prompt enrichment via DashScope API or local Qwen model

### V2V Pipeline Pattern

V2V works by extracting the last frame from input video, using a VLM (InternVL3_5-8B) to generate a description, then feeding both into the I2V/TI2V pipeline. The VLM is loaded on-demand and freed after use.

## Key Constraints

- **Size validation**: Each task only supports specific resolutions (defined in `SUPPORTED_SIZES` in `configs/__init__.py`). A14B tasks use `720*1280`/`1280*720`/`480*832`/`832*480`. TI2V-5B and V2V-5B only support `704*1280`/`1280*704`.
- **VAE compatibility**: A14B models use Wan2.1 VAE (`vae2_1.py`, stride 4×8×8). TI2V-5B uses Wan2.2 VAE (`vae2_2.py`, stride 16×16×4). These are not interchangeable.
- **MoE boundary**: A14B configs define a `boundary` parameter (e.g., 0.875) for switching between high-noise and low-noise expert models during denoising.
- **A14B high/low experts (how it works)**:
  - A14B uses two checkpoints: `high_noise_model` and `low_noise_model` (see `wan_t2v_A14B.py` / `wan_i2v_A14B.py`).
  - During sampling, model selection is timestep-based: use `high_noise_model` when `t >= boundary * num_train_timesteps`, otherwise use `low_noise_model`.
  - Intuition: `high_noise_model` handles early high-noise denoising (global structure/motion), `low_noise_model` handles late low-noise denoising (detail refinement).
  - Default boundaries: `t2v-A14B=0.875`, `i2v-A14B=0.900` (with `num_train_timesteps=1000`, threshold is ~875 / ~900).
  - `guide_scale` can be a tuple `(low, high)`; T2V defaults to `(3.0, 4.0)`, I2V defaults to `(3.5, 3.5)`.
- **Memory flags**: `--offload_model` (auto-set True for single GPU), `--t5_cpu` (keep T5 on CPU), `--convert_model_dtype` (fp16 conversion for lower VRAM).
- **Python >=3.10, numpy <2** required. Flash Attention (`flash_attn`) is required for attention operations.
