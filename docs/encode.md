# Multi-Image Latent Frame Replacement


## Goal

Input N images, replace the first N latent frames in the I2V pipeline, anchoring the beginning of the generated video.

## Implementation Plan

### Status

| # | File | Status | Description |
|---|------|--------|-------------|
| 1 | `wan/textimage2video_mul.py` | DONE | Pipeline class `WanTI2V_Mul`, core method `mi2v` |
| 2 | `wan/configs/__init__.py` | TODO | Add `mi2v-5B` task to `WAN_CONFIGS` / `SUPPORTED_SIZES` |
| 3 | `generate.py` | TODO | Add `elif "mi2v" in args.task` branch, load `--images` arg |
| 4 | `mulI2video.py` | TODO | Standalone entry script for multi-image to video |

### Step 1: `wan/textimage2video_mul.py` (DONE)

Already created. Class `WanTI2V_Mul` with method `mi2v`:
- Input: `imgs` (list of PIL Images)
- Encode each image independently -> concat along temporal dim -> `z_combined` shape `[z_dim, N, H_lat, W_lat]`
- Mask first N latent frames to zero, inject image latents
- Re-anchor after every denoise step

### Step 2: `wan/configs/__init__.py`

Add new task entry so `generate.py` can recognize it:

```python
mi2v_5B = copy.deepcopy(ti2v_5B)

WAN_CONFIGS['mi2v-5B'] = mi2v_5B

SUPPORTED_SIZES['mi2v-5B'] = ('704*1280', '1280*704')
```

Reuses `ti2v_5B` config (Wan2.2 VAE, stride 4x16x16, z_dim=48).

### Step 3: `generate.py`

Two changes:

#### 3a. Add `--images` argument (argparse)

```python
parser.add_argument(
    "--images",
    type=str,
    nargs='+',
    default=None,
    help="Paths to multiple reference images for mi2v task.")
```

#### 3b. Add `mi2v` branch in `generate()` function

Insert before the `else` block (after the `v2v` branch, around line 584):

```python
elif "mi2v" in args.task:
    logging.info("Creating WanTI2V_Mul pipeline.")
    from wan.textimage2video_mul import WanTI2V_Mul
    imgs = [Image.open(p).convert("RGB") for p in args.images]
    wan_mi2v = WanTI2V_Mul(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    logging.info(f"Generating video from {len(imgs)} images ...")
    video = wan_mi2v.generate(
        input_prompt=args.prompt,
        imgs=imgs,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model)
```

Also add validation in `_validate_args`:

```python
if args.task == "mi2v-5B":
    assert args.images is not None and len(args.images) > 0, \
        "Please specify image paths via --images for mi2v task."
```

### Step 4: `mulI2video.py`

Standalone entry script, independent of `generate.py`. Minimal argparse, directly imports `WanTI2V_Mul`:

```python
"""
Usage:
    python mulI2video.py \
        --ckpt_dir ./Wan2.2-TI2V-5B \
        --images frame0.png frame1.png frame2.png frame3.png \
        --prompt "a cat walking in the garden" \
        --size 704*1280 \
        --frame_num 81
"""
```

Arguments:
- `--ckpt_dir`: checkpoint directory (required)
- `--images`: list of image paths (required, nargs='+')
- `--prompt`: text prompt (required)
- `--size`: output resolution, default `704*1280`
- `--frame_num`: number of output frames, default 81
- `--sample_steps`: diffusion steps, default 40
- `--sample_shift`: noise schedule shift, default 5.0
- `--sample_solver`: solver, default `unipc`
- `--sample_guide_scale`: CFG scale, default 5.0
- `--seed`: random seed, default -1
- `--offload_model`: offload to CPU, default True
- `--save_file`: output path, default auto-generated

Flow:
1. Parse args
2. Load config: `from wan.configs import WAN_CONFIGS` -> `cfg = WAN_CONFIGS['ti2v-5B']`
3. Load images: `imgs = [Image.open(p).convert("RGB") for p in args.images]`
4. Create pipeline: `WanTI2V_Mul(config=cfg, checkpoint_dir=args.ckpt_dir, ...)`
5. Generate: `pipeline.generate(input_prompt=args.prompt, imgs=imgs, ...)`
6. Save: `save_video(tensor=video[None], ...)`

---

## Original I2V Mechanism

Pipeline: `wan/textimage2video.py` (class `WanTI2V`, method `i2v`)

### 1. Single Image Encoding (line 477, 512)

```python
img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)  # [3, 1, H, W]
z = self.vae.encode([img])  # z[0] -> [z_dim, 1, H_latent, W_latent]
```

### 2. Noise Construction (line 488-494)

```python
noise = torch.randn(
    self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
    oh // self.vae_stride[1], ow // self.vae_stride[2], ...)
```

Shape: `[z_dim, T_latent, H_latent, W_latent]`

Example (TI2V-5B, 81 frames, 704x1280):
- z_dim=48, T_latent=21, H_latent=44, W_latent=80
- noise shape: `[48, 21, 44, 80]`

### 3. Mask Creation (`wan/utils/utils.py:172-199`)

```python
mask1, mask2 = masks_like([noise], zero=True)
```

`zero=True` sets `mask[:, 0] = 0`, rest = 1.

### 4. Injection (line 551, repeated at line 598 every denoise step)

```python
latent = (1. - mask2[0]) * z[0] + mask2[0] * latent
```

- mask=0 at t=0 -> latent = image latent (clean)
- mask=1 at t>0 -> latent = noise / model prediction

### 5. Timestep Encoding (line 573)

```python
temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
```

mask=0 frames get timestep=0, telling the model these frames are already clean.

## Temporal Compression

VAE temporal stride = 4 for both Wan2.1 and Wan2.2:

```
T_latent = (T_pixel - 1) // stride_t + 1
```

| Latent Frame | Pixel Frames |
|-------------|-------------|
| 0           | 0-3         |
| 1           | 4-7         |
| 2           | 8-11        |
| 3           | 12-15       |

Replacing 4 latent frames anchors the first ~16 pixel frames of the video.

## `wan/textimage2video_mul.py` Details

New file, class `WanTI2V_Mul`, core method `mi2v`.

Compared to original `i2v` in `wan/textimage2video.py`:

| | Original `i2v` | New `mi2v` |
|---|---|---|
| Input | `img`: single PIL Image | `imgs`: list of PIL Images |
| Encoding | 1 image -> `z[0]` shape `[z_dim, 1, H, W]` | N images independently encoded -> `z_combined` shape `[z_dim, N, H, W]` |
| Mask | `masks_like([noise], zero=True)`, dual mask (mask1, mask2), only `t=0` zeroed | Single `torch.ones_like(noise)`, first N frames zeroed |
| Injection | `(1 - mask2[0]) * z[0] + mask2[0] * latent` | `mask * latent + (1 - mask) * z_combined` |
| Validation | None | `assert N <= T_latent` |

### Key Code Sections

#### 1. Preprocess All Images to Same Resolution

All images are resized and center-cropped to `(ow, oh)` determined by the first image:

```python
img_tensors = []
for img in imgs:
    scale = max(ow / img.width, oh / img.height)
    img_resized = img.resize(
        (round(img.width * scale), round(img.height * scale)),
        Image.LANCZOS)
    x1 = (img_resized.width - ow) // 2
    y1 = (img_resized.height - oh) // 2
    img_cropped = img_resized.crop((x1, y1, x1 + ow, y1 + oh))
    t = TF.to_tensor(img_cropped).sub_(0.5).div_(0.5).to(
        self.device).unsqueeze(1)  # [3, 1, H, W]
    img_tensors.append(t)
```

#### 2. Encode Each Image Independently, Concatenate

```python
z_list = []
for t in img_tensors:
    z_i = self.vae.encode([t])  # z_i[0]: [z_dim, 1, H_lat, W_lat]
    z_list.append(z_i[0])
z_combined = torch.cat(z_list, dim=1)  # [z_dim, N, H_lat, W_lat]
```

#### 3. Build Mask: Zero Out First N Frames

Simplified from dual mask (`mask1`, `mask2`) to a single mask tensor:

```python
mask = torch.ones_like(noise)   # [z_dim, T_latent, H_lat, W_lat]
mask[:, :N] = 0.0
```

#### 4. Injection and Re-anchoring

Initial injection before diffusion loop:

```python
latent = mask * noise + (1.0 - mask) * z_combined
```

Re-anchoring after every denoise step:

```python
latent = mask * latent + (1.0 - mask) * z_combined
```

#### 5. Timestep Encoding (Unchanged Logic)

```python
temp_ts = (mask[0][:, ::2, ::2] * timestep).flatten()
```

Uses `mask[0]` (first channel slice, shape `[T_latent, H_lat, W_lat]`) instead of `mask2[0][0]`. Frames where mask=0 get timestep=0, telling the model those frames are clean.

## Design Considerations

### Independent Encoding vs Joint Encoding

| Approach | Pros | Cons |
|----------|------|------|
| Independent encode + concat | Simple, works for arbitrary images | VAE temporal encoder sees no inter-frame relation |
| Stack as `[3, N, H, W]` and encode together | VAE captures temporal coherence | With stride_t=4, need 13 pixel frames to produce 4 latent frames; only works for consecutive video frames |

**Recommendation:** Use independent encoding. It works for any N images (keyframes, different scenes, etc.) and the diffusion model handles temporal coherence during generation.

### Quality Impact

- More anchored frames = stronger constraint on the model
- If the N images have inconsistent motion or style, the model may struggle to produce natural intermediate frames
- Best results when images represent a smooth progression (e.g., keyframes from a planned sequence)

### Memory

- N VAE encode calls instead of 1
- Latent tensor size unchanged (same noise shape)
- No significant VRAM increase since VAE encodes are sequential and small
