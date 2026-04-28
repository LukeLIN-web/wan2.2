# T3 Design — latent pre-encoding under a uniform `video_preprocessing` recipe

**Status: design only. No encode runs until rl2 + luke1 sign off.**

This document covers the seven acceptance points that rl2 specified at 13:28 (msg `ca19f80b`) and the two overrides luke1/rl2 added at 13:30 (msgs `0a7dbc1c`, `6ecb7452`):
1. recipe_id is a first-class artifact (axes explicit, hashed, asserted at load)
2. cross-producing-model uniformity invariant verified empirically
3. recipe values reverse-derived from round2 ckpt's expected input
4. pre-encode subset = explicit group_id list (heldout NEVER encoded → anti-leakage)
5. VAE source cited concretely (path + sha256 + class + constructor args)
6. per-video at-load `recipe_id` assert
7. encode→decode PSNR/SSIM sanity on N samples

---

## 1. Recipe id

The recipe is a frozen, hashable spec. Every axis is explicit; "default" is not allowed. The recipe lives at `humanize/dpo_v0/recipes/wan22_i2v_a14b__round2_v0.yaml` (planned) and is hashed into `recipe_id = sha256(canonical_yaml).hex[:16]`.

| axis | value | rationale (sourced from where) |
|---|---|---|
| `fps` | `16` | `wan/configs/shared_config.py:wan_shared_cfg.sample_fps = 16` (used by I2V-A14B inference). |
| `frame_count_policy` | `first_n` | Plan T3 enumerates `first_n / uniform_sample / stride`. I2V starts from a single conditioning image at t=0, so the *initial frames* are the model's first prediction surface — `first_n` is the only correct policy. |
| `frame_num` | `81` | `wan/configs/shared_config.py:wan_shared_cfg.frame_num = 81`. Constraint `4n+1` (per `generate.py:138`). 81 frames at 16 fps = 5.0625 s. The Wan2.1 VAE's temporal stride (4) compresses 81 → 21 latent frames. |
| `frame_stride` | `1` | When `frame_count_policy = first_n`, take consecutive frames at native rate; only used as a tiebreaker if a video is shorter than `frame_num` (then we explicitly fail per the at-load assert below). |
| `target_resolution` | `480x832` (W×H = 832×480) | I2V-A14B `SUPPORTED_SIZES = ('720*1280', '1280*720', '480*832', '832*480')`. The 480x832 / 832x480 tier is the smallest supported; matches Wan's training data at the lowest VRAM footprint suitable for A100 80GB single-node short DPO with `--ref_offload`. Aspect detected per video; landscape→`832×480`, portrait→`480×832`. |
| `resize_mode` | `letterbox_pad` | Plan enumerates `resize / center_crop / letterbox_pad`. `resize` distorts aspect; `center_crop` may hide the action; `letterbox_pad` keeps full motion intact (critical for "physics" judging). Pad color set below. |
| `pad_color` | `(0, 0, 0)` (black) | Black is the convention used by Wan inference's `interpolate(..., mode='bicubic')` + `torch.zeros(...)` masking pattern in `wan/image2video.py:319-326`. |
| `color_space` | `bt709-tv-range` | All videos in `wmbench/data/videos/<dataset>/<filename>` are H.264 mp4s with rec.709 TV-range YUV. ffmpeg defaults to TV range; we pin it explicitly. |
| `decoder` | `ffmpeg-4.4.2 + imageio-2.37.3` | binary versions captured in manifest `meta.decoder_versions`. We use `imageio-ffmpeg` to ensure consistent decode across machines. |
| `codec_normalization` | `yuv420p_to_rgb_bt709_full` | After ffmpeg YUV→RGB (BT.709), values clamped to [0, 255], then mapped to `[-1, 1]` float per Wan VAE convention (decoder output is `clamp_(-1, 1)`, so the encode side mirrors). |
| `first_frame_is_conditioning` | `true` | Per Wan I2V semantics (`wan/image2video.py:323`): frame[0] is the conditioning image; the temporal mask spans the first 4 latent positions to that single frame. We will **not** re-encode the conditioning image from the video — we use the canonical image we resolved in T2 (the input the models were given). |

The hash is computed over a canonical YAML serialization (sorted keys, no comments, fixed quoting). Any change to any axis changes `recipe_id`.

A pretty-printed example:

```yaml
fps: 16
frame_count_policy: first_n
frame_num: 81
frame_stride: 1
target_resolution: { aspect_ratio_router: { landscape: '832x480', portrait: '480x832' } }
resize_mode: letterbox_pad
pad_color: [0, 0, 0]
color_space: bt709-tv-range
decoder: { ffmpeg: '4.4.2-0ubuntu0.22.04.1', imageio: '2.37.3', imageio_ffmpeg: '0.6.0' }
codec_normalization: yuv420p_to_rgb_bt709_full
first_frame_is_conditioning: true
```

`recipe_id` (placeholder until the YAML is finalized): `<sha256[:16] of canonical bytes>`.

## 2. Cross-producing-model uniformity invariant

Plan: *"a single named `video_preprocessing` recipe is applied uniformly to every retained video regardless of producing model"*.

We verify with a script `verify_recipe_uniformity.py` (separate file, planned but not committed yet) that runs **before** any latent is consumed by training:

1. Sample 5 random `scene_filename`s from the **train+val** preprocess subset.
2. For each scene, locate its 8 dataset variants under `wmbench/data/videos/<dataset>/<scene>.mp4` (the scene appears in 1–3 of these in any single group, but the underlying file exists for all 8 datasets per T0 Q2).
3. Apply the recipe end-to-end (decode → resize/letterbox → normalize → VAE encode) to each variant.
4. Compare per-(scene, dataset) latent statistics:
   - shape: must be **identical** (any mismatch is a bug — assert).
   - dtype: must be `bfloat16` (the dtype we serialize at; assert).
   - mean / std / min / max per latent channel (16): must agree to within (mean Δ < 0.05 \* global std, std ratio ∈ [0.7, 1.3]).
5. Report any (scene, dataset) outlier where the statistical agreement breaks. Failure is a halt: do not proceed to T4 with non-uniform preprocessing.

The shape/dtype assert catches systematic crop-vs-pad disagreement; the moment-level checks catch silent codec-normalization bugs (e.g., one dataset's videos being TV-range vs full-range).

## 3. Round2 ckpt input alignment (rl2 acceptance #3)

The round2 ckpt is `/shared/user63/workspace/data/Wan/finetuned/Wan2.2-I2V-A14B_high_noise_full_round2/epoch-0.safetensors` (28.6 GB). It has no sidecar config. Input expectations are reverse-derived from:

| field | source (file:line) | value |
|---|---|---|
| z_dim (VAE latent channels) | `wan/modules/vae2_1.py:622` (`Wan2_1_VAE(z_dim=16, ...)`) | **16** |
| vae stride (T, H, W) | `wan/configs/wan_i2v_A14B.py:14` (`vae_stride = (4, 8, 8)`) | **(4, 8, 8)** |
| patch_size (DiT) | `wan/configs/wan_i2v_A14B.py:18` (`patch_size = (1, 2, 2)`) | **(1, 2, 2)** |
| transformer dim | `wan/configs/wan_i2v_A14B.py:19` (`dim = 5120`) | 5120 |
| supported pixel sizes | `wan/configs/__init__.py:54` (`'i2v-A14B': ('720*1280', '1280*720', '480*832', '832*480')`) | 4 sizes |
| frame_num | `wan/configs/shared_config.py` (`frame_num = 81`) | **81** (must satisfy 4n+1) |
| sample_fps | `wan/configs/shared_config.py` (`sample_fps = 16`) | **16** |
| conditioning image format | `wan/image2video.py:319-326` | first frame interpolated bicubic to `(h, w)`, concatenated with `zeros(3, F-1, h, w)`, fed to `vae.encode(...)`; mask `[4, 21, h, w]` indicates first-frame coverage |
| conditioning latent shape | `wan/image2video.py:281-287` | `[16, (F-1)/vae_stride[0]+1, h//vae_stride[1]//patch_size[1]*patch_size[1], w//vae_stride[2]//patch_size[2]*patch_size[2]]` = `[16, 21, 60, 104]` for 480x832 |
| video latent shape (DPO target) | derived from VAE.encode of the full 81-frame video at 480x832 | `[16, 21, 60, 104]`, dtype bf16 |

The round2 ckpt is the **fine-tuned** high-noise expert; per the I2V-A14B MoE structure (`wan/configs/wan_i2v_A14B.py:21-22`: `low_noise_checkpoint='low_noise_model'`, `high_noise_checkpoint='high_noise_model'`), the round2 single-file `epoch-0.safetensors` replaces the `high_noise_model/` weights. The **DiT input shape contract is unchanged** by fine-tuning, so the recipe values above are valid.

Sanity load step planned (separate from the encode runner): `safetensors.torch.load_file(epoch-0.safetensors)` and grep keys for `patch_embedding.weight` (or analogous). The first-layer weight shape encodes `(out_dim, in_channels, t_patch, h_patch, w_patch)` from which we re-derive `(z_dim, patch_size)` = `(16, (1,2,2))` and crash if it disagrees with the config above. **This sanity step must pass before encode begins.** It does not modify the ckpt.

## 4. Pre-encode subset (anti-leakage)

**Heldout split is NEVER pre-encoded.** Heldout is reserved for plan Step 6 (regenerate with both round2 baseline and trained ckpt under shared `generation_config`); pre-encoding heldout videos would leak the human-judged target.

Three tiers:

### Tier A — Tiny-overfit subset (T4)
- **Size**: 16 pairs (top of the plan's "8–16" range; gives more diversity without going over the gate threshold).
- **Selection rule**: from `splits/train.json`, deterministic seed `0xdpo-t4`. Stratified by `loser.dataset` to guarantee:
  - 8 pairs where `loser = wan2.2-i2v-a14b` (the DPO improvement signal — 124 available in train).
  - 8 pairs where `winner = wan2.2-i2v-a14b` (Wan-as-winner — 441 available in train; we pick across diverse loser models).
- **Distinct videos**: at most 32 (16 winner videos + 16 loser videos; some may overlap across pairs but at most 32 unique `(video_id, dataset)` pairs).
- The exact 16 pair_ids are dumped to `humanize/dpo_v0/T3_subset.json:tier_a` for rl2 review.

### Tier B — Short DPO subset (T5)
- **Size**: all 2745 train+val pairs from T2's `post_t2_pair.json`.
- **Distinct videos to encode**: 1505 unique `(video_id, dataset)` tuples (computed: 1240 train + 265 val, but with cross-set video reuse; actual unique = 1505).
- **Distinct conditioning images**: 208 (171 train + 37 val scene_filenames; non-overlapping by T1 invariant).
- **Why include val**: val is used for in-loop monitoring during DPO; pre-encoding it removes per-step decode cost. Val pairs share zero prompts with heldout (T1-asserted), so no leakage.
- The 2745 pair_ids and the 1505 (video_id, dataset) tuples and 208 scene_filenames are dumped to `humanize/dpo_v0/T3_subset.json:tier_b` for rl2 review.

### NEVER-encode — Heldout (anti-leakage proof)
- 579 pairs / 309 unique videos / 42 scene_filenames.
- We will dump the heldout group_id and scene lists to `humanize/dpo_v0/T3_subset.json:heldout_excluded` so rl2 can grep that NONE of these appear in tier A or tier B.

The encode runner (planned: `humanize/dpo_v0/encode_latents.py`) will refuse to encode any video whose scene appears in the heldout exclusion list — a runtime assert, not just documentation.

## 5. VAE source (rl2 acceptance #5)

Per luke1's 13:30 directive ("Wan2.2 自带的 vae"), we use **the VAE that ships in the Wan2.2-I2V-A14B model release directory**. For this model, that is the Wan2.1 VAE checkpoint bundled with the model — confirmed by:
- `wan/configs/wan_i2v_A14B.py:14` (`vae_checkpoint = 'Wan2.1_VAE.pth'`)
- `wan/image2video.py:99` (instantiates `Wan2_1_VAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint))`)
- File present at `models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth`.

The Wan2.2 release ships **two VAEs** — `Wan2.1_VAE.pth` (4×8×8 stride, used by the A14B family) and `Wan2.2_VAE.pth` (4×16×16 stride, used by TI2V-5B). The two are NOT interchangeable: feeding 4×16×16 latents to the I2V-A14B DiT would produce wrong-shape activations on the very first patch embedding layer. We therefore interpret luke1's "Wan2.2 自带的 vae" as **the bundled VAE for I2V-A14B**, i.e., `Wan2.1_VAE.pth`. **If luke1 actually intended `Wan2.2_VAE.pth` (4×16×16), the entire pipeline including the round2 ckpt would have to change — please flag explicitly before encode.**

The VAE constructor in our encode runner:

```python
from wan.modules.vae2_1 import Wan2_1_VAE

vae = Wan2_1_VAE(
    z_dim=16,
    vae_pth='/shared/user60/worldmodel/rlvideo/videodpoWan/models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth',
    dtype=torch.bfloat16,
    device='cuda',
)
```

Manifest fields (per rl2 #5: "cite 具体文件路径 + git sha + class/构造参数"):

```json
{
  "vae": {
    "class": "wan.modules.vae2_1.Wan2_1_VAE",
    "checkpoint_path": "/shared/user60/worldmodel/rlvideo/videodpoWan/models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth",
    "checkpoint_sha256": "<sha256 stamped at encode time>",
    "constructor_args": { "z_dim": 16, "dtype": "bfloat16", "device": "cuda" },
    "wan_repo_git_sha": "<git rev-parse HEAD of videodpoWan worktree at encode time>",
    "vae_module_path": "wan/modules/vae2_1.py",
    "vae_module_sha256": "<sha256 of the file at encode time>",
    "stride": [4, 8, 8],
    "z_dim": 16,
    "scale_mean": [<16 floats from vae2_1.py:632-635>],
    "scale_std": [<16 floats from vae2_1.py:636-639>]
  }
}
```

No silent hub fetches anywhere. The runner will refuse to start if the checkpoint sha256 changes between runs (cache-pinned).

## 6. Per-video at-load verification (rl2 acceptance #6)

Each saved latent is a `.safetensors` file with a sidecar `.json` that records:

- `recipe_id` (the 16-char hash from §1)
- `scene_filename`
- `dataset` (producing model)
- `video_id`, `pair_role` ("winner" or "loser")
- `vae.checkpoint_sha256`
- `latent.shape`, `latent.dtype` (bf16)
- `encode_timestamp`, `encoder_git_sha`

The DPO dataset's `__getitem__` will:

1. open the sidecar
2. assert `sidecar.recipe_id == config.training.recipe_id` (raise `RuntimeError` on mismatch — no silent re-encode fallback)
3. assert `sidecar.vae.checkpoint_sha256 == config.training.vae.checkpoint_sha256`
4. assert `sidecar.latent.shape == config.training.latent.expected_shape`

This is the same per-load contract the plan calls "verified per-video at dataset load" (line 19). Failure modes are loud, not silent.

## 7. Encode→decode sanity (rl2 acceptance #7)

After encoding the tier A subset (32 videos), run a one-shot sanity:

1. Pick 5 random latents from tier A (deterministic seed `0xdpo-t3-sanity`).
2. `vae.decode(latent)` → reconstruction (frame range `[-1, 1]`).
3. Re-decode the same source video with the same recipe → ground truth video tensor.
4. Compute per-frame **PSNR(reconstruction vs source)** averaged over frames.
5. Compute per-frame **SSIM(reconstruction vs source)** averaged over frames.
6. **PASS criteria**: PSNR ≥ 28 dB, SSIM ≥ 0.85 — values typical of a continuous-latent VAE on natural video. **FAIL** if any sample drops below; FAIL halts T3 (do not proceed to T4 with a broken VAE).

Outputs: `humanize/dpo_v0/out/<run-timestamp>/t3/sanity.md` with the 5-sample table and side-by-side PNG (frame 0 + frame 40) per sample. SSIM uses color SSIM (per-channel then averaged), not the grayscale shortcut from T2 — at PSNR-tight thresholds, color precision matters per rl2's nudge.

## 8. Outputs and layout

Planned encode outputs (all under `humanize/dpo_v0/out/<run-timestamp>/t3/`):

```
recipe.yaml                             # frozen recipe with recipe_id
recipe.recipe_id.txt                    # plaintext id (for shell asserts)
manifest.json                           # master index: subset, vae, recipe, encode summary
sanity.md                               # encode→decode PSNR/SSIM table
sanity/<scene_id>__<dataset>.png        # side-by-side reconstruction sample (5 files)
uniformity.json                         # cross-model statistical agreement table (5 scenes × 8 datasets)
videos/<scene_id>__<dataset>.safetensors   # video latent
videos/<scene_id>__<dataset>.json          # sidecar
conditioning/<scene_id>.safetensors     # encoded conditioning image (1 per scene_id, shared across datasets)
conditioning/<scene_id>.json
encode.log                              # timestamped log per CLAUDE.md convention
```

The script `humanize/dpo_v0/encode_latents.py` is the encoder. The script `humanize/dpo_v0/verify_recipe_uniformity.py` runs the §2 invariant. Both are added in the implementation pass after rl2 reviews this design.

## 9. CRITICAL — pipeline mismatch surfaced by reading the round2 training script

After luke1's pointer to `/shared/user72/workspace/DiffSynth-Studio/myscipts/inference_I2V_cont*.py`, I read both the inference scripts and the training shell `train_video_start_round2.sh`. **The round2 ckpt is not a vanilla Wan2.2-I2V-A14B fine-tune — it's a V2V-finetune of the high-noise expert under DiffSynth-Studio.** Concretely:

```
# from train_video_start_round2.sh
examples/wanvideo/model_training/train_V2V.py
  --height 480 --width 832
  --num_frames 49                     # NOT 81
  --extra_inputs "video_start"        # video conditioning, not single image
  --max_timestep_boundary 0.358 --min_timestep_boundary 0
  --learning_rate 1e-6
  --remove_prefix_in_ckpt "pipe.dit."
  --output_path .../Wan2.2-I2V-A14B_high_noise_full_round2
```

```python
# from inference_I2V_cont_after_train.py
CONDITION_FRAME = 17
video_start = VideoData(INPUT_VIDEO, height=480, width=832)
video = pipe(
    ...,
    video_start=video_start, video_start_frame=17,
    num_frames=81,
)
video = video[CONDITION_FRAME:]   # slice off conditioning prefix
```

So the round2 ckpt expects:
- **resolution**: 480×832 (training and inference agree) ✓ matches my §1 choice
- **conditioning**: a multi-frame **video clip** (training used `extra_inputs="video_start"`; inference uses `video_start_frame=17`)
- **frames trained**: 49; **frames at inference**: 81 (with first 17 as conditioning, model produces 64 new frames)
- **fps**: ~15 in inference's `save_video(..., fps=15)` (slight mismatch with `wan_shared_cfg.sample_fps=16`; will lock to 16 unless rl2/luke1 says otherwise)
- **training framework**: DiffSynth-Studio's `train_V2V.py`, NOT this repo's `wan/` package

Conversely, the human-eval data was produced by 8 different I2V models conditioned on a **single first-frame image**. The first frames of those 81-frame outputs are essentially identical to the canonical conditioning image (T2 SSIM 0.997+).

This is a real data/model mismatch with two routes forward, **and I need explicit guidance before encoding**:

**Route A — V2V continuation DPO (matches round2's native conditioning)**
- For each retained pair (winner, loser): use the first 17 frames of each video as `video_start`, the remaining 64 frames as the DPO target.
- Practical concern: the first 17 frames of an I2V output are *near-static* (they all came from a single conditioning image). Using them as `video_start` is degenerate — the model would essentially see "17 copies of the conditioning image" as the video prefix.
- Pre-encode subset shape: latent `[16, T_lat, 60, 104]` where T_lat = (49-1)/4+1 = 13 for the 49-frame training window; conditioning latent for the 17-frame prefix similarly.
- Risk: the 17-frame prefix being near-static may not match the openvid clips round2 was trained on (real motion); DPO may push the model in a weird direction.

**Route B — I2V single-frame DPO (matches the human-eval data's conditioning convention)**
- Use the canonical conditioning image (T2-resolved, 250 unique images) as the single conditioning frame for round2.
- Round2 was NOT trained to do single-frame conditioning, but DiffSynth's `extra_inputs="video_start"` mechanism may degrade to single-frame if `video_start_frame=1`. (Untested.)
- Pre-encode subset: full 81-frame latent for each video. The "video_start" is just the canonical first frame, expanded to a 1-frame video_start.
- Risk: round2 may produce poor single-frame I2V output; the DPO ref becomes weak.

**Route C — Use the upstream Wan2.2-I2V-A14B base (not round2)**
- Single-frame I2V is the base model's native conditioning convention (`wan/image2video.py`).
- DPO improves the base on human-eval pairs.
- Loses round2's openvid prior but matches data cleanly.

I cannot pick this without confirmation. **Please pick a route — A / B / C — and confirm.** My default if forced: Route A with `video_start_frame=17` and `num_frames=49` (matching round2's training contract verbatim, accepting the near-static prefix risk).

## 10. Open questions / sign-off requested (other)

1. **VAE confirmation**: `Wan2.1_VAE.pth` bundled at `models/Wan2.2-I2V-A14B/Wan2.1_VAE.pth` is what `inference_I2V_cont_after_train.py:31` cites — confirms my interpretation of "Wan2.2 自带的 vae".
2. **Tier A pair selection seed**: I used `0xdpo-t4` to pick 16 pairs (8 wan-as-loser + 8 wan-as-winner with diverse opponents). The actual list is in `T3_subset.json:tier_a.pair_ids`; reroll if rl2 wants different stratification.
3. **Tier B inclusion of val**: I propose pre-encoding train+val (2745 pairs); rl2 to confirm or restrict to train-only (2286 pairs).
4. **Tied to §9**: if Route A is chosen, the recipe's `frame_num` becomes 49 (training contract) instead of 81; `frame_count_policy` becomes "first 49 frames" with `video_start = first_n(17)`. Recipe shape changes accordingly.
5. **Training framework choice**: round2 was trained with DiffSynth-Studio's `train_V2V.py`. The DPO trainer for v0 most likely lives in DiffSynth too (modifying `train_V2V.py` to add a DPO loss head with frozen ref). The `wan/` package in *this* repo is the upstream Alibaba inference code; it has no training code. The plan implies the DPO trainer goes here in `videodpoWan` — but the training-stack of record is DiffSynth-Studio. **Please confirm where the DPO trainer should live.** This affects T4–T5 commit locations.

Once §9 + §10 are answered I'll commit `recipe.yaml`, the encoder, the uniformity verifier, and run them — posting the encode manifest + sanity table back here for review.
