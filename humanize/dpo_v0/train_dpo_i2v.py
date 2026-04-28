"""Wan2.2-I2V-A14B Direct I2V DPO trainer (v0, round 2 — codex review fixes applied).

Single-file trainer. Loads the original sharded high-noise expert as both
policy (with LoRA on attention q/k/v/o + ffn.0/ffn.2) and frozen reference
(no_grad, optionally CPU-offloaded). Reads pre-encoded tier_a/tier_b
winner+loser latents from the encoder's output dir (recipe_id pinned at
6bef6e104cdd3442), pre-encodes T2 conditioning images via the I2V VAE
once at startup (cached by image_path), encodes prompts via T5 once
(cached by prompt), samples per-pair shared (t, eps) deterministically,
runs the flow-matching DPO loss across 4 forward passes per step under
``torch.amp.autocast("cuda", dtype=bfloat16)`` and applies the AC-5
routing-counter contract on every forward.

Round 2 fixes (Codex review msg 70939ec1, all P0+P1+P3 applied):
  P0 #1 — y conditioning built from cond image VAE latent + 4-channel mask;
          y.shape == [20, F_latent, H_latent, W_latent] passed to all forwards.
  P0 #2 — bf16 autocast wraps every forward; LoRA dtype-safe via
          ``x.to(A.dtype)`` inside the patched forward.
  P0 #3 — Reference defaults to CPU + offload; T5 freed after encoding.
  P1 #4 — LoRA targets matched against Wan's q/k/v/o naming + ffn.{0,2}.
  P1 #5 — LoRA save with module-aligned keys (`<module>.lora_A` / `.lora_B`)
          + metadata (`rank`, `alpha`, `target_modules`).
  P3 #7 — Sampling band restricted to [901, 999] (Wan's grid max is 999).

Multi-GPU launch via torchrun (DistributedSampler over the pair list);
DS Zero-2 wrap deferred to M4 (P2 #6 partial fix tonight).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import gc
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import time
from contextlib import nullcontext


def _mem(stage, device="cuda:0"):
    import torch as _t
    try:
        a = _t.cuda.memory_allocated(device) / 1024**3
        r = _t.cuda.memory_reserved(device) / 1024**3
        m = _t.cuda.max_memory_allocated(device) / 1024**3
        print(f"[mem] {stage}: alloc={a:.2f}GB reserved={r:.2f}GB max_alloc={m:.2f}GB", flush=True)
    except Exception:
        pass

import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # videodpoWan
from dpo_loss import flow_matching_dpo_loss  # noqa: E402

# Recipe pin (must match recipes/recipe_id)
EXPECTED_RECIPE_ID = "6bef6e104cdd3442"
RECIPES_DIR = HERE / "recipes"

# AC-5.U2 raw boundary (switch_DiT_boundary * 1000)
SWITCH_DIT_BOUNDARY_RAW = 900
NUM_TRAIN_TIMESTEPS = 1000
SAMPLING_T_LOW = 901
SAMPLING_T_HIGH = 999  # P3 #7: Wan's grid max is 999, not 1000

# Wan I2V conditioning shapes for tier_a / tier_b at 832x480:
# z latent: [16, 21, 60, 104]; mask: [4, 21, 60, 104]; y = cat(mask, z_cond) → [20, ...]
LATENT_C = 16
MASK_C = 4
Y_C = LATENT_C + MASK_C  # 20

LORA_TARGET_RE = re.compile(r"\.(self_attn|cross_attn)\.(q|k|v|o)$|\.ffn\.(0|2)$")


# ---------- pin / determinism helpers ----------


def _file_sha256(path: pathlib.Path, buf: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_recipe_pin(recipes_dir: pathlib.Path, expected: str = EXPECTED_RECIPE_ID) -> str:
    yaml_bytes = (recipes_dir / "wan22_i2v_a14b__round2_v0.yaml").read_bytes()
    fresh = hashlib.sha256(yaml_bytes).hexdigest()[:16]
    on_disk = (recipes_dir / "recipe_id").read_text(encoding="ascii").strip()
    assert fresh == on_disk == expected, (
        f"recipe pin drift: fresh={fresh}, on_disk={on_disk}, expected={expected}"
    )
    return on_disk


def per_pair_seed(pair_id: str, namespace: str = "dpo-tier_a") -> int:
    h = hashlib.sha256(f"{namespace}:{pair_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def sample_per_pair_t_eps(
    pair_id: str,
    latent_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    namespace: str = "dpo-tier_a",
) -> tuple[int, torch.Tensor]:
    """Sample (t_raw, eps) shared per pair (AC-5.U1/U3).

    t_raw is in [SAMPLING_T_LOW, SAMPLING_T_HIGH] (high-noise sub-band so
    AC-5.U3 routing counter stays at 100% high-noise). eps has shape
    ``latent_shape`` and is generated from a per-pair-deterministic
    CPU generator, then cast to ``dtype`` and moved to ``device``.
    """
    g = torch.Generator(device="cpu").manual_seed(per_pair_seed(pair_id, namespace))
    t_raw = int(torch.randint(low=SAMPLING_T_LOW, high=SAMPLING_T_HIGH + 1, size=(1,), generator=g).item())
    eps_cpu = torch.randn(latent_shape, generator=g, dtype=torch.float32)
    eps = eps_cpu.to(device=device, dtype=dtype)
    return t_raw, eps


def detected_expert_from_raw(raw_timestep: int) -> str:
    return "high_noise" if raw_timestep > SWITCH_DIT_BOUNDARY_RAW else "low_noise"


def linear_flow_matching_noise(z0: torch.Tensor, eps: torch.Tensor, t_raw: int) -> tuple[torch.Tensor, torch.Tensor]:
    """z_t = (1 - frac) * z_0 + frac * eps, target v = eps - z_0 (Wan flow matching, P3 #7 verified)."""
    frac = float(t_raw) / NUM_TRAIN_TIMESTEPS
    z_t = (1.0 - frac) * z0 + frac * eps
    v_target = eps - z0
    return z_t, v_target


# ---------- Conditioning y construction (P0 #1) ----------


def build_y_conditioning(
    cond_image_latent: torch.Tensor,
    latent_T: int,
    latent_H: int,
    latent_W: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the I2V conditioning y = cat(mask, z_cond) with shape [20, T, H, W].

    Mirrors ``wan/image2video.py`` lines 290-326:
      msk[:, 0] = 1, rest = 0
      y_video = VAE.encode(cat([interp(image), zeros(F-1)])); shape [16, T, H, W]
      y = cat(msk_4channel, y_video) along channel dim -> [20, T, H, W]
    """
    # Mask: 4 channels because the upstream construction reshapes a 1-channel
    # mask of length 4*T into a 4-channel tensor of length T (see image2video.py:298-299).
    # In our pre-encoded latent space, T is the *latent* T (e.g., 21). The mask is
    # 1.0 at the conditioning frame slot (frame 0) and 0.0 elsewhere.
    msk = torch.zeros(MASK_C, latent_T, latent_H, latent_W, device=device, dtype=dtype)
    msk[:, 0] = 1.0
    # Concat conditioning latent in front (broadcast first frame across temporal repeat)
    z_cond = cond_image_latent.to(device=device, dtype=dtype)
    if z_cond.shape != (LATENT_C, latent_T, latent_H, latent_W):
        raise ValueError(
            f"cond_image_latent shape mismatch: got {tuple(z_cond.shape)}, expected ({LATENT_C}, {latent_T}, {latent_H}, {latent_W})"
        )
    y = torch.cat([msk, z_cond], dim=0)
    assert y.shape == (Y_C, latent_T, latent_H, latent_W), y.shape
    return y


# ---------- Conditioning image encoder (init-time cache) ----------


def encode_conditioning_image(
    vae,
    image_path: pathlib.Path,
    target_w: int,
    target_h: int,
    frame_num: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Mirror image2video.py:317-326: load image, interp to (h, w), pad zeros for F-1 frames, VAE encode.

    Returns z_cond of shape [16, T_latent, H_latent, W_latent].
    """
    import cv2

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"could not read conditioning image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Bicubic resize to (target_h, target_w)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    img_t = torch.from_numpy(img).float() / 127.5 - 1.0  # [H, W, 3] in [-1, 1]
    img_t = img_t.permute(2, 0, 1).contiguous()  # [3, H, W]
    img_t = img_t.unsqueeze(1)  # [3, 1, H, W]
    # Pad zeros for F-1 frames (per image2video.py L322)
    zeros = torch.zeros(3, frame_num - 1, target_h, target_w, dtype=img_t.dtype)
    video = torch.cat([img_t, zeros], dim=1)  # [3, F, H, W]
    video_dev = video.to(device=device, dtype=dtype)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        z_cond = vae.encode([video_dev])[0]  # [16, F_latent, H_latent, W_latent]
    return z_cond.detach().cpu()


# ---------- Dataset ----------


@dataclasses.dataclass
class PairRecord:
    pair_id: str
    group_id: str
    prompt: str
    cond_image_path: str
    cond_image_md5: str  # rl2 round-2 review: cache cond latents by image_md5 not path
    winner_latent_path: str
    loser_latent_path: str


def load_pair_records(
    latent_manifest_path: pathlib.Path,
    post_t2_pair_path: pathlib.Path,
    t2_image_manifest_path: pathlib.Path,
) -> list[PairRecord]:
    pairs_by_id: dict[str, dict] = {}
    with latent_manifest_path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            pid = entry["pair_id"]
            pairs_by_id.setdefault(pid, {})[entry["role"]] = entry
    pairs_by_id = {pid: roles for pid, roles in pairs_by_id.items() if "winner" in roles and "loser" in roles}

    pair_meta = {r["pair_id"]: r for r in json.loads(post_t2_pair_path.read_bytes())}
    image_manifest = json.loads(t2_image_manifest_path.read_bytes())

    records: list[PairRecord] = []
    for pid, roles in pairs_by_id.items():
        meta = pair_meta[pid]
        gid = meta["group_id"]
        manifest_entry = image_manifest[gid]
        cond_image_path = manifest_entry["image_path"]
        cond_image_md5 = manifest_entry["image_md5"]
        records.append(
            PairRecord(
                pair_id=pid,
                group_id=gid,
                prompt=meta["prompt"],
                cond_image_path=cond_image_path,
                cond_image_md5=cond_image_md5,
                winner_latent_path=roles["winner"]["latent_path"],
                loser_latent_path=roles["loser"]["latent_path"],
            )
        )
    records.sort(key=lambda r: r.pair_id)  # deterministic order
    return records


class TierLatentDataset(Dataset):
    def __init__(self, records: list[PairRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        winner_latent = safetensors.torch.load_file(r.winner_latent_path)["latent"]
        loser_latent = safetensors.torch.load_file(r.loser_latent_path)["latent"]
        return {
            "pair_id": r.pair_id,
            "group_id": r.group_id,
            "prompt": r.prompt,
            "cond_image_path": r.cond_image_path,
            "cond_image_md5": r.cond_image_md5,
            "winner_latent": winner_latent,
            "loser_latent": loser_latent,
        }


def collate_single(batch):
    assert len(batch) == 1, "DPO uses micro_batch=1 (winner+loser already ×2)"
    return batch[0]


# ---------- LoRA injection (P0 #2 dtype-safe + P1 #4 correct names) ----------


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.zeros(base.in_features, rank, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.zeros(rank, base.out_features, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B initialized at zero so adapter contributes 0 at init.
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_a = x.to(self.A.dtype)
        delta = (x_a @ self.A) @ self.B
        return base_out + (self.scale * delta).to(base_out.dtype)


def inject_lora(model: nn.Module, target_re: re.Pattern, rank: int, alpha: float, dtype: torch.dtype, device: torch.device) -> tuple[list[nn.Parameter], list[str]]:
    """Replace matching nn.Linear modules with LoRALinear; return trainable param list + matched names."""
    matched_names: list[str] = []
    lora_params: list[nn.Parameter] = []

    def _walk(parent: nn.Module, prefix: str = ""):
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and target_re.search(full_name):
                lora_module = LoRALinear(child, rank=rank, alpha=alpha, dtype=dtype, device=device)
                setattr(parent, child_name, lora_module)
                matched_names.append(full_name)
                lora_params.append(lora_module.A)
                lora_params.append(lora_module.B)
            else:
                _walk(child, full_name)

    _walk(model)
    return lora_params, matched_names


def collect_lora_state(model: nn.Module) -> tuple[dict[str, torch.Tensor], dict]:
    """Walk model, collect ``<module>.lora_A`` / ``<module>.lora_B`` entries (P1 #5)."""
    state: dict[str, torch.Tensor] = {}
    metadata = {"target_modules": []}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            state[f"{name}.lora_A"] = mod.A.detach().cpu().contiguous()
            state[f"{name}.lora_B"] = mod.B.detach().cpu().contiguous()
            metadata["target_modules"].append(name)
            metadata["rank"] = mod.rank
            metadata["alpha"] = mod.alpha
    return state, metadata


# ---------- Routing counter ----------


@dataclasses.dataclass
class RoutingCounterEntry:
    sampled_timestep_id: int
    raw_timestep: int
    detected_expert: str


class RoutingCounter:
    def __init__(self, halt_on_low_noise: bool = True):
        self.entries: list[RoutingCounterEntry] = []
        self.halt_on_low_noise = halt_on_low_noise
        self.high_count = 0
        self.low_count = 0

    def log(self, sampled_timestep_id: int, raw_timestep: int) -> None:
        detected = detected_expert_from_raw(raw_timestep)
        self.entries.append(RoutingCounterEntry(sampled_timestep_id, raw_timestep, detected))
        if detected == "high_noise":
            self.high_count += 1
        else:
            self.low_count += 1
            if self.halt_on_low_noise:
                raise RuntimeError(
                    f"AC-5.U3 violation: low-noise routing detected at sampled_timestep_id={sampled_timestep_id}, "
                    f"raw_timestep={raw_timestep}; tier_a contract requires 100% high-noise."
                )

    def summary(self) -> dict:
        # Each log() entry corresponds to one sampled (step, t_raw) pair; the 4 model
        # forward passes per pair (policy/ref × winner/loser) all share the same route
        # so the counter is keyed at the sample level, not the forward level.
        return {
            "high_count": self.high_count,
            "low_count": self.low_count,
            "fraction_high_noise": self.high_count / max(1, len(self.entries)),
            "total_samples": len(self.entries),
            "forwards_per_sample": 4,
            "total_forwards": len(self.entries) * 4,
        }


def socket_tail() -> str:
    import socket as _socket

    try:
        return _socket.gethostbyname(_socket.gethostname()).rsplit(".", 1)[-1]
    except OSError:
        return "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=pathlib.Path, default=pathlib.Path("/shared/user63/workspace/data/Wan/Wan2.2-I2V-A14B"))
    p.add_argument("--latent-manifest", type=pathlib.Path, required=True)
    p.add_argument("--post-t2-pair", type=pathlib.Path, required=True)
    p.add_argument("--t2-image-manifest", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, default=HERE / "ckpts")
    p.add_argument("--tier", choices=["tier_a", "tier_b"], default="tier_a")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--ref-on-cpu", type=lambda s: s.lower() == "true", default=True)
    p.add_argument("--seed-namespace", type=str, default="dpo-tier_a")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--target-w", type=int, default=832)
    p.add_argument("--target-h", type=int, default=480)
    p.add_argument("--frame-num", type=int, default=81)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--halt-on-low-noise", type=lambda s: s.lower() == "true", default=True)
    p.add_argument(
        "--cond-image-fallback-root",
        type=pathlib.Path,
        default=None,
        help="If set, look up cond images by basename under this dir when the manifest path is missing.",
    )
    p.add_argument(
        "--enable-grad-ckpt",
        type=lambda s: s.lower() == "true",
        default=False,
        help="Wrap each WanModel block.forward with torch.utils.checkpoint.checkpoint to trade compute for activation memory. Required on 80GB cards for 14B+y conditioning.",
    )
    args = p.parse_args()

    recipe_id = assert_recipe_pin(RECIPES_DIR, EXPECTED_RECIPE_ID)
    print(f"[recipe pin] OK: {recipe_id}", flush=True)

    is_distributed = "WORLD_SIZE" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device(args.device)
        local_rank = 0
        world_size = 1
        rank = 0

    is_main = rank == 0
    dtype = torch.bfloat16

    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / ts
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] world_size={world_size} rank={rank} device={device} run_dir={run_dir}", flush=True)

    # Records
    records_all = load_pair_records(args.latent_manifest, args.post_t2_pair, args.t2_image_manifest)
    # Resolve cond image paths with an optional fallback root: if the original
    # path is missing on this machine, look for the basename under
    # ``--cond-image-fallback-root``. Pairs that resolve nowhere are dropped.
    fallback_root = pathlib.Path(args.cond_image_fallback_root) if args.cond_image_fallback_root else None
    records: list[PairRecord] = []
    dropped: list[str] = []
    for r in records_all:
        original = pathlib.Path(r.cond_image_path)
        resolved = None
        if original.exists():
            resolved = original
        elif fallback_root is not None and (fallback_root / original.name).exists():
            resolved = fallback_root / original.name
        if resolved is not None:
            r2 = dataclasses.replace(r, cond_image_path=str(resolved))
            records.append(r2)
        else:
            dropped.append(f"{r.pair_id} ({r.cond_image_path})")
    if is_main:
        print(
            f"[dataset] {len(records)} pairs (dropped {len(dropped)} with missing cond images)",
            flush=True,
        )
        for d in dropped[:16]:
            print(f"  dropped: {d}", flush=True)
    if not records:
        raise RuntimeError(
            "no pairs left after cond-image existence filter; deploy is incomplete"
        )

    # ---- VAE for cond image encoding (init-time only) ----
    if is_main:
        print(f"[load] VAE for cond image encoding ...", flush=True)
    from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

    vae = Wan2_1_VAE(z_dim=16, vae_pth=str(args.upstream / "Wan2.1_VAE.pth"), dtype=dtype, device=str(device))

    # Cache cond latents by image_md5 (rl2 round-2 review): same md5 = same image bytes,
    # so different paths pointing at the same content share one encode.
    md5_to_path: dict[str, str] = {}
    for r in records:
        if r.cond_image_md5 not in md5_to_path:
            md5_to_path[r.cond_image_md5] = r.cond_image_path
    if is_main:
        print(f"[cond-encode] encoding {len(md5_to_path)} unique cond images (by image_md5) ...", flush=True)
    cond_latent_cache: dict[str, torch.Tensor] = {}
    for md5, ip in md5_to_path.items():
        z_cond_cpu = encode_conditioning_image(
            vae, pathlib.Path(ip), args.target_w, args.target_h, args.frame_num, device, dtype,
        )
        cond_latent_cache[md5] = z_cond_cpu
    # Free VAE GPU memory after init
    del vae
    gc.collect()
    torch.cuda.empty_cache()
    if is_main:
        _mem("after VAE del+gc")

    # ---- T5 (encode all unique prompts at init, then free) ----
    if is_main:
        print(f"[load] T5 encoder ...", flush=True)
    from wan.modules.t5 import T5EncoderModel  # noqa: E402

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=str(args.upstream / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(args.upstream / "google" / "umt5-xxl"),
    )
    unique_prompts = sorted({r.prompt for r in records})
    if is_main:
        print(f"[prompt-encode] encoding {len(unique_prompts)} unique prompts ...", flush=True)
    prompt_cache: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for prompt in unique_prompts:
            ctx_list = text_encoder([prompt], device)
            # Codex round 2 P1: cache to CPU so del text_encoder actually reclaims T5 GPU memory.
            prompt_cache[prompt] = ctx_list[0].detach().cpu()
    # Free T5 (keep encoded tensors on CPU)
    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    if is_main:
        _mem("after T5 del+gc")

    # ---- Policy + reference WanModel ----
    if is_main:
        print(f"[load] policy WanModel from {args.upstream}/high_noise_model ...", flush=True)
    from wan.modules.model import WanModel  # noqa: E402

    policy = WanModel.from_pretrained(args.upstream, subfolder="high_noise_model").to(dtype=dtype, device=device)
    policy.requires_grad_(False)
    if is_main:
        _mem("after policy load")

    # Optional gradient checkpointing on transformer blocks (round-4 fix for 80GB OOM).
    # Wraps each block.forward with torch.utils.checkpoint.checkpoint so activations are
    # recomputed during backward instead of cached. Trades ~2x compute for ~2x activation memory.
    if args.enable_grad_ckpt and hasattr(policy, "blocks"):
        from torch.utils.checkpoint import checkpoint as _ckpt
        for blk in policy.blocks:
            _orig_block_fwd = blk.forward
            def _ckpt_block(*a, _orig=_orig_block_fwd, **kw):
                return _ckpt(_orig, *a, use_reentrant=False, **kw)
            blk.forward = _ckpt_block
        if is_main:
            print(f"[grad-ckpt] wrapped {len(list(policy.blocks))} blocks with torch.utils.checkpoint", flush=True)

    # P1 #4: target Wan's q/k/v/o + ffn.0/ffn.2
    lora_params, matched_names = inject_lora(policy, LORA_TARGET_RE, args.lora_rank, args.lora_alpha, dtype, device)
    if is_main:
        print(f"[lora] injected on {len(matched_names)} modules; sample: {matched_names[:6]}", flush=True)

    if is_main:
        print(f"[load] reference WanModel (frozen, {'CPU' if args.ref_on_cpu else 'GPU'}) ...", flush=True)
    reference = WanModel.from_pretrained(args.upstream, subfolder="high_noise_model").to(
        dtype=dtype, device="cpu" if args.ref_on_cpu else device,
    )
    reference.requires_grad_(False)
    reference.eval()
    if is_main:
        _mem("after reference load")

    if is_distributed:
        # Codex round 2 hint: find_unused_parameters=True is unnecessary overhead since
        # all LoRA params receive grad; flip to False for cleaner DDP behavior.
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=False, broadcast_buffers=False)

    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.999))
    routing_counter = RoutingCounter(halt_on_low_noise=args.halt_on_low_noise)

    dataset = TierLatentDataset(records)
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_single, num_workers=0)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single, num_workers=0)

    losses: list[float] = []
    step = 0
    wall_start = time.time()

    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step // max(1, len(dataset)))
        for data in loader:
            if step >= args.max_steps:
                break
            pid = data["pair_id"]
            wlat = data["winner_latent"].to(device=device, dtype=dtype)
            llat = data["loser_latent"].to(device=device, dtype=dtype)

            # Sample shared (t, eps)
            t_raw, eps = sample_per_pair_t_eps(pid, tuple(wlat.shape), device, dtype, namespace=args.seed_namespace)
            routing_counter.log(sampled_timestep_id=step, raw_timestep=t_raw)
            z_w_t, v_w = linear_flow_matching_noise(wlat, eps, t_raw)
            z_l_t, v_l = linear_flow_matching_noise(llat, eps, t_raw)

            # Build y conditioning (cache keyed by image_md5)
            cond_z = cond_latent_cache[data["cond_image_md5"]]  # [16, T, H, W] CPU bf16
            y = build_y_conditioning(cond_z, wlat.shape[1], wlat.shape[2], wlat.shape[3], device, dtype)

            # Context (T5 cached on CPU; bring to device per step to avoid persistent T5 GPU footprint)
            context = [prompt_cache[data["prompt"]].to(device=device, dtype=dtype, non_blocking=True)]

            seq_len = wlat.shape[1] * wlat.shape[2] * wlat.shape[3] // 4
            t_tensor = torch.tensor([t_raw], device=device, dtype=torch.float32)

            # Reference forwards (no_grad, optionally on CPU)
            try:
                if is_main and step == 0:
                    _mem("step0 pre-ref-forward")
                with torch.no_grad():
                    if args.ref_on_cpu:
                        # Move to GPU just for the two ref forwards
                        reference.to(device)
                        if is_main and step == 0:
                            _mem("step0 ref-on-GPU")
                    with torch.amp.autocast("cuda", dtype=dtype):
                        v_ref_w = reference([z_w_t], t_tensor, context, seq_len, y=[y])[0]
                        v_ref_l = reference([z_l_t], t_tensor, context, seq_len, y=[y])[0]
                    if args.ref_on_cpu:
                        reference.cpu()
                        gc.collect()
                        torch.cuda.empty_cache()
                if is_main and step == 0:
                    _mem("step0 post-ref-back-to-CPU")
            except Exception as e:
                if is_main:
                    print(f"[step {step}] REF FORWARD FAILURE: {e}", flush=True)
                raise

            # Policy forwards (with autograd)
            try:
                with torch.amp.autocast("cuda", dtype=dtype):
                    v_pi_w = policy([z_w_t], t_tensor, context, seq_len, y=[y])[0]
                    if is_main and step == 0:
                        _mem("step0 post-v_pi_w")
                    v_pi_l = policy([z_l_t], t_tensor, context, seq_len, y=[y])[0]
                    if is_main and step == 0:
                        _mem("step0 post-v_pi_l")
            except Exception as e:
                if is_main:
                    print(f"[step {step}] POLICY FORWARD FAILURE: {e}", flush=True)
                raise

            # DPO loss
            loss = flow_matching_dpo_loss(
                v_policy_winner=v_pi_w.unsqueeze(0).float(),
                v_policy_loser=v_pi_l.unsqueeze(0).float(),
                v_reference_winner=v_ref_w.unsqueeze(0).float(),
                v_reference_loser=v_ref_l.unsqueeze(0).float(),
                v_target_winner=v_w.unsqueeze(0).float(),
                v_target_loser=v_l.unsqueeze(0).float(),
                beta=args.beta,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)

            if is_main and step % args.log_every == 0:
                elapsed = time.time() - wall_start
                vram_alloc_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                vram_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
                print(
                    f"[step {step}/{args.max_steps}] pair={pid[:8]} t_raw={t_raw} loss={loss_val:.4f} "
                    f"elapsed={elapsed:.1f}s vram_peak={vram_alloc_gb:.2f}GB reserved={vram_reserved_gb:.2f}GB",
                    flush=True,
                )

            if is_main and step > 0 and step % args.save_every == 0:
                ckpt_path = run_dir / f"lora_step{step}.safetensors"
                state, meta = collect_lora_state(policy.module if is_distributed else policy)
                safetensors.torch.save_file(state, str(ckpt_path), metadata={k: str(v) for k, v in meta.items()})
                print(f"[step {step}] saved {ckpt_path}", flush=True)

            step += 1

    wall_seconds = time.time() - wall_start

    if is_main:
        ckpt_path = run_dir / "lora_final.safetensors"
        state, meta = collect_lora_state(policy.module if is_distributed else policy)
        safetensors.torch.save_file(state, str(ckpt_path), metadata={k: str(v) for k, v in meta.items()})
        manifest = {
            "tier": args.tier,
            "max_steps": args.max_steps,
            "actual_steps": step,
            "lr": args.lr,
            "beta": args.beta,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "ref_on_cpu": args.ref_on_cpu,
            "world_size": world_size,
            # Honest enum (rl2 df979b3d): plan AC-6 names dpo_multi_gpu_zero2 but round-2
            # actually runs plain DDP + ref_offload + DistributedSampler. dpo_multi_gpu_zero2
            # (real DS Zero-2 wrap) is round-3+ aspiration.
            "compute_envelope": "dpo_multi_gpu_ddp" if is_distributed else "single_gpu",
            "parallelism": "ddp" if is_distributed else "none",
            "ref_offload": args.ref_on_cpu,
            "machine_internal_ip_tail": socket_tail(),
            "recipe_id": recipe_id,
            "lora_target_modules_count": len(matched_names),
            "lora_target_modules_sample": matched_names[:8],
            "final_loss": losses[-1] if losses else None,
            "loss_min": min(losses) if losses else None,
            "loss_max": max(losses) if losses else None,
            "loss_mean": sum(losses) / len(losses) if losses else None,
            "routing_counter": routing_counter.summary(),
            "ckpt_path": str(ckpt_path),
            "wall_seconds": round(wall_seconds, 2),
            "p3_sampling_band": [SAMPLING_T_LOW, SAMPLING_T_HIGH],
            "ts_utc": ts,
            "vram_peak_alloc_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 2),
            "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved(device) / 1024**3, 2),
        }
        run_dir.joinpath("run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[done] saved {ckpt_path}", flush=True)
        print(f"[done] manifest at {run_dir / 'run_manifest.json'}", flush=True)
        print(f"[done] routing counter: {routing_counter.summary()}", flush=True)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
