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
  P1 #5 — LoRA save with DiffSynth-native module-aligned keys
          (`<module>.lora_A.weight` / `.lora_B.weight`) + metadata
          (`rank`, `alpha`, `target_modules`).
  P3 #7 — Sampling band restricted to [901, 999] (Wan's grid max is 999).

Multi-GPU launch via torchrun (DistributedSampler over the pair list);
DS Zero-2 wrap deferred to M4 (P2 #6 partial fix tonight).
"""

from __future__ import annotations

import argparse
import contextlib
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
from collections import deque
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


def _wandb_init(args, ts, recipe_id, world_size, run_dir):
    # Rank-0-only. Returns the wandb module on success, None on disable / failure.
    if not args.wandb_project or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] import failed ({e}); skipping.", flush=True)
        return None
    name = args.wandb_run_name or f"{args.tier}-{recipe_id[:8]}-{ts}"
    cfg = {
        "tier": args.tier,
        "lr": args.lr,
        "beta": args.beta,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "num_epochs": args.num_epochs,
        "seed_namespace": args.seed_namespace,
        "world_size": world_size,
        "dit_fsdp": args.dit_fsdp,
        "enable_grad_ckpt": args.enable_grad_ckpt,
        "halt_on_low_noise": args.halt_on_low_noise,
        "recipe_id": recipe_id,
        "training_config_sha256_pin": args.training_config_sha256_pin,
        "pair_ids_sha256_hex16": args.pair_ids_sha256_pin,
        "ts_utc": ts,
        "run_dir": str(run_dir),
    }
    try:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                   name=name, mode=args.wandb_mode, config=cfg, dir=str(run_dir))
        print(f"[wandb] init ok: project={args.wandb_project} entity={args.wandb_entity} "
              f"name={name} mode={args.wandb_mode}", flush=True)
        return wandb
    except Exception as e:
        print(f"[wandb] init failed ({e}); continuing without wandb.", flush=True)
        return None

import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

HERE = pathlib.Path(__file__).resolve().parent
DPO_ROOT = HERE.parent  # humanize/dpo_v0/
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(DPO_ROOT))
sys.path.insert(0, str(DPO_ROOT.parent.parent))  # videodpoWan
from dpo_loss import flow_matching_dpo_loss  # noqa: E402

# Recipe pin (must match recipes/recipe_id)
EXPECTED_RECIPE_ID = "6bef6e104cdd3442"
EXPECTED_VAE_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
EXPECTED_T5_SHA256 = "7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d"
EXPECTED_TOKENIZER_TREE_SHA256 = "d987d207c7b61d346ed38997af29752f8bdde8c7185169b89cbd552c8421c438"
RECIPES_DIR = DPO_ROOT / "recipes"

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


def _tokenizer_tree_sha256(tokenizer_dir: pathlib.Path) -> str:
    h = hashlib.sha256()
    rel_files = sorted(
        p.relative_to(tokenizer_dir).as_posix()
        for p in tokenizer_dir.rglob("*")
        if p.is_file()
    )
    for rel in rel_files:
        h.update(f"{rel}|{_file_sha256(tokenizer_dir / rel)}\n".encode("ascii"))
    return h.hexdigest()


def assert_training_config_pin(
    config_path: pathlib.Path,
    expected_sha256_hex16: str,
) -> dict:
    """Round-4 training_config double-pin (rl2 spec b98b72b1).

    Reads the canonical training_config_round4.yaml, fresh-hashes it under
    the same canonical rules as recipe.yaml, and asserts equality with the
    expected pin. Returns the parsed config dict for hyperparameter override.
    """
    import yaml as _yaml

    config_bytes = config_path.read_bytes()
    fresh = hashlib.sha256(config_bytes).hexdigest()[:16]
    assert fresh == expected_sha256_hex16, (
        f"training_config pin drift: fresh={fresh}, expected={expected_sha256_hex16}, "
        f"path={config_path}"
    )
    return _yaml.safe_load(config_bytes)


def assert_pair_ids_pin(
    pair_ids: list[str],
    expected_sha256_hex16: str,
) -> str:
    """Round-4 subset pin (rl2 spec b98b72b1).

    Hashes the loaded subset's pair_ids under the canonical newline-joined
    form (independent of json.dumps separators / Python implementation) and
    asserts equality with the expected pin emitted by build_round4_tier_b_1k.py.
    """
    canonical = ("\n".join(pair_ids) + "\n").encode("utf-8")
    fresh = hashlib.sha256(canonical).hexdigest()[:16]
    assert fresh == expected_sha256_hex16, (
        f"pair_ids pin drift: fresh={fresh}, expected={expected_sha256_hex16}, "
        f"n_pairs={len(pair_ids)}"
    )
    return fresh


def target_steps_for_epochs(num_epochs: float, steps_per_epoch: int) -> int:
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be > 0, got {num_epochs}")
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")
    return max(1, math.ceil(num_epochs * steps_per_epoch))


def assert_model_asset_pins(upstream: pathlib.Path) -> dict[str, str]:
    pins = {
        "vae_sha256": _file_sha256(upstream / "Wan2.1_VAE.pth"),
        "t5_sha256": _file_sha256(upstream / "models_t5_umt5-xxl-enc-bf16.pth"),
        "tokenizer_tree_sha256": _tokenizer_tree_sha256(upstream / "google" / "umt5-xxl"),
    }
    assert pins["vae_sha256"] == EXPECTED_VAE_SHA256, (
        f"VAE pin drift: actual={pins['vae_sha256']}, expected={EXPECTED_VAE_SHA256}"
    )
    assert pins["t5_sha256"] == EXPECTED_T5_SHA256, (
        f"T5 pin drift: actual={pins['t5_sha256']}, expected={EXPECTED_T5_SHA256}"
    )
    assert pins["tokenizer_tree_sha256"] == EXPECTED_TOKENIZER_TREE_SHA256, (
        "tokenizer tree pin drift: "
        f"actual={pins['tokenizer_tree_sha256']}, expected={EXPECTED_TOKENIZER_TREE_SHA256}"
    )
    return pins


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
    latent_root: pathlib.Path | None = None,
) -> list[PairRecord]:
    """If `latent_root` is given, manifest entries' `latent_path` is treated as
    relative to `latent_root` (basename or partial). This lets a manifest emitted
    on box A (with absolute paths under /home/userA/...) be reused on box B
    without sed-ing every line — pair with `--latent-root <dir>`.
    """
    pairs_by_id: dict[str, dict] = {}
    with latent_manifest_path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            pid = entry["pair_id"]
            if latent_root is not None:
                # Treat latent_path as relative-or-basename; resolve against root.
                lp = pathlib.Path(entry["latent_path"])
                if not lp.is_absolute():
                    entry["latent_path"] = str(latent_root / lp)
                else:
                    # absolute path that may not exist on this box; rebase by basename
                    entry["latent_path"] = str(latent_root / lp.name)
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
        # `enabled=False` makes the adapter a pass-through (= base linear). Used by the
        # ref-via-toggled-LoRA trick to compute reference outputs without holding a
        # second 27 GB WanModel on GPU. Math equivalence: LoRA.B is frozen-equal to
        # what a separate frozen reference would store; setting enabled=False bypasses
        # the delta path entirely so output is byte-identical to the original base linear.
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        x_a = x.to(self.A.dtype)
        delta = (x_a @ self.A) @ self.B
        return base_out + (self.scale * delta).to(base_out.dtype)


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    """Temporarily disable all LoRALinear adapters in `model` so it behaves like the
    frozen base. try/finally guarantees adapters are re-enabled even on exception
    (rl2 sign-off requirement: a leaked disabled state would silently null out training).
    """
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    prev = [m.enabled for m in layers]
    try:
        for m in layers:
            m.enabled = False
        yield
    finally:
        for m, e in zip(layers, prev):
            m.enabled = e


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
    """Walk model, collect DiffSynth-native ``.weight`` LoRA entries.

    Under FSDP, ``named_modules()`` injects ``_fsdp_wrapped_module.`` between every
    wrapped layer and its child. Those segments aren't part of the DiffSynth model
    topology used at load time, so strip them so saved keys match the canonical
    un-wrapped layout regardless of training-time wrapping.
    """
    state: dict[str, torch.Tensor] = {}
    metadata = {"target_modules": []}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            clean_name = name.replace("_fsdp_wrapped_module.", "")
            state[f"{clean_name}.lora_A.weight"] = mod.A.detach().T.cpu().contiguous()
            state[f"{clean_name}.lora_B.weight"] = mod.B.detach().T.cpu().contiguous()
            metadata["target_modules"].append(clean_name)
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
    p.add_argument(
        "--latent-root",
        type=pathlib.Path,
        default=None,
        help="If set, manifest 'latent_path' entries are resolved against this dir (basename or relative). "
             "Lets a manifest emitted on one box be reused on another without sed.",
    )
    p.add_argument("--post-t2-pair", type=pathlib.Path, required=True)
    p.add_argument("--t2-image-manifest", type=pathlib.Path, required=True)
    p.add_argument("--out-dir", type=pathlib.Path, default=DPO_ROOT / "ckpts")
    p.add_argument("--tier", choices=["tier_a", "tier_b"], default="tier_a")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument(
        "--num-epochs",
        type=float,
        default=None,
        help="Train for this many epochs over the per-rank DataLoader. Supports fractional values "
             "such as 0.8. If set, it takes precedence over --max-steps.",
    )
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
    p.add_argument(
        "--dit-fsdp",
        type=lambda s: s.lower() == "true",
        default=False,
        help="Wrap policy WanModel with FSDP FULL_SHARD (use_orig_params=True for LoRA "
             "compat). Shards 14B base weights across world_size: ~3.5GB/rank vs DDP's "
             "~28GB/rank. Required on 8x80GB after grad-ckpt is already enabled to clear "
             "step-0 OOM. Uses wan.distributed.fsdp.shard_model.",
    )
    p.add_argument(
        "--cache-root",
        type=pathlib.Path,
        default=None,
        help="Persistent disk cache for cond VAE latents and T5 prompt encodings, "
             "subdirs keyed by asset sha256 + shape so cache invalidates if VAE/T5/tokenizer "
             "or target_w/h/frame_num change. Default: <DPO_ROOT>/cache. Cache miss => "
             "rank0 encodes + writes; full hit => skip VAE/T5 load entirely (saves ~8 min boot).",
    )
    # Round-4 double-pin args (rl2 spec b98b72b1). All four mutually optional;
    # round-4 launches must set all four together. Round-2 mode (no training_config)
    # still works for smoke tests / single-GPU dev.
    p.add_argument(
        "--training-config-path",
        type=pathlib.Path,
        default=None,
        help="Round-4 training_config_round4.yaml path. When set, hyperparameters "
             "(lr/num_epochs or max_steps/beta/lora_rank/lora_alpha/seed_namespace) are read from "
             "the YAML and override CLI defaults.",
    )
    p.add_argument(
        "--training-config-sha256-pin",
        type=str,
        default=None,
        help="Expected sha256[:16] of the canonical training_config YAML bytes. "
             "Trainer fresh-hashes and asserts equality. Required when --training-config-path is set.",
    )
    p.add_argument(
        "--subset-pair-ids-json",
        type=pathlib.Path,
        default=None,
        help="Round-4 T3_round4_tier_b_1k.json path. When set, pair_records are filtered "
             "to only those in tier_b_round4_1k.pair_ids; sha256 of the canonical "
             "newline-joined form is asserted against --pair-ids-sha256-pin.",
    )
    p.add_argument(
        "--pair-ids-sha256-pin",
        type=str,
        default=None,
        help="Expected sha256[:16] of the canonical newline-joined pair_ids. "
             "Required when --subset-pair-ids-json is set.",
    )
    # Wandb (rank-0 only; failures are non-fatal so training never depends on wandb).
    p.add_argument("--wandb-project", type=str, default=None,
                   help="Wandb project name. If unset (or --wandb-mode disabled), wandb is skipped.")
    p.add_argument("--wandb-entity", type=str, default=None,
                   help="Wandb entity (user or team).")
    p.add_argument("--wandb-mode", type=str, default="online",
                   choices=["online", "offline", "disabled"],
                   help="Wandb mode. 'disabled' is a no-op even if --wandb-project is set.")
    p.add_argument("--wandb-run-name", type=str, default=None,
                   help="Wandb run name. Default: <tier>-<recipe[:8]>-<ts>.")
    args = p.parse_args()

    recipe_id = assert_recipe_pin(RECIPES_DIR, EXPECTED_RECIPE_ID)
    print(f"[recipe pin] OK: {recipe_id}", flush=True)

    # Validate paired pin args before expensive model asset hashing so bad launch
    # commands fail immediately and tests can exercise the guard with fake paths.
    if args.training_config_path is not None and args.training_config_sha256_pin is None:
        raise SystemExit(
            "--training-config-path set without --training-config-sha256-pin; both required together."
        )
    if args.training_config_sha256_pin is not None and args.training_config_path is None:
        raise SystemExit(
            "--training-config-sha256-pin set without --training-config-path; both required together."
        )
    if args.subset_pair_ids_json is not None and args.pair_ids_sha256_pin is None:
        raise SystemExit(
            "--subset-pair-ids-json set without --pair-ids-sha256-pin; both required together."
        )
    if args.pair_ids_sha256_pin is not None and args.subset_pair_ids_json is None:
        raise SystemExit(
            "--pair-ids-sha256-pin set without --subset-pair-ids-json; both required together."
        )

    asset_pins = assert_model_asset_pins(args.upstream)
    print(
        "[asset pins] OK: "
        f"vae={asset_pins['vae_sha256'][:16]} "
        f"t5={asset_pins['t5_sha256'][:16]} "
        f"tokenizer_tree={asset_pins['tokenizer_tree_sha256'][:16]}",
        flush=True,
    )

    # Round-4 training_config double-pin (rl2 spec b98b72b1). When the operator
    # passes --training-config-path, hyperparameters in the YAML override CLI
    # defaults and the pin is asserted. Single-arg-set: requires --training-config-sha256-pin.
    training_config_dict: dict | None = None
    if args.training_config_path is not None:
        if args.training_config_sha256_pin is None:
            raise SystemExit(
                "--training-config-path set without --training-config-sha256-pin; both required together."
            )
        training_config_dict = assert_training_config_pin(
            args.training_config_path, args.training_config_sha256_pin
        )
        has_num_epochs = "num_epochs" in training_config_dict or "epochs" in training_config_dict
        if has_num_epochs and "max_steps" in training_config_dict:
            raise SystemExit(
                "training_config must set only one of num_epochs/epochs or max_steps."
            )
        # Override CLI defaults with YAML values (YAML is the source of truth for round-4).
        for cli_attr, yaml_key in (
            ("lr", "lr"),
            ("max_steps", "max_steps"),
            ("num_epochs", "num_epochs"),
            ("beta", "beta"),
            ("lora_rank", "lora_rank"),
            ("lora_alpha", "lora_alpha"),
            ("seed_namespace", "seed_namespace"),
        ):
            if yaml_key in training_config_dict:
                setattr(args, cli_attr, training_config_dict[yaml_key])
        if "epochs" in training_config_dict:
            args.num_epochs = training_config_dict["epochs"]
        if "max_steps" in training_config_dict and not has_num_epochs:
            args.num_epochs = None
        print(
            f"[training_config pin] OK: {args.training_config_sha256_pin} "
            f"(lr={args.lr} num_epochs={args.num_epochs} "
            f"max_steps={args.max_steps if args.num_epochs is None else None} beta={args.beta} "
            f"lora_rank={args.lora_rank} lora_alpha={args.lora_alpha} "
            f"seed_namespace={args.seed_namespace} "
            f"max_pairs={training_config_dict.get('max_pairs')})",
            flush=True,
        )
    elif args.training_config_sha256_pin is not None:
        raise SystemExit(
            "--training-config-sha256-pin set without --training-config-path; both required together."
        )

    if args.subset_pair_ids_json is not None and args.pair_ids_sha256_pin is None:
        raise SystemExit(
            "--subset-pair-ids-json set without --pair-ids-sha256-pin; both required together."
        )
    if args.pair_ids_sha256_pin is not None and args.subset_pair_ids_json is None:
        raise SystemExit(
            "--pair-ids-sha256-pin set without --subset-pair-ids-json; both required together."
        )

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
    wandb_mod = None
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[init] world_size={world_size} rank={rank} device={device} run_dir={run_dir}", flush=True)
        wandb_mod = _wandb_init(args, ts, recipe_id, world_size, run_dir)

    # Records
    records_all = load_pair_records(
        args.latent_manifest, args.post_t2_pair, args.t2_image_manifest,
        latent_root=args.latent_root,
    )

    # Round-4 subset filter (rl2 spec b98b72b1). When --subset-pair-ids-json is
    # set, restrict records to the canonical 1k subset and assert pair_ids pin.
    if args.subset_pair_ids_json is not None:
        subset_data = json.loads(args.subset_pair_ids_json.read_bytes())
        subset_pair_ids = subset_data["tier_b_round4_1k"]["pair_ids"]
        assert_pair_ids_pin(subset_pair_ids, args.pair_ids_sha256_pin)
        subset_set = set(subset_pair_ids)
        before = len(records_all)
        records_all = [r for r in records_all if r.pair_id in subset_set]
        if is_main_pre := (int(os.environ.get("LOCAL_RANK", "0")) == 0):
            print(
                f"[pair_ids pin] OK: {args.pair_ids_sha256_pin} "
                f"(subset={len(subset_pair_ids)}, manifest_intersect={len(records_all)}, "
                f"manifest_total={before})",
                flush=True,
            )
        if len(records_all) < len(subset_pair_ids):
            missing = sorted(subset_set - {r.pair_id for r in records_all})
            raise RuntimeError(
                f"latent manifest is missing {len(missing)} of {len(subset_pair_ids)} subset pair_ids; "
                f"sample missing: {missing[:5]}"
            )

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

    # ---- cond latent cache (disk, keyed by VAE sha256 + shape + image md5) ----
    # Cache cond latents by image_md5 (rl2 round-2 review): same md5 = same image bytes,
    # so different paths pointing at the same content share one encode.
    md5_to_path: dict[str, str] = {}
    for r in records:
        if r.cond_image_md5 not in md5_to_path:
            md5_to_path[r.cond_image_md5] = r.cond_image_path

    cache_root = pathlib.Path(args.cache_root) if args.cache_root else (DPO_ROOT / "cache")
    cond_cache_dir = cache_root / "cond_latent" / (
        f"{asset_pins['vae_sha256'][:16]}_{args.target_w}x{args.target_h}_{args.frame_num}f"
    )
    cond_cache_dir.mkdir(parents=True, exist_ok=True)

    cond_to_encode = [(md5, ip) for md5, ip in md5_to_path.items()
                      if not (cond_cache_dir / f"{md5}.pt").exists()]
    if is_main:
        print(f"[cond-cache] dir={cond_cache_dir} hit={len(md5_to_path) - len(cond_to_encode)}/"
              f"{len(md5_to_path)} miss={len(cond_to_encode)}", flush=True)

    if cond_to_encode:
        if is_main:
            print(f"[load] VAE for cond image encoding ({len(cond_to_encode)} miss) ...", flush=True)
        from wan.modules.vae2_1 import Wan2_1_VAE  # noqa: E402

        vae = Wan2_1_VAE(z_dim=16, vae_pth=str(args.upstream / "Wan2.1_VAE.pth"), dtype=dtype, device=str(device))
        if is_main:
            print(f"[cond-encode] encoding {len(cond_to_encode)} new cond images (rank0 only) ...", flush=True)
            for md5, ip in cond_to_encode:
                z_cond_cpu = encode_conditioning_image(
                    vae, pathlib.Path(ip), args.target_w, args.target_h, args.frame_num, device, dtype,
                )
                tmp = cond_cache_dir / f".{md5}.pt.tmp"
                torch.save(z_cond_cpu, tmp)
                tmp.replace(cond_cache_dir / f"{md5}.pt")  # atomic publish
        del vae
        gc.collect()
        torch.cuda.empty_cache()
        if is_main:
            _mem("after VAE del+gc")
    elif is_main:
        print("[cond-cache] full hit — skipping VAE load entirely", flush=True)

    if is_distributed:
        dist.barrier()  # rank0 finishes writing before non-zero ranks read

    # All ranks load cond latents from disk
    cond_latent_cache: dict[str, torch.Tensor] = {}
    for md5 in md5_to_path:
        cond_latent_cache[md5] = torch.load(
            cond_cache_dir / f"{md5}.pt", map_location="cpu", weights_only=True
        )

    # ---- T5 prompt cache (disk, keyed by T5+tokenizer sha256 + text_len + prompt md5) ----
    unique_prompts = sorted({r.prompt for r in records})
    prompt_cache_dir = cache_root / "prompt_t5" / (
        f"{asset_pins['t5_sha256'][:16]}_{asset_pins['tokenizer_tree_sha256'][:16]}_t512"
    )
    prompt_cache_dir.mkdir(parents=True, exist_ok=True)
    # Map prompt -> stable hash filename (md5 of UTF-8 bytes; collision-resistant enough for cache).
    prompt_to_key = {p: hashlib.md5(p.encode("utf-8")).hexdigest() for p in unique_prompts}
    prompt_to_encode = [p for p in unique_prompts
                        if not (prompt_cache_dir / f"{prompt_to_key[p]}.pt").exists()]
    if is_main:
        print(f"[prompt-cache] dir={prompt_cache_dir} hit={len(unique_prompts) - len(prompt_to_encode)}/"
              f"{len(unique_prompts)} miss={len(prompt_to_encode)}", flush=True)

    if prompt_to_encode:
        if is_main:
            print(f"[load] T5 encoder ({len(prompt_to_encode)} miss) ...", flush=True)
        from wan.modules.t5 import T5EncoderModel  # noqa: E402

        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=device,
            checkpoint_path=str(args.upstream / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(args.upstream / "google" / "umt5-xxl"),
        )
        if is_main:
            print(f"[prompt-encode] encoding {len(prompt_to_encode)} new prompts (rank0 only) ...", flush=True)
            with torch.no_grad():
                for prompt in prompt_to_encode:
                    ctx_list = text_encoder([prompt], device)
                    enc = ctx_list[0].detach().cpu()
                    tmp = prompt_cache_dir / f".{prompt_to_key[prompt]}.pt.tmp"
                    torch.save(enc, tmp)
                    tmp.replace(prompt_cache_dir / f"{prompt_to_key[prompt]}.pt")
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
        if is_main:
            _mem("after T5 del+gc")
    elif is_main:
        print("[prompt-cache] full hit — skipping T5 load entirely", flush=True)

    if is_distributed:
        dist.barrier()  # rank0 finishes writing before non-zero ranks read

    # All ranks load prompt encodings from disk
    prompt_cache: dict[str, torch.Tensor] = {}
    for prompt, key in prompt_to_key.items():
        prompt_cache[prompt] = torch.load(
            prompt_cache_dir / f"{key}.pt", map_location="cpu", weights_only=True
        )

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

    # rl9 precondition for ref-via-disabled-LoRA: no dropout in policy. WanModel uses
    # RMSNorm (no running stats) and flash attention without attention_dropout, so this
    # should be empty — but assert defensively in case a future commit adds dropout.
    has_dropout = any(
        isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout))
        for m in policy.modules()
    )
    assert not has_dropout, (
        "ref-via-disabled-LoRA assumes no dropout in policy; revisit if WanModel adds dropout"
    )
    if is_main:
        print("[ref-via-toggled-LoRA] precondition OK: policy has no dropout module", flush=True)

    if is_distributed:
        if args.dit_fsdp:
            # FSDP FULL_SHARD: ~28GB base / world_size per rank (vs DDP's full 28GB/rank).
            # use_lora=True -> use_orig_params=True so LoRA Parameter objects stay live for
            # AdamW + lora_disabled() context (frozen base + trainable LoRA mixed requires_grad).
            from wan.distributed.fsdp import shard_model
            policy = shard_model(policy, device_id=local_rank, use_lora=True)
            if is_main:
                print("[fsdp] policy wrapped: FULL_SHARD use_orig_params=True bf16/fp32-reduce", flush=True)
        else:
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

    steps_per_epoch = len(loader)
    if args.num_epochs is not None:
        args.num_epochs = float(args.num_epochs)
        target_steps = target_steps_for_epochs(args.num_epochs, steps_per_epoch)
        control_mode = "epoch"
    else:
        if args.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {args.max_steps}")
        target_steps = int(args.max_steps)
        control_mode = "step"
    target_epochs = target_steps / steps_per_epoch
    if is_main:
        print(
            f"[schedule] control={control_mode} num_epochs={args.num_epochs} "
            f"max_steps={args.max_steps if control_mode == 'step' else None} "
            f"steps_per_epoch={steps_per_epoch} target_steps={target_steps} "
            f"target_epochs={target_epochs:.6g}",
            flush=True,
        )

    losses: list[float] = []
    acc_window: deque[int] = deque(maxlen=50)
    step = 0
    wall_start = time.time()

    def _save_lora(ckpt_path):
        # Under FSDP FULL_SHARD, LoRA Parameters are sharded across ranks; collect_lora_state
        # would silently grab local shards. summon_full_params is a collective so ALL ranks
        # must enter; rank0_only=True materializes full params on rank 0 only (others see
        # empty), and only rank 0 writes the safetensors file.
        if args.dit_fsdp and is_distributed:
            from torch.distributed.fsdp import FullyShardedDataParallel as _FSDP
            with _FSDP.summon_full_params(policy, writeback=False, rank0_only=True):
                if is_main:
                    state, meta = collect_lora_state(policy)
                    safetensors.torch.save_file(state, str(ckpt_path), metadata={k: str(v) for k, v in meta.items()})
        elif is_main:
            inner = policy.module if is_distributed else policy
            state, meta = collect_lora_state(inner)
            safetensors.torch.save_file(state, str(ckpt_path), metadata={k: str(v) for k, v in meta.items()})

    epoch_index = 0
    while step < target_steps:
        if sampler is not None:
            sampler.set_epoch(epoch_index)
        for batch_in_epoch, data in enumerate(loader):
            if step >= target_steps:
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

            # ----- Sequential DPO with ref-via-disabled-LoRA (v8 architecture) -----
            #
            # Original 4-forward DPO had two issues on 80GB cards:
            #   1. Separate reference WanModel = 27 GB redundant weight on GPU
            #   2. v_pi_w + v_pi_l autograd graphs alive simultaneously = 2x activations
            #
            # v8 fix:
            #   Pass A — 4 no-grad forwards on POLICY (with LoRA toggled off for ref pair)
            #            collect scalar MSEs only (no activations retained)
            #   Pass B — single forward+backward on policy(z_w_t) with grad-coef c_w
            #   Pass C — single forward+backward on policy(z_l_t) with grad-coef c_l
            #            grads accumulate into LoRA params via .grad += chain rule
            #
            # Math (sigmoid loss decomposition):
            #   delta = (mse_pi_l - mse_pi_w) - (mse_ref_l - mse_ref_w)
            #   L     = -log_sigmoid(beta * delta)
            #   c_w   = +beta * sigmoid(-beta*delta)        (= -dL/d(mse_pi_w))
            #   c_l   = -beta * sigmoid(-beta*delta)
            #   So scalar grad coefs precomputed -> two independent backward passes
            #   accumulate into the same .grad with no double-activation residency.

            mse_target_w = v_w  # [16, T, H, W]
            mse_target_l = v_l

            try:
                if is_main and step == 0:
                    _mem("step0 pre-passA")
                # Pass A: 4 no-grad forwards on POLICY for scalar MSEs.
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=dtype):
                        # ref via disabled LoRA (mathematically equivalent to frozen base)
                        with lora_disabled(policy.module if is_distributed else policy):
                            v_ref_w_ng = policy([z_w_t], t_tensor, context, seq_len, y=[y])[0]
                            mse_ref_w = (v_ref_w_ng.float() - mse_target_w.float()).pow(2).mean()
                            del v_ref_w_ng
                            v_ref_l_ng = policy([z_l_t], t_tensor, context, seq_len, y=[y])[0]
                            mse_ref_l = (v_ref_l_ng.float() - mse_target_l.float()).pow(2).mean()
                            del v_ref_l_ng
                        # policy with LoRA enabled
                        v_pi_w_ng = policy([z_w_t], t_tensor, context, seq_len, y=[y])[0]
                        mse_pi_w_scalar = (v_pi_w_ng.float() - mse_target_w.float()).pow(2).mean()
                        del v_pi_w_ng
                        v_pi_l_ng = policy([z_l_t], t_tensor, context, seq_len, y=[y])[0]
                        mse_pi_l_scalar = (v_pi_l_ng.float() - mse_target_l.float()).pow(2).mean()
                        del v_pi_l_ng
                if is_main and step == 0:
                    _mem("step0 post-passA")

                # Compute scalar grad coefficients
                delta_val = (mse_pi_l_scalar - mse_pi_w_scalar) - (mse_ref_l - mse_ref_w)
                # logit = beta * delta
                logit = float(args.beta) * float(delta_val.item())
                # loss = -log(sigmoid(logit)) = -F.logsigmoid(logit_t).item()  (scalar)
                logit_t = torch.tensor([logit], dtype=torch.float32, device=device)
                loss_val = float((-F.logsigmoid(logit_t)).item())
                # dL/dlogit = -sigmoid(-logit)  (so d/d(mse_pi_l) = -sigmoid(-logit)*beta,
                #                                d/d(mse_pi_w) = +sigmoid(-logit)*beta)
                sig_neg = float(torch.sigmoid(-logit_t).item())
                c_w = +sig_neg * float(args.beta)
                c_l = -sig_neg * float(args.beta)

                # Pass B + Pass C: forward + backward separately for winner / loser, summing grads.
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", dtype=dtype):
                    v_pi_w_grad = policy([z_w_t], t_tensor, context, seq_len, y=[y])[0]
                    if is_main and step == 0:
                        _mem("step0 post-passB-fwd")
                mse_pi_w_grad = (v_pi_w_grad.float() - mse_target_w.float()).pow(2).mean()
                (c_w * mse_pi_w_grad).backward()
                del v_pi_w_grad, mse_pi_w_grad
                if is_main and step == 0:
                    _mem("step0 post-passB-bwd")

                with torch.amp.autocast("cuda", dtype=dtype):
                    v_pi_l_grad = policy([z_l_t], t_tensor, context, seq_len, y=[y])[0]
                    if is_main and step == 0:
                        _mem("step0 post-passC-fwd")
                mse_pi_l_grad = (v_pi_l_grad.float() - mse_target_l.float()).pow(2).mean()
                (c_l * mse_pi_l_grad).backward()
                del v_pi_l_grad, mse_pi_l_grad
                if is_main and step == 0:
                    _mem("step0 post-passC-bwd")
            except Exception as e:
                if is_main:
                    print(f"[step {step}] DPO FAILURE: {e}", flush=True)
                raise

            # Grad norm + clip every step. Clipping is mandatory under r5's β=1000:
            # naive grad scale is ~10000x r4, so unclipped step would diverge to NaN.
            # FSDP shards params; sum local sum-of-squares then all-reduce -> sqrt = global L2.
            sq = torch.zeros((), device=device, dtype=torch.float32)
            for p in policy.parameters():
                if p.requires_grad and p.grad is not None:
                    sq = sq + p.grad.detach().float().pow(2).sum()
            if is_distributed:
                dist.all_reduce(sq, op=dist.ReduceOp.SUM)
            grad_norm = float(sq.sqrt().item())

            max_grad_norm = 1.0
            grad_finite = math.isfinite(grad_norm)
            if not grad_finite:
                # +inf or NaN: clipping would propagate NaN (inf * 0 = NaN). Drop the step.
                if is_main:
                    print(f"[step {step}] SKIP optimizer.step: grad_norm={grad_norm} (non-finite)", flush=True)
                optimizer.zero_grad()
            elif grad_norm > max_grad_norm:
                scale = max_grad_norm / (grad_norm + 1e-6)
                for p in policy.parameters():
                    if p.requires_grad and p.grad is not None:
                        p.grad.detach().mul_(scale)

            if grad_finite:
                optimizer.step()

            # loss_val computed scalar-style above (no autograd through loss)
            losses.append(loss_val)
            acc_window.append(int(logit > 0.0))
            acc_win = sum(acc_window) / len(acc_window)

            if is_main and step % args.log_every == 0:
                elapsed = time.time() - wall_start
                vram_alloc_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                vram_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
                epoch_progress = epoch_index + (batch_in_epoch + 1) / steps_per_epoch
                print(
                    f"[step {step}/{target_steps}] epoch={epoch_progress:.4g}/{target_epochs:.4g} "
                    f"pair={pid[:8]} t_raw={t_raw} loss={loss_val:.4f} "
                    f"gnorm={grad_norm:.3g} margin={logit:.3g} "
                    f"elapsed={elapsed:.1f}s vram_peak={vram_alloc_gb:.2f}GB reserved={vram_reserved_gb:.2f}GB",
                    flush=True,
                )
                if wandb_mod is not None:
                    # DPO standard decomposition (log p ≈ -mse, const omitted; sign-equivalent):
                    #   reward_chosen   = β·(log π_w − log π_ref_w) ≈ β·(mse_ref_w − mse_pi_w)
                    #   reward_rejected = β·(log π_l − log π_ref_l) ≈ β·(mse_ref_l − mse_pi_l)
                    #   margin          = reward_chosen − reward_rejected = logit
                    # Healthy training: chosen ↑, rejected ↓, margin ↑, accuracy → 1.
                    beta_v = float(args.beta)
                    mse_pi_w_v = float(mse_pi_w_scalar.item())
                    mse_pi_l_v = float(mse_pi_l_scalar.item())
                    mse_ref_w_v = float(mse_ref_w.item())
                    mse_ref_l_v = float(mse_ref_l.item())
                    chosen_reward = beta_v * (mse_ref_w_v - mse_pi_w_v)
                    rejected_reward = beta_v * (mse_ref_l_v - mse_pi_l_v)
                    try:
                        wandb_mod.log({
                            "loss": loss_val,
                            "t_raw": t_raw,
                            "logit": logit,
                            "margin": logit,
                            "delta": float(delta_val.item()),
                            "grad_norm": grad_norm,
                            "grad_finite": 1.0 if grad_finite else 0.0,
                            "chosen_logp": -mse_pi_w_v,
                            "rejected_logp": -mse_pi_l_v,
                            "chosen_reward": chosen_reward,
                            "rejected_reward": rejected_reward,
                            "accuracy": 1.0 if logit > 0 else 0.0,
                            "acc_win50": acc_win,
                            "mse_pi_w": mse_pi_w_v,
                            "mse_pi_l": mse_pi_l_v,
                            "mse_ref_w": mse_ref_w_v,
                            "mse_ref_l": mse_ref_l_v,
                            "c_w": c_w,
                            "elapsed_s": elapsed,
                            "vram_peak_alloc_gb": vram_alloc_gb,
                            "vram_peak_reserved_gb": vram_reserved_gb,
                        }, step=step)
                    except Exception as e:
                        print(f"[wandb] log failed at step {step}: {e}", flush=True)

            # FSDP collective requires all ranks to enter summon_full_params together,
            # so the gating drops `is_main` here; _save_lora handles rank-0-only IO internally.
            if step > 0 and step % args.save_every == 0:
                ckpt_path = run_dir / f"lora_step{step}.safetensors"
                _save_lora(ckpt_path)
                if is_main:
                    print(f"[step {step}] saved {ckpt_path}", flush=True)

            step += 1
        epoch_index += 1

    wall_seconds = time.time() - wall_start

    # _save_lora is a FSDP collective when --dit-fsdp is set (all ranks enter); manifest
    # write below stays rank-0-only.
    ckpt_path = run_dir / "lora_final.safetensors"
    _save_lora(ckpt_path)

    if is_main:
        manifest = {
            "tier": args.tier,
            "control_mode": control_mode,
            "num_epochs": args.num_epochs,
            "max_steps": args.max_steps if control_mode == "step" else None,
            "target_steps": target_steps,
            "steps_per_epoch": steps_per_epoch,
            "actual_steps": step,
            "actual_epochs": step / steps_per_epoch,
            "lr": args.lr,
            "beta": args.beta,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "world_size": world_size,
            # Honest enum (rl2 df979b3d): plan AC-6 names dpo_multi_gpu_zero2 but round-2
            # actually runs plain DDP + sequential-DPO + ref-via-disabled-LoRA. The
            # dpo_multi_gpu_zero2 envelope (DS Zero-2 wrap) remains a round-3+ aspiration.
            "compute_envelope": (
                "dpo_multi_gpu_fsdp" if (is_distributed and args.dit_fsdp)
                else ("dpo_multi_gpu_ddp" if is_distributed else "single_gpu")
            ),
            "parallelism": (
                "fsdp" if (is_distributed and args.dit_fsdp)
                else ("ddp" if is_distributed else "none")
            ),
            # v8 architecture: no separate reference WanModel. Ref forwards run on policy
            # with LoRA disabled (= byte-identical to frozen base because base is never
            # touched). Saves 27 GB of redundant ref weight on GPU.
            "ref_strategy": "lora-toggle-on-policy",
            "dpo_execution": "sequential-grad-coef",
            "ref_offload": False,  # legacy field; no separate ref to offload
            "machine_internal_ip_tail": socket_tail(),
            "recipe_id": recipe_id,
            "training_config_sha256_pin": args.training_config_sha256_pin,
            "training_config_path": str(args.training_config_path) if args.training_config_path else None,
            "subset_pair_ids_json": str(args.subset_pair_ids_json) if args.subset_pair_ids_json else None,
            "pair_ids_sha256_hex16": args.pair_ids_sha256_pin,
            "round_tag": (training_config_dict or {}).get("round_tag"),
            "vae_sha256": asset_pins["vae_sha256"],
            "t5_sha256": asset_pins["t5_sha256"],
            "tokenizer_tree_sha256": asset_pins["tokenizer_tree_sha256"],
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

        if wandb_mod is not None:
            try:
                for k, v in manifest.items():
                    # wandb summary accepts JSON-ish scalars/lists/dicts.
                    wandb_mod.summary[k] = v
                wandb_mod.finish()
            except Exception as e:
                print(f"[wandb] finish failed: {e}", flush=True)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
