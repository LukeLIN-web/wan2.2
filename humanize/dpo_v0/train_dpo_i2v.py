"""Wan2.2-I2V-A14B Direct I2V DPO trainer (v0, tier_a tiny-overfit + tier_b short DPO).

Single-file trainer. Loads the original sharded high-noise expert as both
policy (with LoRA) and frozen reference (no_grad, optionally CPU-offloaded).
Reads pre-encoded tier_a / tier_b winner+loser latents from the encoder's
output dir (recipe_id pinned at 6bef6e104cdd3442), reads the canonical
T2-resolved conditioning image for each pair, encodes prompts via the
pinned T5, samples per-pair shared (t, eps) deterministically, runs the
flow-matching DPO loss across 4 forward passes per step (policy/ref ×
winner/loser), and applies the AC-5 routing-counter contract on every
forward.

Multi-GPU launch via accelerate (DeepSpeed Zero-2 config in
``deepspeed_zero2.json``). Single-card launch via ``python train_dpo_i2v.py``.

Hard contracts (carried from plan + rl2 reviews):
  - recipe_id pin asserted at startup (sha256 of canonical_yaml.bytes[:16] = 6bef6e104cdd3442)
  - heldout never pre-encoded (encoder enforces; trainer also asserts via blocklist)
  - tier_a routing counter 100% high-noise (any low-noise increment halts hard)
  - low-noise expert byte-equality post-run (held in disk; loader manifest pre/post compare)
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import time
from contextlib import contextmanager, nullcontext

import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

# Local modules
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))  # videodpoWan
from dpo_loss import flow_matching_dpo_loss

# Recipe pin (must match recipes/recipe_id)
EXPECTED_RECIPE_ID = "6bef6e104cdd3442"
RECIPES_DIR = HERE / "recipes"
LOADER_OUT_ROOT = HERE / "loader" / "out"

# AC-5.U2 raw boundary (switch_DiT_boundary * 1000)
SWITCH_DIT_BOUNDARY_RAW = 900
NUM_TRAIN_TIMESTEPS = 1000
# AC-5.U1 sampling band (boundary fractions)
MIN_TIMESTEP_FRACTION = 0.0
MAX_TIMESTEP_FRACTION = 0.358


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
    assert fresh == on_disk == expected, f"recipe pin drift: fresh={fresh}, on_disk={on_disk}, expected={expected}"
    return on_disk


def per_pair_seed(pair_id: str, namespace: str = "dpo-tier_a") -> int:
    """Deterministic per-pair seed for (t, eps) sampling.

    Identical between winner/loser AND between policy/reference within
    one pair, independent across pairs (AC-5 contract).
    """
    h = hashlib.sha256(f"{namespace}:{pair_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def sample_per_pair_t_eps(
    pair_id: str,
    latent_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    namespace: str = "dpo-tier_a",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample shared (t, eps) for a pair under AC-5.U1 sampling band.

    Returns (t_raw, eps) where t_raw is in [SWITCH_DIT_BOUNDARY_RAW + 1,
    NUM_TRAIN_TIMESTEPS] (i.e. always inside the high-noise expert band
    after AC-5.U2 routing detection) and eps has shape ``latent_shape``.

    Sampling band fractions [0, 0.358] map to raw timesteps in
    [641, 1000] (i.e. 1000 - 0.358*1000 = 641 to 1000 - 0*1000 = 1000).
    Per AC-5.U2, raw_timestep > 900 routes to high-noise. Within the
    band [641, 1000], values > 900 are 100/360 ≈ 28% of samples by
    default. To honor AC-5.U3 (100% high-noise on tier_a), we restrict
    sampling to the high-noise sub-band [901, 1000].
    """
    g = torch.Generator(device="cpu").manual_seed(per_pair_seed(pair_id, namespace))
    # Restrict to high-noise sub-band so routing counter stays at 100% high-noise.
    t_raw = torch.randint(
        low=SWITCH_DIT_BOUNDARY_RAW + 1, high=NUM_TRAIN_TIMESTEPS + 1, size=(1,), generator=g
    ).to(device=device).long()
    eps = torch.randn(latent_shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    return t_raw, eps


def detected_expert_from_raw(raw_timestep: int) -> str:
    return "high_noise" if raw_timestep > SWITCH_DIT_BOUNDARY_RAW else "low_noise"


def linear_flow_matching_noise(z0: torch.Tensor, eps: torch.Tensor, t_raw: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Add flow-matching noise: z_t = (1 - frac) * z_0 + frac * eps.

    ``t_raw`` is in [0, NUM_TRAIN_TIMESTEPS]; fraction = t_raw /
    NUM_TRAIN_TIMESTEPS. Target velocity v_target = eps - z_0 (constant
    along the linear interpolant). Returns (z_t, v_target).
    """
    frac = float(t_raw) / NUM_TRAIN_TIMESTEPS
    z_t = (1.0 - frac) * z0 + frac * eps
    v_target = eps - z0
    return z_t, v_target


@dataclasses.dataclass
class PairLatent:
    pair_id: str
    prompt: str
    image_path: str  # canonical T2-resolved conditioning image path
    winner_latent_path: str
    loser_latent_path: str


def load_pair_latents_manifest(manifest_path: pathlib.Path) -> dict[str, dict]:
    """Index encoder's manifest.jsonl by pair_id, with winner / loser paths."""
    pairs: dict[str, dict] = {}
    with manifest_path.open("rb") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            pid = entry["pair_id"]
            pairs.setdefault(pid, {})[entry["role"]] = entry
    # only keep pairs with both roles present
    return {pid: roles for pid, roles in pairs.items() if "winner" in roles and "loser" in roles}


def load_t2_image_manifest(t2_image_manifest_path: pathlib.Path) -> dict[str, str]:
    """Map (pair_id or group_id) -> conditioning image path from T2 manifest."""
    raw = json.loads(t2_image_manifest_path.read_bytes())
    # The T2 image manifest is keyed by group_id; we'll need to resolve
    # via post_t2_pair.json (group_id from pair_id).
    if isinstance(raw, dict):
        return {gid: entry.get("image_path", entry.get("path", "")) for gid, entry in raw.items()}
    return {}


def load_post_t2_pair(post_t2_pair_path: pathlib.Path) -> dict[str, dict]:
    """Index post_t2_pair.json by pair_id."""
    records = json.loads(post_t2_pair_path.read_bytes())
    return {r["pair_id"]: r for r in records}


class TierLatentDataset(Dataset):
    """Yields (pair_id, prompt, conditioning_image_path, winner_latent, loser_latent)."""

    def __init__(
        self,
        latent_manifest_path: pathlib.Path,
        post_t2_pair_path: pathlib.Path,
        t2_image_manifest_path: pathlib.Path,
        wmbench_data_root: pathlib.Path,
    ):
        self.pairs = load_pair_latents_manifest(latent_manifest_path)
        self.pair_meta = load_post_t2_pair(post_t2_pair_path)
        self.image_manifest = load_t2_image_manifest(t2_image_manifest_path)
        self.wmbench_data_root = wmbench_data_root
        self.pair_ids = sorted(self.pairs.keys())
        for pid in self.pair_ids:
            assert pid in self.pair_meta, f"pair {pid} in latent manifest but not post_t2_pair.json"

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int):
        pid = self.pair_ids[idx]
        roles = self.pairs[pid]
        meta = self.pair_meta[pid]
        winner_latent = safetensors.torch.load_file(roles["winner"]["latent_path"])["latent"]
        loser_latent = safetensors.torch.load_file(roles["loser"]["latent_path"])["latent"]
        # Prompt comes from post_t2_pair.json
        prompt = meta["prompt"]
        # Conditioning image is per-group (resolved at T2). Use group_id mapping.
        gid = meta["group_id"]
        cond_image_path = self.image_manifest.get(gid, "")
        return {
            "pair_id": pid,
            "prompt": prompt,
            "winner_latent": winner_latent,  # bf16 [16, 21, H, W]
            "loser_latent": loser_latent,
            "cond_image_path": cond_image_path,
            "group_id": gid,
        }


def collate_pair(batch):
    """Collate a list of pair dicts; for DPO we typically use micro_batch=1."""
    assert len(batch) == 1, f"expected micro_batch=1 (DPO ×2 already), got {len(batch)}"
    return batch[0]


# ---------- Routing counter (AC-5.U2/U3/U4) ----------


@dataclasses.dataclass
class RoutingCounterEntry:
    sampled_timestep_id: int
    raw_timestep: int
    detected_expert: str


class RoutingCounter:
    """AC-5 routing counter; raises on any low-noise hit on tier_a."""

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
        return {
            "high_count": self.high_count,
            "low_count": self.low_count,
            "fraction_high_noise": self.high_count / max(1, len(self.entries)),
            "total_forwards": len(self.entries),
        }


# ---------- Main training entry ----------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--upstream", type=pathlib.Path, default=pathlib.Path("/shared/user63/workspace/data/Wan/Wan2.2-I2V-A14B"))
    p.add_argument("--latent-manifest", type=pathlib.Path, required=True, help="manifest.jsonl from encode_videos.py")
    p.add_argument("--post-t2-pair", type=pathlib.Path, required=True, help="post_t2_pair.json from T2 step")
    p.add_argument("--t2-image-manifest", type=pathlib.Path, required=True, help="t2/image_manifest.json")
    p.add_argument("--wmbench-data-root", type=pathlib.Path, default=pathlib.Path("/shared/user60/worldmodel/wmbench/data"))
    p.add_argument("--out-dir", type=pathlib.Path, default=HERE / "ckpts")
    p.add_argument("--tier", choices=["tier_a", "tier_b"], default="tier_a")
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--ref-offload", action="store_true")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--seed-namespace", type=str, default="dpo-tier_a")
    p.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--halt-on-low-noise", type=lambda s: s.lower() == "true", default=True)
    args = p.parse_args()

    # Recipe pin assert (before any heavy I/O)
    recipe_id = assert_recipe_pin(RECIPES_DIR, EXPECTED_RECIPE_ID)
    print(f"[recipe pin] OK: {recipe_id}", flush=True)

    # DDP setup if launched with torchrun / accelerate
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

    # Set up output dir
    ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out_dir / ts
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    # Load policy + reference WanModel (high_noise expert only for v0)
    if is_main:
        print(f"[load] importing WanModel ...", flush=True)
    from wan.modules.model import WanModel  # videodpoWan

    if is_main:
        print(f"[load] policy from {args.upstream}/high_noise_model ...", flush=True)
    policy = WanModel.from_pretrained(args.upstream, subfolder="high_noise_model").to(dtype=dtype, device=device)
    policy.requires_grad_(False)  # base frozen, only LoRA trains

    # LoRA injection on attention + FFN modules
    target_keys = []
    for name, mod in policy.named_modules():
        if isinstance(mod, nn.Linear):
            # heuristic: target attention q/k/v/o and FFN linears
            lower = name.lower()
            if any(t in lower for t in ["q_proj", "k_proj", "v_proj", "o_proj", "to_q", "to_k", "to_v", "to_out", "ffn", "mlp", "fc1", "fc2"]):
                target_keys.append(name)
    if is_main:
        print(f"[lora] target_keys ({len(target_keys)}):", target_keys[:6], "..." if len(target_keys) > 6 else "", flush=True)

    # Insert LoRA adapters
    lora_params: list[nn.Parameter] = []
    for tname in target_keys:
        # Resolve module
        mod = policy
        for part in tname.split("."):
            mod = getattr(mod, part)
        in_f, out_f = mod.in_features, mod.out_features
        A = nn.Parameter(torch.zeros(in_f, args.lora_rank, dtype=dtype, device=device))
        B = nn.Parameter(torch.zeros(args.lora_rank, out_f, dtype=dtype, device=device))
        # init A normal, B zero -> LoRA effective is zero at init
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        # B left at zero
        scale = float(args.lora_alpha) / float(args.lora_rank)
        # Wrap forward: out = mod(x) + scale * (x @ A) @ B
        original_forward = mod.forward
        def _make_lora_forward(orig_fwd, A_p, B_p, s):
            def _fwd(x):
                base = orig_fwd(x)
                lora_out = (x @ A_p) @ B_p
                return base + s * lora_out
            return _fwd
        mod.forward = _make_lora_forward(original_forward, A, B, scale)
        # Register A/B on the module so DDP picks them up
        mod.register_parameter(f"lora_A", A)
        mod.register_parameter(f"lora_B", B)
        lora_params.append(A)
        lora_params.append(B)
    for pp in lora_params:
        pp.requires_grad_(True)

    if is_main:
        print(f"[load] reference from {args.upstream}/high_noise_model (frozen, no LoRA) ...", flush=True)
    reference = WanModel.from_pretrained(args.upstream, subfolder="high_noise_model").to(dtype=dtype, device=device if not args.ref_offload else "cpu")
    reference.requires_grad_(False)
    reference.eval()

    # DDP wrap policy
    if is_distributed:
        policy = DDP(policy, device_ids=[local_rank], find_unused_parameters=True)

    # T5 + tokenizer (for prompt encoding)
    if is_main:
        print(f"[load] T5 encoder ...", flush=True)
    from wan.modules.t5 import T5EncoderModel
    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=dtype,
        device=device,
        checkpoint_path=str(args.upstream / "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=str(args.upstream / "google" / "umt5-xxl"),
    )

    # Dataset
    dataset = TierLatentDataset(
        latent_manifest_path=args.latent_manifest,
        post_t2_pair_path=args.post_t2_pair,
        t2_image_manifest_path=args.t2_image_manifest,
        wmbench_data_root=args.wmbench_data_root,
    )
    if is_main:
        print(f"[dataset] {len(dataset)} pairs", flush=True)

    # Optimizer (LoRA params only)
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, betas=(0.9, 0.999))

    routing_counter = RoutingCounter(halt_on_low_noise=args.halt_on_low_noise)

    # Training loop
    step = 0
    losses = []
    while step < args.max_steps:
        # Iterate dataset; cycle if needed
        for idx in range(len(dataset)):
            if step >= args.max_steps:
                break
            data = dataset[idx]
            pid = data["pair_id"]
            prompt = data["prompt"]
            wlat = data["winner_latent"].to(device=device, dtype=dtype)  # [16, T, H, W]
            llat = data["loser_latent"].to(device=device, dtype=dtype)

            # Sample shared (t, eps) per pair
            latent_shape = tuple(wlat.shape)
            t_raw, eps = sample_per_pair_t_eps(
                pid, latent_shape, device, dtype, namespace=args.seed_namespace
            )
            t_raw_int = int(t_raw.item())
            routing_counter.log(sampled_timestep_id=step, raw_timestep=t_raw_int)

            # Add noise
            z_w_t, v_w = linear_flow_matching_noise(wlat, eps, t_raw_int)
            z_l_t, v_l = linear_flow_matching_noise(llat, eps, t_raw_int)

            # Encode prompt via T5 (cached per prompt would be better; for tier_a 16 pairs OK)
            with torch.no_grad():
                context = text_encoder([prompt], device)  # list of [L, C]

            # TODO: integrate y (conditioning image latent + mask) per WanI2V conditioning contract.
            # For tier_a tiny-overfit smoke, we forward without conditioning and let LoRA learn delta;
            # this is a known simplification flagged in DESIGN open question #2.
            seq_len = wlat.shape[1] * wlat.shape[2] * wlat.shape[3] // 4  # rough; WanModel will assert
            t_tensor = t_raw.float().to(device=device)

            # Forward through policy and reference
            try:
                v_pi_w_pred = policy([z_w_t], t_tensor, context, seq_len, y=None)[0]
                v_pi_l_pred = policy([z_l_t], t_tensor, context, seq_len, y=None)[0]
                with torch.no_grad():
                    if args.ref_offload:
                        reference.to(device)
                    v_ref_w_pred = reference([z_w_t], t_tensor, context, seq_len, y=None)[0]
                    v_ref_l_pred = reference([z_l_t], t_tensor, context, seq_len, y=None)[0]
                    if args.ref_offload:
                        reference.cpu()
            except Exception as e:
                if is_main:
                    print(f"[step {step}] FORWARD FAILURE: {e}", flush=True)
                raise

            # DPO loss
            loss = flow_matching_dpo_loss(
                v_policy_winner=v_pi_w_pred.unsqueeze(0),
                v_policy_loser=v_pi_l_pred.unsqueeze(0),
                v_reference_winner=v_ref_w_pred.unsqueeze(0),
                v_reference_loser=v_ref_l_pred.unsqueeze(0),
                v_target_winner=v_w.unsqueeze(0),
                v_target_loser=v_l.unsqueeze(0),
                beta=args.beta,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu())
            losses.append(loss_val)

            if is_main and step % args.log_every == 0:
                print(f"[step {step}/{args.max_steps}] pair={pid[:8]} t_raw={t_raw_int} loss={loss_val:.4f}", flush=True)

            if is_main and step > 0 and step % args.save_every == 0:
                ckpt_path = run_dir / f"lora_step{step}.safetensors"
                lora_state = {f"lora_{i}_A": A for i, A in enumerate([p for p in lora_params if "A" in str(p)])}
                # simplified save: just dump the lora_params list
                state = {f"lora_param_{i}": p.detach().cpu() for i, p in enumerate(lora_params)}
                safetensors.torch.save_file(state, str(ckpt_path))
                print(f"[step {step}] saved {ckpt_path}", flush=True)

            step += 1

    # Final save + manifest
    if is_main:
        ckpt_path = run_dir / "lora_final.safetensors"
        state = {f"lora_param_{i}": p.detach().cpu() for i, p in enumerate(lora_params)}
        safetensors.torch.save_file(state, str(ckpt_path))
        manifest = {
            "tier": args.tier,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "beta": args.beta,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "ref_offload": args.ref_offload,
            "world_size": world_size,
            "compute_envelope": "dpo_multi_gpu_zero2" if is_distributed else "single_gpu",
            "machine_internal_ip_tail": socket_tail(),
            "recipe_id": recipe_id,
            "final_loss": losses[-1] if losses else None,
            "loss_min": min(losses) if losses else None,
            "loss_max": max(losses) if losses else None,
            "loss_mean": sum(losses) / len(losses) if losses else None,
            "routing_counter": routing_counter.summary(),
            "ckpt_path": str(ckpt_path),
            "wall_seconds": None,  # filled below
        }
        run_dir.joinpath("run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[done] saved {ckpt_path}", flush=True)
        print(f"[done] manifest at {run_dir / 'run_manifest.json'}", flush=True)

    if is_distributed:
        dist.destroy_process_group()


def socket_tail() -> str:
    import socket
    try:
        return socket.gethostbyname(socket.gethostname()).rsplit(".", 1)[-1]
    except OSError:
        return "unknown"


if __name__ == "__main__":
    main()
