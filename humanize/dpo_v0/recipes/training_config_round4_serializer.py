"""Canonical serializer for the round-4 DPO training config (rl2 spec b98b72b1).

Round-4 dispatch (lukedecision.md A2 + A1#5 + B1) introduces a new training
config layer separate from the data-preprocessing recipe.  The recipe.yaml
remains immutable across rounds (recipe_id = 6bef6e104cdd3442 frozen at
round-2); training-knob changes (lr / max_steps / max_pairs / etc.) get a
fresh per-round training_config.yaml with its own sha256[:16] pin.

Trainer startup performs **two** independent asserts:

  1. recipe_id pin: data preprocessing canonical sha must equal the
     immutable round-2 anchor (BL-20260428-canonical-recipe-pin-runtime-reread).
  2. training_config_sha256 pin: per-round training-knob canonical sha must
     equal the on-disk pin file emitted at build time.

Either assert failing halts the trainer.  Both pins live under
``humanize/dpo_v0/recipes/`` so the round-4 training config is co-located
with the data recipe it composes with.

Canonicality rules are byte-equal to the round-2 ``canonical_serializer.py``
(rl2 acceptance criteria, msg ``22e2a02c``):

* ``yaml.safe_dump(..., sort_keys=True, default_flow_style=False,
  allow_unicode=False, width=2**31 - 1, indent=2, line_break='\n')``
* PyYAML 6.x major-version pinned
* trailing newline preserved as-is

Usage::

    python training_config_round4_serializer.py emit
    python training_config_round4_serializer.py verify
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import yaml

YAML_MAJOR_REQUIRED = 6
TRAINING_CONFIG_FILENAME = "training_config_round4.yaml"
TRAINING_CONFIG_ID_FILENAME = "training_config_sha256_pin"
HERE = pathlib.Path(__file__).resolve().parent


# Round-4 training knobs (lukedecision.md A2 + A1#5).  Anything the trainer
# reads that isn't a v8 architecture invariant goes here.  v8 Hammer 1/2/3
# (ref-via-disabled-LoRA / sequential DPO / grad-ckpt monkey-patch) are not
# parameterized — they are hard invariants pinned by `v8-trainer-invariants.md`
# and changing them requires a separate sign-off chain.
TRAINING_CONFIG = {
    "beta": 0.1,
    "dpo_loss_kind": "sigmoid",
    "lora_alpha": 16,
    "lora_rank": 16,
    "lr": 5.0e-5,
    "max_pairs": 1000,
    "max_steps": 200,
    "micro_batch": 1,
    "round_tag": "round-4",
    "sampling_band": [901, 999],
    "seed_namespace": "round4-tier_b-1k",
    "subset_pair_ids_sha256_hex16": "cf5d3e5fd528a3e0",
}

REQUIRED_AXES = (
    "beta",
    "dpo_loss_kind",
    "lora_alpha",
    "lora_rank",
    "lr",
    "max_pairs",
    "max_steps",
    "micro_batch",
    "round_tag",
    "sampling_band",
    "seed_namespace",
    "subset_pair_ids_sha256_hex16",
)


def _check_yaml_version() -> None:
    major = int(yaml.__version__.split(".")[0])
    if major != YAML_MAJOR_REQUIRED:
        raise RuntimeError(
            f"PyYAML major version mismatch: required {YAML_MAJOR_REQUIRED}, got {yaml.__version__}"
        )


def canonical_bytes(config: dict) -> bytes:
    _check_yaml_version()
    missing = [axis for axis in REQUIRED_AXES if axis not in config]
    if missing:
        raise ValueError(f"training_config is missing required axes: {missing}")
    text = yaml.safe_dump(
        config,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=False,
        width=2**31 - 1,
        indent=2,
        line_break="\n",
    )
    return text.encode("utf-8")


def training_config_sha256_hex16(config_bytes: bytes) -> str:
    return hashlib.sha256(config_bytes).hexdigest()[:16]


def emit(out_dir: pathlib.Path = HERE) -> tuple[pathlib.Path, pathlib.Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / TRAINING_CONFIG_FILENAME
    id_path = out_dir / TRAINING_CONFIG_ID_FILENAME
    data = canonical_bytes(TRAINING_CONFIG)
    sha16 = training_config_sha256_hex16(data)
    config_path.write_bytes(data)
    id_path.write_text(sha16 + "\n", encoding="ascii")
    return config_path, id_path, sha16


def verify(out_dir: pathlib.Path = HERE) -> tuple[bool, str, str]:
    config_path = out_dir / TRAINING_CONFIG_FILENAME
    id_path = out_dir / TRAINING_CONFIG_ID_FILENAME
    on_disk_config = config_path.read_bytes()
    on_disk_id = id_path.read_text(encoding="ascii").strip()
    fresh_id = training_config_sha256_hex16(on_disk_config)
    return on_disk_id == fresh_id, on_disk_id, fresh_id


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in {"emit", "verify"}:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[1] == "emit":
        config_path, id_path, value = emit()
        print(f"wrote {config_path}")
        print(f"wrote {id_path}")
        print(f"training_config_sha256_hex16 = {value}")
        return 0
    ok, on_disk, fresh = verify()
    if ok:
        print(f"OK: {on_disk}")
        return 0
    print(f"MISMATCH: on_disk={on_disk} fresh={fresh}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
