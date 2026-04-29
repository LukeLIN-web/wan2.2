#!/usr/bin/env python3
"""Strip ``_fsdp_wrapped_module.`` prefix segments from a LoRA safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


FSDP_TAG = "_fsdp_wrapped_module."


def strip_keys(state: dict) -> tuple[dict, int]:
    out: dict = {}
    n_changed = 0
    for k, v in state.items():
        nk = k.replace(FSDP_TAG, "")
        if nk != k:
            n_changed += 1
        if nk in out:
            raise ValueError(
                f"key collision after strip: {nk!r} (original {k!r}); "
                "src has duplicate post-strip keys"
            )
        out[nk] = v
    return out, n_changed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    args = p.parse_args()

    state = load_file(str(args.src))
    cleaned, n_changed = strip_keys(state)
    print(
        f"[strip] keys total={len(state)} stripped={n_changed} "
        f"src={args.src} dst={args.dst}"
    )
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    save_file(cleaned, str(args.dst))
    print(f"[strip] wrote {args.dst} ({args.dst.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
