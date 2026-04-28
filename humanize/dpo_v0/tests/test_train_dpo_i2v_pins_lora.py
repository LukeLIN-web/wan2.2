from __future__ import annotations

import hashlib
import pathlib
import sys

import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # humanize/dpo_v0/

from train import train_dpo_i2v as trainer  # noqa: E402


def test_collect_lora_state_emits_diffsynth_native_weight_keys():
    base = nn.Linear(3, 5, bias=False)
    lora = trainer.LoRALinear(base, rank=2, alpha=2, dtype=torch.float32, device=torch.device("cpu"))
    with torch.no_grad():
        lora.A.copy_(torch.arange(6, dtype=torch.float32).reshape(3, 2))
        lora.B.copy_(torch.arange(10, dtype=torch.float32).reshape(2, 5))

    model = nn.Module()
    model.block = lora

    state, meta = trainer.collect_lora_state(model)
    assert sorted(state) == ["block.lora_A.weight", "block.lora_B.weight"]
    assert state["block.lora_A.weight"].shape == (2, 3)
    assert state["block.lora_B.weight"].shape == (5, 2)
    assert torch.equal(state["block.lora_A.weight"], lora.A.detach().T)
    assert torch.equal(state["block.lora_B.weight"], lora.B.detach().T)
    assert meta["rank"] == 2
    assert meta["alpha"] == 2
    assert meta["target_modules"] == ["block"]


def test_tokenizer_tree_sha256_uses_sorted_relpaths(tmp_path):
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "b.txt").write_text("two", encoding="utf-8")
    sub = tok / "nested"
    sub.mkdir()
    (sub / "a.txt").write_text("one", encoding="utf-8")

    rows = []
    for rel in ["b.txt", "nested/a.txt"]:
        rows.append(f"{rel}|{hashlib.sha256((tok / rel).read_bytes()).hexdigest()}\n")
    expected = hashlib.sha256("".join(rows).encode("ascii")).hexdigest()
    assert trainer._tokenizer_tree_sha256(tok) == expected
