"""Unit test: sequential DPO grad equivalent to parallel 4-forward DPO.

rl2 sign-off mandate (msg `e6dde036`): "single-pass-vs-sequential 数学等价
必须 verified before commit". A bug here would silently flip gradient direction
or scale, far worse than an OOM (training would proceed but minimize the
wrong objective).

Design:
  - Build a tiny ``MiniNet`` that has both a frozen 'base' linear + a
    trainable LoRA adapter (mimicking LoRALinear). Two of them so we can
    treat one as 'policy' and one as 'frozen reference' - except in
    sequential mode we use the same model with LoRA toggled off.
  - Run 4 forwards (parallel mode) and compute one combined backward
    through the canonical flow_matching DPO loss; capture grads.
  - Run 4 no-grad forwards + 2 with-grad forwards (sequential mode)
    using the scalar grad-coef decomposition; capture grads.
  - Assert torch.allclose(parallel_grad, sequential_grad, rtol=1e-4) for
    every trainable param.
"""

from __future__ import annotations

import contextlib
import math
import sys
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dpo_loss import flow_matching_dpo_loss  # noqa: E402


class MiniLoRALinear(nn.Module):
    """Mirror of LoRALinear in train_dpo_i2v.py for a self-contained test."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 4.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = float(alpha) / float(rank)
        self.A = nn.Parameter(torch.zeros(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B starts at zero (LoRA init) but we randomize for a non-trivial test
        nn.init.normal_(self.B, std=0.02)
        self.enabled = True

    def forward(self, x):
        base_out = self.base(x)
        if not self.enabled:
            return base_out
        delta = (x @ self.A) @ self.B
        return base_out + self.scale * delta


@contextlib.contextmanager
def lora_disabled(model: nn.Module):
    layers = [m for m in model.modules() if isinstance(m, MiniLoRALinear)]
    prev = [m.enabled for m in layers]
    try:
        for m in layers:
            m.enabled = False
        yield
    finally:
        for m, e in zip(layers, prev):
            m.enabled = e


class MiniNet(nn.Module):
    def __init__(self, dim: int = 16, rank: int = 4):
        super().__init__()
        self.l1 = MiniLoRALinear(dim, dim, rank=rank)
        self.l2 = MiniLoRALinear(dim, dim, rank=rank)

    def forward(self, x):
        return self.l2(F.gelu(self.l1(x)))


def _grads_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    """Capture .grad of all trainable params, .clone()'d so optimizer.zero_grad doesn't wipe."""
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}


def _zero_grad(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def _run_parallel(net: MiniNet, x_w, x_l, v_w, v_l, beta: float) -> dict[str, torch.Tensor]:
    """Original 4-forward DPO. ref forwards via lora_disabled."""
    _zero_grad(net)
    with lora_disabled(net):
        v_ref_w = net(x_w)
        v_ref_l = net(x_l)
    # Detach ref outputs (they don't carry grad anyway, but make explicit)
    v_ref_w = v_ref_w.detach()
    v_ref_l = v_ref_l.detach()
    v_pi_w = net(x_w)
    v_pi_l = net(x_l)
    loss = flow_matching_dpo_loss(
        v_policy_winner=v_pi_w.unsqueeze(0),
        v_policy_loser=v_pi_l.unsqueeze(0),
        v_reference_winner=v_ref_w.unsqueeze(0),
        v_reference_loser=v_ref_l.unsqueeze(0),
        v_target_winner=v_w.unsqueeze(0),
        v_target_loser=v_l.unsqueeze(0),
        beta=beta,
    )
    loss.backward()
    return _grads_snapshot(net), float(loss.detach().item())


def _run_sequential(net: MiniNet, x_w, x_l, v_w, v_l, beta: float) -> dict[str, torch.Tensor]:
    """Sequential DPO with scalar grad-coef decomposition (ref via lora_disabled)."""
    # Pass A: 4 no-grad forwards, capture scalar MSEs.
    with torch.no_grad():
        with lora_disabled(net):
            mse_ref_w = (net(x_w).float() - v_w.float()).pow(2).mean()
            mse_ref_l = (net(x_l).float() - v_l.float()).pow(2).mean()
        mse_pi_w_scalar = (net(x_w).float() - v_w.float()).pow(2).mean()
        mse_pi_l_scalar = (net(x_l).float() - v_l.float()).pow(2).mean()

    delta = (mse_pi_l_scalar - mse_pi_w_scalar) - (mse_ref_l - mse_ref_w)
    logit = beta * delta.item()
    logit_t = torch.tensor([logit], dtype=torch.float32)
    loss_val = float((-F.logsigmoid(logit_t)).item())
    sig_neg = float(torch.sigmoid(-logit_t).item())
    c_w = +sig_neg * beta
    c_l = -sig_neg * beta

    # Pass B: forward+backward winner with grad coef c_w.
    _zero_grad(net)
    v_pi_w = net(x_w)
    mse_pi_w = (v_pi_w.float() - v_w.float()).pow(2).mean()
    (c_w * mse_pi_w).backward()

    # Pass C: forward+backward loser with grad coef c_l (grads accumulate).
    v_pi_l = net(x_l)
    mse_pi_l = (v_pi_l.float() - v_l.float()).pow(2).mean()
    (c_l * mse_pi_l).backward()

    return _grads_snapshot(net), loss_val


def test_sequential_equivalent_to_parallel():
    torch.manual_seed(20260427)
    net = MiniNet(dim=16, rank=4)
    x_w = torch.randn(8, 16)
    x_l = torch.randn(8, 16)
    v_w = torch.randn(8, 16)
    v_l = torch.randn(8, 16)
    beta = 0.05

    # We need two independent runs to compare grads. Save initial state to ensure
    # we don't perturb between runs.
    init_state = {n: p.detach().clone() for n, p in net.named_parameters()}

    grads_par, loss_par = _run_parallel(net, x_w, x_l, v_w, v_l, beta)

    # Reset model state (in case Pass B/C accumulated state we don't want)
    for n, p in net.named_parameters():
        p.data.copy_(init_state[n])

    grads_seq, loss_seq = _run_sequential(net, x_w, x_l, v_w, v_l, beta)

    print(f"parallel loss   = {loss_par:.6f}")
    print(f"sequential loss = {loss_seq:.6f}")
    assert math.isclose(loss_par, loss_seq, rel_tol=1e-5, abs_tol=1e-6), (
        f"loss mismatch: parallel={loss_par} sequential={loss_seq}"
    )

    assert set(grads_par.keys()) == set(grads_seq.keys()), "grad key mismatch"
    for k in grads_par:
        gp = grads_par[k]
        gs = grads_seq[k]
        assert gp.shape == gs.shape, f"shape mismatch for {k}: {gp.shape} vs {gs.shape}"
        ok = torch.allclose(gp, gs, rtol=1e-4, atol=1e-6)
        if not ok:
            diff = (gp - gs).abs().max().item()
            print(f"FAIL {k}: max abs diff = {diff:.3e}, max abs gp = {gp.abs().max().item():.3e}")
        assert ok, f"grad mismatch for {k}"
        print(f"  {k}: max_abs_diff = {(gp - gs).abs().max().item():.3e}, "
              f"|gp|_max = {gp.abs().max().item():.3e}")

    print("\nALL GRADS MATCH (rtol=1e-4)")


if __name__ == "__main__":
    test_sequential_equivalent_to_parallel()
    print("PASS")
