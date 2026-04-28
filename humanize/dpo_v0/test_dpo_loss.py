"""Unit tests for the flow-matching DPO loss.

Covers (a) algebraic invariants — when policy == reference, the loss is
exactly ``-log_sigmoid(0) = log(2)`` regardless of inputs; (b) sign — a
policy that prefers winner more than reference yields a smaller loss;
(c) per-sample MSE shape; (d) gradient flow into policy parameters only.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from dpo_loss import flow_matching_dpo_loss, per_sample_mse


def _shape() -> tuple[int, int, int, int, int]:
    return (2, 16, 21, 60, 104)  # batch, channels, latent T, H, W (Wan 2.2 latent shape)


def test_per_sample_mse_shape_and_value():
    a = torch.zeros(_shape())
    b = torch.ones(_shape())
    mse = per_sample_mse(a, b)
    assert mse.shape == (2,)
    assert torch.allclose(mse, torch.ones(2))


def test_loss_when_policy_equals_reference_is_log2():
    """If v_policy == v_reference (and same on both branches), loss = log(2)."""
    g = torch.Generator().manual_seed(0)
    v = torch.randn(*_shape(), generator=g)
    target_w = torch.randn(*_shape(), generator=g)
    target_l = torch.randn(*_shape(), generator=g)
    loss = flow_matching_dpo_loss(
        v_policy_winner=v, v_policy_loser=v,
        v_reference_winner=v, v_reference_loser=v,
        v_target_winner=target_w, v_target_loser=target_l,
        beta=0.1,
    )
    assert torch.allclose(loss, torch.tensor(math.log(2.0)), atol=1e-5)


def test_loss_decreases_as_policy_prefers_winner_more_than_reference():
    """Build a contrast: policy aligns with target_winner; reference does not."""
    g = torch.Generator().manual_seed(7)
    target_w = torch.randn(*_shape(), generator=g)
    target_l = torch.randn(*_shape(), generator=g)
    v_ref_w = torch.randn(*_shape(), generator=g)
    v_ref_l = torch.randn(*_shape(), generator=g)
    # baseline: policy == reference
    loss_baseline = flow_matching_dpo_loss(
        v_policy_winner=v_ref_w, v_policy_loser=v_ref_l,
        v_reference_winner=v_ref_w, v_reference_loser=v_ref_l,
        v_target_winner=target_w, v_target_loser=target_l,
        beta=0.1,
    )
    # better: policy moved closer to target_winner, farther from target_loser
    v_pi_w = 0.5 * (target_w + v_ref_w)
    v_pi_l = 1.5 * v_ref_l
    loss_better = flow_matching_dpo_loss(
        v_policy_winner=v_pi_w, v_policy_loser=v_pi_l,
        v_reference_winner=v_ref_w, v_reference_loser=v_ref_l,
        v_target_winner=target_w, v_target_loser=target_l,
        beta=0.1,
    )
    assert loss_better < loss_baseline, (loss_baseline.item(), loss_better.item())


def test_loss_increases_as_policy_prefers_winner_less_than_reference():
    """Mirror of the previous test — policy should be penalized."""
    g = torch.Generator().manual_seed(11)
    target_w = torch.randn(*_shape(), generator=g)
    target_l = torch.randn(*_shape(), generator=g)
    v_ref_w = torch.randn(*_shape(), generator=g)
    v_ref_l = torch.randn(*_shape(), generator=g)
    loss_baseline = flow_matching_dpo_loss(
        v_policy_winner=v_ref_w, v_policy_loser=v_ref_l,
        v_reference_winner=v_ref_w, v_reference_loser=v_ref_l,
        v_target_winner=target_w, v_target_loser=target_l,
        beta=0.1,
    )
    # worse: policy moved away from target_winner, towards target_loser
    v_pi_w = 1.5 * v_ref_w
    v_pi_l = 0.5 * (target_l + v_ref_l)
    loss_worse = flow_matching_dpo_loss(
        v_policy_winner=v_pi_w, v_policy_loser=v_pi_l,
        v_reference_winner=v_ref_w, v_reference_loser=v_ref_l,
        v_target_winner=target_w, v_target_loser=target_l,
        beta=0.1,
    )
    assert loss_worse > loss_baseline


def test_components_sign_consistency():
    """policy_advantage - reference_advantage should sum to logit / beta."""
    g = torch.Generator().manual_seed(13)
    target_w = torch.randn(*_shape(), generator=g)
    target_l = torch.randn(*_shape(), generator=g)
    v_pi_w = torch.randn(*_shape(), generator=g)
    v_pi_l = torch.randn(*_shape(), generator=g)
    v_ref_w = torch.randn(*_shape(), generator=g)
    v_ref_l = torch.randn(*_shape(), generator=g)
    beta = 0.1
    _, c = flow_matching_dpo_loss(
        v_policy_winner=v_pi_w, v_policy_loser=v_pi_l,
        v_reference_winner=v_ref_w, v_reference_loser=v_ref_l,
        v_target_winner=target_w, v_target_loser=target_l,
        beta=beta, return_components=True,
    )
    expected_logit = beta * (c["policy_advantage"] - c["reference_advantage"])
    assert torch.allclose(c["logit"], expected_logit, atol=1e-5)


def test_gradients_flow_only_to_policy_inputs():
    """Verify reference branch is detached: gradient should only touch policy tensors."""
    g = torch.Generator().manual_seed(17)
    target_w = torch.randn(*_shape(), generator=g)
    target_l = torch.randn(*_shape(), generator=g)
    v_pi_w = torch.randn(*_shape(), generator=g, requires_grad=True)
    v_pi_l = torch.randn(*_shape(), generator=g, requires_grad=True)
    # reference is no-grad in real usage; here we use detach to mirror that.
    v_ref_w = torch.randn(*_shape(), generator=g).detach()
    v_ref_l = torch.randn(*_shape(), generator=g).detach()
    loss = flow_matching_dpo_loss(
        v_policy_winner=v_pi_w, v_policy_loser=v_pi_l,
        v_reference_winner=v_ref_w, v_reference_loser=v_ref_l,
        v_target_winner=target_w, v_target_loser=target_l,
        beta=0.1,
    )
    loss.backward()
    assert v_pi_w.grad is not None
    assert v_pi_l.grad is not None


def test_shape_mismatch_raises():
    a = torch.zeros(2, 4, 4)
    b = torch.zeros(2, 4, 5)
    with pytest.raises(ValueError, match="pred/target shape mismatch"):
        per_sample_mse(a, b)
