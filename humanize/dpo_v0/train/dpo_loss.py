"""Flow-matching DPO loss for Wan 2.2 I2V.

Implements the v0 DPO loss with the policy / reference contrast worked out
for flow-matching parameterizations. The key identity:

  log p_pi(z) - log p_ref(z) ≈ -beta_inv * (||v_pi(z_t,t) - v_target||² - ||v_ref(z_t,t) - v_target||²) / 2

so the DPO log-ratio in the loss telescopes to differences of squared
prediction errors. We work directly in MSE space and let the ``beta``
coefficient absorb the constant.

The accepted form (rl2 sign-checked in msg ``8e40d34c``) is::

  L = -log_sigmoid(beta * (
        (-||v_pi(z_w_t,t) - v_w||² + ||v_pi(z_l_t,t) - v_l||²)
      - (-||v_ref(z_w_t,t) - v_w||² + ||v_ref(z_l_t,t) - v_l||²)
      ))

The first parenthesis says "policy prefers winner over loser" (small
winner MSE / large loser MSE → positive); the second is the same for
reference. The difference is "policy prefers winner more than reference
does"; log_sigmoid is monotone so larger argument means lower loss.

Per-pair shared (t, eps) lives outside this module — the caller passes in
the four already-computed velocity predictions and the two velocity
targets.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE reduced over all dims except the leading batch dim."""
    if pred.shape != target.shape:
        raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
    diff = pred - target
    return (diff ** 2).flatten(start_dim=1).mean(dim=1)


def flow_matching_dpo_loss(
    v_policy_winner: torch.Tensor,
    v_policy_loser: torch.Tensor,
    v_reference_winner: torch.Tensor,
    v_reference_loser: torch.Tensor,
    v_target_winner: torch.Tensor,
    v_target_loser: torch.Tensor,
    beta: float = 0.1,
    return_components: bool = False,
):
    """Compute the per-pair flow-matching DPO loss.

    Each ``v_*`` is the velocity field predicted by the policy / reference
    on the noised winner / loser latent at the same per-pair shared
    ``(t, eps)``; ``v_target_*`` is the corresponding ground-truth
    velocity ``z_1 - z_0`` for that latent.

    Returns the mean batch loss and (optionally) a dict of components for
    logging: per-sample MSEs, the policy / reference advantage, the raw
    DPO logit, and the implied policy-over-reference probability.
    """
    mse_pi_w = per_sample_mse(v_policy_winner, v_target_winner)
    mse_pi_l = per_sample_mse(v_policy_loser, v_target_loser)
    mse_ref_w = per_sample_mse(v_reference_winner, v_target_winner)
    mse_ref_l = per_sample_mse(v_reference_loser, v_target_loser)

    policy_advantage = -mse_pi_w + mse_pi_l        # > 0 when policy prefers winner
    reference_advantage = -mse_ref_w + mse_ref_l   # > 0 when reference prefers winner
    logit = beta * (policy_advantage - reference_advantage)
    loss = -F.logsigmoid(logit).mean()

    if return_components:
        with torch.no_grad():
            implied_p = torch.sigmoid(logit)
        return loss, {
            "mse_policy_winner": mse_pi_w.mean().detach(),
            "mse_policy_loser": mse_pi_l.mean().detach(),
            "mse_reference_winner": mse_ref_w.mean().detach(),
            "mse_reference_loser": mse_ref_l.mean().detach(),
            "policy_advantage": policy_advantage.mean().detach(),
            "reference_advantage": reference_advantage.mean().detach(),
            "logit": logit.mean().detach(),
            "implied_winner_prob": implied_p.mean().detach(),
        }
    return loss
