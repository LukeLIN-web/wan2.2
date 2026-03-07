"""StyleAdapter: CLIP 768d → per-layer self-attn shift/scale modulation.

Converts a pre-extracted CLIP ViT-L/14 CLS embedding [B, 768] into
per-layer (Δshift, Δscale) pairs [B, 3072] for additive modulation
of the DiT's self-attention input in layers 0-10.

Design:
  CLIP ViT-L/14 768d (frozen, offline)
      ↓
  Shared trunk: Linear(768, 1024) → GELU → Linear(1024, 256)
      ↓
  Per-layer heads (11 heads, layers 0-10):
      Linear(256, 3072×2) → chunk → (Δshift, Δscale)
      ↓ × sigmoid(beta_logit) × strength
      ↓
  additive modulation on self-attn input in layers 0-10
"""

import random

import torch
import torch.nn as nn

STYLE_INJECT_LAYERS = set(range(0, 11))  # layers 0, 1, ..., 10
NUM_STYLE_LAYERS = 11


class StyleAdapter(nn.Module):
    """Project CLIP embedding into per-layer self-attn shift/scale modulation.

    Args:
        clip_dim: Input dimension from CLIP encoder (default 768).
        dim: DiT hidden dimension (must match model, default 3072).
        num_layers: Number of DiT layers to modulate (default 11, layers 0-10).
        state_dropout: Probability of replacing input with null embedding during training.
    """

    TRUNK_DIM = 1024
    NECK_DIM = 256

    def __init__(self, clip_dim: int = 768, dim: int = 3072,
                 num_layers: int = NUM_STYLE_LAYERS, state_dropout: float = 0.1):
        super().__init__()
        self.clip_dim = clip_dim
        self.dim = dim
        self.num_layers = num_layers
        self.state_dropout = state_dropout

        # Shared trunk: [B, 768] → [B, 256]
        self.trunk = nn.Sequential(
            nn.Linear(clip_dim, self.TRUNK_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.TRUNK_DIM, self.NECK_DIM),
        )

        # Per-layer heads: [B, 256] → [B, dim * 2] each
        self.heads = nn.ModuleList([
            nn.Linear(self.NECK_DIM, dim * 2) for _ in range(num_layers)
        ])

        # Learnable gate: sigmoid(-2) ≈ 0.12, starts small for safety
        self.beta_logit = nn.Parameter(torch.tensor(-2.0))

        # Learnable null embedding for dropout (better than zero vector)
        self.null_style_emb = nn.Parameter(torch.randn(clip_dim) * 0.01)

        self._init_weights()

    def _init_weights(self):
        # Xavier init for trunk
        for layer in self.trunk:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Small init for heads → near-zero initial output
        for head in self.heads:
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)
            with torch.no_grad():
                head.weight.mul_(0.01)

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.beta_logit)

    def forward(self, clip_emb: torch.Tensor, strength: float = 1.0):
        """Forward pass.

        Args:
            clip_emb: [B, 768] CLIP CLS embedding.
            strength: Inference-time scaling factor (default 1.0).

        Returns:
            List of (Δshift, Δscale) tuples, one per layer.
            Each Δshift, Δscale has shape [B, dim].
        """
        B = clip_emb.shape[0]

        # Dropout: replace with learnable null embedding
        if self.training and self.state_dropout > 0:
            if random.random() < self.state_dropout:
                clip_emb = self.null_style_emb.unsqueeze(0).expand(B, -1)

        feat = self.trunk(clip_emb)  # [B, 256]
        beta = torch.sigmoid(self.beta_logit) * strength

        modulations = []
        for head in self.heads:
            mod = head(feat)                                  # [B, dim * 2]
            delta_shift, delta_scale = mod.chunk(2, dim=-1)   # each [B, dim]
            modulations.append((delta_shift * beta, delta_scale * beta))

        return modulations
