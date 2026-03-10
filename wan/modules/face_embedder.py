"""FaceEmbedder: ArcFace 512d → gated identity tokens for DiT cross-attention.

Converts a pre-extracted ArcFace embedding [B, 512] into a small set of
identity tokens [B, K, dim] that are concatenated to the text/state context
ONLY in DiT layers 17-19, with a learnable sigmoid gate (zero-terminal init).

Design:
  ArcFace 512d (frozen, offline)
      ↓
  MLP: Linear(512, 1024) → GELU → Linear(1024, K × dim)  [zero-init output]
      ↓ reshape
  [B, K=4, 3072]
      ↓ × sigmoid(alpha_logit)   # learnable gate, init ≈ 0.018 (logit=-4)
      ↓
  concat to context in layers 17-19 only
"""

import torch
import torch.nn as nn

NUM_FACE_TOKENS = 4
FACE_INJECT_LAYERS = set(range(17, 20))  # layers 17, 18, 19


class FaceEmbedder(nn.Module):
    """Project ArcFace embedding into gated DiT cross-attention tokens.

    Args:
        face_dim: Input dimension from ArcFace encoder (default 512).
        dim: DiT hidden dimension (must match model, default 3072).
        num_tokens: Number of output tokens per sample (default 4).
        state_dropout: Probability of zeroing entire face embedding during training.
    """

    HIDDEN_DIM = 1024

    def __init__(self, face_dim: int = 512, dim: int = 3072,
                 num_tokens: int = NUM_FACE_TOKENS, state_dropout: float = 0.1):
        super().__init__()
        self.face_dim = face_dim
        self.dim = dim
        self.num_tokens = num_tokens
        self.state_dropout = state_dropout

        # MLP: [B, 512] → [B, K * 3072]
        self.mlp = nn.Sequential(
            nn.Linear(face_dim, self.HIDDEN_DIM),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.HIDDEN_DIM, num_tokens * dim),
        )

        # Learnable gate: sigmoid(-4) ≈ 0.018 (zero-terminal init)
        self.alpha_logit = nn.Parameter(torch.tensor([-4.0]))

        self._init_weights()

    def _init_weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Zero-terminal init: output layer weight AND bias = 0
        # At step 0, FaceEmbedder outputs exactly 0 → DiT behaves as vanilla
        final_linear = self.mlp[-1]
        nn.init.zeros_(final_linear.weight)
        nn.init.zeros_(final_linear.bias)

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(self, face_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            face_emb: [B, 512] ArcFace embedding (zero vector if no face).

        Returns:
            [B, num_tokens, dim] gated identity tokens.
        """
        B = face_emb.shape[0]

        # Dropout: zero out entire face vector during training
        if self.training and self.state_dropout > 0:
            drop_mask = torch.rand(B, 1, device=face_emb.device) > self.state_dropout
            face_emb = face_emb * drop_mask.float()

        tokens = self.mlp(face_emb)                          # [B, K * dim]
        tokens = tokens.reshape(B, self.num_tokens, self.dim) # [B, K, dim]

        # Gated output
        alpha = torch.sigmoid(self.alpha_logit)
        tokens = alpha * tokens

        return tokens
