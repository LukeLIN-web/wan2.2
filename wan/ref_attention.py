"""
Reference Attention for cross-shot identity preservation (training-free).

Unlike intershot KV concat (which mixes reference tokens into self-attention
and conflicts with RoPE), Reference Attention computes a SEPARATE cross-attention
using the DiT's existing self-attention weights and adds the result as a
scaled residual.

Flow:
  Shot 0: generate normally, cache first-frame K,V (pre-RoPE) at last step
          using existing intershot cache mechanism.
  Shot N: at each denoising step, for layers in REF_ATTN_LAYERS:
          1. Normalize current hidden state with block.norm1
          2. Q = self_attn.norm_q(self_attn.q(normalized_x))  [no RoPE]
          3. K_ref, V_ref = cached from Shot 0  [no RoPE]
          4. out = flash_attention(Q, K_ref, V_ref)  [pure semantic matching]
          5. out = self_attn.o(out)
          6. x = x + alpha * out
"""

import torch
from .modules.attention import flash_attention

# Default layers for reference attention (middle layers, identity-sensitive)
REF_ATTN_LAYERS = {15, 16, 17, 18, 19, 20}

# Default blending strength
DEFAULT_ALPHA = 0.3


def build_ref_attn_hook(block, ref_k, ref_v, alpha=DEFAULT_ALPHA):
    """Build a reference cross-attention hook for a specific block.

    Args:
        block: WanAttentionBlock instance (captures self_attn weights)
        ref_k: Cached reference K [B, L_ref, N, D], pre-RoPE, on CPU
        ref_v: Cached reference V [B, L_ref, N, D], on CPU
        alpha: Blending strength for the residual

    Returns:
        Callable hook: x -> x + alpha * ref_cross_attn(x, ref_k, ref_v)
    """
    sa = block.self_attn
    norm1 = block.norm1

    def hook(x):
        device = x.device
        b = x.size(0)
        n, d = sa.num_heads, sa.head_dim

        # Normalize current features (match self-attention input distribution)
        x_norm = norm1(x).to(x.dtype)

        # Compute Q from current state (no RoPE — pure semantic matching)
        q = sa.norm_q(sa.q(x_norm)).view(b, -1, n, d)

        # Move reference K,V to device
        k = ref_k.to(device=device, dtype=q.dtype)
        v = ref_v.to(device=device, dtype=q.dtype)

        # Separate reference cross-attention
        out = flash_attention(q, k, v)
        out = out.flatten(2)
        out = sa.o(out)

        # Scaled residual
        return x + alpha * out

    return hook


def build_ref_attn_config_for_model(ref_kv_cache, layers=None, alpha=DEFAULT_ALPHA):
    """Build ref_attn_config dict to pass to WanModel.forward().

    Args:
        ref_kv_cache: dict {layer_idx: (K, V)} from Shot 0's intershot cache.
            K shape: [B, L_ref, N, D], V shape: [B, L_ref, N, D]
        layers: set of layer indices (default: REF_ATTN_LAYERS)
        alpha: blending strength

    Returns:
        dict suitable for WanModel.forward(ref_attn_config=...)
    """
    if ref_kv_cache is None:
        return None
    return {
        'layers': layers or REF_ATTN_LAYERS,
        'kv_cache': ref_kv_cache,
        'alpha': alpha,
    }
