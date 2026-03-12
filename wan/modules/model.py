# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.amp.autocast('cuda', enabled=False)
def rope_apply_cached_kv(pk, grid_sizes, freqs, temporal_shift):
    """Apply RoPE to cached (pre-RoPE) K with a temporal phase shift (TcRoPE).

    The cached K has shape [B, num_cached_tokens, num_heads, head_dim].
    We infer num_cached_frames from the token count and spatial dims (H, W)
    taken from grid_sizes. The temporal RoPE frequencies are shifted by
    `temporal_shift` positions, making the model perceive cached frames as
    coming from a different temporal phase.

    Args:
        pk: Cached pre-RoPE K, shape [B, L_cache, N, D]
        grid_sizes: Current grid sizes [B, 3] — we use (H, W) from here
        freqs: Full RoPE frequency table [max_seq, C/num_heads/2]
        temporal_shift: Integer phase offset for temporal frequencies
    Returns:
        pk with RoPE applied (phase-shifted), same shape
    """
    n = pk.size(2)
    c = pk.size(3) // 2

    # split freqs into temporal, H, W
    freq_parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f_cur, h, w) in enumerate(grid_sizes.tolist()):
        hw = int(h * w)
        num_cached_tokens = pk.size(1)
        num_cf = num_cached_tokens // hw
        if num_cf == 0:
            output.append(pk[i])
            continue

        seq_len = num_cf * hw
        pk_i = torch.view_as_complex(
            pk[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))

        j = temporal_shift
        freqs_i = torch.cat([
            freq_parts[0][j:j + num_cf].view(num_cf, 1, 1, -1).expand(num_cf, int(h), int(w), -1),
            freq_parts[1][:int(h)].view(1, int(h), 1, -1).expand(num_cf, int(h), int(w), -1),
            freq_parts[2][:int(w)].view(1, 1, int(w), -1).expand(num_cf, int(h), int(w), -1),
        ], dim=-1).reshape(seq_len, 1, -1)

        pk_i = torch.view_as_real(pk_i * freqs_i).flatten(2)
        # If there are extra padding tokens beyond seq_len, keep them unchanged
        if num_cached_tokens > seq_len:
            pk_i = torch.cat([pk_i, pk[i, seq_len:]])
        output.append(pk_i)

    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs,
                prev_kv=None, cache_kv=False, cache_frame_indices=None,
                cache_strip_rope=False, cache_tcrope_shift=0,
                prev_kv_head_mask=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            prev_kv(tuple, optional): (K, V) from previous shot, each [B, L_cache, N, D]
            cache_kv(bool): Whether to extract KV for caching
            cache_frame_indices(list, optional): Temporal frame indices to cache
            cache_strip_rope(bool): If True, cache K without RoPE (Plan B)
            cache_tcrope_shift(int): Temporal phase offset for TcRoPE on cached K.
                When > 0 and prev_kv contains pre-RoPE K, applies RoPE with
                shifted temporal frequencies before injection.
            prev_kv_head_mask(Tensor, optional): Boolean mask [num_heads] selecting
                which heads receive injected KV. Non-selected heads' KV is zeroed.
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q_rope = rope_apply(q, grid_sizes, freqs)
        k_rope = rope_apply(k, grid_sizes, freqs)

        # Head importance profiling: record per-head face attention affinity
        if hasattr(self, '_profile_bbox_indices') and self._profile_bbox_indices is not None:
            with torch.no_grad():
                bbox_idx = self._profile_bbox_indices.to(q_rope.device)
                hw = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
                q_f0 = q_rope[0, :hw].float()       # [H*W, n, d]
                k_bbox = k_rope[0, bbox_idx].float()  # [num_bbox, n, d]
                # Per-head mean dot product: higher = head attends more to face
                scores = torch.einsum('qnd,knd->nqk', q_f0, k_bbox) / math.sqrt(d)
                self._profile_importance = scores.mean(dim=(1, 2)).cpu()  # [n]

        # Extract frame KV for inter-shot caching
        # Plan A (cache_strip_rope=False): cache K with RoPE
        # Plan B (cache_strip_rope=True): cache K without RoPE (semantic-only)
        cached_kv_out = None
        if cache_kv and cache_frame_indices is not None:
            cache_k_source = k if cache_strip_rope else k_rope
            cached_kv_out = self._extract_frame_kv(
                cache_k_source, v, grid_sizes, cache_frame_indices)

        # Inject previous shot's KV cache
        actual_k = k_rope
        actual_v = v
        actual_k_lens = seq_lens

        if prev_kv is not None:
            prev_k, prev_v = prev_kv
            # TcRoPE: apply RoPE with temporal phase shift to pre-RoPE cached K
            if cache_tcrope_shift > 0:
                prev_k = rope_apply_cached_kv(
                    prev_k, grid_sizes, freqs, cache_tcrope_shift)
            # Head-wise injection mask: zero out non-selected heads
            if prev_kv_head_mask is not None:
                mask = prev_kv_head_mask.view(1, 1, n, 1).to(
                    device=prev_k.device, dtype=prev_k.dtype)
                prev_k = prev_k * mask
                prev_v = prev_v * mask
            actual_k = torch.cat([prev_k, k_rope], dim=1)
            actual_v = torch.cat([prev_v, v], dim=1)
            actual_k_lens = seq_lens + prev_k.size(1)

        x = flash_attention(
            q=q_rope,
            k=actual_k,
            v=actual_v,
            k_lens=actual_k_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x, cached_kv_out

    def _extract_frame_kv(self, k, v, grid_sizes, frame_indices):
        """Extract KV tokens for specific temporal frames."""
        results_k, results_v = [], []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            hw = int(h * w)
            indices = []
            for fi in frame_indices:
                if fi < f:
                    start = int(fi * hw)
                    indices.extend(range(start, start + hw))
            idx_t = torch.tensor(indices, device=k.device, dtype=torch.long)
            results_k.append(k[i].index_select(0, idx_t))
            results_v.append(v[i].index_select(0, idx_t))
        return (torch.stack(results_k), torch.stack(results_v))


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        prev_kv=None,
        cache_kv=False,
        cache_frame_indices=None,
        cache_strip_rope=False,
        cache_tcrope_shift=0,
        prev_kv_head_mask=None,
        face_context=None,
        face_context_lens=None,
        style_shift=None,
        style_scale=None,
        predx0_hook=None,
        ref_attn_hook=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            prev_kv(tuple, optional): Previous shot KV for inter-shot attention
            cache_kv(bool): Whether to cache KV from this block
            cache_frame_indices(list, optional): Frame indices to cache
            cache_strip_rope(bool): If True, cache K without RoPE (Plan B)
            cache_tcrope_shift(int): Temporal phase offset for TcRoPE on cached K
            prev_kv_head_mask(Tensor, optional): Boolean mask [num_heads] for
                head-wise KV injection. Non-selected heads' cached KV is zeroed.
            face_context(Tensor, optional): Context with face tokens appended,
                shape [B, L_text + L_face, C]. Used only in face injection layers.
            face_context_lens(Tensor, optional): Corresponding context lengths
                for face_context.
            style_shift(Tensor, optional): Additive shift for self-attn input,
                shape [B, C]. From StyleAdapter, broadcast over seq_len.
            style_scale(Tensor, optional): Additive scale for self-attn input,
                shape [B, C]. From StyleAdapter, broadcast over seq_len.
            predx0_hook(callable, optional): Called after self-attention residual,
                before cross-attention. Receives x, returns modified x.
                Used for Pred-x0 feature caching/injection.
            ref_attn_hook(callable, optional): Called after predx0_hook,
                before cross-attention. Computes reference cross-attention
                using cached Shot 0 K,V and adds as scaled residual.
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        sa_input = self.norm1(x).float()
        sa_scale = 1 + e[1].squeeze(2)
        sa_shift = e[0].squeeze(2)
        if style_shift is not None:
            sa_shift = sa_shift + style_shift.float().unsqueeze(1)
            sa_scale = sa_scale + style_scale.float().unsqueeze(1)
        y, cached_kv = self.self_attn(
            (sa_input * sa_scale + sa_shift).to(x.dtype),
            seq_lens, grid_sizes, freqs,
            prev_kv=prev_kv, cache_kv=cache_kv,
            cache_frame_indices=cache_frame_indices,
            cache_strip_rope=cache_strip_rope,
            cache_tcrope_shift=cache_tcrope_shift,
            prev_kv_head_mask=prev_kv_head_mask)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # Pred-x0 hook: after self-attn residual, before cross-attn
        if predx0_hook is not None:
            x = predx0_hook(x)

        # Reference attention hook: separate cross-attn with Shot 0 K,V
        if ref_attn_hook is not None:
            x = ref_attn_hook(x)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        # Use face-augmented context if provided for this block
        ctx = face_context if face_context is not None else context
        ctx_lens = face_context_lens if face_context_lens is not None else context_lens
        x = cross_attn_ffn(x, ctx, ctx_lens, e)
        return x, cached_kv


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


def _build_predx0_hook(predx0_config, block_idx, device):
    """Build a hook for Pred-x0 feature caching or injection at a specific block."""
    mode = predx0_config.get('mode')

    if mode == 'cache':
        cache_indices = predx0_config['cache_indices'].to(device)
        step_cache = predx0_config['_step_cache']

        def cache_hook(x):
            step_cache[block_idx] = x[0, cache_indices, :].detach().float().cpu()
            return x

        return cache_hook

    if mode == 'inject':
        step_cache = predx0_config.get('step_cache')
        if step_cache is None or block_idx not in step_cache:
            return None

        cached_feat = step_cache[block_idx].to(device=device)
        inject_indices = predx0_config['inject_indices'].to(device)
        inject_weights = predx0_config['inject_weights'].to(device)
        alpha = predx0_config.get('alpha', 0.05)
        n_face = len(predx0_config['cache_indices'])

        # Identity-normalized: subtract mean across face tokens
        identity = cached_feat - cached_feat.mean(dim=0, keepdim=True)

        # Expand identity to all non-first frames
        n_frames_inject = len(inject_indices) // n_face
        identity_expanded = identity.unsqueeze(0).expand(
            n_frames_inject, -1, -1
        ).reshape(-1, identity.size(-1))

        def inject_hook(x):
            delta = alpha * inject_weights.unsqueeze(-1) * identity_expanded.to(x.dtype)
            x[:, inject_indices, :] = x[:, inject_indices, :] + delta
            return x

        return inject_hook

    return None


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        state_embedding=None,
        face_embedding=None,
        face_inject_layers=None,
        intershot_kv_cache=None,
        cache_kv=False,
        cache_frame_indices=None,
        intershot_layers=None,
        cache_strip_rope=False,
        cache_tcrope_shift=0,
        intershot_head_mask=None,
        style_modulation=None,
        style_inject_layers=None,
        predx0_config=None,
        ref_attn_config=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            state_embedding (Tensor, *optional*):
                State embedding tokens from StateEmbedder, shape [B, K, dim].
                Concatenated to text context for cross-attention conditioning.
            face_embedding (Tensor, *optional*):
                Gated face identity tokens from FaceEmbedder, shape [B, K_face, dim].
                Injected into cross-attention context only in face_inject_layers.
            face_inject_layers (set, *optional*):
                Set of layer indices where face tokens are injected (default: {15..20}).
            intershot_kv_cache (dict, optional):
                {layer_idx: (K, V)} from previous shot
            cache_kv (bool):
                Whether to extract KV cache from this forward pass
            cache_frame_indices (list, optional):
                Temporal frame indices to cache
            intershot_layers (set, optional):
                Set of layer indices that participate in inter-shot attention
            cache_strip_rope (bool):
                If True, cache K without RoPE (Plan B: semantic-only matching)
            cache_tcrope_shift (int):
                Temporal phase offset for TcRoPE on cached K (0 = no shift)
            intershot_head_mask (dict, optional):
                {layer_idx: BoolTensor[num_heads]} selecting which heads receive
                injected AnchorKV. Non-selected heads' cached KV is zeroed out.
            style_modulation (list, optional):
                List of (Δshift, Δscale) tuples from StyleAdapter, one per style layer.
                Each Δshift/Δscale has shape [B, dim].
            style_inject_layers (set, optional):
                Set of layer indices for style injection (default: {0..10}).
            predx0_config (dict, optional):
                Pred-x0 feature injection config. Keys:
                - mode: 'cache' or 'inject'
                - layers: set of layer indices
                - cache_indices: LongTensor, frame-0 face token indices
                - _step_cache: mutable dict, populated during cache mode
                - step_cache: dict {layer: tensor}, features to inject
                - inject_indices: LongTensor, non-first-frame face indices
                - inject_weights: FloatTensor, soft Gaussian mask
                - alpha: float, blend weight
            ref_attn_config (dict, optional):
                Reference attention config for cross-shot identity. Keys:
                - layers: set of layer indices (default: {15..20})
                - kv_cache: dict {layer_idx: (K, V)}, cached Shot 0 K,V
                - alpha: float, blending strength (default: 0.3)

        Returns:
            List[Tensor] or (List[Tensor], dict):
                Denoised video tensors. If cache_kv=True, also returns new KV cache dict.
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # inject state embedding (cross-attn conditioning)
        if state_embedding is not None:
            context = torch.cat([context, state_embedding], dim=1)

        # build face-augmented context (used only in face_inject_layers)
        _face_context = None
        _face_context_lens = None
        if face_embedding is not None:
            from .face_embedder import FACE_INJECT_LAYERS
            _face_inject = face_inject_layers if face_inject_layers is not None else FACE_INJECT_LAYERS
            _face_context = torch.cat([context, face_embedding], dim=1)
        else:
            _face_inject = set()

        # build style modulation lookup (used only in style_inject_layers)
        if style_modulation is not None:
            from .style_adapter import STYLE_INJECT_LAYERS
            _style_inject = style_inject_layers if style_inject_layers is not None else STYLE_INJECT_LAYERS
            _style_inject_min = min(_style_inject)
        else:
            _style_inject = set()
            _style_inject_min = 0

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)

        new_cache = {}
        for block_idx, block in enumerate(self.blocks):
            use_intershot = (intershot_layers is not None
                             and block_idx in intershot_layers)

            prev_kv = None
            if use_intershot and intershot_kv_cache is not None:
                prev_kv = intershot_kv_cache.get(block_idx)
                if prev_kv is not None:
                    pk, pv = prev_kv
                    prev_kv = (pk.to(device=x.device),
                               pv.to(device=x.device))

            do_cache = cache_kv and use_intershot

            # Pass face-augmented context only to face injection layers
            face_kw = {}
            if block_idx in _face_inject:
                face_kw = dict(face_context=_face_context,
                               face_context_lens=_face_context_lens)

            # Pass style modulation only to style injection layers
            style_kw = {}
            if block_idx in _style_inject:
                layer_offset = block_idx - _style_inject_min
                style_kw = dict(style_shift=style_modulation[layer_offset][0],
                                style_scale=style_modulation[layer_offset][1])

            # Pred-x0 feature hook
            predx0_hook = None
            if predx0_config is not None and block_idx in predx0_config.get('layers', set()):
                predx0_hook = _build_predx0_hook(predx0_config, block_idx, device)

            # Reference attention hook
            ref_hook = None
            if ref_attn_config is not None and block_idx in ref_attn_config.get('layers', set()):
                ref_kv = ref_attn_config.get('kv_cache', {}).get(block_idx)
                if ref_kv is not None:
                    from ..ref_attention import build_ref_attn_hook
                    ref_hook = build_ref_attn_hook(
                        block, ref_kv[0], ref_kv[1],
                        alpha=ref_attn_config.get('alpha', 0.3))

            # Head-wise injection mask for this layer
            layer_head_mask = None
            if intershot_head_mask is not None and block_idx in intershot_head_mask:
                layer_head_mask = intershot_head_mask[block_idx]

            x, cached_kv = block(
                x, **kwargs,
                prev_kv=prev_kv,
                cache_kv=do_cache,
                cache_frame_indices=cache_frame_indices,
                cache_strip_rope=cache_strip_rope,
                cache_tcrope_shift=cache_tcrope_shift if use_intershot else 0,
                prev_kv_head_mask=layer_head_mask,
                **face_kw,
                **style_kw,
                predx0_hook=predx0_hook,
                ref_attn_hook=ref_hook)

            if cached_kv is not None:
                new_cache[block_idx] = cached_kv

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        if cache_kv:
            return [u.float() for u in x], new_cache
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
