# Wan2.2 Architecture

## Overview

Wan2.2 是统一的 DiT (Diffusion Transformer) 视频生成框架，支持 T2V / I2V / TI2V / S2V 等任务。核心特点：**纯 Transformer，无独立 temporal attention / motion modules / 3D conv**。

代码位置：`wan/modules/model.py`

---

## 1. WanModel

```
WanModel (ModelMixin, ConfigMixin)
├── patch_embedding: Conv3d(in_dim, dim, kernel=patch_size)
├── text_embedding: Linear(text_dim→dim) → SiLU → Linear(dim→dim)
├── time_embedding: sinusoidal → Linear(freq_dim→dim) → SiLU → Linear(dim→dim)
├── N × WanAttentionBlock
└── head: LayerNorm → Linear(dim→patch_prod*out_dim)
```

### 模型配置

| Variant | dim | ffn_dim | heads | layers | patch_size | VAE stride | in_dim |
|---------|-----|---------|-------|--------|------------|------------|--------|
| T2V A14B | 5120 | 13824 | 40 | 40 | (1,2,2) | (4,8,8) | 16 |
| I2V A14B | 5120 | 13824 | 40 | 40 | (1,2,2) | (4,8,8) | 16 |
| TI2V 5B | 3072 | 14336 | 24 | 30 | (1,2,2) | (4,16,16) | 48 |
| S2V 14B | 5120 | 13824 | 40 | 40 | (1,2,2) | (4,8,8) | 16 |

### Forward 流程

输入：`x: List[Tensor]` [C, F, H, W], `t: [B]`, `context: List[Tensor]` [L, text_dim]

1. Patch embed: Conv3d → flatten → pad to seq_len
2. Time embed: sinusoidal + MLP → 6 个 modulation 向量
3. Text embed: Linear projection → pad to text_len=512
4. RoPE 计算：按 (temporal, H, W) 三维分配频率
5. N × WanAttentionBlock
6. Head: modulation + projection → unpatchify

---

## 2. WanAttentionBlock

```python
class WanAttentionBlock:
    norm1 → self_attn → (modulation gate)   # residual
    norm3 → cross_attn                       # residual
    norm2 → ffn → (modulation gate)          # residual
    modulation: Parameter [1, 6, dim]        # 6 个: shift/scale/gate × 2 (self_attn, ffn)
```

**Modulation**：时间条件通过 `e = (1 + modulation) * time_embed` 调制每个 block 的 norm shift/scale 和输出 gate。

---

## 3. Attention

代码：`wan/modules/attention.py`

### Self-Attention

```python
class WanSelfAttention:
    q, k, v, o: Linear(dim, dim)
    norm_q, norm_k: WanRMSNorm  # QK norm
```

**关键：没有分离的 temporal / spatial attention**。所有 (F×H×W) patch 拼成一个序列，统一做 global self-attention。时空区分完全靠 RoPE。

### Cross-Attention

```python
class WanCrossAttention(WanSelfAttention):
    # Q from video, K/V from text
```

### Flash Attention

支持 Flash Attention v2/v3，可变长序列 (`seq_lens`)。

---

## 4. RoPE (3D Rotary Position Embedding)

时空分离完全由 RoPE 实现，head_dim 按比例分给三个维度：

```python
d_t = head_dim - 4 * (head_dim // 6)   # temporal（最大份额）
d_h = 2 * (head_dim // 6)               # spatial H
d_w = 2 * (head_dim // 6)               # spatial W
```

对 TI2V-5B (head_dim=128)：d_t=96, d_h=42, d_w=42 (有 overlap)。

每个维度独立计算 freq → complex rotation → 拼接。这使模型能区分时间和空间位置关系，无需独立 temporal attention。

---

## 5. A14B 双模型 (MoE)

T2V-A14B 和 I2V-A14B 使用两个独立的 WanModel：

- `high_noise_model`: t >= boundary（A14B boundary=0.875）→ 负责 global motion / layout
- `low_noise_model`: t < boundary → 负责 detail refinement

推理时按 timestep 切换。DPO 训练只微调 `low_noise_model`。

TI2V-5B 是单模型，无此机制。

---

## 6. VAE

### Wan2.1 VAE (T2V / I2V)

- z_dim=16, c_dim=128, stride=(4,8,8)
- Causal Conv3d（只用过去/当前帧）
- Encoder3d → bottleneck → Decoder3d

### Wan2.2 VAE (TI2V)

- z_dim=48, c_dim=160, stride=(4,16,16)
- 更高空间压缩率
- 48 通道 latent 有 hardcoded mean/std 归一化

---

## 7. T5 Text Encoder

UMT5-XXL：dim=4096, ffn=10240, heads=64, layers=24, vocab=256384 (multilingual)。

Relative position embedding (32 buckets, bidirectional)。输出 [L, 4096] → WanModel text_embedding 投影到 model dim。

---

## 8. Diffusion Schedule

Flow Matching：`x_t = (1-σ)*x_0 + σ*ε`，模型预测 flow velocity `v = ε - x_0`。

Sigma shift：`σ_shifted = shift * σ / (1 + (shift-1) * σ)`

| Variant | shift | sample_steps | guidance_scale |
|---------|-------|-------------|----------------|
| T2V A14B | 12.0 | 40 | (3.0, 4.0) |
| I2V A14B | 5.0 | 40 | (3.5, 3.5) |
| TI2V 5B | 5.0 | 50 | 5.0 |

Solver: FlowDPMSolver++ (order 2, midpoint) 或 FlowUniPC。

---

## 9. LoRA 可 Target 的模块

```
每个 WanAttentionBlock:
├── self_attn.q / k / v / o     ← LoRA target
├── cross_attn.q / k / v / o   ← LoRA target
└── ffn.0 / ffn.2              ← LoRA target
```

TI2V-5B 实测：LoRA rank=128 → 322.4M / 5322.2M = 6.1%。

**不存在**独立的 temporal attention / motion module / 3D conv 可以单独 target。
