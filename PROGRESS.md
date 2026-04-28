# PROGRESS — 经验教训沉淀

每条记录附 commit ID，避免同样的问题犯两次。

---

## 2026-04-28 — DPO round-4 step-0 OOM + 12 min cold boot

**Commit**: `e05a9a5` — feat(dpo_v0/train): FSDP for policy + disk cache for cond/T5

### 问题 1: step-0 OOM (78.68 / 80 GB)

**症状**: 8×80GB H100 上 round-4 trainer 在 step-0 forward 第二个 policy pass 中爆，已经
开了 grad-ckpt + sequential DPO + ref-via-disabled-LoRA (单 model + scalar grad-coef
两个独立 backward) — 这已是 DPO 最省显存的拓扑，仍卡在 ~78.7 GB。

**根因**: 14B Wan2.2-I2V bf16 = ~28 GB, **每张卡都持久占一份** (DDP). passA 完后
activation 不能释 (要等 DPO loss)，passB forward 时同时 hold 两份 activations，alloc
27 → 56 GB；加上 9.86 GB allocator fragmentation，超 80 GB cap。

**走过的弯路**:
- v2: 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — log 里出
  `Warning: expandable_segments not supported on this platform`，**这台机器的驱动/CUDA
  组合不支持这个 flag**，等于 no-op。日后看到这条 warning 直接放弃这条路。

**解法**: `--dit-fsdp true`，`wan/distributed/fsdp.py::shard_model(use_lora=True)`
把 policy WanModel 包成 FSDP FULL_SHARD with `use_orig_params=True`。
- LoRA Parameter 对象在 FSDP wrap 后仍是同一组 Python 对象 (`use_orig_params=True`
  的关键)，AdamW + `lora_disabled()` context 全部继续工作
- 14B / 8 rank = 3.5 GB/rank 持久占用 (vs DDP 的 28 GB)
- 实测 step-0 post-passB-fwd peak 从 65.7 GB → 43.2 GB，整体 vram_peak 58.3 GB

### 问题 2: FSDP 下 collect_lora_state 拿不到 full tensor

**根因**: `use_orig_params=True` 让 LoRA Parameter 仍可见，但 `.data` 是 sharded local
slice。直接 save 会写出残缺文件。

**解法**: 新 `_save_lora()` 在 FSDP 路径下走
`FSDP.summon_full_params(policy, writeback=False, rank0_only=True)` — collective 必须
全 rank 进入，rank0 拿全量 tensor 写盘，其它 rank 看到空。原本 `if is_main and ...:`
的 save gate 必须放宽到全 rank 都进入 collective，rank0 才在 with 里做 IO。

**坑**: 同样适用于 final-save。原 final-save 在 `if is_main:` 块里包含 ckpt + manifest
两件事，要拆开 — `_save_lora(ckpt_path)` 必须全 rank 调用，manifest 仍 rank0-only。

**验证 (smoke test)**: `--save-every 1` 跑 step 0/1/2，三次 save 文件 size 完全一致
(146 MB)，与 LoRA rank=16 × 400 modules × bf16 计算值匹配 → 没漏写 / 没只写 shard。

### 问题 3: cold boot 12 分钟

**根因**: `train_dpo_i2v.py` 里 cond-encode 是单 rank 内串行 for-loop，**且 8 个 rank
全部独立编码同样 164 张 cond image**，VAE 是确定性的所以输出 byte-identical，纯浪费。
T5 prompt-encode 同款问题。

**解法**: Tier 1 disk cache，`humanize/dpo_v0/cache/{cond_latent,prompt_t5}/<asset_sha256[:16]>_<shape>/<key>.pt`：
- 命名包含 VAE/T5/tokenizer sha256 + target_w/h/frame_num/text_len，asset 一漂 cache
  自动失效
- rank0 encode + 原子 publish (`tmp.replace(final)`)，`dist.barrier()` 等待，全 rank
  从 disk 加载
- **全 hit 时连 VAE / T5 都不 load** (那条耗时分支整个跳过)

**收益**: 第一次 cold ~12 min (rank0 only 编码替代 8× 重复); 第二次起 kill-restart
~3 min (`[cond-cache] full hit — skipping VAE load entirely` + 同款 T5)。

### 后续避雷

- 看到 `expandable_segments not supported on this platform` warning → 立即换方案，
  不要寄望它生效
- 80GB 卡跑 14B+ LoRA DPO → 默认带 `--dit-fsdp true`
- 任何 `summon_full_params` 调用必须**全 rank 进入**，IO 在 with 内 rank0-only
- 任何 "for in dict over identical work × N rank" 模式 → 强制 disk cache + dist.barrier
