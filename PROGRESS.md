# PROGRESS — 经验教训沉淀

每条记录附 commit ID，避免同样的问题犯两次。不要写这么多废话, 就写正确方法就行. 不要写别的.

---

## 2026-04-28 — DPO 14B step-0 OOM / cold boot

**Commit**: `e05a9a5`

- 80GB 卡跑 14B+LoRA DPO: 默认 `--dit-fsdp true`，用 `wan/distributed/fsdp.py::shard_model(use_lora=True)`。
- FSDP 下保存 LoRA: `FSDP.summon_full_params(policy, writeback=False, rank0_only=True)`，全 rank 进 collective，rank0 在 with 内做 IO。
- 全 rank 重复的 init 编码 (cond VAE / prompt T5)：rank0 encode + 原子写盘 + `dist.barrier()` + 全 rank 读盘。cache key 含 asset sha256 + shape。
- 看到 `expandable_segments not supported on this platform` 直接换方案，不要寄望它生效。
