# PROGRESS — 经验教训沉淀

每条记录附 commit ID，避免同样的问题犯两次。不要写这么多废话, 就写正确方法就行. 不要写别的.

---

## 2026-04-28 — DPO trainer 接 wandb

**Commit**: `0f88d69`

- `train_dpo_i2v.py`: 新增 `--wandb-{project,entity,mode,run-name}` 四个 CLI；`_wandb_init` 仅在 rank0 调用，import / init / log / finish 全 try-except，wandb 任何失败都不影响训练。
- 每 `--log-every` 步 `wandb.log({loss, t_raw, logit, delta, mse_pi/ref_{w,l}, c_w, vram_peak})`；收尾把 `manifest` 全写到 `wandb.summary` 再 `finish()`。
- launcher 默认 `WANDB_PROJECT=wanrl WANDB_ENTITY=lukelin`；没 `WANDB_API_KEY` 也没 `~/.netrc` 就自动降级 `offline`。
- 不要把 wandb log 放在所有 rank：FSDP 下非 rank0 调用会触发 wandb 多进程冲突；`if is_main:` 守好。

---

## 2026-04-28 — DPO 14B step-0 OOM / cold boot

**Commit**: `e05a9a5`

- 80GB 卡跑 14B+LoRA DPO: 默认 `--dit-fsdp true`，用 `wan/distributed/fsdp.py::shard_model(use_lora=True)`。
- FSDP 下保存 LoRA: `FSDP.summon_full_params(policy, writeback=False, rank0_only=True)`，全 rank 进 collective，rank0 在 with 内做 IO。
- 全 rank 重复的 init 编码 (cond VAE / prompt T5)：rank0 encode + 原子写盘 + `dist.barrier()` + 全 rank 读盘。cache key 含 asset sha256 + shape。

---

## 2026-04-28 — Inference startup shard sha 重计算 / smoke pin 太严

**Commit**: `8a531e2`

- 上游 shard 的 ckpt sha 走 `humanize/dpo_v0/file_sha_cache.py`，sidecar `<upstream_root>/file_sha.cache.json`，key=(size, mtime_ns)；聚合 byte stream 与 cold recompute 一致，AC-7.3 manifest 字节稳定。
- 入口 `inference_smoke.py` 不再持有缓存实现，所有 inference 路径 (含 `heldout_regen` transitively) 共享同一 util。
- argparse 路径不再硬钉 `--num-frames` / `--fps`，eval 通过 `--gen-config-json` 仍然钉 81/16；smoke 可下调 `(n-1) %% 4 == 0` 的合法值。
- 老 (`dpo_v0/inference_smoke.py`) 与新 (`dpo_v0/eval/inference_smoke.py`) layout 都能 import：用 `importlib.util.spec_from_file_location` 走父目录探测；`RECIPES_DIR` 同样探测 `recipes/recipe_id` sidecar。
