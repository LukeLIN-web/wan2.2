#!/usr/bin/env bash
# Round-4 v3 lr=1e-5 β=100 sweep. Forks lr1e5 control with:
#   * beta 1000 → 100 (training_config_round4_lr1e5_beta100.yaml, sha256[:16]=19c2a6d6bba0a59e)
#   * everything else identical (lr=1e-5, lora_rank=16, num_samples=800 → 200 steps at 4 ranks)
# nproc_per_node auto-detected from nvidia-smi; override with NPROC_PER_NODE env var.
# Decision: diagnose β=1000 lr=1e-5 control (run 54xoj0uw) failure mode — clip at max_grad_norm=1.0
# fires nearly every step under β=1000, degenerating updates to sign-SGD; β=100 should drop raw
# grad ~10x and let real magnitude through. Pair with β=10 sweep to bracket the right scale.
# Pins: recipe=6bef6e104cdd3442, train_cfg=19c2a6d6bba0a59e, pair_ids=cf5d3e5fd528a3e0

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env wmbench-swift-train
resolve_wandb
cd "$DPO_DIR"

expect_recipe="6bef6e104cdd3442"
expect_train_cfg="19c2a6d6bba0a59e"
expect_pair_ids="cf5d3e5fd528a3e0"
verify_pins "$expect_recipe" "$expect_train_cfg" "$expect_pair_ids" \
  "$DPO_DIR/recipes/training_config_round4_lr1e5_beta100_sha256_pin"

TRAINER_PY="$DPO_DIR/train/train_dpo_i2v.py"
[[ -f "$TRAINER_PY" ]] || TRAINER_PY="$DPO_DIR/train_dpo_i2v.py"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l)"
  fi
  NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
  [[ "$NPROC_PER_NODE" -ge 1 ]] || NPROC_PER_NODE=1
fi
echo "[launch] nproc_per_node=$NPROC_PER_NODE"

nohup torchrun --nproc_per_node="$NPROC_PER_NODE" "$TRAINER_PY" \
  --tier tier_b \
  --upstream "$HOME/Wan2.2-I2V-A14B" \
  --latent-manifest "$DPO_DIR/latents/20260428T164038Z/tier_b_round4_1k/manifest.jsonl" \
  --post-t2-pair "$HOME/T0_T3_root/t2/post_t2_pair.json" \
  --t2-image-manifest "$HOME/T0_T3_root/t2/image_manifest.json" \
  --cond-image-fallback-root "$HOME/cond_imgs" \
  --training-config-path "$DPO_DIR/recipes/training_config_round4_lr1e5_beta100.yaml" \
  --training-config-sha256-pin "$expect_train_cfg" \
  --subset-pair-ids-json "$DPO_DIR/out/round4/20260428T160839Z/T3_round4_tier_b_1k.json" \
  --pair-ids-sha256-pin "$expect_pair_ids" \
  --enable-grad-ckpt true \
  --dit-fsdp true \
  --halt-on-low-noise true \
  --out-dir "$DPO_DIR/ckpts" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-mode "$WANDB_MODE_OPT" \
  --wandb-run-name "round4-lr1e5-beta100-${NPROC_PER_NODE}rank-$(date -u +%Y%m%dT%H%M%SZ)" \
  > "$LOG_FILE" 2>&1 &

print_launch_info "$!"
