#!/usr/bin/env bash
# Round-4 tier_b 1k DPO training (v3 — FSDP for policy WanModel) on 8 ranks.
# r5 pins: recipe=6bef6e104cdd3442, train_cfg=7a265387fb8eef44, pair_ids=cf5d3e5fd528a3e0
# beta lowered 1000 → 100 after run 9if26kr9 showed sigmoid saturation + grad spikes (see docs/).

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env wan
resolve_wandb
cd "$DPO_DIR"

expect_recipe="6bef6e104cdd3442"
expect_train_cfg="7a265387fb8eef44"
expect_pair_ids="cf5d3e5fd528a3e0"
verify_pins "$expect_recipe" "$expect_train_cfg" "$expect_pair_ids" \
  "$DPO_DIR/recipes/training_config_round4_beta100_sha256_pin"

nohup torchrun --nproc_per_node=8 train/train_dpo_i2v.py \
  --tier tier_b \
  --upstream "$HOME/Wan2.2-I2V-A14B" \
  --latent-manifest "$DPO_DIR/latents/20260428T164038Z/tier_b_round4_1k/manifest.jsonl" \
  --post-t2-pair "$HOME/T0_T3_root/t2/post_t2_pair.json" \
  --t2-image-manifest "$HOME/T0_T3_root/t2/image_manifest.json" \
  --cond-image-fallback-root "$HOME/cond_imgs" \
  --training-config-path "$DPO_DIR/recipes/training_config_round4_beta100.yaml" \
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
  > "$LOG_FILE" 2>&1 &

print_launch_info "$!"
