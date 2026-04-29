#!/usr/bin/env bash
# Round-4 Option D smoke (1-step, β=0.1, lr=1e-5, lora_rank=8, FSDP).
# Hammer 1 invariant: with LoRA B init = 0, policy ≡ ref ⇒ DPO loss = ln(2) ≈ 0.6931.
#   loss = 0.6931 ± 1e-3 → pass (safe to relaunch under Option A)
#   otherwise           → FSDP shard breaks ref forward; halt.
# Pins (round-4 r6 smokeD): recipe=6bef6e104cdd3442, train_cfg=9f1aafc8ddc66884, pair_ids=cf5d3e5fd528a3e0

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env wan
resolve_wandb
cd "$DPO_DIR"

expect_recipe="6bef6e104cdd3442"
expect_train_cfg="9f1aafc8ddc66884"
expect_pair_ids="cf5d3e5fd528a3e0"
verify_pins "$expect_recipe" "$expect_train_cfg" "$expect_pair_ids" \
  "$DPO_DIR/recipes/training_config_smokeD_sha256_pin"

nohup torchrun --nproc_per_node=8 train/train_dpo_i2v.py \
  --tier tier_b \
  --upstream "$HOME/Wan2.2-I2V-A14B" \
  --latent-manifest "$DPO_DIR/latents/20260428T164038Z/tier_b_round4_1k/manifest.jsonl" \
  --post-t2-pair "$HOME/T0_T3_root/t2/post_t2_pair.json" \
  --t2-image-manifest "$HOME/T0_T3_root/t2/image_manifest.json" \
  --cond-image-fallback-root "$HOME/cond_imgs" \
  --training-config-path "$DPO_DIR/recipes/training_config_smokeD.yaml" \
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
echo "verify: grep -E 'step 0.*loss=' $LOG_FILE  # expect loss = 0.6931 ± 1e-3 (Hammer 1 invariant)"
