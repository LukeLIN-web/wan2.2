#!/usr/bin/env bash
# Round-4 v3 lr=1e-5 control on juyi-finetune (4×A100). Mirrors v3 FSDP launcher with:
#   * lr 5e-5 → 1e-5 (training_config_round4_lr1e5.yaml, sha256[:16]=88276313696b6245)
#   * max_steps=200 → num_epochs=0.8 (same 200 optimizer steps on the 1k-pair/4-rank run)
#   * --nproc_per_node 8 → 4
# Decision: lukedecision task #39 — diagnose v3 FSDP lr=5e-5 vs lr=1e-5 stability at β=1000.
# Pins (round-4 r5 lr=1e-5 control): recipe=6bef6e104cdd3442, train_cfg=88276313696b6245, pair_ids=cf5d3e5fd528a3e0

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
# juyi-finetune env: torch 2.11.0+cu130, flash_attn 2.8.3, diffsynth, torchvision 0.26.0.
activate_env wmbench-swift-train
resolve_wandb
cd "$DPO_DIR"

expect_recipe="6bef6e104cdd3442"
expect_train_cfg="88276313696b6245"
expect_pair_ids="cf5d3e5fd528a3e0"
verify_pins "$expect_recipe" "$expect_train_cfg" "$expect_pair_ids" \
  "$DPO_DIR/recipes/training_config_round4_lr1e5_sha256_pin"

# juyi-finetune's worktree predates rl1's commit 52cfe68 ("split flat layout"),
# so the trainer may live at the pre-split path.
TRAINER_PY="$DPO_DIR/train/train_dpo_i2v.py"
[[ -f "$TRAINER_PY" ]] || TRAINER_PY="$DPO_DIR/train_dpo_i2v.py"

nohup torchrun --nproc_per_node=4 "$TRAINER_PY" \
  --tier tier_b \
  --upstream "$HOME/Wan2.2-I2V-A14B" \
  --latent-manifest "$DPO_DIR/latents/20260428T164038Z/tier_b_round4_1k/manifest.jsonl" \
  --post-t2-pair "$HOME/T0_T3_root/t2/post_t2_pair.json" \
  --t2-image-manifest "$HOME/T0_T3_root/t2/image_manifest.json" \
  --cond-image-fallback-root "$HOME/cond_imgs" \
  --training-config-path "$DPO_DIR/recipes/training_config_round4_lr1e5.yaml" \
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
  --wandb-run-name "round4-lr1e5-control-juyi-finetune-4rank-$(date -u +%Y%m%dT%H%M%SZ)" \
  > "$LOG_FILE" 2>&1 &

print_launch_info "$!"
