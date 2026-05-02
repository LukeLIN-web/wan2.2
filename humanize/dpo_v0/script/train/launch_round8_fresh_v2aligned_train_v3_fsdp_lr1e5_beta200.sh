#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env wan
resolve_wandb
cd "$DPO_DIR"

expect_recipe="6bef6e104cdd3442"          # inherited recipe_id pin
expect_train_cfg="3d3ebb6907c48501"       # sha256[:16] of round-8 fresh v2aligned beta200 yaml
expect_pair_ids="a262b7153f58c37f"        # sampler output, b-mode=repeat (982 unique, 42 B-class dups)

ROUND7_OUT_DIR="${ROUND7_OUT_DIR:-$(ls -1d "$DPO_DIR"/out/round7/*/ 2>/dev/null | sort | tail -1 | sed 's:/$::')}"
SUBSET_PAIR_IDS_JSON="${SUBSET_PAIR_IDS_JSON:-${ROUND7_OUT_DIR}/T3_round7_v2aligned_1024.json}"
PAIR_IDS_PIN_FILE="${PAIR_IDS_PIN_FILE:-${ROUND7_OUT_DIR}/pair_ids_round7_v2aligned_1024_sha256_hex16_pin}"
export PAIR_IDS_PIN_FILE

if [[ ! -f "$SUBSET_PAIR_IDS_JSON" ]]; then
  echo "FATAL: SUBSET_PAIR_IDS_JSON not found: $SUBSET_PAIR_IDS_JSON" >&2
  echo "       Run script/sample/round7_v2aligned_1024.py first." >&2
  exit 3
fi
if [[ ! -f "$PAIR_IDS_PIN_FILE" ]]; then
  echo "FATAL: PAIR_IDS_PIN_FILE not found: $PAIR_IDS_PIN_FILE" >&2
  exit 3
fi
verify_pins "$expect_recipe" "$expect_train_cfg" "$expect_pair_ids" \
  "$DPO_DIR/recipes/training_config_round8_fresh_v2aligned_beta200_sha256_pin"

LATENT_MANIFEST="${LATENT_MANIFEST:-}"
if [[ -z "$LATENT_MANIFEST" ]]; then
  LATENT_MANIFEST="$(ls -1 "$DPO_DIR"/latents/*/tier_b_round7_v2aligned_1024/manifest.jsonl 2>/dev/null | sort | tail -1)"
fi
if [[ ! -f "$LATENT_MANIFEST" ]]; then
  echo "FATAL: LATENT_MANIFEST not found: $LATENT_MANIFEST" >&2
  echo "       Build the union manifest first (concat round-4 + round-5 manifests, filter to tier_b_round7_v2aligned_1024)." >&2
  exit 4
fi
echo "[round8] LATENT_MANIFEST=$LATENT_MANIFEST"
echo "[round8] SUBSET_PAIR_IDS_JSON=$SUBSET_PAIR_IDS_JSON"

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

MASTER_PORT="${MASTER_PORT:-29500}"

nohup torchrun --master-port "$MASTER_PORT" --nproc_per_node="$NPROC_PER_NODE" "$TRAINER_PY" \
  --tier tier_b \
  --upstream "$HOME/Wan2.2-I2V-A14B" \
  --latent-manifest "$LATENT_MANIFEST" \
  --post-t2-pair "$HOME/T0_T3_root/t2/post_t2_pair.json" \
  --t2-image-manifest "$HOME/T0_T3_root/t2/image_manifest.json" \
  --cond-image-fallback-root "$HOME/cond_imgs" \
  --training-config-path "$DPO_DIR/recipes/training_config_round8_fresh_v2aligned_beta200.yaml" \
  --training-config-sha256-pin "$expect_train_cfg" \
  --subset-pair-ids-json "$SUBSET_PAIR_IDS_JSON" \
  --pair-ids-sha256-pin "$expect_pair_ids" \
  --allow-repeated-pair-ids true \
  --save-optimizer-state true \
  --save-every 32 \
  --enable-grad-ckpt true \
  --dit-fsdp true \
  --halt-on-low-noise true \
  --out-dir "$DPO_DIR/ckpts" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-mode "$WANDB_MODE_OPT" \
  --wandb-run-name "round8-fresh-v2aligned-lr1e5-beta200-${NPROC_PER_NODE}rank-$(date -u +%Y%m%dT%H%M%SZ)" \
  > "$LOG_FILE" 2>&1 &

print_launch_info "$!"
