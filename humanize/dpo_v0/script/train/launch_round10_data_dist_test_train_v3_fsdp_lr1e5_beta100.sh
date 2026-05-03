#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env wan
resolve_wandb
cd "$DPO_DIR"

expect_recipe="6bef6e104cdd3442"          # inherited recipe_id pin
expect_train_cfg="faedd80b62b45994"       # sha256[:16] of round-10 data-dist-test yaml
expect_pair_ids="cf5d3e5fd528a3e0"        # round-4 sampler output (1000 seed-shuffled, A-heavy natural distribution)

ROUND4_OUT_DIR="${ROUND4_OUT_DIR:-$(ls -1d "$DPO_DIR"/out/round4/*/ 2>/dev/null | sort | tail -1 | sed 's:/$::')}"
SUBSET_PAIR_IDS_JSON="${SUBSET_PAIR_IDS_JSON:-${ROUND4_OUT_DIR}/T3_round4_tier_b_1k.json}"
PAIR_IDS_PIN_FILE="${PAIR_IDS_PIN_FILE:-${ROUND4_OUT_DIR}/pair_ids_sha256_hex16_pin}"
export PAIR_IDS_PIN_FILE

if [[ ! -f "$SUBSET_PAIR_IDS_JSON" ]]; then
  echo "FATAL: SUBSET_PAIR_IDS_JSON not found: $SUBSET_PAIR_IDS_JSON" >&2
  echo "       Round-10 reuses round-4 pair_ids; ensure the round-4 sampler output is present" >&2
  echo "       under \$DPO_DIR/out/round4/<ts>/T3_round4_tier_b_1k.json (sha256[:16] cf5d3e5fd528a3e0)." >&2
  exit 3
fi
if [[ ! -f "$PAIR_IDS_PIN_FILE" ]]; then
  echo "FATAL: PAIR_IDS_PIN_FILE not found: $PAIR_IDS_PIN_FILE" >&2
  exit 3
fi
verify_pins "$expect_recipe" "$expect_train_cfg" "$expect_pair_ids" \
  "$DPO_DIR/recipes/training_config_round10_data_dist_test_beta100_sha256_pin"

LATENT_MANIFEST="${LATENT_MANIFEST:-}"
if [[ -z "$LATENT_MANIFEST" ]]; then
  LATENT_MANIFEST="$(ls -1 "$DPO_DIR"/latents/*/tier_b_round4_1k/manifest.jsonl 2>/dev/null | sort | tail -1)"
fi
if [[ ! -f "$LATENT_MANIFEST" ]]; then
  echo "FATAL: LATENT_MANIFEST not found: $LATENT_MANIFEST" >&2
  echo "       Round-10 reuses round-4's tier_b_round4_1k latent manifest unchanged (no re-staging)." >&2
  exit 4
fi
echo "[round10] LATENT_MANIFEST=$LATENT_MANIFEST"
echo "[round10] SUBSET_PAIR_IDS_JSON=$SUBSET_PAIR_IDS_JSON"

LATENT_MANIFEST_SHA="$(sha256sum "$LATENT_MANIFEST" | awk '{print $1}')"
SEED_NAMESPACE_FROM_YAML="$(awk -F': *' '/^seed_namespace:/{print $2}' \
  "$DPO_DIR/recipes/training_config_round10_data_dist_test_beta100.yaml")"

PRELAUNCH_MANIFEST="$LOG_DIR/round10_prelaunch_manifest_$(date -u +%Y%m%dT%H%M%SZ).json"
cat > "$PRELAUNCH_MANIFEST" <<JSON
{
  "round_tag": "round-10-data-dist-test",
  "training_config_sha256_hex16": "$expect_train_cfg",
  "subset_pair_ids_sha256_hex16": "$expect_pair_ids",
  "seed_namespace": "$SEED_NAMESPACE_FROM_YAML",
  "latent_manifest_path": "$LATENT_MANIFEST",
  "latent_manifest_sha256": "$LATENT_MANIFEST_SHA",
  "fresh_init": true,
  "machine_internal_ip_tail": "$(hostname -I 2>/dev/null | awk '{print $1}' | awk -F. '{print $4}')",
  "launcher_script": "$(basename "${BASH_SOURCE[0]}")",
  "ts_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
echo "[round10] pre-launch provenance manifest: $PRELAUNCH_MANIFEST"

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
  --training-config-path "$DPO_DIR/recipes/training_config_round10_data_dist_test_beta100.yaml" \
  --training-config-sha256-pin "$expect_train_cfg" \
  --subset-pair-ids-json "$SUBSET_PAIR_IDS_JSON" \
  --pair-ids-sha256-pin "$expect_pair_ids" \
  --allow-repeated-pair-ids true \
  --save-optimizer-state true \
  --save-every 50 \
  --enable-grad-ckpt true \
  --dit-fsdp true \
  --halt-on-low-noise true \
  --out-dir "$DPO_DIR/ckpts" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-mode "$WANDB_MODE_OPT" \
  --wandb-run-name "round10-data-dist-test-lr1e5-beta100-${NPROC_PER_NODE}rank-$(date -u +%Y%m%dT%H%M%SZ)" \
  > "$LOG_FILE" 2>&1 &

print_launch_info "$!"
