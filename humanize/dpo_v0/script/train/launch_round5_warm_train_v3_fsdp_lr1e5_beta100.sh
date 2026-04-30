#!/usr/bin/env bash
# Round-5 warm-start launcher: β=100 lora_final → +1202 fresh pair_ids.
#
# Continues training from the round-4 winner (β=100 lora_final, +0.114 axes-
# avg / PTV sign-test p=0.039 on the full 42-prompt validation, see
# docs/experiment-results/round4_v3_lr1e5_full42_validation.md). Trains on the
# 1202 cond-image-present pairs that were NOT used in round-4 (= 2745 raw
# setminus round-4 1k filtered to disk-present), built by
# dataprocessing/build_round5_warm_setminus.py.
#
# Topology (per luke1 lock 2026-04-30):
# - 8-rank FSDP on juyi-videorl (`MAX_STEPS=150` = 1202 / 8 floor).
# - β=100 / lr=1e-5 / lora_rank=16 / sampling_band=[901,999] inherited from
#   round-4 r5 lr1e5 beta100 recipe. recipe_id unchanged 6bef6e104cdd3442.
# - --init-lora-from points at the round-4 final ckpt; --save-optimizer-state
#   true so future round-N+1 can warm-resume momentum (round-4 lora_final did
#   NOT save optimizer state, so this run starts AdamW from scratch — first
#   ~5 step optimizer warmup transient is expected).
#
# Required env (override defaults if path differs on your box):
# - VIDEODPOWAN_ROOT (default $HOME/videodpoWan-task20)
# - SUBSET_PAIR_IDS_JSON: full path to the round-5 official 1202 JSON
#   (T3_round5_warm_official_1202.json under out/round5/<UTC>/). The launcher
#   refuses to start if not set, since the UTC dir is build-time and cannot
#   be hardcoded across operator runs.
# - LATENT_MANIFEST: full path to the round-5 1202 latent manifest
#   (latents/<UTC>/tier_b_round5_warm_1202/manifest.jsonl). Same reason.
# - INIT_LORA_FROM (default round-4 β=100 lora_final):
#     $HOME/videodpoWan-task20/humanize/dpo_v0/ckpts/20260429T234925Z/lora_final.safetensors
#
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env wan
resolve_wandb
cd "$DPO_DIR"

# Round-5 pin chain.
expect_recipe="6bef6e104cdd3442"
expect_train_cfg="88cc04e3b8dc9100"
expect_pair_ids="680a7eec8090d48b"

# Locate the round-5 build artifacts. Latest UTC dir under out/round5/ is the
# expected path; operator can override via SUBSET_PAIR_IDS_JSON.
ROUND5_OUT_DIR="${ROUND5_OUT_DIR:-$(ls -1d "$DPO_DIR"/out/round5/*/ 2>/dev/null | sort | tail -1 | sed 's:/$::')}"
SUBSET_PAIR_IDS_JSON="${SUBSET_PAIR_IDS_JSON:-${ROUND5_OUT_DIR}/T3_round5_warm_official_1202.json}"
PAIR_IDS_PIN_FILE="${PAIR_IDS_PIN_FILE:-${ROUND5_OUT_DIR}/pair_ids_cond_image_present_sha256_hex16_pin}"
export PAIR_IDS_PIN_FILE

if [[ ! -f "$SUBSET_PAIR_IDS_JSON" ]]; then
  echo "FATAL: SUBSET_PAIR_IDS_JSON not found: $SUBSET_PAIR_IDS_JSON" >&2
  echo "       Run dataprocessing/build_round5_warm_setminus.py first." >&2
  exit 3
fi
if [[ ! -f "$PAIR_IDS_PIN_FILE" ]]; then
  echo "FATAL: PAIR_IDS_PIN_FILE not found: $PAIR_IDS_PIN_FILE" >&2
  exit 3
fi
verify_pins "$expect_recipe" "$expect_train_cfg" "$expect_pair_ids" \
  "$DPO_DIR/recipes/training_config_round5_warm_beta100_sha256_pin"

# Latent manifest (operator must point at the encode output).
LATENT_MANIFEST="${LATENT_MANIFEST:-}"
if [[ -z "$LATENT_MANIFEST" ]]; then
  # Fallback: find the most recent round-5 manifest.
  LATENT_MANIFEST="$(ls -1 "$DPO_DIR"/latents/*/tier_b_round5_warm_1202/manifest.jsonl 2>/dev/null | sort | tail -1)"
fi
if [[ ! -f "$LATENT_MANIFEST" ]]; then
  echo "FATAL: LATENT_MANIFEST not found: $LATENT_MANIFEST" >&2
  echo "       Run encode_videos.py --tier tier_b_subset --rank/--world-size first." >&2
  exit 4
fi
echo "[round5] LATENT_MANIFEST=$LATENT_MANIFEST"
echo "[round5] SUBSET_PAIR_IDS_JSON=$SUBSET_PAIR_IDS_JSON"

INIT_LORA_FROM="${INIT_LORA_FROM:-$DPO_DIR/ckpts/20260429T234925Z/lora_final.safetensors}"
if [[ ! -f "$INIT_LORA_FROM" ]]; then
  echo "FATAL: INIT_LORA_FROM not found: $INIT_LORA_FROM" >&2
  exit 5
fi
echo "[round5] INIT_LORA_FROM=$INIT_LORA_FROM"

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
  --training-config-path "$DPO_DIR/recipes/training_config_round5_warm_beta100.yaml" \
  --training-config-sha256-pin "$expect_train_cfg" \
  --subset-pair-ids-json "$SUBSET_PAIR_IDS_JSON" \
  --pair-ids-sha256-pin "$expect_pair_ids" \
  --init-lora-from "$INIT_LORA_FROM" \
  --save-optimizer-state true \
  --enable-grad-ckpt true \
  --dit-fsdp true \
  --halt-on-low-noise true \
  --out-dir "$DPO_DIR/ckpts" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-mode "$WANDB_MODE_OPT" \
  --wandb-run-name "round5-warm-lr1e5-beta100-${NPROC_PER_NODE}rank-$(date -u +%Y%m%dT%H%M%SZ)" \
  > "$LOG_FILE" 2>&1 &

print_launch_info "$!"
