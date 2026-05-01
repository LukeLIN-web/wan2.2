#!/usr/bin/env bash
# Round-6 v2aligned launcher: β=100 lora_final → +800 fresh class-balanced pair_ids.
#
# Continues training from the round-4 winner (β=100 lora_final, same warm-init
# as round-5) on a class-rebalanced 800-pair subset of the cond-present 2202
# pool. Per-class quotas match the eval-v2 distribution
# (A:279/B:167/C:112/D:93/E:19/F:74/G:56=800; sampler
# script/sample/round6_class_balanced.py).
#
# Topology (per round6_plan.md):
# - 8-rank FSDP on juyi-videorl (target_steps = ceil(800/8) = 100, no wrap).
# - β=100 / lr=1e-5 / lora_rank=16 / sampling_band=[901,999] inherited from
#   round-5; recipe_id unchanged 6bef6e104cdd3442.
# - --init-lora-from round-4 β=100 lora_final (NOT --ref-lora-from per
#   round6_plan.md (i)-default; round-7 may use --ref-lora-from).
# - --save-optimizer-state true so round-7 can warm-resume momentum.
# - --save-every 20 (5 ckpts: step 20/40/60/80/lora_final, denser than
#   round-4/5 save_every=50 per round6_plan.md decision).
#
# Required env (override defaults if path differs on your box):
# - VIDEODPOWAN_ROOT (default $HOME/videodpoWan-task20)
# - LATENT_MANIFEST: full path to the round-6 union latent manifest
#   (latents/<UTC>/tier_b_round6_v2aligned_800/manifest.jsonl). The launcher
#   auto-globs the latest UTC dir if not set.
# - SUBSET_PAIR_IDS_JSON / PAIR_IDS_PIN_FILE / ROUND6_OUT_DIR: auto-locate
#   latest UTC dir under out/round6/ if not set.
# - INIT_LORA_FROM (default round-4 β=100 lora_final at
#   ckpts/20260429T234925Z/lora_final.safetensors).
#
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env wan
resolve_wandb
cd "$DPO_DIR"

# Round-6 pin chain (locked).
expect_recipe="6bef6e104cdd3442"
expect_train_cfg="fa4dcf26a8f8e3e7"
expect_pair_ids="2749aeb2bb192148"

# Locate the round-6 sampler artifacts. Latest UTC dir under out/round6/.
ROUND6_OUT_DIR="${ROUND6_OUT_DIR:-$(ls -1d "$DPO_DIR"/out/round6/*/ 2>/dev/null | sort | tail -1 | sed 's:/$::')}"
SUBSET_PAIR_IDS_JSON="${SUBSET_PAIR_IDS_JSON:-${ROUND6_OUT_DIR}/T3_round6_v2aligned_800.json}"
PAIR_IDS_PIN_FILE="${PAIR_IDS_PIN_FILE:-${ROUND6_OUT_DIR}/pair_ids_round6_v2aligned_800_sha256_hex16_pin}"
export PAIR_IDS_PIN_FILE

if [[ ! -f "$SUBSET_PAIR_IDS_JSON" ]]; then
  echo "FATAL: SUBSET_PAIR_IDS_JSON not found: $SUBSET_PAIR_IDS_JSON" >&2
  echo "       Run script/sample/round6_class_balanced.py first." >&2
  exit 3
fi
if [[ ! -f "$PAIR_IDS_PIN_FILE" ]]; then
  echo "FATAL: PAIR_IDS_PIN_FILE not found: $PAIR_IDS_PIN_FILE" >&2
  exit 3
fi
verify_pins "$expect_recipe" "$expect_train_cfg" "$expect_pair_ids" \
  "$DPO_DIR/recipes/training_config_round6_v2aligned_beta100_sha256_pin"

# Latent manifest. Operator-overridable; otherwise auto-glob the latest
# round-6 union manifest (a02ca9b1... build emits at
# latents/<UTC>/tier_b_round6_v2aligned_800/manifest.jsonl).
LATENT_MANIFEST="${LATENT_MANIFEST:-}"
if [[ -z "$LATENT_MANIFEST" ]]; then
  LATENT_MANIFEST="$(ls -1 "$DPO_DIR"/latents/*/tier_b_round6_v2aligned_800/manifest.jsonl 2>/dev/null | sort | tail -1)"
fi
if [[ ! -f "$LATENT_MANIFEST" ]]; then
  echo "FATAL: LATENT_MANIFEST not found: $LATENT_MANIFEST" >&2
  echo "       Build the union manifest first (concat round-4 + round-5 manifests, filter to round-6 800)." >&2
  exit 4
fi
echo "[round6] LATENT_MANIFEST=$LATENT_MANIFEST"
echo "[round6] SUBSET_PAIR_IDS_JSON=$SUBSET_PAIR_IDS_JSON"

INIT_LORA_FROM="${INIT_LORA_FROM:-$DPO_DIR/ckpts/20260429T234925Z/lora_final.safetensors}"
if [[ ! -f "$INIT_LORA_FROM" ]]; then
  echo "FATAL: INIT_LORA_FROM not found: $INIT_LORA_FROM" >&2
  exit 5
fi
echo "[round6] INIT_LORA_FROM=$INIT_LORA_FROM"

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
  --training-config-path "$DPO_DIR/recipes/training_config_round6_v2aligned_beta100.yaml" \
  --training-config-sha256-pin "$expect_train_cfg" \
  --subset-pair-ids-json "$SUBSET_PAIR_IDS_JSON" \
  --pair-ids-sha256-pin "$expect_pair_ids" \
  --init-lora-from "$INIT_LORA_FROM" \
  --save-optimizer-state true \
  --save-every 20 \
  --enable-grad-ckpt true \
  --dit-fsdp true \
  --halt-on-low-noise true \
  --out-dir "$DPO_DIR/ckpts" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --wandb-mode "$WANDB_MODE_OPT" \
  --wandb-run-name "round6-v2aligned-lr1e5-beta100-${NPROC_PER_NODE}rank-$(date -u +%Y%m%dT%H%M%SZ)" \
  > "$LOG_FILE" 2>&1 &

print_launch_info "$!"
