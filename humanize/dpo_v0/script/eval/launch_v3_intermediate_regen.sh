#!/usr/bin/env bash
# Round-4 v3 intermediate-step heldout regen (juyi-videorl, 8-rank).
#
# Generates 42 trained heldout videos for each of step50 / step100 / step150
# from the v3 training run (ckpts/20260429T023311Z/), reusing the v3 final-ckpt
# baseline videos via --baseline-from. Goal: locate the inflection step where
# pair-pref has converged but PhyJudge degradation hasn't kicked in yet, so
# round-5 can early-stop instead of paying for a full re-train at lower β/rank.
#
# Per-step wall ~75 min trained-only on 8-rank (~14 min/video / 8 ranks * 42
# prompts + 1 build + 1 attach). 3 steps serial ≈ 4h.
#
# Env overrides (all optional unless noted):
#   CKPT_ROOT      v3 training run dir (default: $REPO_ROOT/.../ckpts/20260429T023311Z)
#   BASELINE_FROM  prior heldout_regen run dir whose 42 baselines should be reused
#   OUT_ROOT       per-step output root (default: ~/gen_out/v3_intermediate_<ts>)
#   STEPS_TO_EVAL space-separated step ids (default: "50 100 150")
#   T0_T3_ROOT     T0_T3_root path on this box (default: $HOME/T0_T3_root)
#   UPSTREAM       canonical Wan2.2-I2V-A14B root (default: $HOME/Wan2.2-I2V-A14B)

set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env wan
cd "$DPO_DIR"

CKPT_ROOT="${CKPT_ROOT:-$DPO_DIR/ckpts/20260429T023311Z}"
BASELINE_FROM="${BASELINE_FROM:-$HOME/gen_out/v3_full_strip_20260429T162925Z/20260429T162927Z}"
OUT_ROOT="${OUT_ROOT:-$HOME/gen_out/v3_intermediate_${TS_UTC}}"
STEPS_TO_EVAL="${STEPS_TO_EVAL:-50 100 150}"
T0_T3_ROOT="${T0_T3_ROOT:-$HOME/T0_T3_root}"
UPSTREAM="${UPSTREAM:-$HOME/Wan2.2-I2V-A14B}"
COND_IMG_FALLBACK="${COND_IMG_FALLBACK:-$HOME/cond_imgs}"

if [[ ! -d "$BASELINE_FROM/heldout_regen" ]]; then
  echo "FATAL: BASELINE_FROM=$BASELINE_FROM does not contain heldout_regen/" >&2
  echo "       expected layout: <BASELINE_FROM>/heldout_regen/<prompt_id>/baseline/..." >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"
{
  echo "[paths] DPO_DIR=$DPO_DIR"
  echo "[paths] CKPT_ROOT=$CKPT_ROOT"
  echo "[paths] BASELINE_FROM=$BASELINE_FROM"
  echo "[paths] OUT_ROOT=$OUT_ROOT"
  echo "[paths] STEPS_TO_EVAL='$STEPS_TO_EVAL'"
} | tee -a "$LOG_FILE"

run_one_step() {
  local step="$1"
  local raw="$CKPT_ROOT/lora_step${step}.safetensors"
  local stripped="$CKPT_ROOT/lora_step${step}_stripped.safetensors"
  local step_out="$OUT_ROOT/step${step}"

  echo "=== step ${step} ===" | tee -a "$LOG_FILE"
  strip_fsdp_ckpt "$raw" "$stripped" 2>&1 | tee -a "$LOG_FILE"

  mkdir -p "$step_out"
  echo "[regen] step${step} trained-only -> $step_out" | tee -a "$LOG_FILE"
  torchrun --nproc_per_node=8 humanize/dpo_v0/eval/heldout_regen.py \
    --t0-t3-root "$T0_T3_ROOT" \
    --out-dir "$step_out" \
    --upstream "$UPSTREAM" \
    --trained-lora "$stripped" \
    --adapter python_api \
    --cache-pipe --mode-batched \
    --baseline-from "$BASELINE_FROM" \
    --cond-image-fallback-root "$COND_IMG_FALLBACK" \
    --compute-envelope multi_gpu_inference_seed_parallel \
    2>&1 | tee -a "$LOG_FILE"
}

for STEP in $STEPS_TO_EVAL; do
  run_one_step "$STEP"
done

echo "[done] all steps regen complete -> $OUT_ROOT" | tee -a "$LOG_FILE"
echo "[next] mirror to juyi-finetune, then bash launch_v3_intermediate_score.sh" | tee -a "$LOG_FILE"
