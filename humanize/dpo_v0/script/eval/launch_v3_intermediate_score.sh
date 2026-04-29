#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_common.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

init_paths "${BASH_SOURCE[0]}"
activate_env physground
cd "$DPO_DIR"

if [[ -z "${RUN_ROOT:-}" ]]; then
  echo "FATAL: set RUN_ROOT to the v3_intermediate_<ts> dir mirrored from juyi-videorl" >&2
  exit 2
fi
if [[ ! -d "$RUN_ROOT" ]]; then
  echo "FATAL: RUN_ROOT=$RUN_ROOT does not exist" >&2
  exit 2
fi

VIDEODPO_DIR="${VIDEODPO_DIR:-$HOME/videodpo}"
SCORER="${SCORER:-$VIDEODPO_DIR/benchmark/physground_score.py}"
STEPS_TO_SCORE="${STEPS_TO_SCORE:-50 100 150}"
AXES="${AXES:-SA PTV persistence inertia momentum}"
BATCH_AXES="${BATCH_AXES:---batch-axes}"

if [[ ! -f "$SCORER" ]]; then
  echo "FATAL: scorer not found at $SCORER (set VIDEODPO_DIR or SCORER)" >&2
  exit 2
fi

{
  echo "[paths] RUN_ROOT=$RUN_ROOT"
  echo "[paths] SCORER=$SCORER"
  echo "[paths] STEPS_TO_SCORE='$STEPS_TO_SCORE'"
} | tee -a "$LOG_FILE"

for STEP in $STEPS_TO_SCORE; do
  RUN_GLOB="$RUN_ROOT/step${STEP}/*/heldout_regen/*/trained/**/manifest.json"
  OUT_DIR="$RUN_ROOT/scores/step${STEP}"
  echo "=== score step ${STEP} ===" | tee -a "$LOG_FILE"
  echo "[score] run-glob=$RUN_GLOB" | tee -a "$LOG_FILE"
  echo "[score] out-dir=$OUT_DIR" | tee -a "$LOG_FILE"
  mkdir -p "$OUT_DIR"
  # shellcheck disable=SC2086
  python "$SCORER" \
    --run-glob "$RUN_GLOB" \
    --axes $AXES \
    $BATCH_AXES \
    --out-dir "$OUT_DIR" \
    --skip-existing \
    2>&1 | tee -a "$LOG_FILE"
done

echo "[done] scoring complete -> $RUN_ROOT/scores/step{50,100,150}/<ts>/results.jsonl" | tee -a "$LOG_FILE"
