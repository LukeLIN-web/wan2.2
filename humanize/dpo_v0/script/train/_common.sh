#!/usr/bin/env bash
# Shared helpers for round-4 launch scripts.
# Source from a launcher; do not execute directly.

# Resolve repo paths and a per-launch timestamped log file.
# Sets: REPO_ROOT, DPO_DIR, LOG_DIR, LOG_FILE.
init_paths() {
  local script_path="$1"
  REPO_ROOT="${VIDEODPOWAN_ROOT:-$HOME/videodpoWan-task20}"
  DPO_DIR="$REPO_ROOT/humanize/dpo_v0"
  LOG_DIR="${LOG_DIR:-$DPO_DIR/logs}"
  mkdir -p "$LOG_DIR"
  local stem
  stem="$(basename "$script_path" .sh)"
  LOG_FILE="${LOG_FILE:-$LOG_DIR/${stem}_$(date -u +%Y%m%dT%H%M%SZ).log}"
}

# Activate conda env and export CUDA alloc config.
activate_env() {
  local env_name="${1:-wan}"
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/bin/activate" "$env_name"
  export VIDEODPOWAN_ROOT="$REPO_ROOT"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

# Resolve wandb config; falls back to offline if no API key + no ~/.netrc.
# Sets: WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE_OPT.
resolve_wandb() {
  WANDB_PROJECT="${WANDB_PROJECT:-wanrl}"
  WANDB_ENTITY="${WANDB_ENTITY:-lukelin}"
  WANDB_MODE_OPT="${WANDB_MODE:-online}"
  local token_file="$REPO_ROOT/wandbtoken"
  if [[ -z "${WANDB_API_KEY:-}" && -r "$token_file" ]]; then
    WANDB_API_KEY="$(tr -d '[:space:]' < "$token_file")"
    export WANDB_API_KEY
    echo "[wandb] loaded API key from $token_file" >&2
  fi
  if [[ "$WANDB_MODE_OPT" == "online" && -z "${WANDB_API_KEY:-}" && ! -f "$HOME/.netrc" ]]; then
    echo "[wandb] no WANDB_API_KEY and no ~/.netrc — falling back to offline mode" >&2
    WANDB_MODE_OPT="offline"
  fi
}

# Verify a single pin file matches its expected sha256/id; exit 2 on mismatch.
_verify_pin() {
  local label="$1" expected="$2" pin_file="$3"
  local actual
  actual="$(tr -d '[:space:]' < "$pin_file")"
  if [[ "$actual" != "$expected" ]]; then
    echo "FATAL: $label mismatch (expected $expected, got $actual)" >&2
    exit 2
  fi
  printf '%s' "$actual"
}

# Verify all three pins (recipe, training_config, pair_ids).
# Args: <expect_recipe> <expect_train_cfg> <expect_pair_ids> <train_cfg_pin_file>
# Env override (round-5+): PAIR_IDS_PIN_FILE can point at a non-round-4
# pair_ids pin file; default keeps the round-4 path so existing launchers are
# unchanged.
verify_pins() {
  local expect_recipe="$1" expect_train_cfg="$2" expect_pair_ids="$3" train_cfg_pin="$4"
  local recipe_pin="$DPO_DIR/recipes/recipe_id"
  local pair_ids_pin="${PAIR_IDS_PIN_FILE:-$DPO_DIR/out/round4/20260428T160839Z/pair_ids_sha256_hex16_pin}"
  local actual_recipe actual_train_cfg actual_pair_ids
  actual_recipe="$(_verify_pin recipe_id "$expect_recipe" "$recipe_pin")"
  actual_train_cfg="$(_verify_pin training_config_sha256 "$expect_train_cfg" "$train_cfg_pin")"
  actual_pair_ids="$(_verify_pin pair_ids_sha256_hex16 "$expect_pair_ids" "$pair_ids_pin")"
  echo "[pins] recipe=$actual_recipe train_cfg=$actual_train_cfg pair_ids=$actual_pair_ids OK"
}

# Print PID + log + monitor hint after a nohup launch.
print_launch_info() {
  local pid="$1"
  echo "PID=$pid"
  echo "log=$LOG_FILE"
  echo "monitor: tail -f $LOG_FILE | grep -E '\\[step|\\[fsdp|OutOfMemory|Traceback|Error|loss=|max_alloc=|halt|RoutingCounter'"
}
