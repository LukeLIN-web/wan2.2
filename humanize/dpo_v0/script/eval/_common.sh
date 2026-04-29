#!/usr/bin/env bash

_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

init_paths() {
  local script_path="$1"
  REPO_ROOT="${VIDEODPOWAN_ROOT:-$HOME/videodpoWan-task20}"
  DPO_DIR="$REPO_ROOT/humanize/dpo_v0"
  LOG_DIR="${LOG_DIR:-$DPO_DIR/logs}"
  mkdir -p "$LOG_DIR"
  TS_UTC="${TS_UTC:-$(date -u +%Y%m%dT%H%M%SZ)}"
  local stem
  stem="$(basename "$script_path" .sh)"
  LOG_FILE="${LOG_FILE:-$LOG_DIR/${stem}_${TS_UTC}.log}"
}

activate_env() {
  local env_name="${1:-wan}"
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/bin/activate" "$env_name"
  export VIDEODPOWAN_ROOT="$REPO_ROOT"
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

strip_fsdp_ckpt() {
  local src="$1" dst="$2"
  if [[ ! -f "$src" ]]; then
    echo "FATAL: missing src ckpt $src" >&2
    return 2
  fi
  if [[ -f "$dst" ]]; then
    echo "[strip] reusing $dst"
    return 0
  fi
  python "$_COMMON_DIR/strip_fsdp_prefix.py" --src "$src" --dst "$dst"
}

print_launch_info() {
  local pid="$1"
  echo "PID=$pid"
  echo "log=$LOG_FILE"
}
