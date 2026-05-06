#!/usr/bin/env bash
# Submit GraphFF/LSTM Mingling benchmark jobs for every camera.
#
# Usage:
#   bash slurm/submit_lstm_mingling_all.sh
#
# Optional:
#   DRY_RUN=1 bash slurm/submit_lstm_mingling_all.sh
#   EXTRA_EXPORTS='NUM_EPOCHS=600,PATIENCE=50' bash slurm/submit_lstm_mingling_all.sh
#   EXCLUDE_NODES='gpu[36-45]' bash slurm/submit_lstm_mingling_all.sh
#   SBATCH_ARGS='--constraint=a40' bash slurm/submit_lstm_mingling_all.sh

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

SCRIPT="$PROJECT_ROOT/slurm/run_vitpose_dataframe_5fold.sbatch"
EXCLUDE_NODES="${EXCLUDE_NODES:-gpu[36-45]}"
SBATCH_EXTRA_ARGS=()
if [[ -n "${SBATCH_ARGS:-}" ]]; then
  read -r -a SBATCH_EXTRA_ARGS <<< "$SBATCH_ARGS"
fi
if [[ -n "$EXCLUDE_NODES" ]]; then
  SBATCH_EXTRA_ARGS+=(--exclude="$EXCLUDE_NODES")
fi
DATASETS=(
  "mingling1/cam06"
  "mingling1/cam08"
  "mingling1/cam10"
  "mingling2/cam01"
  "mingling2/cam03"
)

[[ -f "$SCRIPT" ]] || { echo "[ERROR] missing Slurm script: $SCRIPT" >&2; exit 2; }

for dataset in "${DATASETS[@]}"; do
  dataset_dir="$PROJECT_ROOT/data/$dataset"
  [[ -f "$dataset_dir/features.csv" ]] || { echo "[ERROR] missing: $dataset_dir/features.csv" >&2; exit 2; }
  [[ -f "$dataset_dir/GT.csv" ]] || { echo "[ERROR] missing: $dataset_dir/GT.csv" >&2; exit 2; }
  [[ -f "$dataset_dir/group_names.txt" ]] || { echo "[ERROR] missing: $dataset_dir/group_names.txt" >&2; exit 2; }
  [[ -f "$dataset_dir/scene_continuity.csv" ]] || { echo "[ERROR] missing: $dataset_dir/scene_continuity.csv" >&2; exit 2; }

  export_arg="ALL,DATASET=$dataset"
  if [[ -n "${EXTRA_EXPORTS:-}" ]]; then
    export_arg="$export_arg,$EXTRA_EXPORTS"
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'sbatch'
    for arg in "${SBATCH_EXTRA_ARGS[@]}"; do
      printf ' %q' "$arg"
    done
    printf ' --export=%q %q\n' "$export_arg" "$SCRIPT"
  else
    sbatch "${SBATCH_EXTRA_ARGS[@]}" --export="$export_arg" "$SCRIPT"
  fi
done
