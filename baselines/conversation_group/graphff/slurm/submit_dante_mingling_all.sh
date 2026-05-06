#!/usr/bin/env bash
# Submit DANTE Mingling benchmark jobs for every camera.
#
# Usage:
#   bash slurm/submit_dante_mingling_all.sh
#
# Optional:
#   DRY_RUN=1 bash slurm/submit_dante_mingling_all.sh
#   DANTE_SCRIPT=slurm/run_dante_mingling_5fold.sbatch bash slurm/submit_dante_mingling_all.sh
#   EXTRA_EXPORTS='EPOCHS=600,PATIENCE=50' bash slurm/submit_dante_mingling_all.sh

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

DANTE_SCRIPT="${DANTE_SCRIPT:-slurm/run_dante_mingling_cpu_5fold.sbatch}"
SCRIPT="$PROJECT_ROOT/$DANTE_SCRIPT"
DATASETS=(
  "mingling1/cam06"
  "mingling1/cam08"
  "mingling1/cam10"
  "mingling2/cam01"
  "mingling2/cam03"
)

[[ -f "$SCRIPT" ]] || { echo "[ERROR] missing Slurm script: $SCRIPT" >&2; exit 2; }

for dataset in "${DATASETS[@]}"; do
  dataset_dir="$PROJECT_ROOT/DANTE-master/datasets/$dataset"
  [[ -f "$dataset_dir/DS_utils/features.txt" ]] || { echo "[ERROR] missing: $dataset_dir/DS_utils/features.txt" >&2; exit 2; }
  [[ -f "$dataset_dir/DS_utils/group_names.txt" ]] || { echo "[ERROR] missing: $dataset_dir/DS_utils/group_names.txt" >&2; exit 2; }
  for fold in 0 1 2 3 4; do
    [[ -f "$dataset_dir/fold_$fold/train.p" ]] || { echo "[ERROR] missing: $dataset_dir/fold_$fold/train.p" >&2; exit 2; }
    [[ -f "$dataset_dir/fold_$fold/val.p" ]] || { echo "[ERROR] missing: $dataset_dir/fold_$fold/val.p" >&2; exit 2; }
    [[ -f "$dataset_dir/fold_$fold/test.p" ]] || { echo "[ERROR] missing: $dataset_dir/fold_$fold/test.p" >&2; exit 2; }
  done

  export_arg="ALL,DATASET=$dataset"
  if [[ -n "${EXTRA_EXPORTS:-}" ]]; then
    export_arg="$export_arg,$EXTRA_EXPORTS"
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'sbatch --export=%q %q\n' "$export_arg" "$SCRIPT"
  else
    sbatch --export="$export_arg" "$SCRIPT"
  fi
done
