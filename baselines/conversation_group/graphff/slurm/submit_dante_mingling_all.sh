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
DANTE_DATA_ROOT="${DANTE_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/DANTE}"
DANTE_EXPERIMENT_ROOT="${DANTE_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/DANTE/experiments}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/home/nfs/zli33/slurm_outputs/dante}"
DATASETS=(
  "mingling1/cam06"
  "mingling1/cam08"
  "mingling1/cam10"
  "mingling2/cam01"
  "mingling2/cam03"
)

[[ -f "$SCRIPT" ]] || { echo "[ERROR] missing Slurm script: $SCRIPT" >&2; exit 2; }
[[ -d "$DANTE_DATA_ROOT" ]] || { echo "[ERROR] data root not found: $DANTE_DATA_ROOT" >&2; exit 2; }

# Slurm rejects a job outright if the --output directory does not already exist.
mkdir -p "$SLURM_LOG_DIR"
echo "data root:       $DANTE_DATA_ROOT"
echo "experiment root: $DANTE_EXPERIMENT_ROOT"
echo "slurm log dir:   $SLURM_LOG_DIR"

for dataset in "${DATASETS[@]}"; do
  dataset_dir="$DANTE_DATA_ROOT/$dataset"
  [[ -f "$dataset_dir/DS_utils/features.txt" ]] || { echo "[ERROR] missing: $dataset_dir/DS_utils/features.txt" >&2; exit 2; }
  [[ -f "$dataset_dir/DS_utils/group_names.txt" ]] || { echo "[ERROR] missing: $dataset_dir/DS_utils/group_names.txt" >&2; exit 2; }
  for fold in 0 1 2 3 4; do
    [[ -f "$dataset_dir/fold_$fold/train.p" ]] || { echo "[ERROR] missing: $dataset_dir/fold_$fold/train.p" >&2; exit 2; }
    [[ -f "$dataset_dir/fold_$fold/val.p" ]] || { echo "[ERROR] missing: $dataset_dir/fold_$fold/val.p" >&2; exit 2; }
    [[ -f "$dataset_dir/fold_$fold/test.p" ]] || { echo "[ERROR] missing: $dataset_dir/fold_$fold/test.p" >&2; exit 2; }
  done

  export_arg="ALL,DATASET=$dataset,DANTE_DATA_ROOT=$DANTE_DATA_ROOT,DANTE_EXPERIMENT_ROOT=$DANTE_EXPERIMENT_ROOT"
  if [[ -n "${EXTRA_EXPORTS:-}" ]]; then
    export_arg="$export_arg,$EXTRA_EXPORTS"
  fi

  # Passed explicitly so an overridden SLURM_LOG_DIR actually takes effect; these
  # override the literal path in the script's own #SBATCH directives.
  out_arg="$SLURM_LOG_DIR/slurm-%x-%A_%a.out"
  err_arg="$SLURM_LOG_DIR/slurm-%x-%A_%a.err"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'sbatch --output=%q --error=%q --export=%q %q\n' \
      "$out_arg" "$err_arg" "$export_arg" "$SCRIPT"
  else
    sbatch --output="$out_arg" --error="$err_arg" --export="$export_arg" "$SCRIPT"
  fi
done
