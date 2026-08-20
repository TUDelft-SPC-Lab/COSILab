#!/usr/bin/env bash
# Submit DANTE Mingling benchmark jobs.
#
# Usage:
#   bash slurm/submit_dante.sh <cam> [fold]
#
#   <cam>   6 | 06 | 8 | 08 | 10 | 1 | 01 | 3 | 03 | cam06 | all
#           The Mingling session is derived from the camera number:
#             06, 08, 10 -> mingling1
#             01, 03     -> mingling2
#   [fold]  optional single fold 0-4; omit to run all five
#
# Examples:
#   bash slurm/submit_dante.sh 6            # cam06, all 5 folds
#   bash slurm/submit_dante.sh all          # all 5 cameras, 25 tasks
#   bash slurm/submit_dante.sh 6 2          # cam06, fold 2 only
#   DRY_RUN=1 bash slurm/submit_dante.sh all
#   USE_GPU=1 bash slurm/submit_dante.sh 6
#   RUN_ID=2 bash slurm/submit_dante.sh all
#
# Environment overrides:
#   RUN_ID=1                  output goes to <exp_root>/exp_$RUN_ID/...
#   OVERWRITE=1               0 refuses to run when a fold directory exists
#   USE_GPU=0                 1 requests a GPU and runs TensorFlow on it
#   DRY_RUN=0                 1 prints the sbatch lines without submitting
#   DANTE_DATA_ROOT=...       benchmark artifacts to read
#   DANTE_EXPERIMENT_ROOT=... where runs are written
#   SLURM_LOG_DIR=...         Slurm stdout/stderr
#   EXTRA_EXPORTS='EPOCHS=600,PATIENCE=50'
#   SBATCH_ARGS='--constraint=...'

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

SCRIPT="$PROJECT_ROOT/slurm/run_dante.sbatch"
DANTE_DATA_ROOT="${DANTE_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/DANTE}"
DANTE_EXPERIMENT_ROOT="${DANTE_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/DANTE/experiments}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/home/nfs/zli33/slurm_outputs/dante}"
RUN_ID="${RUN_ID:-1}"
OVERWRITE="${OVERWRITE:-1}"
USE_GPU="${USE_GPU:-0}"

ALL_CAMS=(06 08 10 01 03)

usage() {
  sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}

# camera number -> "<session>/cam<nn>". Zero-pads so 6 -> 06 and 1 -> 01, which
# keeps 1 (cam01, mingling2) and 10 (cam10, mingling1) distinct.
resolve_dataset() {
  local cam="${1#cam}"
  [[ "$cam" =~ ^[0-9]{1,2}$ ]] || return 1
  cam="$(printf '%02d' "$((10#$cam))")"
  case "$cam" in
    06|08|10) echo "mingling1/cam$cam" ;;
    01|03)    echo "mingling2/cam$cam" ;;
    *)        return 1 ;;
  esac
}

[[ $# -ge 1 ]] || usage
case "${1:-}" in -h|--help|help) usage 0 ;; esac

CAM_ARG="$1"
FOLD_ARG="${2:-}"

if [[ -n "$FOLD_ARG" ]]; then
  [[ "$FOLD_ARG" =~ ^[0-4]$ ]] || { echo "[ERROR] fold must be 0-4, got: $FOLD_ARG" >&2; exit 2; }
  ARRAY_SPEC="${FOLD_ARG}-${FOLD_ARG}"
else
  ARRAY_SPEC="0-4"
fi

CAMS=()
if [[ "$CAM_ARG" == "all" ]]; then
  CAMS=("${ALL_CAMS[@]}")
else
  CAMS=("$CAM_ARG")
fi

DATASETS=()
for cam in "${CAMS[@]}"; do
  dataset="$(resolve_dataset "$cam")" || {
    echo "[ERROR] unknown camera: $cam (expected one of ${ALL_CAMS[*]}, or all)" >&2
    exit 2
  }
  DATASETS+=("$dataset")
done

[[ -f "$SCRIPT" ]] || { echo "[ERROR] missing Slurm script: $SCRIPT" >&2; exit 2; }
[[ -d "$DANTE_DATA_ROOT" ]] || { echo "[ERROR] data root not found: $DANTE_DATA_ROOT" >&2; exit 2; }

# resource profile; these override the #SBATCH defaults in run_dante.sbatch
SBATCH_RESOURCE_ARGS=()
if [[ "$USE_GPU" == "1" ]]; then
  SBATCH_RESOURCE_ARGS=(--gres=gpu:1 --mem=32G --cpus-per-task=2 --time=16:00:00)
elif [[ "$USE_GPU" != "0" ]]; then
  echo "[ERROR] USE_GPU must be 0 or 1, got: $USE_GPU" >&2
  exit 2
fi

SBATCH_EXTRA_ARGS=()
if [[ -n "${SBATCH_ARGS:-}" ]]; then
  read -r -a SBATCH_EXTRA_ARGS <<< "$SBATCH_ARGS"
fi

# Slurm rejects a job outright if the --output directory does not already exist.
mkdir -p "$SLURM_LOG_DIR"

echo "data root:       $DANTE_DATA_ROOT"
echo "experiment root: $DANTE_EXPERIMENT_ROOT/exp_$RUN_ID"
echo "slurm log dir:   $SLURM_LOG_DIR"
echo "cameras:         ${DATASETS[*]}"
echo "folds:           $ARRAY_SPEC"
echo "gpu:             $USE_GPU"
echo "overwrite:       $OVERWRITE"
echo

for dataset in "${DATASETS[@]}"; do
  dataset_dir="$DANTE_DATA_ROOT/$dataset"
  [[ -f "$dataset_dir/DS_utils/features.txt" ]] || { echo "[ERROR] missing: $dataset_dir/DS_utils/features.txt" >&2; exit 2; }
  [[ -f "$dataset_dir/DS_utils/group_names.txt" ]] || { echo "[ERROR] missing: $dataset_dir/DS_utils/group_names.txt" >&2; exit 2; }
  for fold in $(seq "${ARRAY_SPEC%-*}" "${ARRAY_SPEC#*-}"); do
    for part in train val test; do
      [[ -f "$dataset_dir/fold_$fold/$part.p" ]] || { echo "[ERROR] missing: $dataset_dir/fold_$fold/$part.p" >&2; exit 2; }
    done
  done

  job_name="dante-$(basename "$dataset")"
  export_arg="ALL,DATASET=$dataset,RUN_ID=$RUN_ID,OVERWRITE=$OVERWRITE,USE_GPU=$USE_GPU"
  export_arg="$export_arg,DANTE_DATA_ROOT=$DANTE_DATA_ROOT,DANTE_EXPERIMENT_ROOT=$DANTE_EXPERIMENT_ROOT"
  if [[ -n "${EXTRA_EXPORTS:-}" ]]; then
    export_arg="$export_arg,$EXTRA_EXPORTS"
  fi

  sbatch_args=(
    --job-name="$job_name"
    --array="$ARRAY_SPEC"
    --output="$SLURM_LOG_DIR/slurm-%x-%A_%a.out"
    --error="$SLURM_LOG_DIR/slurm-%x-%A_%a.err"
    "${SBATCH_RESOURCE_ARGS[@]}"
    "${SBATCH_EXTRA_ARGS[@]}"
    --export="$export_arg"
    "$SCRIPT"
  )

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'sbatch'
    printf ' %q' "${sbatch_args[@]}"
    printf '\n'
  else
    sbatch "${sbatch_args[@]}"
  fi
done
