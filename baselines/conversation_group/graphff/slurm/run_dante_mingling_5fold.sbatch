#!/usr/bin/env bash
#SBATCH --job-name=dante-mingling-5fold
#SBATCH --array=0-4
#SBATCH --partition=all
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/nfs/zli33/slurm_outputs/dante/slurm-%x-%A_%a.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/dante/slurm-%x-%A_%a.err
#
# The log directory above is a literal (Slurm does not expand variables in
# #SBATCH directives) and must exist at submit time.
#
# Usage:
#   sbatch --export=ALL,DATASET=mingling1/cam06 slurm/run_dante_mingling_5fold.sbatch
#   sbatch --export=ALL,DATASET=mingling2/cam01 slurm/run_dante_mingling_5fold.sbatch
#
# RUN_ID is shared by all five tasks, so the folds land side by side under one
# pair_predictions_$RUN_ID. Do NOT pass FOLD here: it would override the array
# index and make all five tasks train the same fold.
#
# Optional overrides are forwarded to run_dante_mingling_fold.sbatch:
#   APPTAINER_IMAGE=/path/to/image.sif
#     Default: $PROJECT_ROOT/apptainer/deep_fformation_dante.sif
#   DATASET=mingling1/cam06 or mingling2/cam01
#   RUN_ID=1
#   OVERWRITE=1
#   DANTE_DATA_ROOT=/path/to/DANTE
#   DANTE_EXPERIMENT_ROOT=/path/to/experiments
#   EPOCHS=600
#   BATCH_SIZE=1024
#   PATIENCE=50
#   MIN_DELTA=0.0
#   F1_EVAL_EVERY=10
#   NO_POINTNET=0
#   SYMMETRIC=0
#   DANTE_PYTHON_BIN=/path/to/python-inside-container

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}}"
DATASET="${DATASET:-mingling1/cam06}"

echo "Dispatching DANTE dataset ${DATASET}, fold ${FOLD} via ${PROJECT_ROOT}/slurm/run_dante_mingling_fold.sbatch"

export PROJECT_ROOT
export FOLD
export DATASET
export RUN_ID="${RUN_ID:-1}"
export OVERWRITE="${OVERWRITE:-1}"
export DANTE_DATA_ROOT="${DANTE_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/DANTE}"
export DANTE_EXPERIMENT_ROOT="${DANTE_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/DANTE/experiments}"
exec bash "${PROJECT_ROOT}/slurm/run_dante_mingling_fold.sbatch"
