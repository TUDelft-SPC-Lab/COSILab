#!/usr/bin/env bash
#SBATCH --job-name=dante-mingling-cpu-5fold
#SBATCH --array=0-4
#SBATCH --partition=all
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/nfs/zli33/slurm_outputs/dante/slurm-%x-%A_%a.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/dante/slurm-%x-%A_%a.err
#
# CPU-only DANTE 5-fold wrapper.
#
# The log directory above is a literal (Slurm does not expand variables in
# #SBATCH directives) and must exist at submit time.
#
# Usage:
#   sbatch --export=ALL,DATASET=mingling1/cam06 slurm/run_dante_mingling_cpu_5fold.sbatch
#   sbatch --export=ALL,DATASET=mingling2/cam01 slurm/run_dante_mingling_cpu_5fold.sbatch
#
# Override EPOCHS/PATIENCE/F1_EVAL_EVERY for sensitivity runs.
# RUN_ID is shared by all five tasks, so the folds land side by side under one
# pair_predictions_$RUN_ID. Do NOT pass FOLD here: it would override the array
# index and make all five tasks train the same fold.

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

export PROJECT_ROOT
export FOLD="${FOLD:-${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}}"
export DATASET="${DATASET:-mingling1/cam06}"
export RUN_ID="${RUN_ID:-1}"
export OVERWRITE="${OVERWRITE:-1}"
export DANTE_DATA_ROOT="${DANTE_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/DANTE}"
export DANTE_EXPERIMENT_ROOT="${DANTE_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/DANTE/experiments}"
export EPOCHS="${EPOCHS:-600}"
export PATIENCE="${PATIENCE:-50}"
export F1_EVAL_EVERY="${F1_EVAL_EVERY:-10}"
export DANTE_USE_GPU=0

exec bash "$PROJECT_ROOT/slurm/run_dante_mingling_fold.sbatch"
