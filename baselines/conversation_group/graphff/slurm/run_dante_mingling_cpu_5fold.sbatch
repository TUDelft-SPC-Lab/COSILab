#!/usr/bin/env bash
#SBATCH --job-name=dante-mingling-cpu-5fold
#SBATCH --array=0-4
#SBATCH --partition=all
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#
# CPU-only DANTE 5-fold wrapper.
#
# Usage:
#   sbatch --export=ALL,DATASET=mingling1/cam06 slurm/run_dante_mingling_cpu_5fold.sbatch
#   sbatch --export=ALL,DATASET=mingling2/cam01 slurm/run_dante_mingling_cpu_5fold.sbatch
#
# Override EPOCHS/PATIENCE/F1_EVAL_EVERY for sensitivity runs.

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
export EPOCHS="${EPOCHS:-600}"
export PATIENCE="${PATIENCE:-50}"
export F1_EVAL_EVERY="${F1_EVAL_EVERY:-10}"
export DANTE_USE_GPU=0

exec bash "$PROJECT_ROOT/slurm/run_dante_mingling_fold.sbatch"
