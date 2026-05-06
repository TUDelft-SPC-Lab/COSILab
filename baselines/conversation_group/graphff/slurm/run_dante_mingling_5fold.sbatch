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
#SBATCH --output=logs/slurm-%x-%A_%a.out
#SBATCH --error=logs/slurm-%x-%A_%a.err
#
# Usage:
#   sbatch --export=ALL,DATASET=mingling1/cam06 slurm/run_dante_mingling_5fold.sbatch
#   sbatch --export=ALL,DATASET=mingling2/cam01 slurm/run_dante_mingling_5fold.sbatch
#
# Optional overrides are forwarded to run_dante_mingling_fold.sbatch:
#   APPTAINER_IMAGE=/path/to/image.sif
#     Default: $PROJECT_ROOT/apptainer/deep_fformation_dante.sif
#   DATASET=mingling1/cam06 or mingling2/cam01
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
exec bash "${PROJECT_ROOT}/slurm/run_dante_mingling_fold.sbatch"
