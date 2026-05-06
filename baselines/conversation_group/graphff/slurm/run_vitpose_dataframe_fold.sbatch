#!/usr/bin/env bash
#SBATCH --job-name=graphff-mingling-f0
#SBATCH --partition=all
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/slurm-%x-%A.out
#SBATCH --error=logs/slurm-%x-%A.err
#
# Usage:
#   sbatch --export=ALL,FOLD=0,DATASET=mingling2/cam01 slurm/run_vitpose_dataframe_fold.sbatch
#
# Optional overrides:
#   APPTAINER_IMAGE=/path/to/image.sif
#     Default: $PROJECT_ROOT/apptainer/deep_fformation_dante.sif
#   DATASET=mingling1/cam06 or mingling2/cam01
#   TRAIN=1
#   DATASET_MAKE=1
#   SEQ_LEN=10
#   FRAME_STRIDE=20
#   HIDDEN_DIM=8
#   THRESHOLD=1.0
#   NUM_EPOCHS=600
#   BATCH_SIZE=128
#   LR=0.001
#   PATIENCE=50
#   MIN_DELTA=0.0
#   F1_EVAL_EVERY=10     # 0 disables epoch-end validation F1; final test F1 still runs
#   DETECT_ANOMALY=0     # 1 enables PyTorch anomaly detection for debugging
#   OUTPUT_SUBDIR=output/mingling2_cam01

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

APPTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJECT_ROOT/apptainer/deep_fformation_dante.sif}"

FOLD="${FOLD:-0}"
DATASET="${DATASET:-mingling1/cam06}"
DATASET_LABEL="${DATASET//\//_}"
TRAIN="${TRAIN:-1}"
DATASET_MAKE="${DATASET_MAKE:-1}"
SEQ_LEN="${SEQ_LEN:-10}"
FRAME_STRIDE="${FRAME_STRIDE:-20}"
HIDDEN_DIM="${HIDDEN_DIM:-8}"
THRESHOLD="${THRESHOLD:-1.0}"
NUM_EPOCHS="${NUM_EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-0.001}"
PATIENCE="${PATIENCE:-50}"
MIN_DELTA="${MIN_DELTA:-0.0}"
F1_EVAL_EVERY="${F1_EVAL_EVERY:-10}"
DETECT_ANOMALY="${DETECT_ANOMALY:-0}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-output/${DATASET_LABEL}}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/logs/cache}"
RUNTIME_TMP_ROOT="${RUNTIME_TMP_ROOT:-/tmp/graphff_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_$$}"

cd "$PROJECT_ROOT"
mkdir -p logs
mkdir -p \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/mpl" \
  "$CACHE_ROOT/cuda" \
  "$CACHE_ROOT/tmp" \
  "$RUNTIME_TMP_ROOT"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PY_SCRIPT="$PROJECT_ROOT/main_parallel.py"
DATASET_DIR="$PROJECT_ROOT/data/$DATASET"

[[ -f "$APPTAINER_IMAGE" ]] || { echo "[ERROR] image not found: $APPTAINER_IMAGE" >&2; exit 2; }
[[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] python script not found: $PY_SCRIPT" >&2; exit 2; }
[[ -d "$DATASET_DIR" ]] || { echo "[ERROR] dataset dir not found: $DATASET_DIR" >&2; exit 2; }
[[ -f "$DATASET_DIR/features.csv" ]] || { echo "[ERROR] missing: $DATASET_DIR/features.csv" >&2; exit 2; }
[[ -f "$DATASET_DIR/GT.csv" ]] || { echo "[ERROR] missing: $DATASET_DIR/GT.csv" >&2; exit 2; }
[[ -f "$DATASET_DIR/group_names.txt" ]] || { echo "[ERROR] missing: $DATASET_DIR/group_names.txt" >&2; exit 2; }
[[ -f "$DATASET_DIR/scene_continuity.csv" ]] || { echo "[ERROR] missing: $DATASET_DIR/scene_continuity.csv" >&2; exit 2; }

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}"
echo "project_root=$PROJECT_ROOT"
echo "apptainer_image=$APPTAINER_IMAGE"
echo "dataset=$DATASET"
echo "dataset_dir=$DATASET_DIR"
echo "fold=$FOLD"
echo "train=$TRAIN"
echo "dataset_make=$DATASET_MAKE"
echo "seq_len=$SEQ_LEN"
echo "frame_stride=$FRAME_STRIDE"
echo "hidden_dim=$HIDDEN_DIM"
echo "threshold=$THRESHOLD"
echo "num_epochs=$NUM_EPOCHS"
echo "batch_size=$BATCH_SIZE"
echo "lr=$LR"
echo "patience=$PATIENCE"
echo "min_delta=$MIN_DELTA"
echo "f1_eval_every=$F1_EVAL_EVERY"
echo "detect_anomaly=$DETECT_ANOMALY"
echo "output_subdir=$OUTPUT_SUBDIR"
echo "cache_root=$CACHE_ROOT"
echo "runtime_tmp_root=$RUNTIME_TMP_ROOT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

apptainer_env_args=(
  --env PROJECT_ROOT="$PROJECT_ROOT"
  --env PYTHONPATH="$PROJECT_ROOT"
  --env FOLD="$FOLD"
  --env GRAPHFF_DATASET="$DATASET"
  --env GRAPHFF_TRAIN="$TRAIN"
  --env GRAPHFF_DATASET_MAKE="$DATASET_MAKE"
  --env GRAPHFF_SEQ_LEN="$SEQ_LEN"
  --env GRAPHFF_FRAME_STRIDE="$FRAME_STRIDE"
  --env GRAPHFF_HIDDEN_DIM="$HIDDEN_DIM"
  --env GRAPHFF_THRESHOLD="$THRESHOLD"
  --env GRAPHFF_NUM_EPOCHS="$NUM_EPOCHS"
  --env GRAPHFF_BATCH_SIZE="$BATCH_SIZE"
  --env GRAPHFF_LR="$LR"
  --env GRAPHFF_PATIENCE="$PATIENCE"
  --env GRAPHFF_MIN_DELTA="$MIN_DELTA"
  --env GRAPHFF_F1_EVAL_EVERY="$F1_EVAL_EVERY"
  --env GRAPHFF_DETECT_ANOMALY="$DETECT_ANOMALY"
  --env GRAPHFF_OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_SUBDIR"
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg"
  --env TORCH_HOME="$CACHE_ROOT/torch"
  --env MPLCONFIGDIR="$CACHE_ROOT/mpl"
  --env CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
  --env TMPDIR="$RUNTIME_TMP_ROOT"
)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  apptainer_env_args+=(--env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi

apptainer exec --cleanenv --nv \
  "${apptainer_env_args[@]}" \
  -B "$HOME:$HOME" \
  -B "$PROJECT_ROOT:$PROJECT_ROOT" \
  -B "$RUNTIME_TMP_ROOT:$RUNTIME_TMP_ROOT" \
  "$APPTAINER_IMAGE" \
  bash -lc '
    set -euo pipefail
    cd "$PROJECT_ROOT"
    echo "container_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi
    else
      echo "[WARN] nvidia-smi is not available inside the container"
    fi
    find_python_with_torch() {
      local candidates=(
        "/opt/venv/bin/python"
        "/usr/local/bin/python"
        "/usr/local/bin/python3"
        "/opt/conda/bin/python"
        "/miniconda/bin/python"
        "/venv/bin/python"
      )
      if command -v python >/dev/null 2>&1; then
        candidates+=("$(command -v python)")
      fi
      if command -v python3 >/dev/null 2>&1; then
        candidates+=("$(command -v python3)")
      fi
      for py in "${candidates[@]}"; do
        [[ -x "$py" ]] || continue
        if "$py" -c "import torch" >/dev/null 2>&1; then
          echo "$py"
          return 0
        fi
      done
      return 1
    }
    PYBIN="$(find_python_with_torch || true)"
    if [[ -z "${PYBIN}" ]]; then
      echo "[ERROR] Could not find a Python interpreter with torch inside the container."
      exit 3
    fi
    echo "python_bin=${PYBIN}"
    "${PYBIN}" -V
    "${PYBIN}" - <<'"'"'PY'"'"'
import torch

print("torch_cuda_is_available:", torch.cuda.is_available())
print("torch_cuda_device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed: torch cannot see a CUDA GPU inside the container.")
PY
    "${PYBIN}" main_parallel.py "$FOLD"
  '
