#!/usr/bin/env bash
#SBATCH --job-name=dante
#SBATCH --array=0-4
#SBATCH --partition=insy,all
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/nfs/zli33/slurm_outputs/dante/slurm-%x-%A_%a.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/dante/slurm-%x-%A_%a.err
#
# One DANTE camera, one array task per fold. Normally submitted through
# slurm/submit_dante.sh, which resolves the camera number, sets GPU/CPU
# resources, and creates the log directory. Direct use:
#
#   sbatch --export=ALL,DATASET=mingling1/cam06 slurm/run_dante.sbatch
#   sbatch --array=2-2 --export=ALL,DATASET=mingling1/cam06 slurm/run_dante.sbatch
#
# The defaults above are the CPU profile. submit_dante.sh overrides --gres,
# --mem, --cpus-per-task and --time when USE_GPU=1. The #SBATCH log directory is
# a literal because Slurm does not expand variables in these directives; it must
# exist at submit time, and an explicit `sbatch --output=` overrides it.
#
# The fold comes from the array index. Do NOT pass FOLD to select a single fold,
# it would apply to every task; narrow the array instead (--array=2-2).
#
# Environment overrides:
#   RUN_ID=1             # output goes to <exp_root>/exp_$RUN_ID/$DATASET/fold_$FOLD
#   OVERWRITE=1          # 0 refuses to run when that fold directory already exists
#   USE_GPU=0            # 1 runs TensorFlow on a GPU (needs --gres=gpu:1 too)
#   DANTE_DATA_ROOT=/path/to/DANTE
#   DANTE_EXPERIMENT_ROOT=/path/to/experiments
#   EPOCHS=600  BATCH_SIZE=1024  PATIENCE=50  MIN_DELTA=0.0  F1_EVAL_EVERY=10
#   NO_POINTNET=0  SYMMETRIC=0
#   APPTAINER_IMAGE=/path/to/image.sif
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

APPTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJECT_ROOT/apptainer/deep_fformation_dante.sif}"

# the fold is the array index; FOLD is only a fallback for non-array use
FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
DATASET="${DATASET:-mingling1/cam06}"
RUN_ID="${RUN_ID:-1}"
OVERWRITE="${OVERWRITE:-1}"
DANTE_DATA_ROOT="${DANTE_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/DANTE}"
DANTE_EXPERIMENT_ROOT="${DANTE_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/DANTE/experiments}"
EPOCHS="${EPOCHS:-600}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
PATIENCE="${PATIENCE:-50}"
MIN_DELTA="${MIN_DELTA:-0.0}"
F1_EVAL_EVERY="${F1_EVAL_EVERY:-10}"
NO_POINTNET="${NO_POINTNET:-0}"
SYMMETRIC="${SYMMETRIC:-0}"
DANTE_USE_GPU="${USE_GPU:-${DANTE_USE_GPU:-0}}"
DANTE_PYTHON_BIN="${DANTE_PYTHON_BIN:-}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/logs/cache}"
RUNTIME_TMP_ROOT="${RUNTIME_TMP_ROOT:-/tmp/dante_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_$$}"

cd "$PROJECT_ROOT"
mkdir -p logs
mkdir -p \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/mpl" \
  "$CACHE_ROOT/tf" \
  "$CACHE_ROOT/cuda" \
  "$CACHE_ROOT/tmp" \
  "$RUNTIME_TMP_ROOT"

# Clear node-local scratch on exit, however the task ends. Guarded by the path
# prefix so an explicitly supplied RUNTIME_TMP_ROOT is never removed.
cleanup_scratch() {
  case "$RUNTIME_TMP_ROOT" in
    /tmp/dante_*)
      rm -rf "$RUNTIME_TMP_ROOT"
      echo "cleared node scratch: $RUNTIME_TMP_ROOT"
      ;;
    *)
      echo "leaving node scratch in place (not under /tmp/dante_): $RUNTIME_TMP_ROOT"
      ;;
  esac
}
trap cleanup_scratch EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

DANTE_ROOT="$PROJECT_ROOT/DANTE-master"
DATASET_DIR="$DANTE_DATA_ROOT/$DATASET"
FOLD_DIR="$DATASET_DIR/fold_$FOLD"
RUN_SCRIPT="$DANTE_ROOT/deep_fformation/run_models.py"
DATASET_OUT_DIR="$DANTE_EXPERIMENT_ROOT/exp_$RUN_ID/$DATASET"
DATASET_LOG_DIR="$DATASET_OUT_DIR/logs"
FOLD_LOG="$DATASET_LOG_DIR/fold_${FOLD}.log"

[[ -f "$APPTAINER_IMAGE" ]] || { echo "[ERROR] image not found: $APPTAINER_IMAGE" >&2; exit 2; }
[[ -f "$RUN_SCRIPT" ]] || { echo "[ERROR] DANTE train script not found: $RUN_SCRIPT" >&2; exit 2; }
[[ -d "$DATASET_DIR" ]] || { echo "[ERROR] DANTE dataset dir not found: $DATASET_DIR" >&2; exit 2; }
[[ -f "$DATASET_DIR/DS_utils/features.txt" ]] || { echo "[ERROR] missing: $DATASET_DIR/DS_utils/features.txt" >&2; exit 2; }
[[ -f "$DATASET_DIR/DS_utils/group_names.txt" ]] || { echo "[ERROR] missing: $DATASET_DIR/DS_utils/group_names.txt" >&2; exit 2; }
[[ -f "$FOLD_DIR/train.p" ]] || { echo "[ERROR] missing: $FOLD_DIR/train.p" >&2; exit 2; }
[[ -f "$FOLD_DIR/val.p" ]] || { echo "[ERROR] missing: $FOLD_DIR/val.p" >&2; exit 2; }
[[ -f "$FOLD_DIR/test.p" ]] || { echo "[ERROR] missing: $FOLD_DIR/test.p" >&2; exit 2; }

# Apptainer needs both bind sources to exist on the host before the exec.
mkdir -p "$DANTE_EXPERIMENT_ROOT" "$DATASET_LOG_DIR"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} ARRAY=${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA}"
echo "project_root=$PROJECT_ROOT"
echo "apptainer_image=$APPTAINER_IMAGE"
echo "dante_root=$DANTE_ROOT"
echo "dataset=$DATASET"
echo "dante_data_root=$DANTE_DATA_ROOT"
echo "dante_experiment_root=$DANTE_EXPERIMENT_ROOT"
echo "dataset_dir=$DATASET_DIR"
echo "fold=$FOLD"
echo "run_id=$RUN_ID"
echo "overwrite=$OVERWRITE"
echo "fold_output_dir=$DATASET_OUT_DIR/fold_$FOLD"
echo "fold_log=$FOLD_LOG"
echo "epochs=$EPOCHS"
echo "batch_size=$BATCH_SIZE"
echo "patience=$PATIENCE"
echo "min_delta=$MIN_DELTA"
echo "f1_eval_every=$F1_EVAL_EVERY"
echo "no_pointnet=$NO_POINTNET"
echo "symmetric=$SYMMETRIC"
echo "dante_use_gpu=$DANTE_USE_GPU"
echo "dante_python_bin=${DANTE_PYTHON_BIN:-auto}"
echo "cache_root=$CACHE_ROOT"
echo "runtime_tmp_root=$RUNTIME_TMP_ROOT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

apptainer_env_args=(
  --env PROJECT_ROOT="$PROJECT_ROOT"
  --env DANTE_ROOT="$DANTE_ROOT"
  --env DATASET="$DATASET"
  --env FOLD="$FOLD"
  --env RUN_ID="$RUN_ID"
  --env OVERWRITE="$OVERWRITE"
  --env DANTE_DATA_ROOT="$DANTE_DATA_ROOT"
  --env DANTE_EXPERIMENT_ROOT="$DANTE_EXPERIMENT_ROOT"
  --env EPOCHS="$EPOCHS"
  --env BATCH_SIZE="$BATCH_SIZE"
  --env PATIENCE="$PATIENCE"
  --env MIN_DELTA="$MIN_DELTA"
  --env F1_EVAL_EVERY="$F1_EVAL_EVERY"
  --env NO_POINTNET="$NO_POINTNET"
  --env SYMMETRIC="$SYMMETRIC"
  --env DANTE_USE_GPU="$DANTE_USE_GPU"
  --env DANTE_PYTHON_BIN="$DANTE_PYTHON_BIN"
  --env PYTHONPATH="$DANTE_ROOT/deep_fformation:$DANTE_ROOT/datasets:$PROJECT_ROOT"
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg"
  --env MPLCONFIGDIR="$CACHE_ROOT/mpl"
  --env CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
  --env TMPDIR="$RUNTIME_TMP_ROOT"
  --env TF_FORCE_GPU_ALLOW_GROWTH=true
)
apptainer_gpu_args=()
if [[ "$DANTE_USE_GPU" == "1" ]]; then
  apptainer_gpu_args+=(--nv)
elif [[ "$DANTE_USE_GPU" == "0" ]]; then
  apptainer_env_args+=(--env CUDA_VISIBLE_DEVICES="")
else
  echo "[ERROR] USE_GPU must be 0 or 1, got: $DANTE_USE_GPU" >&2
  exit 2
fi
if [[ "$DANTE_USE_GPU" == "1" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  apptainer_env_args+=(--env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi

# ${arr[@]+"${arr[@]}"} rather than "${arr[@]}": apptainer_gpu_args is empty on
# CPU, and expanding an empty array under `set -u` aborts on bash < 4.4
apptainer exec --cleanenv \
  ${apptainer_gpu_args[@]+"${apptainer_gpu_args[@]}"} \
  "${apptainer_env_args[@]}" \
  -B "$HOME:$HOME" \
  -B "$PROJECT_ROOT:$PROJECT_ROOT" \
  -B "$RUNTIME_TMP_ROOT:$RUNTIME_TMP_ROOT" \
  -B "$DANTE_DATA_ROOT:$DANTE_DATA_ROOT" \
  -B "$DANTE_EXPERIMENT_ROOT:$DANTE_EXPERIMENT_ROOT" \
  "$APPTAINER_IMAGE" \
  bash -lc '
    set -euo pipefail
    cd "$DANTE_ROOT/deep_fformation"
    if [[ "$DANTE_USE_GPU" == "0" ]]; then
      export CUDA_VISIBLE_DEVICES=""
      echo "DANTE GPU disabled; running TensorFlow on CPU."
    fi
    echo "container_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
    if [[ "$DANTE_USE_GPU" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi
    elif [[ "$DANTE_USE_GPU" == "1" ]]; then
      echo "[WARN] nvidia-smi is not available inside the container"
    else
      echo "Skipping nvidia-smi because DANTE_USE_GPU=0"
    fi
    find_python_with_dante_deps() {
      local candidates=(
        "/opt/conda/envs/dante_tf1/bin/python"
        "/opt/conda/envs/py371/bin/python"
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
        if "$py" -c "import tensorflow, keras" >/dev/null 2>&1; then
          echo "$py"
          return 0
        fi
      done
      return 1
    }
    if [[ -n "${DANTE_PYTHON_BIN:-}" ]]; then
      PYBIN="$DANTE_PYTHON_BIN"
      if [[ ! -x "$PYBIN" ]]; then
        echo "[ERROR] DANTE_PYTHON_BIN is not executable inside the container: $PYBIN"
        exit 3
      fi
      if ! "$PYBIN" -c "import tensorflow, keras" >/dev/null 2>&1; then
        echo "[ERROR] DANTE_PYTHON_BIN cannot import tensorflow and keras: $PYBIN"
        exit 3
      fi
    else
      PYBIN="$(find_python_with_dante_deps || true)"
    fi
    if [[ -z "${PYBIN}" ]]; then
      echo "[ERROR] Could not find a Python interpreter with tensorflow and keras inside the container."
      exit 3
    fi
    echo "python_bin=${PYBIN}"
    "${PYBIN}" -V
    "${PYBIN}" - <<'"'"'PY'"'"'
import tensorflow as tf
import keras

print("tensorflow_version:", tf.__version__)
print("keras_version:", keras.__version__)
PY
    args=(
      -d "$DATASET"
      -f "$FOLD"
      -e "$EPOCHS"
      --batch_size "$BATCH_SIZE"
      --patience "$PATIENCE"
      --min_delta "$MIN_DELTA"
      --f1_eval_every "$F1_EVAL_EVERY"
      --run_id "$RUN_ID"
    )
    if [[ "$NO_POINTNET" == "1" ]]; then
      args+=("-p")
    fi
    if [[ "$SYMMETRIC" == "1" ]]; then
      args+=("-s")
    fi
    if [[ "$OVERWRITE" == "0" ]]; then
      args+=("--no-overwrite")
    fi
    "${PYBIN}" run_models.py "${args[@]}"
  ' 2>&1 | tee "$FOLD_LOG"
