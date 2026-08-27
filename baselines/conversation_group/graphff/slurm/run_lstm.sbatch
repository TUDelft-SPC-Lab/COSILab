#!/usr/bin/env bash
#SBATCH --job-name=lstm
#SBATCH --array=0-4
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --time=32:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --output=/home/nfs/zli33/slurm_outputs/lstm/slurm-%x-%A_%a.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/lstm/slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zli33@tudelft.nl
#SBATCH --signal=B:USR1@120
#
# One LSTM/GraphFF camera, one array task per fold. Normally submitted through
# slurm/submit_lstm.sh, which resolves the camera number and creates the log
# directory. Direct use:
#
#   sbatch --export=ALL,DATASET=mingling1/cam06 slurm/run_lstm.sbatch
#   sbatch --array=2-2 --export=ALL,DATASET=mingling1/cam06 slurm/run_lstm.sbatch
#
# Unlike DANTE this pipeline is GPU-only: it asserts torch.cuda.is_available()
# inside the container and fails the task if no GPU is visible.
#
# The #SBATCH log directory is a literal because Slurm does not expand variables
# in these directives; it must exist at submit time, and an explicit
# `sbatch --output=` overrides it.
#
# The fold comes from the array index. Do NOT pass FOLD to select a single fold,
# it would apply to every task; narrow the array instead (--array=2-2).
#
# Mail goes out once per array (END,FAIL), not once per task; submit_lstm.sh
# overrides --mail-user/--mail-type from MAIL_USER/MAIL_TYPE.
#
# Every fold ends with a [FOLD STATUS] line in its fold log and a matching
# <logs>/fold_<n>.status file, so COMPLETED / TIMEOUT / FAILED / CANCELLED is
# readable without guessing from where the log stops.
#
# Environment overrides:
#   RUN_ID=1             # output goes to <exp_root>/exp_$RUN_ID/$DATASET/fold_$FOLD
#   OVERWRITE=1          # 0 refuses to run when that fold directory already exists
#   GRAPHFF_DATA_ROOT=/path/to/LSTM
#   GRAPHFF_EXPERIMENT_ROOT=/path/to/experiments
#   TRAIN=1              # 0 reloads a saved model and only re-evaluates
#   DATASET_MAKE=1       # 0 reuses the cached split tensors under <exp_root>/_cache
#   SEQ_LEN=10  FRAME_STRIDE=20  HIDDEN_DIM=8  THRESHOLD=1.0
#   NUM_EPOCHS=600  BATCH_SIZE=128  LR=0.001  PATIENCE=50  MIN_DELTA=0.0
#   F1_EVAL_EVERY=10     # 0 disables epoch-end validation F1; final F1 still runs
#   DETECT_ANOMALY=0     # 1 enables PyTorch anomaly detection for debugging
#   APPTAINER_IMAGE=/path/to/image.sif
#     Default: /tudelft.net/staff-umbrella/neon/apptainer/deep_fformation_dante.sif

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

APPTAINER_IMAGE="${APPTAINER_IMAGE:-/tudelft.net/staff-umbrella/neon/apptainer/deep_fformation_dante.sif}"

# the fold is the array index; FOLD is only a fallback for non-array use
FOLD="${SLURM_ARRAY_TASK_ID:-${FOLD:-0}}"
DATASET="${DATASET:-mingling1/cam06}"
RUN_ID="${RUN_ID:-1}"
OVERWRITE="${OVERWRITE:-1}"
GRAPHFF_DATA_ROOT="${GRAPHFF_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/LSTM}"
GRAPHFF_EXPERIMENT_ROOT="${GRAPHFF_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/LSTM/experiments}"
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

# Clear node-local scratch on exit, however the task ends. Guarded by the path
# prefix so an explicitly supplied RUNTIME_TMP_ROOT is never removed.
cleanup_scratch() {
  case "$RUNTIME_TMP_ROOT" in
    /tmp/graphff_*)
      rm -rf "$RUNTIME_TMP_ROOT"
      echo "cleared node scratch: $RUNTIME_TMP_ROOT"
      ;;
    *)
      echo "leaving node scratch in place (not under /tmp/graphff_): $RUNTIME_TMP_ROOT"
      ;;
  esac
}

# How the fold ended has to be readable from the log itself: a task killed at the
# wall clock limit simply stops mid-epoch and looks exactly like one still
# running. --signal=B:USR1@120 warns this shell shortly before the limit, so the
# SIGTERM that follows can be reported as TIMEOUT rather than as a cancellation.
FOLD_START_EPOCH="$(date +%s)"
TIMEOUT_WARNED=0
TERM_SIGNALLED=0

on_timeout_warning() {
  TIMEOUT_WARNED=1
  echo "[FOLD WARN] wall clock limit is near; Slurm terminates this task shortly"
}
on_terminate() {
  TERM_SIGNALLED=1
}
trap on_timeout_warning USR1
trap on_terminate TERM INT

# Written to the Slurm output and appended to the fold log, so either file on its
# own says how the fold ended. The fold log may not exist yet when an early
# preflight check fails, hence the guard.
log_status_line() {
  echo "$1"
  if [[ -n "${FOLD_LOG:-}" && -f "${FOLD_LOG:-}" ]]; then
    echo "$1" >> "$FOLD_LOG"
  fi
}

on_exit() {
  local rc=$?
  local end_epoch elapsed status
  end_epoch="$(date +%s)"
  elapsed="$(printf '%02d:%02d:%02d' \
    $(( (end_epoch - FOLD_START_EPOCH) / 3600 )) \
    $(( (end_epoch - FOLD_START_EPOCH) % 3600 / 60 )) \
    $(( (end_epoch - FOLD_START_EPOCH) % 60 )))"

  # a SIGTERM decides the outcome even when the pipeline still reports 0: bash
  # keeps running after a trapped signal, so rc alone would call a killed fold
  # completed
  if [[ "$TERM_SIGNALLED" == "1" && "$TIMEOUT_WARNED" == "1" ]]; then
    status=TIMEOUT
  elif [[ "$TERM_SIGNALLED" == "1" ]]; then
    status=CANCELLED
  elif [[ "$rc" -eq 0 ]]; then
    status=COMPLETED
  else
    status=FAILED
  fi

  log_status_line "[FOLD STATUS] model=lstm dataset=${DATASET:-NA} fold=${FOLD:-NA} run_id=${RUN_ID:-NA} status=$status exit_code=$rc elapsed=$elapsed end=$(date -Is) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}}_${SLURM_ARRAY_TASK_ID:-NA} host=$(hostname)"

  if [[ -n "${DATASET_LOG_DIR:-}" && -d "${DATASET_LOG_DIR:-}" ]]; then
    {
      echo "status=$status"
      echo "exit_code=$rc"
      echo "model=lstm"
      echo "dataset=${DATASET:-NA}"
      echo "fold=${FOLD:-NA}"
      echo "run_id=${RUN_ID:-NA}"
      echo "job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}}_${SLURM_ARRAY_TASK_ID:-NA}"
      echo "host=$(hostname)"
      echo "elapsed=$elapsed"
      echo "end=$(date -Is)"
    } > "$DATASET_LOG_DIR/fold_${FOLD}.status"
  fi

  cleanup_scratch

  # Slurm (and the FAIL mail) must see a non-zero status for anything that did
  # not finish, including a kill the pipeline swallowed
  if [[ "$status" != "COMPLETED" && "$rc" -eq 0 ]]; then
    exit 1
  fi
}
trap on_exit EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PY_SCRIPT="$PROJECT_ROOT/main_parallel.py"
DATASET_DIR="$GRAPHFF_DATA_ROOT/$DATASET"
DATASET_OUT_DIR="$GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID/$DATASET"
DATASET_LOG_DIR="$DATASET_OUT_DIR/logs"
FOLD_LOG="$DATASET_LOG_DIR/fold_${FOLD}.log"

[[ -f "$APPTAINER_IMAGE" ]] || { echo "[ERROR] image not found: $APPTAINER_IMAGE" >&2; exit 2; }
[[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] python script not found: $PY_SCRIPT" >&2; exit 2; }
[[ -d "$DATASET_DIR" ]] || { echo "[ERROR] dataset dir not found: $DATASET_DIR" >&2; exit 2; }
[[ -f "$DATASET_DIR/features.csv" ]] || { echo "[ERROR] missing: $DATASET_DIR/features.csv" >&2; exit 2; }
[[ -f "$DATASET_DIR/GT.csv" ]] || { echo "[ERROR] missing: $DATASET_DIR/GT.csv" >&2; exit 2; }
[[ -f "$DATASET_DIR/group_names.txt" ]] || { echo "[ERROR] missing: $DATASET_DIR/group_names.txt" >&2; exit 2; }
[[ -f "$DATASET_DIR/scene_continuity.csv" ]] || { echo "[ERROR] missing: $DATASET_DIR/scene_continuity.csv" >&2; exit 2; }

# Apptainer needs both bind sources to exist on the host before the exec.
mkdir -p "$GRAPHFF_EXPERIMENT_ROOT" "$DATASET_LOG_DIR"

# start the fold log fresh, then append everything else; the begin/status pair
# brackets the run so a truncated log is obvious
: > "$FOLD_LOG"
rm -f "$DATASET_LOG_DIR/fold_${FOLD}.status"
log_status_line "[FOLD BEGIN] model=lstm dataset=$DATASET fold=$FOLD run_id=$RUN_ID start=$(date -Is) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}}_${SLURM_ARRAY_TASK_ID:-NA} host=$(hostname) time_limit=$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' | head -n 1)"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} ARRAY=${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA}"
echo "project_root=$PROJECT_ROOT"
echo "apptainer_image=$APPTAINER_IMAGE"
echo "dataset=$DATASET"
echo "graphff_data_root=$GRAPHFF_DATA_ROOT"
echo "graphff_experiment_root=$GRAPHFF_EXPERIMENT_ROOT"
echo "dataset_dir=$DATASET_DIR"
echo "fold=$FOLD"
echo "run_id=$RUN_ID"
echo "overwrite=$OVERWRITE"
echo "fold_output_dir=$DATASET_OUT_DIR/fold_$FOLD"
echo "fold_log=$FOLD_LOG"
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
echo "cache_root=$CACHE_ROOT"
echo "runtime_tmp_root=$RUNTIME_TMP_ROOT"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

apptainer_env_args=(
  --env PROJECT_ROOT="$PROJECT_ROOT"
  --env PYTHONPATH="$PROJECT_ROOT"
  --env FOLD="$FOLD"
  --env RUN_ID="$RUN_ID"
  --env GRAPHFF_DATASET="$DATASET"
  --env GRAPHFF_DATA_ROOT="$GRAPHFF_DATA_ROOT"
  --env GRAPHFF_EXPERIMENT_ROOT="$GRAPHFF_EXPERIMENT_ROOT"
  --env GRAPHFF_OVERWRITE="$OVERWRITE"
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
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg"
  --env TORCH_HOME="$CACHE_ROOT/torch"
  --env MPLCONFIGDIR="$CACHE_ROOT/mpl"
  --env CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
  --env TMPDIR="$RUNTIME_TMP_ROOT"
  # the image's %environment sets PATH but not LD_LIBRARY_PATH; without conda's
  # own lib dir the loader falls back to the container's Ubuntu 18.04 libstdc++
  # (GLIBCXX <= 3.4.25) and scipy fails with "GLIBCXX_3.4.26 not found"
  --env LD_LIBRARY_PATH="/opt/conda/envs/py371/lib"
)
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  apptainer_env_args+=(--env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
fi

apptainer exec --cleanenv --nv \
  "${apptainer_env_args[@]}" \
  -B "$HOME:$HOME" \
  -B "$PROJECT_ROOT:$PROJECT_ROOT" \
  -B "$RUNTIME_TMP_ROOT:$RUNTIME_TMP_ROOT" \
  -B "$GRAPHFF_DATA_ROOT:$GRAPHFF_DATA_ROOT" \
  -B "$GRAPHFF_EXPERIMENT_ROOT:$GRAPHFF_EXPERIMENT_ROOT" \
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
  ' 2>&1 | tee -a "$FOLD_LOG"
