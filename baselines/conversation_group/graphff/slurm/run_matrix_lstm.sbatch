#!/usr/bin/env bash
#SBATCH --job-name=lstm-matrix
#SBATCH --array=0-4
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/nfs/zli33/slurm_outputs/lstm/slurm-%x-%A_%a.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/lstm/slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --signal=B:USR1@120
#
# Cross-camera evaluation of already-trained LSTM/GraphFF models. One array task
# per *evaluation* camera: that task loads the camera's data once and scores all
# five training cameras' models on it, for every fold. Nothing is trained.
#
# Normally submitted through slurm/submit_matrix_lstm.sh, which also queues the
# dependent report job. Direct use:
#
#   sbatch --export=ALL,RUN_ID=1 slurm/run_matrix_lstm.sbatch
#   sbatch --array=1-1 --export=ALL,RUN_ID=1 slurm/run_matrix_lstm.sbatch
#
# The evaluation camera comes from the array index (06, 08, 10, 01, 03). Do NOT
# pass EVAL_CAM to select one, it would apply to every task; narrow the array
# instead (--array=1-1).
#
# Unlike training this job is CPU by default -- inference on a 522-parameter
# model is dispatch-bound and the scene scoring is pure Python. USE_GPU=1 asks
# for a GPU; it needs --gres=gpu:1 too, which submit_matrix_lstm.sh adds.
#
# Missing checkpoints do not fail the task: the cell is recorded as
# status=missing_checkpoint and the report names it.
#
# Environment overrides:
#   RUN_ID=1                 # which experiment to read, i.e. exp_$RUN_ID
#   TRAIN_CAMS=all           # which trained models to score (all | 06,08)
#   FOLDS=all                # which folds to score (all | 0,2)
#   USE_GPU=0                # 1 runs inference on a GPU
#   REBUILD_DATA=0           # 1 rebuilds split tensors instead of using _cache
#   PAIR_FILES=1             # 0 skips the evaluations/<train>@<test>.csv files
#   DISTANCE_RESCALE=1       # 0 feeds each model the evaluation camera's own
#                            # distance scale; reproduces the uncorrected numbers
#   DISTANCE_SCALERS=$PROJECT_ROOT/config/mingling_distance_scalers.json
#                            # from scripts/compute_distance_scalers.py; must sit
#                            # under a bound directory, PROJECT_ROOT by default
#   GRAPHFF_DATA_ROOT=/path/to/LSTM
#   GRAPHFF_EXPERIMENT_ROOT=/path/to/experiments
#   SEQ_LEN=10  FRAME_STRIDE=20
#   APPTAINER_IMAGE=/path/to/image.sif

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

ALL_CAMS=(06 08 10 01 03)
# the evaluation camera is the array index; EVAL_CAM is only a fallback for
# non-array use
EVAL_CAM="${EVAL_CAM:-${ALL_CAMS[${SLURM_ARRAY_TASK_ID:-0}]}}"
RUN_ID="${RUN_ID:-1}"
TRAIN_CAMS="${TRAIN_CAMS:-all}"
FOLDS="${FOLDS:-all}"
USE_GPU="${USE_GPU:-0}"
REBUILD_DATA="${REBUILD_DATA:-0}"
PAIR_FILES="${PAIR_FILES:-1}"
DISTANCE_RESCALE="${DISTANCE_RESCALE:-1}"
DISTANCE_SCALERS="${DISTANCE_SCALERS:-$PROJECT_ROOT/config/mingling_distance_scalers.json}"
SEQ_LEN="${SEQ_LEN:-10}"
FRAME_STRIDE="${FRAME_STRIDE:-20}"
GRAPHFF_DATA_ROOT="${GRAPHFF_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/LSTM}"
GRAPHFF_EXPERIMENT_ROOT="${GRAPHFF_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/LSTM}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/logs/cache}"
RUNTIME_TMP_ROOT="${RUNTIME_TMP_ROOT:-/tmp/graphff_matrix_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_$$}"

cd "$PROJECT_ROOT"
mkdir -p logs
mkdir -p \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/torch" \
  "$CACHE_ROOT/mpl" \
  "$CACHE_ROOT/cuda" \
  "$RUNTIME_TMP_ROOT"

cleanup_scratch() {
  case "$RUNTIME_TMP_ROOT" in
    /tmp/graphff_matrix_*)
      rm -rf "$RUNTIME_TMP_ROOT"
      echo "cleared node scratch: $RUNTIME_TMP_ROOT"
      ;;
    *)
      echo "leaving node scratch in place (not under /tmp/graphff_matrix_): $RUNTIME_TMP_ROOT"
      ;;
  esac
}

# A task killed at the wall clock limit stops mid-camera and looks exactly like
# one still running; --signal=B:USR1@120 warns this shell first so the SIGTERM
# that follows can be reported as TIMEOUT rather than as a cancellation.
TASK_START_EPOCH="$(date +%s)"
TIMEOUT_WARNED=0
TERM_SIGNALLED=0
on_timeout_warning() {
  TIMEOUT_WARNED=1
  echo "[TASK WARN] wall clock limit is near; Slurm terminates this task shortly"
}
on_terminate() { TERM_SIGNALLED=1; }
trap on_timeout_warning USR1
trap on_terminate TERM INT

log_status_line() {
  echo "$1"
  if [[ -n "${TASK_LOG:-}" && -f "${TASK_LOG:-}" ]]; then
    echo "$1" >> "$TASK_LOG"
  fi
}

on_exit() {
  local rc=$?
  local end_epoch elapsed status
  end_epoch="$(date +%s)"
  elapsed="$(printf '%02d:%02d:%02d' \
    $(( (end_epoch - TASK_START_EPOCH) / 3600 )) \
    $(( (end_epoch - TASK_START_EPOCH) % 3600 / 60 )) \
    $(( (end_epoch - TASK_START_EPOCH) % 60 )))"

  if [[ "$TERM_SIGNALLED" == "1" && "$TIMEOUT_WARNED" == "1" ]]; then
    status=TIMEOUT
  elif [[ "$TERM_SIGNALLED" == "1" ]]; then
    status=CANCELLED
  elif [[ "$rc" -eq 0 ]]; then
    status=COMPLETED
  else
    status=FAILED
  fi

  log_status_line "[TASK STATUS] model=lstm stage=matrix eval_cam=cam${EVAL_CAM} run_id=${RUN_ID:-NA} status=$status exit_code=$rc elapsed=$elapsed end=$(date -Is) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}}_${SLURM_ARRAY_TASK_ID:-NA} host=$(hostname)"

  if [[ -n "${TASK_LOG_DIR:-}" && -d "${TASK_LOG_DIR:-}" ]]; then
    {
      echo "status=$status"
      echo "exit_code=$rc"
      echo "model=lstm"
      echo "stage=matrix"
      echo "eval_cam=cam${EVAL_CAM}"
      echo "run_id=${RUN_ID:-NA}"
      echo "job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}}_${SLURM_ARRAY_TASK_ID:-NA}"
      echo "host=$(hostname)"
      echo "elapsed=$elapsed"
      echo "end=$(date -Is)"
    } > "$TASK_LOG_DIR/matrix_eval_cam${EVAL_CAM}.status"
  fi

  cleanup_scratch

  if [[ "$status" != "COMPLETED" && "$rc" -eq 0 ]]; then
    exit 1
  fi
}
trap on_exit EXIT

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

PY_SCRIPT="$PROJECT_ROOT/scripts/evaluate_matrix_lstm.py"
# must match graphff_paths.evaluation_results_dir
RESULTS_ROOT="$GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID/evaluations/results"
TASK_LOG_DIR="$RESULTS_ROOT/logs"
TASK_LOG="$TASK_LOG_DIR/matrix_eval_cam${EVAL_CAM}.log"

[[ -f "$APPTAINER_IMAGE" ]] || { echo "[ERROR] image not found: $APPTAINER_IMAGE" >&2; exit 2; }
[[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] python script not found: $PY_SCRIPT" >&2; exit 2; }
[[ -d "$GRAPHFF_DATA_ROOT" ]] || { echo "[ERROR] data root not found: $GRAPHFF_DATA_ROOT" >&2; exit 2; }
[[ -d "$GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID" ]] || {
  echo "[ERROR] experiment not found: $GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID" >&2
  echo "        There is nothing to evaluate; train it first or pass another RUN_ID." >&2
  exit 2
}
if [[ "$DISTANCE_RESCALE" == "1" ]]; then
  [[ -f "$DISTANCE_SCALERS" ]] || {
    echo "[ERROR] distance scaler manifest not found: $DISTANCE_SCALERS" >&2
    echo "        Build it once with scripts/compute_distance_scalers.py, or set" >&2
    echo "        DISTANCE_RESCALE=0 to evaluate without it (uncorrected numbers)." >&2
    exit 2
  }
elif [[ "$DISTANCE_RESCALE" != "0" ]]; then
  echo "[ERROR] DISTANCE_RESCALE must be 0 or 1, got: $DISTANCE_RESCALE" >&2
  exit 2
fi

mkdir -p "$TASK_LOG_DIR"
: > "$TASK_LOG"
rm -f "$TASK_LOG_DIR/matrix_eval_cam${EVAL_CAM}.status"
log_status_line "[TASK BEGIN] model=lstm stage=matrix eval_cam=cam${EVAL_CAM} run_id=$RUN_ID start=$(date -Is) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}}_${SLURM_ARRAY_TASK_ID:-NA} host=$(hostname)"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} ARRAY=${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA}"
echo "project_root=$PROJECT_ROOT"
echo "apptainer_image=$APPTAINER_IMAGE"
echo "graphff_data_root=$GRAPHFF_DATA_ROOT"
echo "graphff_experiment_root=$GRAPHFF_EXPERIMENT_ROOT"
echo "run_id=$RUN_ID"
echo "eval_cam=cam${EVAL_CAM}"
echo "train_cams=$TRAIN_CAMS"
echo "folds=$FOLDS"
echo "results_root=$RESULTS_ROOT"
echo "task_log=$TASK_LOG"
echo "use_gpu=$USE_GPU"
echo "rebuild_data=$REBUILD_DATA"
echo "pair_files=$PAIR_FILES"
echo "distance_rescale=$DISTANCE_RESCALE"
echo "distance_scalers=$DISTANCE_SCALERS"
echo "seq_len=$SEQ_LEN"
echo "frame_stride=$FRAME_STRIDE"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

apptainer_env_args=(
  --env PROJECT_ROOT="$PROJECT_ROOT"
  --env PYTHONPATH="$PROJECT_ROOT"
  # --cleanenv hides the host environment, and the error path names the image
  --env APPTAINER_IMAGE="$APPTAINER_IMAGE"
  --env RUN_ID="$RUN_ID"
  --env EVAL_CAM="$EVAL_CAM"
  --env TRAIN_CAMS="$TRAIN_CAMS"
  --env FOLDS="$FOLDS"
  --env USE_GPU="$USE_GPU"
  --env REBUILD_DATA="$REBUILD_DATA"
  --env PAIR_FILES="$PAIR_FILES"
  --env DISTANCE_RESCALE="$DISTANCE_RESCALE"
  --env DISTANCE_SCALERS="$DISTANCE_SCALERS"
  --env SEQ_LEN="$SEQ_LEN"
  --env FRAME_STRIDE="$FRAME_STRIDE"
  --env GRAPHFF_DATA_ROOT="$GRAPHFF_DATA_ROOT"
  --env GRAPHFF_EXPERIMENT_ROOT="$GRAPHFF_EXPERIMENT_ROOT"
  --env GRAPHFF_FRAME_STRIDE="$FRAME_STRIDE"
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg"
  --env TORCH_HOME="$CACHE_ROOT/torch"
  --env MPLCONFIGDIR="$CACHE_ROOT/mpl"
  --env CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
  --env TMPDIR="$RUNTIME_TMP_ROOT"
  # the image's %environment sets PATH but not LD_LIBRARY_PATH; without conda's
  # own lib dir the loader falls back to the container's Ubuntu 18.04 libstdc++
  # and scipy fails with "GLIBCXX_3.4.26 not found"
  --env LD_LIBRARY_PATH="/opt/conda/envs/py371/lib"
)
apptainer_gpu_args=()
if [[ "$USE_GPU" == "1" ]]; then
  apptainer_gpu_args+=(--nv)
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    apptainer_env_args+=(--env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
  fi
elif [[ "$USE_GPU" == "0" ]]; then
  apptainer_env_args+=(--env CUDA_VISIBLE_DEVICES="")
else
  echo "[ERROR] USE_GPU must be 0 or 1, got: $USE_GPU" >&2
  exit 2
fi

# ${arr[@]+"${arr[@]}"} rather than "${arr[@]}": the array is empty on CPU, and
# expanding an empty array under `set -u` aborts on bash < 4.4
apptainer exec --cleanenv \
  ${apptainer_gpu_args[@]+"${apptainer_gpu_args[@]}"} \
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
    PYBIN="${PYTHON_BIN:-/opt/conda/envs/py371/bin/python}"
    if [[ ! -x "$PYBIN" ]]; then
      echo "[ERROR] no python at $PYBIN inside $APPTAINER_IMAGE"
      echo "        Set PYTHON_BIN if this image lays its environments out differently."
      exit 3
    fi
    echo "python_bin=${PYBIN}"
    # a capability check, not a version check: USE_GPU=1 must not quietly run on
    # CPU after the job has claimed a GPU
    if [[ "$USE_GPU" == "1" ]]; then
      "${PYBIN}" - <<'"'"'PY'"'"'
import torch

print("torch_cuda_is_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed: USE_GPU=1 but torch sees no GPU.")
PY
    fi
    args=(
      --exp-id "$RUN_ID"
      --eval-cam "$EVAL_CAM"
      --train-cam "$TRAIN_CAMS"
      --fold "$FOLDS"
      --seq-len "$SEQ_LEN"
      --frame-stride "$FRAME_STRIDE"
      --device "$([[ "$USE_GPU" == "1" ]] && echo cuda || echo cpu)"
    )
    if [[ "$REBUILD_DATA" == "1" ]]; then
      args+=(--rebuild-data)
    fi
    if [[ "$PAIR_FILES" == "0" ]]; then
      args+=(--no-pair-files)
    fi
    if [[ "$DISTANCE_RESCALE" == "1" ]]; then
      args+=(--distance-scalers "$DISTANCE_SCALERS")
    else
      args+=(--no-distance-rescale)
    fi
    "${PYBIN}" scripts/evaluate_matrix_lstm.py "${args[@]}"
  ' 2>&1 | tee -a "$TASK_LOG"
