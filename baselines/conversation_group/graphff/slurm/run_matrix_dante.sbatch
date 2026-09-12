#!/usr/bin/env bash
#SBATCH --job-name=dante-matrix
#SBATCH --array=0-4
#SBATCH --qos=medium
#SBATCH --partition=insy,general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/nfs/zli33/slurm_outputs/dante/slurm-%x-%A_%a.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/dante/slurm-%x-%A_%a.err
#SBATCH --mail-type=END,FAIL,ARRAY_TASKS
#SBATCH --signal=B:USR1@120
#
# Cross-camera evaluation of already-trained DANTE models. One array task per
# *evaluation* camera: that task reads the camera's fold-k test blocks and scores
# all five training cameras' models on them. Nothing is trained.
#
# Normally submitted through slurm/submit_matrix_dante.sh, which also queues the
# dependent report job. Direct use:
#
#   sbatch --export=ALL,RUN_ID=1 slurm/run_matrix_dante.sbatch
#   sbatch --array=1-1 --export=ALL,RUN_ID=1 slurm/run_matrix_dante.sbatch
#
# The evaluation camera comes from the array index (06, 08, 10, 01, 03). Do NOT
# pass EVAL_CAM to select one, it would apply to every task; narrow the array
# instead (--array=1-1).
#
# CPU by default, matching the training profile. USE_GPU=1 runs TensorFlow on a
# GPU; it needs --gres=gpu:1 too, which submit_matrix_dante.sh adds.
#
# Missing checkpoints do not fail the task: the cell is recorded as
# status=missing_checkpoint and the report names it.
#
# Environment overrides:
#   RUN_ID=1                 # which experiment to read, i.e. exp_$RUN_ID
#   TRAIN_CAMS=all           # which trained models to score (all | 06,08)
#   FOLDS=all                # which folds to score (all | 0,2)
#   USE_GPU=0                # 1 runs TensorFlow on a GPU
#   NO_POINTNET=0            # 1 reads <camera>/no_pointnet/fold_<k> models
#   PAIR_FILES=1             # 0 skips the evaluations/<train>@<test>.csv files
#   DANTE_DATA_ROOT=/path/to/DANTE
#   DANTE_EXPERIMENT_ROOT=/path/to/experiments
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

APPTAINER_IMAGE="${APPTAINER_IMAGE:-/tudelft.net/staff-umbrella/neon/apptainer/deep_fformation_dante.sif}"

ALL_CAMS=(06 08 10 01 03)
# the evaluation camera is the array index; EVAL_CAM is only a fallback for
# non-array use
EVAL_CAM="${EVAL_CAM:-${ALL_CAMS[${SLURM_ARRAY_TASK_ID:-0}]}}"
RUN_ID="${RUN_ID:-1}"
TRAIN_CAMS="${TRAIN_CAMS:-all}"
FOLDS="${FOLDS:-all}"
NO_POINTNET="${NO_POINTNET:-0}"
PAIR_FILES="${PAIR_FILES:-1}"
DANTE_DATA_ROOT="${DANTE_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/DANTE}"
DANTE_EXPERIMENT_ROOT="${DANTE_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/DANTE}"
DANTE_USE_GPU="${USE_GPU:-${DANTE_USE_GPU:-0}}"
DANTE_PYTHON_BIN="${DANTE_PYTHON_BIN:-}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/logs/cache}"
RUNTIME_TMP_ROOT="${RUNTIME_TMP_ROOT:-/tmp/dante_matrix_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}_$$}"

cd "$PROJECT_ROOT"
mkdir -p logs
mkdir -p \
  "$CACHE_ROOT/xdg" \
  "$CACHE_ROOT/mpl" \
  "$CACHE_ROOT/tf" \
  "$CACHE_ROOT/cuda" \
  "$RUNTIME_TMP_ROOT"

cleanup_scratch() {
  case "$RUNTIME_TMP_ROOT" in
    /tmp/dante_matrix_*)
      rm -rf "$RUNTIME_TMP_ROOT"
      echo "cleared node scratch: $RUNTIME_TMP_ROOT"
      ;;
    *)
      echo "leaving node scratch in place (not under /tmp/dante_matrix_): $RUNTIME_TMP_ROOT"
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

  log_status_line "[TASK STATUS] model=dante stage=matrix eval_cam=cam${EVAL_CAM} run_id=${RUN_ID:-NA} status=$status exit_code=$rc elapsed=$elapsed end=$(date -Is) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}}_${SLURM_ARRAY_TASK_ID:-NA} host=$(hostname)"

  if [[ -n "${TASK_LOG_DIR:-}" && -d "${TASK_LOG_DIR:-}" ]]; then
    {
      echo "status=$status"
      echo "exit_code=$rc"
      echo "model=dante"
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

DANTE_ROOT="$PROJECT_ROOT/DANTE-master"
PY_SCRIPT="$PROJECT_ROOT/scripts/evaluate_matrix_dante.py"
# must match dante_paths.evaluation_results_dir
RESULTS_ROOT="$DANTE_EXPERIMENT_ROOT/exp_$RUN_ID/evaluations/results"
TASK_LOG_DIR="$RESULTS_ROOT/logs"
TASK_LOG="$TASK_LOG_DIR/matrix_eval_cam${EVAL_CAM}.log"

[[ -f "$APPTAINER_IMAGE" ]] || { echo "[ERROR] image not found: $APPTAINER_IMAGE" >&2; exit 2; }
[[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] python script not found: $PY_SCRIPT" >&2; exit 2; }
[[ -d "$DANTE_DATA_ROOT" ]] || { echo "[ERROR] data root not found: $DANTE_DATA_ROOT" >&2; exit 2; }
[[ -d "$DANTE_EXPERIMENT_ROOT/exp_$RUN_ID" ]] || {
  echo "[ERROR] experiment not found: $DANTE_EXPERIMENT_ROOT/exp_$RUN_ID" >&2
  echo "        There is nothing to evaluate; train it first or pass another RUN_ID." >&2
  exit 2
}

mkdir -p "$TASK_LOG_DIR"
: > "$TASK_LOG"
rm -f "$TASK_LOG_DIR/matrix_eval_cam${EVAL_CAM}.status"
log_status_line "[TASK BEGIN] model=dante stage=matrix eval_cam=cam${EVAL_CAM} run_id=$RUN_ID start=$(date -Is) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-NA}}_${SLURM_ARRAY_TASK_ID:-NA} host=$(hostname)"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} ARRAY=${SLURM_ARRAY_JOB_ID:-NA}_${SLURM_ARRAY_TASK_ID:-NA}"
echo "project_root=$PROJECT_ROOT"
echo "apptainer_image=$APPTAINER_IMAGE"
echo "dante_root=$DANTE_ROOT"
echo "dante_data_root=$DANTE_DATA_ROOT"
echo "dante_experiment_root=$DANTE_EXPERIMENT_ROOT"
echo "run_id=$RUN_ID"
echo "eval_cam=cam${EVAL_CAM}"
echo "train_cams=$TRAIN_CAMS"
echo "folds=$FOLDS"
echo "results_root=$RESULTS_ROOT"
echo "task_log=$TASK_LOG"
echo "no_pointnet=$NO_POINTNET"
echo "pair_files=$PAIR_FILES"
echo "dante_use_gpu=$DANTE_USE_GPU"
echo "dante_python_bin=${DANTE_PYTHON_BIN:-auto}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

apptainer_env_args=(
  --env PROJECT_ROOT="$PROJECT_ROOT"
  --env DANTE_ROOT="$DANTE_ROOT"
  # --cleanenv hides the host environment, and the error path names the image
  --env APPTAINER_IMAGE="$APPTAINER_IMAGE"
  --env RUN_ID="$RUN_ID"
  --env EVAL_CAM="$EVAL_CAM"
  --env TRAIN_CAMS="$TRAIN_CAMS"
  --env FOLDS="$FOLDS"
  --env NO_POINTNET="$NO_POINTNET"
  --env PAIR_FILES="$PAIR_FILES"
  --env DANTE_DATA_ROOT="$DANTE_DATA_ROOT"
  --env DANTE_EXPERIMENT_ROOT="$DANTE_EXPERIMENT_ROOT"
  --env DANTE_USE_GPU="$DANTE_USE_GPU"
  --env DANTE_PYTHON_BIN="$DANTE_PYTHON_BIN"
  --env PYTHONPATH="$DANTE_ROOT/deep_fformation:$DANTE_ROOT/datasets:$PROJECT_ROOT/scripts"
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg"
  --env MPLCONFIGDIR="$CACHE_ROOT/mpl"
  --env CUDA_CACHE_PATH="$CACHE_ROOT/cuda"
  --env TMPDIR="$RUNTIME_TMP_ROOT"
  --env TF_FORCE_GPU_ALLOW_GROWTH=true
)
apptainer_gpu_args=()
if [[ "$DANTE_USE_GPU" == "1" ]]; then
  apptainer_gpu_args+=(--nv)
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    apptainer_env_args+=(--env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES")
  fi
elif [[ "$DANTE_USE_GPU" == "0" ]]; then
  apptainer_env_args+=(--env CUDA_VISIBLE_DEVICES="")
else
  echo "[ERROR] USE_GPU must be 0 or 1, got: $DANTE_USE_GPU" >&2
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
  -B "$DANTE_DATA_ROOT:$DANTE_DATA_ROOT" \
  -B "$DANTE_EXPERIMENT_ROOT:$DANTE_EXPERIMENT_ROOT" \
  "$APPTAINER_IMAGE" \
  bash -lc '
    set -euo pipefail
    cd "$PROJECT_ROOT"
    if [[ "$DANTE_USE_GPU" == "0" ]]; then
      export CUDA_VISIBLE_DEVICES=""
      echo "DANTE GPU disabled; running TensorFlow on CPU."
    fi
    # the TensorFlow 1.x / Keras environment of the image; the image also exports
    # DANTE_PYTHON_BIN, and --env passes an empty one through when the host has
    # none, hence :- rather than -
    PYBIN="${DANTE_PYTHON_BIN:-/opt/conda/envs/dante_tf1/bin/python}"
    if [[ ! -x "$PYBIN" ]]; then
      echo "[ERROR] no python at $PYBIN inside $APPTAINER_IMAGE"
      echo "        Set DANTE_PYTHON_BIN if this image lays its environments out differently."
      exit 3
    fi
    echo "python_bin=${PYBIN}"
    args=(
      --exp-id "$RUN_ID"
      --eval-cam "$EVAL_CAM"
      --train-cam "$TRAIN_CAMS"
      --fold "$FOLDS"
    )
    if [[ "$NO_POINTNET" == "1" ]]; then
      args+=(--no-pointnet)
    fi
    if [[ "$PAIR_FILES" == "0" ]]; then
      args+=(--no-pair-files)
    fi
    "${PYBIN}" scripts/evaluate_matrix_dante.py "${args[@]}"
  ' 2>&1 | tee -a "$TASK_LOG"
