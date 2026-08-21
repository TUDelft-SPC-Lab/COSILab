#!/usr/bin/env bash
# Train LSTM/GraphFF locally, no Slurm. Folds run sequentially in one process.
#
# Usage:
#   bash scripts/train_lstm_local.sh --cam=<cam> [--fold=<n>]
#
#   --cam=<cam>   required. 06 | 08 | 10 | 01 | 03 | all
#                 Unpadded numbers are zero-padded, so --cam=6 means --cam=06.
#   --fold=<n>    optional single fold 0-4; omit to run all five in sequence
#
# Everything lives under one local directory (LOCAL_ROOT, default COSILab/data):
#
#   $LOCAL_ROOT/LSTM/mingling1/cam06/{features,GT}.csv ...   input
#   $LOCAL_ROOT/deep_fformation_dante.sif                    optional container
#   $LOCAL_ROOT/experiments/exp_<RUN_ID>/...                 output
#
# If $LOCAL_ROOT/LSTM is absent but $LOCAL_ROOT/mingling1 exists, LOCAL_ROOT is
# used as the data root directly.
#
# The container is used when the .sif and the apptainer command are both present;
# otherwise the script falls back to a local Python that can import torch. Either
# way training runs on CPU unless a CUDA GPU is visible -- unlike the Slurm job,
# this script does not require one.
#
# Examples:
#   bash scripts/train_lstm_local.sh --cam=06 --fold=0
#   bash scripts/train_lstm_local.sh --cam=06
#   NUM_EPOCHS=20 F1_EVAL_EVERY=0 bash scripts/train_lstm_local.sh --cam=06 --fold=0
#   DRY_RUN=1 bash scripts/train_lstm_local.sh --cam=all
#
# Speed: the forward runs seq_len x num_neighbors = 310 LSTMCell calls in Python,
# so it is dispatch-bound -- extra CPU threads do not help, but a larger batch
# does (~3x at BATCH_SIZE=1024). Raising it changes the optimisation, so keep the
# default 128 for runs meant to match the cluster.
#
# Environment overrides:
#   LOCAL_ROOT=...       root holding input, sif and output (default COSILab/data)
#   RUN_ID=1             output goes to $LOCAL_ROOT/experiments/exp_$RUN_ID/...
#   OVERWRITE=1          0 refuses to run when a fold directory already exists
#   PYTHON=python3       interpreter for the no-container fallback
#   USE_CONTAINER=auto   auto | 1 | 0
#   DRY_RUN=0            1 prints what would run
#   NUM_EPOCHS=600  BATCH_SIZE=128  LR=0.001  PATIENCE=50  MIN_DELTA=0.0
#   SEQ_LEN=10  FRAME_STRIDE=20  HIDDEN_DIM=8  THRESHOLD=1.0
#   F1_EVAL_EVERY=10     0 disables the epoch-end F1 (the slowest part per epoch)
#   TRAIN=1  DATASET_MAKE=1  DETECT_ANOMALY=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
# graphff -> conversation_group -> baselines -> COSILab
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$PROJECT_ROOT/../../.." && pwd)/data}"

RUN_ID="${RUN_ID:-1}"
OVERWRITE="${OVERWRITE:-1}"
PYTHON="${PYTHON:-python3}"
USE_CONTAINER="${USE_CONTAINER:-auto}"
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

ALL_CAMS=(06 08 10 01 03)

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit "${1:-2}"
}

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

CAM_ARG=""
FOLD_ARG=""

require_value() {
  [[ "$2" -ge 2 ]] || { echo "[ERROR] $1 requires a value, e.g. $1=06" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cam=*)   CAM_ARG="${1#*=}"; shift ;;
    --cam)     require_value --cam $#; CAM_ARG="$2"; shift 2 ;;
    --fold=*)  FOLD_ARG="${1#*=}"; shift ;;
    --fold)    require_value --fold $#; FOLD_ARG="$2"; shift 2 ;;
    -h|--help|help) usage 0 ;;
    --) shift; break ;;
    -*) echo "[ERROR] unknown option: $1" >&2; usage 2 ;;
    *)  echo "[ERROR] unexpected positional argument: $1" >&2
        echo "        arguments are named, e.g. --cam=06 --fold=2" >&2
        usage 2 ;;
  esac
done

[[ -n "$CAM_ARG" ]] || { echo "[ERROR] --cam is required, e.g. --cam=06 or --cam=all" >&2; usage 2; }

if [[ -n "$FOLD_ARG" ]]; then
  [[ "$FOLD_ARG" =~ ^[0-4]$ ]] || { echo "[ERROR] fold must be 0-4, got: $FOLD_ARG" >&2; exit 2; }
  FOLDS=("$FOLD_ARG")
else
  FOLDS=(0 1 2 3 4)
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

# accept either <root>/LSTM/mingling1/... or <root>/mingling1/...
if [[ -d "$LOCAL_ROOT/LSTM" ]]; then
  GRAPHFF_DATA_ROOT="${GRAPHFF_DATA_ROOT:-$LOCAL_ROOT/LSTM}"
else
  GRAPHFF_DATA_ROOT="${GRAPHFF_DATA_ROOT:-$LOCAL_ROOT}"
fi
GRAPHFF_EXPERIMENT_ROOT="${GRAPHFF_EXPERIMENT_ROOT:-$LOCAL_ROOT/experiments}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$LOCAL_ROOT/deep_fformation_dante.sif}"

PY_SCRIPT="$PROJECT_ROOT/main_parallel.py"
[[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] python script not found: $PY_SCRIPT" >&2; exit 2; }
[[ -d "$GRAPHFF_DATA_ROOT" ]] || {
  echo "[ERROR] data root not found: $GRAPHFF_DATA_ROOT" >&2
  echo "        Put the LSTM input folder at \$LOCAL_ROOT/LSTM, or set GRAPHFF_DATA_ROOT." >&2
  exit 2
}

for dataset in "${DATASETS[@]}"; do
  for required in features.csv GT.csv group_names.txt scene_continuity.csv; do
    [[ -f "$GRAPHFF_DATA_ROOT/$dataset/$required" ]] || {
      echo "[ERROR] missing: $GRAPHFF_DATA_ROOT/$dataset/$required" >&2; exit 2; }
  done
done

# ------------------------- pick how to run python -------------------------
RUNNER=""
if [[ "$USE_CONTAINER" != "0" ]] && [[ -f "$APPTAINER_IMAGE" ]] && command -v apptainer >/dev/null 2>&1; then
  RUNNER="container"
elif [[ "$USE_CONTAINER" == "1" ]]; then
  echo "[ERROR] USE_CONTAINER=1 but the container is unusable:" >&2
  [[ -f "$APPTAINER_IMAGE" ]] || echo "        image not found: $APPTAINER_IMAGE" >&2
  command -v apptainer >/dev/null 2>&1 || echo "        apptainer is not on PATH" >&2
  exit 2
else
  if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "[ERROR] no usable runtime: $APPTAINER_IMAGE is absent (or apptainer missing)," >&2
    echo "        and '$PYTHON' cannot import torch. Set PYTHON=... or place the .sif." >&2
    exit 2
  fi
  RUNNER="python"
fi

NV_ARGS=()
if [[ "$RUNNER" == "container" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  NV_ARGS=(--nv)
fi

mkdir -p "$GRAPHFF_EXPERIMENT_ROOT"

echo "project root:    $PROJECT_ROOT"
echo "local root:      $LOCAL_ROOT"
echo "data root:       $GRAPHFF_DATA_ROOT"
echo "experiment root: $GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID"
echo "runner:          $RUNNER$([[ ${#NV_ARGS[@]} -gt 0 ]] && echo ' (--nv)')"
[[ "$RUNNER" == "container" ]] && echo "image:           $APPTAINER_IMAGE"
echo "cameras:         ${DATASETS[*]}"
echo "folds:           ${FOLDS[*]}"
echo "epochs:          $NUM_EPOCHS (batch $BATCH_SIZE, patience $PATIENCE, f1_eval_every $F1_EVAL_EVERY)"
echo

run_fold() {
  local dataset="$1" fold="$2"
  local log_dir="$GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID/$dataset/logs"
  mkdir -p "$log_dir"
  local log="$log_dir/fold_${fold}.log"

  local env_pairs=(
    "PYTHONPATH=$PROJECT_ROOT"
    "RUN_ID=$RUN_ID"
    "GRAPHFF_DATASET=$dataset"
    "GRAPHFF_DATA_ROOT=$GRAPHFF_DATA_ROOT"
    "GRAPHFF_EXPERIMENT_ROOT=$GRAPHFF_EXPERIMENT_ROOT"
    "GRAPHFF_OVERWRITE=$OVERWRITE"
    "GRAPHFF_TRAIN=$TRAIN"
    "GRAPHFF_DATASET_MAKE=$DATASET_MAKE"
    "GRAPHFF_SEQ_LEN=$SEQ_LEN"
    "GRAPHFF_FRAME_STRIDE=$FRAME_STRIDE"
    "GRAPHFF_HIDDEN_DIM=$HIDDEN_DIM"
    "GRAPHFF_THRESHOLD=$THRESHOLD"
    "GRAPHFF_NUM_EPOCHS=$NUM_EPOCHS"
    "GRAPHFF_BATCH_SIZE=$BATCH_SIZE"
    "GRAPHFF_LR=$LR"
    "GRAPHFF_PATIENCE=$PATIENCE"
    "GRAPHFF_MIN_DELTA=$MIN_DELTA"
    "GRAPHFF_F1_EVAL_EVERY=$F1_EVAL_EVERY"
    "GRAPHFF_DETECT_ANOMALY=$DETECT_ANOMALY"
  )

  echo "=== $dataset fold $fold -> $log"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    if [[ "$RUNNER" == "container" ]]; then
      echo "    apptainer exec --cleanenv ${NV_ARGS[*]+${NV_ARGS[*]} }-B ... $APPTAINER_IMAGE \\"
      echo "      /opt/conda/envs/py371/bin/python main_parallel.py $fold"
    else
      echo "    (cd $PROJECT_ROOT && env ${env_pairs[*]} $PYTHON main_parallel.py $fold)"
    fi
    return 0
  fi

  local env_args=()
  local pair
  if [[ "$RUNNER" == "container" ]]; then
    for pair in "${env_pairs[@]}"; do env_args+=(--env "$pair"); done
    apptainer exec --cleanenv \
      ${NV_ARGS[@]+"${NV_ARGS[@]}"} \
      "${env_args[@]}" \
      -B "$PROJECT_ROOT:$PROJECT_ROOT" \
      -B "$LOCAL_ROOT:$LOCAL_ROOT" \
      --pwd "$PROJECT_ROOT" \
      "$APPTAINER_IMAGE" \
      /opt/conda/envs/py371/bin/python main_parallel.py "$fold" 2>&1 | tee "$log"
  else
    (cd "$PROJECT_ROOT" && env "${env_pairs[@]}" "$PYTHON" main_parallel.py "$fold") 2>&1 | tee "$log"
  fi
}

started=$(date +%s)
for dataset in "${DATASETS[@]}"; do
  for fold in "${FOLDS[@]}"; do
    run_fold "$dataset" "$fold"
    echo
  done
done
elapsed=$(( $(date +%s) - started ))
printf 'done in %dh%02dm%02ds\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60))
echo "results under $GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID"
echo
echo "aggregate with:"
echo "  GRAPHFF_EXPERIMENT_ROOT=$GRAPHFF_EXPERIMENT_ROOT \\"
echo "    python3 scripts/aggregate_lstm_mingling_results.py --run-id $RUN_ID"
