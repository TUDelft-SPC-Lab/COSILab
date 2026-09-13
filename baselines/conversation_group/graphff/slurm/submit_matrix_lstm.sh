#!/usr/bin/env bash
# Submit the LSTM/GraphFF camera x camera evaluation of an already-trained
# experiment, plus the report job that turns its cells into the 5x5 matrix.
#
# Nothing is trained here: every checkpoint of exp_$RUN_ID is scored on every
# camera's fold-matched held-out test block.
#
# Usage:
#   bash slurm/submit_matrix_lstm.sh [--eval-cam=<cam>] [--train-cam=<cams>] [--fold=<n>]
#
#   --eval-cam=<cam>    which camera to score models ON. 06|08|10|01|03|all
#                       (default all: one array task per camera)
#   --train-cam=<cams>  which trained models to score. all, or a comma-separated
#                       list (default all)
#   --fold=<n>          single fold 0-4, or a comma-separated list; omit for all
#
# Examples:
#   bash slurm/submit_matrix_lstm.sh                      # full 5x5, exp_1
#   RUN_ID=2 bash slurm/submit_matrix_lstm.sh             # full 5x5, exp_2
#   bash slurm/submit_matrix_lstm.sh --eval-cam=06        # one column of the matrix
#   bash slurm/submit_matrix_lstm.sh --train-cam=06 --eval-cam=08   # one cell's folds
#   DRY_RUN=1 bash slurm/submit_matrix_lstm.sh
#   REPORT_ONLY=1 bash slurm/submit_matrix_lstm.sh        # rebuild the matrix only
#
# Per-pair metrics go to $GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID/evaluations as
# <train>@<test>.csv, and the matrix tables to that directory's results/.
#
# Environment overrides:
#   RUN_ID=1                  which experiment to read and write results for
#   USE_GPU=0                 1 requests a GPU and runs inference on it
#   REBUILD_DATA=0            1 rebuilds split tensors instead of using _cache
#   PAIR_FILES=1              0 skips the evaluations/<train>@<test>.csv files
#   DISTANCE_RESCALE=1        0 evaluates without converting the evaluation
#                             camera's distance scale to the model's own; the
#                             numbers are then not comparable across cameras
#   DISTANCE_SCALERS=...      manifest from scripts/compute_distance_scalers.py
#   NO_REPORT=0               1 submits only the evaluation array
#   REPORT_ONLY=0             1 submits only the report (no evaluation)
#   DRY_RUN=0                 1 prints the sbatch lines without submitting
#   MAIL_USER=zli33@tudelft.nl  empty disables mail entirely
#   MAIL_TYPE=END,FAIL,ARRAY_TASKS
#   GRAPHFF_DATA_ROOT=...     benchmark artifacts to read
#   GRAPHFF_EXPERIMENT_ROOT=... where the trained experiment lives
#   SLURM_LOG_DIR=...         Slurm stdout/stderr
#   SEQ_LEN=10  FRAME_STRIDE=20   must match what the models were trained with
#   MATRIX_METRICS=f1_1,f1_2_3,auc
#   EXTRA_EXPORTS='TRAIN_CAMS=06,FOLDS=0'
#   SBATCH_ARGS='--constraint=...'

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

EVAL_SCRIPT="$PROJECT_ROOT/slurm/run_matrix_lstm.sbatch"
REPORT_SCRIPT="$PROJECT_ROOT/slurm/run_matrix_report.sbatch"
GRAPHFF_DATA_ROOT="${GRAPHFF_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/LSTM}"
GRAPHFF_EXPERIMENT_ROOT="${GRAPHFF_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/LSTM}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/home/nfs/zli33/slurm_outputs/lstm}"
RUN_ID="${RUN_ID:-1}"
USE_GPU="${USE_GPU:-0}"
REBUILD_DATA="${REBUILD_DATA:-0}"
PAIR_FILES="${PAIR_FILES:-1}"
DISTANCE_RESCALE="${DISTANCE_RESCALE:-1}"
DISTANCE_SCALERS="${DISTANCE_SCALERS:-$PROJECT_ROOT/config/mingling_distance_scalers.json}"
NO_REPORT="${NO_REPORT:-0}"
REPORT_ONLY="${REPORT_ONLY:-0}"
SEQ_LEN="${SEQ_LEN:-10}"
FRAME_STRIDE="${FRAME_STRIDE:-20}"
MATRIX_METRICS="${MATRIX_METRICS:-f1_1,f1_2_3,auc}"
# `-` not `:-`: MAIL_USER= (explicitly empty) means "no mail", not "use the default"
MAIL_USER="${MAIL_USER-zli33@tudelft.nl}"
MAIL_TYPE="${MAIL_TYPE:-END,FAIL,ARRAY_TASKS}"

ALL_CAMS=(06 08 10 01 03)

usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit "${1:-2}"
}

# camera number -> session, for the checkpoint filenames
session_of() {
  case "$1" in
    06|08|10) echo mingling1 ;;
    01|03)    echo mingling2 ;;
    *)        return 1 ;;
  esac
}

pad_cam() {
  local cam="${1#cam}"
  [[ "$cam" =~ ^[0-9]{1,2}$ ]] || return 1
  printf '%02d' "$((10#$cam))"
}

index_of_cam() {
  local target="$1" index=0
  for cam in "${ALL_CAMS[@]}"; do
    if [[ "$cam" == "$target" ]]; then
      echo "$index"
      return 0
    fi
    index=$((index + 1))
  done
  return 1
}

EVAL_CAM_ARG="all"
TRAIN_CAM_ARG="all"
FOLD_ARG="all"

require_value() {
  [[ "$2" -ge 2 ]] || { echo "[ERROR] $1 requires a value, e.g. $1=06" >&2; exit 2; }
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-cam=*)  EVAL_CAM_ARG="${1#*=}"; shift ;;
    --eval-cam)    require_value --eval-cam $#; EVAL_CAM_ARG="$2"; shift 2 ;;
    --train-cam=*) TRAIN_CAM_ARG="${1#*=}"; shift ;;
    --train-cam)   require_value --train-cam $#; TRAIN_CAM_ARG="$2"; shift 2 ;;
    --fold=*)      FOLD_ARG="${1#*=}"; shift ;;
    --fold)        require_value --fold $#; FOLD_ARG="$2"; shift 2 ;;
    -h|--help|help) usage 0 ;;
    --) shift; break ;;
    -*) echo "[ERROR] unknown option: $1" >&2; usage 2 ;;
    *)  echo "[ERROR] unexpected positional argument: $1" >&2
        echo "        arguments are named, e.g. --eval-cam=06 --fold=2" >&2
        usage 2 ;;
  esac
done

[[ -f "$EVAL_SCRIPT" ]] || { echo "[ERROR] missing Slurm script: $EVAL_SCRIPT" >&2; exit 2; }
[[ -f "$REPORT_SCRIPT" ]] || { echo "[ERROR] missing Slurm script: $REPORT_SCRIPT" >&2; exit 2; }
EXPERIMENT_DIR="$GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID"
[[ -d "$EXPERIMENT_DIR" ]] || {
  echo "[ERROR] experiment not found: $EXPERIMENT_DIR" >&2
  echo "        Train it first, or pass RUN_ID=<id> of the run that holds the models." >&2
  exit 2
}

# which array tasks to submit; one array per camera keeps --eval-cam=06,10
# working without assuming the indices are contiguous
ARRAY_SPECS=()
if [[ "$EVAL_CAM_ARG" == "all" ]]; then
  ARRAY_SPECS=("0-4")
else
  IFS=',' read -r -a requested <<< "$EVAL_CAM_ARG"
  for raw in "${requested[@]}"; do
    cam="$(pad_cam "$raw")" || { echo "[ERROR] unknown camera: $raw" >&2; exit 2; }
    index="$(index_of_cam "$cam")" || {
      echo "[ERROR] unknown camera: $raw (expected one of ${ALL_CAMS[*]}, or all)" >&2
      exit 2
    }
    ARRAY_SPECS+=("$index-$index")
  done
fi

# Preflight the checkpoints so a half-trained experiment is visible now rather
# than only in the report. Missing ones are a warning: the rest of the table is
# still worth computing, and the report records exactly which cells are affected.
stride_suffix=""
if [[ "$FRAME_STRIDE" != "1" ]]; then
  stride_suffix="_stride=$FRAME_STRIDE"
fi
found=0
missing=()
for cam in "${ALL_CAMS[@]}"; do
  session="$(session_of "$cam")"
  for fold in 0 1 2 3 4; do
    name="dataset=${session}_cam${cam}_seq=${SEQ_LEN}${stride_suffix}_model_fold${fold}.pt"
    if [[ -f "$EXPERIMENT_DIR/cam$cam/fold_$fold/$name" ]] \
       || [[ -f "$GRAPHFF_EXPERIMENT_ROOT/_cache/cam$cam/$name" ]]; then
      found=$((found + 1))
    else
      missing+=("cam$cam/fold_$fold")
    fi
  done
done
if [[ "$found" -eq 0 ]]; then
  echo "[ERROR] no LSTM checkpoints found under $EXPERIMENT_DIR" >&2
  echo "        Expected files like cam06/fold_0/dataset=mingling1_cam06_seq=${SEQ_LEN}${stride_suffix}_model_fold0.pt" >&2
  echo "        Check RUN_ID, SEQ_LEN and FRAME_STRIDE against the training run." >&2
  exit 2
fi

# Without the manifest every off-diagonal cell reads the wrong distance scale and
# collapses, so a missing one is refused here rather than discovered in the table.
if [[ "$DISTANCE_RESCALE" == "1" && ! -f "$DISTANCE_SCALERS" ]]; then
  echo "[ERROR] distance scaler manifest not found: $DISTANCE_SCALERS" >&2
  echo "        Build it once (a few minutes, CPU only):" >&2
  echo "          python scripts/compute_distance_scalers.py" >&2
  echo "        or set DISTANCE_RESCALE=0 to evaluate without it." >&2
  exit 2
fi

# Slurm rejects a job outright if the --output directory does not already exist.
if [[ ! -d "$SLURM_LOG_DIR" ]]; then
  if mkdir -p "$SLURM_LOG_DIR" 2>/dev/null; then
    echo "created slurm log dir: $SLURM_LOG_DIR"
  elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[WARN] slurm log dir is missing and not creatable here: $SLURM_LOG_DIR" >&2
  else
    echo "[ERROR] slurm log dir does not exist and could not be created: $SLURM_LOG_DIR" >&2
    exit 2
  fi
fi

echo "data root:       $GRAPHFF_DATA_ROOT"
echo "experiment:      $EXPERIMENT_DIR"
echo "evaluations:     $EXPERIMENT_DIR/evaluations"
echo "results:         $EXPERIMENT_DIR/evaluations/results"
echo "slurm log dir:   $SLURM_LOG_DIR"
echo "eval cameras:    $EVAL_CAM_ARG"
echo "train cameras:   $TRAIN_CAM_ARG"
echo "folds:           $FOLD_ARG"
echo "gpu:             $USE_GPU"
echo "distance scale:  $([[ "$DISTANCE_RESCALE" == "1" ]] && echo "$DISTANCE_SCALERS" || echo "<not rescaled>")"
echo "matrix metrics:  $MATRIX_METRICS"
echo "checkpoints:     $found of 25 found"
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[WARN] ${#missing[@]} checkpoints are missing: ${missing[*]}"
  echo "[WARN] those cells stay empty and are named in the results file."
fi
echo "mail:            ${MAIL_USER:-<disabled>} (${MAIL_TYPE})"
echo

SBATCH_MAIL_ARGS=()
if [[ -n "$MAIL_USER" ]]; then
  SBATCH_MAIL_ARGS=(--mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE")
else
  SBATCH_MAIL_ARGS=(--mail-type=NONE)
fi

SBATCH_EXTRA_ARGS=()
if [[ -n "${SBATCH_ARGS:-}" ]]; then
  read -r -a SBATCH_EXTRA_ARGS <<< "$SBATCH_ARGS"
fi

SBATCH_RESOURCE_ARGS=()
if [[ "$USE_GPU" == "1" ]]; then
  SBATCH_RESOURCE_ARGS=(--gres=gpu:1)
elif [[ "$USE_GPU" != "0" ]]; then
  echo "[ERROR] USE_GPU must be 0 or 1, got: $USE_GPU" >&2
  exit 2
fi

# sbatch --export takes a comma-separated list, so a value that itself contains a
# comma -- a camera list, a fold list, the metric list -- is cut at the first one
# and the remainder is read as further variable names to import. Everything the
# job needs therefore travels through the environment that ALL propagates, which
# has no such parsing. EXTRA_EXPORTS keeps the inline form: it is the documented
# escape hatch and its values must not contain commas.
TRAIN_CAMS="$TRAIN_CAM_ARG"
FOLDS="$FOLD_ARG"
export RUN_ID TRAIN_CAMS FOLDS USE_GPU REBUILD_DATA PAIR_FILES
export DISTANCE_RESCALE DISTANCE_SCALERS SEQ_LEN FRAME_STRIDE
export GRAPHFF_DATA_ROOT GRAPHFF_EXPERIMENT_ROOT
export MODEL=lstm MATRIX_METRICS

export_arg="ALL"
if [[ -n "${EXTRA_EXPORTS:-}" ]]; then
  export_arg="$export_arg,$EXTRA_EXPORTS"
fi

eval_job_ids=()
if [[ "$REPORT_ONLY" != "1" ]]; then
  for spec in "${ARRAY_SPECS[@]}"; do
    sbatch_args=(
      --job-name=lstm-matrix
      --array="$spec"
      --output="$SLURM_LOG_DIR/slurm-%x-%A_%a.out"
      --error="$SLURM_LOG_DIR/slurm-%x-%A_%a.err"
    )
    sbatch_args+=("${SBATCH_MAIL_ARGS[@]}")
    if [[ ${#SBATCH_RESOURCE_ARGS[@]} -gt 0 ]]; then
      sbatch_args+=("${SBATCH_RESOURCE_ARGS[@]}")
    fi
    if [[ ${#SBATCH_EXTRA_ARGS[@]} -gt 0 ]]; then
      sbatch_args+=("${SBATCH_EXTRA_ARGS[@]}")
    fi
    sbatch_args+=(--export="$export_arg" "$EVAL_SCRIPT")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'sbatch --parsable'
      printf ' %q' "${sbatch_args[@]}"
      printf '\n'
      eval_job_ids+=("<array-job-id>")
    else
      job_id="$(sbatch --parsable "${sbatch_args[@]}")"
      echo "submitted evaluation array $job_id (tasks $spec)"
      eval_job_ids+=("$job_id")
    fi
  done
fi

if [[ "$NO_REPORT" == "1" ]]; then
  echo
  echo "report not submitted (NO_REPORT=1). Build it later with:"
  echo "  RUN_ID=$RUN_ID MODEL=lstm sbatch slurm/run_matrix_report.sbatch"
  exit 0
fi

report_args=(
  --job-name=lstm-matrix-report
  --output="$SLURM_LOG_DIR/slurm-%x-%j.out"
  --error="$SLURM_LOG_DIR/slurm-%x-%j.err"
)
report_args+=("${SBATCH_MAIL_ARGS[@]}")
if [[ ${#eval_job_ids[@]} -gt 0 ]]; then
  # afterany, not afterok: a partial matrix that names what is missing beats no
  # matrix at all, and naming it is the report's purpose
  dependency="afterany"
  for job_id in "${eval_job_ids[@]}"; do
    dependency="$dependency:$job_id"
  done
  report_args+=(--dependency="$dependency" --kill-on-invalid-dep=yes)
fi
report_args+=(--export="$export_arg" "$REPORT_SCRIPT")

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'sbatch'
  printf ' %q' "${report_args[@]}"
  printf '\n'
else
  report_id="$(sbatch --parsable "${report_args[@]}")"
  echo "submitted report job $report_id"
  echo
  echo "results will appear in $EXPERIMENT_DIR/evaluations:"
  echo "  <train>@<test>.csv                       per-pair metrics, one row per fold"
  echo "  results/lstm_mingling_matrix_report.txt  tables, missing checkpoints, warnings"
  echo "  results/lstm_mingling_matrix_f1_1.csv    rendered 5x5 table, F1 at T=1"
  echo "  results/lstm_mingling_matrix_f1_2_3.csv  rendered 5x5 table, F1 at T=2/3"
  echo "  results/lstm_mingling_matrix_auc.csv     rendered 5x5 table, AUC"
  echo "  results/lstm_mingling_matrix_long.csv    tidy mean/std per cell"
  echo "  results/lstm_mingling_matrix_cells.csv   every cell, with its status"
fi
