#!/usr/bin/env bash
# Submit LSTM/GraphFF Mingling benchmark jobs.
#
# Usage:
#   bash slurm/submit_lstm.sh --cam=<cam> [--fold=<n>]
#
#   --cam=<cam>   required. 06 | 08 | 10 | 01 | 03 | all
#                 Unpadded numbers are zero-padded, so --cam=6 means --cam=06.
#                 A cam06 form is accepted too. The Mingling session is derived
#                 from the camera number:
#                   06, 08, 10 -> mingling1
#                   01, 03     -> mingling2
#   --fold=<n>    optional single fold 0-4; omit to run all five
#
# Examples:
#   bash slurm/submit_lstm.sh --cam=06              # cam06, all 5 folds
#   bash slurm/submit_lstm.sh --cam=all             # all 5 cameras, 25 tasks
#   bash slurm/submit_lstm.sh --cam=06 --fold=2     # cam06, fold 2 only
#   DRY_RUN=1 bash slurm/submit_lstm.sh --cam=all
#   RUN_ID=2 bash slurm/submit_lstm.sh --cam=all
#
# This pipeline is GPU-only; the job asserts torch.cuda.is_available() inside the
# container and fails the task if no GPU is visible.
#
# Environment overrides:
#   RUN_ID=1                    output goes to <exp_root>/exp_$RUN_ID/...
#   OVERWRITE=1                 0 refuses to run when a fold directory exists
#   DRY_RUN=0                   1 prints the sbatch lines without submitting
#   GRAPHFF_DATA_ROOT=...       benchmark artifacts to read
#   GRAPHFF_EXPERIMENT_ROOT=... where runs are written
#   SLURM_LOG_DIR=...           Slurm stdout/stderr
#   EXCLUDE_NODES='gpu[36-45]'  set empty to disable
#   EXTRA_EXPORTS='NUM_EPOCHS=600,PATIENCE=50'
#   SBATCH_ARGS='--constraint=...'

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
  PROJECT_ROOT="$PROJECT_ROOT"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

SCRIPT="$PROJECT_ROOT/slurm/run_lstm.sbatch"
GRAPHFF_DATA_ROOT="${GRAPHFF_DATA_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_2/baselines/LSTM}"
GRAPHFF_EXPERIMENT_ROOT="${GRAPHFF_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/LSTM/experiments}"
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/home/nfs/zli33/slurm_outputs/lstm}"
RUN_ID="${RUN_ID:-1}"
OVERWRITE="${OVERWRITE:-1}"
EXCLUDE_NODES="${EXCLUDE_NODES-gpu[36-45]}"

ALL_CAMS=(06 08 10 01 03)

# print the header comment block, so help text and the file never drift apart
usage() {
  awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"
  exit "${1:-2}"
}

# camera number -> "<session>/cam<nn>". Zero-pads so 6 -> 06 and 1 -> 01, which
# keeps 1 (cam01, mingling2) and 10 (cam10, mingling1) distinct.
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
  # $1 = flag name, $2 = remaining arg count
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
  ARRAY_SPEC="${FOLD_ARG}-${FOLD_ARG}"
else
  ARRAY_SPEC="0-4"
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

[[ -f "$SCRIPT" ]] || { echo "[ERROR] missing Slurm script: $SCRIPT" >&2; exit 2; }
[[ -d "$GRAPHFF_DATA_ROOT" ]] || { echo "[ERROR] data root not found: $GRAPHFF_DATA_ROOT" >&2; exit 2; }

SBATCH_EXTRA_ARGS=()
if [[ -n "${SBATCH_ARGS:-}" ]]; then
  read -r -a SBATCH_EXTRA_ARGS <<< "$SBATCH_ARGS"
fi
if [[ -n "$EXCLUDE_NODES" ]]; then
  SBATCH_EXTRA_ARGS+=(--exclude="$EXCLUDE_NODES")
fi

# Slurm rejects a job outright if the --output directory does not already exist.
# Normally it is already there, so only try to create it when it is missing, and
# say something useful rather than leaking a raw mkdir error when we cannot.
if [[ ! -d "$SLURM_LOG_DIR" ]]; then
  if mkdir -p "$SLURM_LOG_DIR" 2>/dev/null; then
    echo "created slurm log dir: $SLURM_LOG_DIR"
  elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[WARN] slurm log dir is missing and not creatable here: $SLURM_LOG_DIR" >&2
    echo "       harmless for a dry run; it must exist before a real submission." >&2
  else
    echo "[ERROR] slurm log dir does not exist and could not be created: $SLURM_LOG_DIR" >&2
    echo "        Create it on the cluster, or set SLURM_LOG_DIR to a writable path." >&2
    exit 2
  fi
fi

echo "data root:       $GRAPHFF_DATA_ROOT"
echo "experiment root: $GRAPHFF_EXPERIMENT_ROOT/exp_$RUN_ID"
echo "slurm log dir:   $SLURM_LOG_DIR"
echo "cameras:         ${DATASETS[*]}"
echo "folds:           $ARRAY_SPEC"
echo "overwrite:       $OVERWRITE"
echo "exclude nodes:   ${EXCLUDE_NODES:-<none>}"
echo

for dataset in "${DATASETS[@]}"; do
  dataset_dir="$GRAPHFF_DATA_ROOT/$dataset"
  for required in features.csv GT.csv group_names.txt scene_continuity.csv; do
    [[ -f "$dataset_dir/$required" ]] || { echo "[ERROR] missing: $dataset_dir/$required" >&2; exit 2; }
  done

  job_name="lstm-$(basename "$dataset")"
  export_arg="ALL,DATASET=$dataset,RUN_ID=$RUN_ID,OVERWRITE=$OVERWRITE"
  export_arg="$export_arg,GRAPHFF_DATA_ROOT=$GRAPHFF_DATA_ROOT,GRAPHFF_EXPERIMENT_ROOT=$GRAPHFF_EXPERIMENT_ROOT"
  if [[ -n "${EXTRA_EXPORTS:-}" ]]; then
    export_arg="$export_arg,$EXTRA_EXPORTS"
  fi

  # built up in steps rather than one literal: expanding a possibly-empty array
  # under `set -u` aborts on bash < 4.4, which is what the cluster runs
  sbatch_args=(
    --job-name="$job_name"
    --array="$ARRAY_SPEC"
    --output="$SLURM_LOG_DIR/slurm-%x-%A_%a.out"
    --error="$SLURM_LOG_DIR/slurm-%x-%A_%a.err"
  )
  if [[ ${#SBATCH_EXTRA_ARGS[@]} -gt 0 ]]; then
    sbatch_args+=("${SBATCH_EXTRA_ARGS[@]}")
  fi
  sbatch_args+=(--export="$export_arg" "$SCRIPT")

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'sbatch'
    printf ' %q' "${sbatch_args[@]}"
    printf '\n'
  else
    sbatch "${sbatch_args[@]}"
  fi
done
