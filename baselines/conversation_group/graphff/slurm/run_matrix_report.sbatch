#!/usr/bin/env bash
#SBATCH --job-name=matrix-report
#SBATCH --qos=short
#SBATCH --partition=insy,general
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/nfs/zli33/slurm_outputs/slurm-%x-%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/slurm-%x-%j.err
#SBATCH --mail-type=END,FAIL
#
# Reduces the evaluation cells of one experiment to the 5x5 matrix and the
# results file. Shared by both pipelines: MODEL picks which cells to read and
# which experiment root to bind.
#
# Normally submitted by slurm/submit_matrix_{lstm,dante}.sh with a dependency on
# the evaluation array, so it runs once every task has finished. Direct use:
#
#   sbatch --export=ALL,MODEL=lstm,RUN_ID=1 slurm/run_matrix_report.sbatch
#
# The dependency is afterany, not afterok, on purpose: a partial matrix that
# names what is missing is more useful than no matrix at all, and naming it is
# this job's whole point.
#
# It runs in the py371 environment for both pipelines -- the report needs pandas
# and DANTE's dante_tf1 environment has none. It reads only files, so it is
# cheap and safe to re-run.
#
# Environment overrides:
#   MODEL=lstm               # lstm | dante, required
#   RUN_ID=1                 # which experiment to report, i.e. exp_$RUN_ID
#   MATRIX_METRICS=f1_1,f1_2_3,auc
#   MATRIX_DECIMALS=3
#   GRAPHFF_EXPERIMENT_ROOT=/path/to/experiments   # MODEL=lstm
#   DANTE_EXPERIMENT_ROOT=/path/to/experiments     # MODEL=dante
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
MODEL="${MODEL:-}"
RUN_ID="${RUN_ID:-1}"
MATRIX_METRICS="${MATRIX_METRICS:-f1_1,f1_2_3,auc}"
MATRIX_DECIMALS="${MATRIX_DECIMALS:-3}"
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/logs/cache}"

case "$MODEL" in
  lstm)
    EXPERIMENT_ROOT="${GRAPHFF_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/LSTM}"
    ;;
  dante)
    EXPERIMENT_ROOT="${DANTE_EXPERIMENT_ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/B2_pipeline/DANTE}"
    ;;
  *)
    echo "[ERROR] MODEL must be lstm or dante, got: '${MODEL}'" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
mkdir -p "$CACHE_ROOT/xdg" "$CACHE_ROOT/mpl"

PY_SCRIPT="$PROJECT_ROOT/scripts/build_matrix_report.py"
# must match {graphff,dante}_paths.evaluation_results_dir
RESULTS_ROOT="$EXPERIMENT_ROOT/exp_$RUN_ID/evaluations/results"
TASK_LOG_DIR="$RESULTS_ROOT/logs"
TASK_LOG="$TASK_LOG_DIR/matrix_report.log"

[[ -f "$APPTAINER_IMAGE" ]] || { echo "[ERROR] image not found: $APPTAINER_IMAGE" >&2; exit 2; }
[[ -f "$PY_SCRIPT" ]] || { echo "[ERROR] python script not found: $PY_SCRIPT" >&2; exit 2; }
[[ -d "$RESULTS_ROOT/cells" ]] || {
  echo "[ERROR] no evaluation cells under $RESULTS_ROOT/cells" >&2
  echo "        The evaluation array wrote nothing; check its Slurm logs." >&2
  exit 2
}

mkdir -p "$TASK_LOG_DIR"
: > "$TASK_LOG"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}"
echo "project_root=$PROJECT_ROOT"
echo "model=$MODEL"
echo "run_id=$RUN_ID"
echo "experiment_root=$EXPERIMENT_ROOT"
echo "results_root=$RESULTS_ROOT"
echo "matrix_metrics=$MATRIX_METRICS"
echo "matrix_decimals=$MATRIX_DECIMALS"

apptainer exec --cleanenv \
  --env PROJECT_ROOT="$PROJECT_ROOT" \
  --env APPTAINER_IMAGE="$APPTAINER_IMAGE" \
  --env MODEL="$MODEL" \
  --env RUN_ID="$RUN_ID" \
  --env MATRIX_METRICS="$MATRIX_METRICS" \
  --env MATRIX_DECIMALS="$MATRIX_DECIMALS" \
  --env GRAPHFF_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  --env DANTE_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  --env XDG_CACHE_HOME="$CACHE_ROOT/xdg" \
  --env MPLCONFIGDIR="$CACHE_ROOT/mpl" \
  --env LD_LIBRARY_PATH="/opt/conda/envs/py371/lib" \
  -B "$HOME:$HOME" \
  -B "$PROJECT_ROOT:$PROJECT_ROOT" \
  -B "$EXPERIMENT_ROOT:$EXPERIMENT_ROOT" \
  "$APPTAINER_IMAGE" \
  bash -lc '
    set -euo pipefail
    cd "$PROJECT_ROOT"
    # py371 for both pipelines: the report needs pandas, dante_tf1 has none
    PYBIN="${PYTHON_BIN:-/opt/conda/envs/py371/bin/python}"
    if [[ ! -x "$PYBIN" ]]; then
      echo "[ERROR] no python at $PYBIN inside $APPTAINER_IMAGE"
      echo "        Set PYTHON_BIN if this image lays its environments out differently."
      exit 3
    fi
    echo "python_bin=${PYBIN}"
    "${PYBIN}" scripts/build_matrix_report.py \
      --model "$MODEL" \
      --exp-id "$RUN_ID" \
      --matrix-metrics "$MATRIX_METRICS" \
      --matrix-decimals "$MATRIX_DECIMALS"
  ' 2>&1 | tee -a "$TASK_LOG"
