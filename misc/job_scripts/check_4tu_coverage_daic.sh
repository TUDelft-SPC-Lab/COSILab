#!/bin/bash
#SBATCH --job-name="4tu_coverage"
#SBATCH --time=24:00:00
#SBATCH --account=daic-eemcs-insy-prb
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/zli33/linuxhome/slurm_outputs/misc/4tu_coverage_%j.out
#SBATCH --error=/home/zli33/linuxhome/slurm_outputs/misc/4tu_coverage_%j.err

# DAIC: full MD5 check of which cosilab_project folders are already fully
# duplicated in 4TU_upload (misc/check_4tu_coverage.py). CPU only and read-only:
# it never deletes or changes anything under ROOT.
#
#   mkdir -p /home/zli33/linuxhome/slurm_outputs/misc   # once; Slurm needs it for the logs
#   sbatch misc/job_scripts/check_4tu_coverage_daic.sh
#   sbatch misc/job_scripts/check_4tu_coverage_daic.sh --folders raw_video data_temp
#   sbatch misc/job_scripts/check_4tu_coverage_daic.sh --summary-depth 2
#
# Any arguments are passed straight to check_4tu_coverage.py (--folders,
# --exclude, --summary-depth, --quick-only, ...).
#
# Resuming: hashes are cached in $OUT_DIR/hash_cache.sqlite. If the job hits the
# time limit, submit the same command again and it continues from the cache.
# Keep OUT_DIR the same between runs or the cache is not found.
#
# Environment overrides:
#   ROOT=/tudelft.net/staff-umbrella/neon/cosilab_project
#   REF=$ROOT/4TU_upload
#   OUT_DIR=/home/zli33/linuxhome/4tu_coverage
#   WORKERS=$SLURM_CPUS_PER_TASK   parallel hashing threads (I/O bound; more
#                                  than ~16 rarely helps on network storage)
#   PROJECT_ROOT=/home/zli33/linuxhome/projects/COSILab   checkout to run from
#   PYTHON_BIN=python3             any Python >= 3.8; the script is stdlib only

set -euo pipefail

ROOT="${ROOT:-/tudelft.net/staff-umbrella/neon/cosilab_project}"
REF="${REF:-${ROOT}/4TU_upload}"
OUT_DIR="${OUT_DIR:-/home/zli33/linuxhome/4tu_coverage}"
WORKERS="${WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/zli33/linuxhome/projects/COSILab}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

PY_SCRIPT="${PROJECT_ROOT}/misc/check_4tu_coverage.py"

[[ -f "${PY_SCRIPT}" ]] || { echo "[ERROR] script not found: ${PY_SCRIPT} (set PROJECT_ROOT)" >&2; exit 2; }
[[ -d "${ROOT}" ]] || { echo "[ERROR] ROOT not found: ${ROOT}" >&2; exit 2; }
[[ -d "${REF}" ]] || { echo "[ERROR] REF not found: ${REF}" >&2; exit 2; }
command -v "${PYTHON_BIN}" >/dev/null || { echo "[ERROR] no ${PYTHON_BIN} on this node (set PYTHON_BIN)" >&2; exit 2; }
"${PYTHON_BIN}" -c 'import sys; sys.exit(sys.version_info < (3, 8))' || {
    echo "[ERROR] $("${PYTHON_BIN}" --version 2>&1) is too old; need Python >= 3.8 (set PYTHON_BIN)" >&2
    exit 2
}

mkdir -p "${OUT_DIR}"

echo "[INFO] host         = $(hostname)"
echo "[INFO] job          = ${SLURM_JOB_ID:-NA}"
echo "[INFO] python       = $(command -v "${PYTHON_BIN}") ($("${PYTHON_BIN}" --version 2>&1))"
echo "[INFO] root         = ${ROOT}"
echo "[INFO] ref          = ${REF}"
echo "[INFO] out_dir      = ${OUT_DIR}"
echo "[INFO] workers      = ${WORKERS}"
echo "[INFO] extra args   = $*"
echo "[INFO] git commit   = $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo NA)"
echo "[INFO] started      = $(date -Is)"

# -u: unbuffered, so progress lines show up in the Slurm log as they happen
"${PYTHON_BIN}" -u "${PY_SCRIPT}" \
    --root "${ROOT}" \
    --ref "${REF}" \
    --out "${OUT_DIR}" \
    --workers "${WORKERS}" \
    "$@"

echo "[INFO] finished     = $(date -Is)"
echo "[INFO] results      = ${OUT_DIR}/summary.csv, ${OUT_DIR}/files.csv"
