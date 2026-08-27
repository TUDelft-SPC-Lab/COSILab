#!/bin/bash
#SBATCH --job-name="qwen25-probe"
#SBATCH --time=01:00:00
#SBATCH --partition=ewi-insy,all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:2
#SBATCH --output=/home/zli33/linuxhome/slurm_outputs/intention/qwen25_probe_%j.out
#SBATCH --error=/home/zli33/linuxhome/slurm_outputs/intention/qwen25_probe_%j.err

# Validate a rebuilt image before making it the production backend image.
set -euo pipefail

PROJECT_ROOT="/home/zli33/linuxhome/projects/COSILab"
SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/qwen2.5-omni-inference.sif"
VIDEO="/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp/video_clips_30s_timestamp_fixed/seg_134500_134530/camera_06-mingle_session_1.mp4"
IMAGE="/tudelft.net/staff-umbrella/neon/cosilab_project/B1_pipeline/participant_imgs/participant_10.png"
BACKEND="qwen3b"
FRAMES="4,8,16,32"

usage() {
    echo "Usage: sbatch $0 --backend qwen3b|qwen7b [options]" >&2
    echo "  --video PATH    Override the representative COSILab clip" >&2
    echo "  --image PATH    Override participant_10.png" >&2
    echo "  --frames CSV    Default: 4,8,16,32" >&2
    echo "  --sif-path PATH Override the rebuilt container image" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video) VIDEO="${2:?Missing value for --video}"; shift 2 ;;
        --image) IMAGE="${2:?Missing value for --image}"; shift 2 ;;
        --backend) BACKEND="${2:?Missing value for --backend}"; shift 2 ;;
        --frames) FRAMES="${2:?Missing value for --frames}"; shift 2 ;;
        --sif-path) SIF_PATH="${2:?Missing value for --sif-path}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option: $1" >&2; usage; exit 2 ;;
    esac
done

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "[ERROR] Submit this script with sbatch." >&2
    exit 1
fi
if [[ "${BACKEND}" != "qwen3b" && "${BACKEND}" != "qwen7b" ]]; then
    echo "[ERROR] --backend must be qwen3b or qwen7b." >&2
    exit 2
fi
for path in "${SIF_PATH}" "${VIDEO}"; do
    [[ -f "${path}" ]] || { echo "[ERROR] File not found: ${path}" >&2; exit 1; }
done
if [[ -n "${IMAGE}" && ! -f "${IMAGE}" ]]; then
    echo "[ERROR] Image not found: ${IMAGE}" >&2
    exit 1
fi

probe_args=(
    python /workspace/baselines/intention/apptainer/probe_qwen25_vision.py
    --video "${VIDEO}"
    --backend "${BACKEND}"
    --frames "${FRAMES}"
)
[[ -z "${IMAGE}" ]] || probe_args+=(--image "${IMAGE}")

echo "[INFO] sif=${SIF_PATH} backend=${BACKEND} frames=${FRAMES} video=${VIDEO}"
srun apptainer exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind /home/zli33/linuxhome:/home/zli33/linuxhome \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    --bind /tudelft.net/staff-umbrella/GesBench:/tudelft.net/staff-umbrella/GesBench \
    --pwd /workspace \
    --env HF_HOME=/tudelft.net/staff-umbrella/neon/zonghuan/.cache/huggingface \
    --env TRANSFORMERS_CACHE=/tudelft.net/staff-umbrella/neon/zonghuan/.cache/huggingface \
    --env NUMBA_CACHE_DIR=/tmp \
    --env TRITON_CACHE_DIR=/tmp \
    --env PYTHONPATH=/workspace/baselines/intention/src \
    "${SIF_PATH}" \
    "${probe_args[@]}"
