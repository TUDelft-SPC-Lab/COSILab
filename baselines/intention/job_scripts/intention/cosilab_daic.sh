#!/bin/bash
#SBATCH --job-name="intention"
#SBATCH --time=10:00:00
#SBATCH --partition=ewi-insy,all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=8GB
#SBATCH --gres=gpu:nvidia_rtx_pro_6000:1
#SBATCH --mail-type=END
#SBATCH --output=/home/zli33/linuxhome/slurm_outputs/intention/slurm_%j.out
#SBATCH --error=/home/zli33/linuxhome/slurm_outputs/intention/slurm_%j.err

# Intention inference on the DAIC environment.
#
# Neither what runs nor what is asked is fixed here: --backend picks the model,
# and lib/model_backends.sh maps that name to a container image; --mode picks the
# task variant. The SLURM header above sizes the job for a local multimodal model
# on one GPU.
#
# This file is only the environment: the SLURM header above (which sbatch must
# read literally, so it cannot be computed) plus the three host paths. All the
# actual logic and the full option list live in lib/intention_job.sh.
#
# Run `bash job_scripts/intention/cosilab_daic.sh --help` for the options.
#
#   sbatch job_scripts/intention/cosilab_daic.sh --backend gemma --mode pa
#   sbatch job_scripts/intention/cosilab_daic.sh --backend qwen7b --mode fa --index-range 0-250

# Absolute host paths. These are NOT container paths: the source below runs on
# the host, before apptainer is involved. PROJECT_ROOT is what gets bound to
# /workspace inside the container.
PROJECT_ROOT="/home/zli33/linuxhome/projects/COSILab"
HOME_ROOT="/home/zli33/linuxhome"
SLURM_OUTPUT_ROOT="/home/zli33/linuxhome/slurm_outputs"

# Sourced (not exec'd) so the body inherits "$@" and the SLURM job environment.
# Referenced by absolute path rather than via $0, because sbatch copies this
# script to a spool directory where $0 no longer points into the repo.
job_body="${PROJECT_ROOT}/baselines/intention/job_scripts/lib/intention_job.sh"
if [[ ! -f "${job_body}" ]]; then
    echo "[ERROR] Job body not found: ${job_body}" >&2
    echo "        Fix PROJECT_ROOT near the bottom of cosilab_daic.sh," >&2
    echo "        or sync the repo to that path on this environment." >&2
    exit 1
fi
source "${job_body}"
