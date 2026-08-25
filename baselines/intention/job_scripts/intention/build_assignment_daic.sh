#!/bin/bash
#SBATCH --job-name="build_assignment"
#SBATCH --time=1:00:00
#SBATCH --account=testusers
#SBATCH --partition=ewi-insy,all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END
#SBATCH --output=/home/zli33/linuxhome/slurm_outputs/intention/assignment_%j.out
#SBATCH --error=/home/zli33/linuxhome/slurm_outputs/intention/assignment_%j.err

# Build the balanced persona x clip design.
#
# No --gres: the builder loads no model. It stats media paths and shuffles
# integers, so it is CPU-only and short. It also runs unchanged on a login node,
# which is the quickest way to use --explain:
#
#   sbatch job_scripts/intention/build_assignment_daic.sh --mode pa
#   bash   job_scripts/intention/build_assignment_daic.sh --mode pa --explain 2
#
# --mode is required. A design is only valid for the mode it was validated
# against, and the sweep that runs it must match.
#
# This file is only the environment: the SLURM header above (which sbatch must
# read literally, so it cannot be computed) plus PROJECT_ROOT. The design and all
# the logic live in lib/build_assignment_job.sh.

# Absolute host path, NOT a container path: the source below runs on the host,
# before apptainer is involved. PROJECT_ROOT is bound to /workspace inside.
PROJECT_ROOT="/home/zli33/linuxhome/projects/COSILab"

# Sourced (not exec'd) so the body inherits "$@" and the SLURM job environment.
# Referenced by absolute path rather than via $0, because sbatch copies this
# script to a spool directory where $0 no longer points into the repo.
job_body="${PROJECT_ROOT}/baselines/intention/job_scripts/lib/build_assignment_job.sh"
if [[ ! -f "${job_body}" ]]; then
    echo "[ERROR] Job body not found: ${job_body}" >&2
    echo "        Fix PROJECT_ROOT near the bottom of build_assignment_daic.sh," >&2
    echo "        or sync the repo to that path on this environment." >&2
    exit 1
fi
source "${job_body}"
