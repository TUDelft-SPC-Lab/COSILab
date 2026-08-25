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

# Build the balanced persona x clip design that persona.sh runs.
#
# This is step one of the persona workflow; nothing else here writes an
# assignment. The other two scripts in this directory are:
#
#   cosilab_daic.sh  runs one Slurm job WITHOUT personas
#   persona.sh       launches packed jobs WITH personas, from this file or a spec
#
# No --gres: the builder loads no model. It stats media paths and shuffles
# integers, so it is CPU-only and short. It also runs unchanged on a login node,
# which is the quickest way to use --explain:
#
#   sbatch job_scripts/intention/build_assignment_daic.sh --mode pa
#   bash   job_scripts/intention/build_assignment_daic.sh --mode pa --explain 2
#
# --explain N prints, path by path, where the first N candidate clips' media was
# looked for, then exits without building anything. Reach for it when validation
# rejects everything: the usual cause is a media/local prefix pair that does not
# match the manifest, which leaves the raw URL in place and silently resolves it
# against the wrong directory, so every clip looks missing when only a flag is
# wrong.
#
# --mode is required. A design is only valid for the mode it was validated
# against, because what makes a clip usable depends on what that mode needs to
# ground the question. It is recorded in the assignment's __meta__, and
# persona.sh refuses to run a design whose mode disagrees with its own.
#
# THE DESIGN IS FIXED in the block below, not on the command line, so every
# assignment built from this repo comes from a design that is written down and
# version-controlled rather than from whatever flags one submission carried.
# Change it by editing that block and rebuilding; the flags that would override
# it are rejected. Everything downstream reads the design back out of the
# assignment JSON, so these numbers appear in this file only.
#
# The remaining arguments are forwarded to build_assignment.py, so --seed /
# --output / --explain / --no-audio / --skip-validation work here.

set -euo pipefail

# Absolute host path, NOT a container path: everything here runs on the host,
# before apptainer is involved. PROJECT_ROOT is bound to /workspace inside.
PROJECT_ROOT="/home/zli33/linuxhome/projects/COSILab"

CONTAINER_PROJECT_ROOT="/workspace"
PACKAGE_SUBDIR="baselines/intention"

# Run with bash rather than sbatch -- which is how --explain is meant to be used
# -- this file is the real one, so the checkout it belongs to is discoverable
# from BASH_SOURCE and the hardcoded PROJECT_ROOT above is only a fallback. Under
# sbatch it is NOT: the script has been copied to a spool directory, so the
# hardcoded path is the only truth. Same rule as cosilab_daic.sh.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    _script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    _checkout="$(cd -- "${_script_dir}/../../../.." && pwd)"
    if [[ -f "${_checkout}/${PACKAGE_SUBDIR}/job_scripts/lib/intention_paths.sh" ]]; then
        PROJECT_ROOT="${_checkout}"
    fi
fi
# Any image with a Python will do -- the builder imports nothing heavier than the
# standard library -- so the smallest one is used rather than a backend's.
SIF_PATH="${SIF_PATH:-/tudelft.net/staff-umbrella/neon/apptainer/gemma.sif}"

# Same file the inference job and persona.sh read, so the assignment is validated
# against exactly the paths inference will use. A prefix that stopped matching
# would otherwise leave every clip unusable here and equally unusable there,
# which is the failure this sharing exists to make impossible.
source "${PROJECT_ROOT}/${PACKAGE_SUBDIR}/job_scripts/lib/intention_paths.sh"

SRC_DIR="${PROJECT_ROOT}/${PACKAGE_SUBDIR}/src"
CONTAINER_SRC_DIR="${CONTAINER_PROJECT_ROOT}/${PACKAGE_SUBDIR}/src"
builder="${SRC_DIR}/intention_inference/build_assignment.py"

# ---------------------------------------------------------------------------
# THE DESIGN. Edit here, rebuild, and everything downstream follows the file.
#
# The two sides of the balance must agree, or the builder refuses to run:
#
#   personas x CLIPS_PER_PERSONA == NUM_CLIPS x PERSONAS_PER_CLIP
#   1000     x 6                == 1000      x 6                  = 6000 inferences
#
# CLIP_POOL_RANGE is the slice of the manifest NUM_CLIPS are sampled from; only
# clips whose media actually resolves are eligible.
NUM_CLIPS="1000"
PERSONAS_PER_CLIP="6"
CLIPS_PER_PERSONA="6"
PERSONA_RANGE="1-1000"
CLIP_POOL_RANGE="0-4000"
# ---------------------------------------------------------------------------

# Fixed means fixed: a design flag on the command line is an error rather than a
# silent override, since argparse would otherwise let the last value win and the
# resulting assignment would no longer match the block above. Checked before the
# environment below, so a bad flag is reported as a bad flag even on a machine
# without the image.
for arg in "$@"; do
    case "${arg}" in
        --num-clips|--num-clips=*|\
        --personas-per-clip|--personas-per-clip=*|\
        --clips-per-persona|--clips-per-persona=*|\
        --persona-range|--persona-range=*|\
        --clip-pool-range|--clip-pool-range=*)
            echo "[ERROR] ${arg%%=*} is fixed by the design block in" >&2
            echo "        the design block in this file:" >&2
            echo "          ${NUM_CLIPS} clip(s), ${PERSONAS_PER_CLIP} persona(s) per clip," >&2
            echo "          ${CLIPS_PER_PERSONA} clip(s) per persona, personas ${PERSONA_RANGE}" >&2
            echo "        Edit that block and rebuild to change the design." >&2
            exit 1
            ;;
    esac
done

# Answered before the --mode gate below, which would otherwise turn `--help` into
# an error about a missing flag.
for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            echo "Usage: sbatch $0 --mode <fa|sa|pa> [--seed N] [--output PATH]" >&2
            echo "       bash   $0 --mode <fa|sa|pa> --explain 2" >&2
            echo "" >&2
            echo "Builds the balanced persona x clip design that persona.sh runs." >&2
            echo "" >&2
            echo "  --mode fa|sa|pa   Required. A design is only valid for the mode it was" >&2
            echo "                    validated against; persona.sh refuses a mismatch." >&2
            echo "  --explain N       Print where the first N candidate clips' media was" >&2
            echo "                    looked for, then exit without building. Use this when" >&2
            echo "                    validation rejects everything." >&2
            echo "  --seed N          Reproducible draw of the same design. Default: 42" >&2
            echo "  --output PATH     Default: ${INTENTION_ASSIGNMENT_JSON}" >&2
            echo "  --no-audio, --skip-validation   Forwarded to build_assignment.py" >&2
            echo "" >&2
            echo "The design itself (${NUM_CLIPS} clips, ${PERSONAS_PER_CLIP} personas per clip," >&2
            echo "${CLIPS_PER_PERSONA} clips per persona, personas ${PERSONA_RANGE}) is fixed in the" >&2
            echo "design block in this file and cannot be set on the command line." >&2
            exit 0
            ;;
    esac
done

# --mode is forwarded, not defaulted: a design carries the mode it was validated
# for, and guessing one here would produce a file that looks usable by any sweep.
if [[ "$*" != *--mode* ]]; then
    echo "[ERROR] --mode is required: a design is only valid for the mode it was" >&2
    echo "        validated against, because what makes a clip usable depends on" >&2
    echo "        what the mode needs to ground the question." >&2
    echo "        For example: sbatch $0 --mode pa" >&2
    exit 1
fi

if [[ ! -f "${SIF_PATH}" ]]; then
    echo "[ERROR] SIF not found: ${SIF_PATH}" >&2
    exit 1
fi
if [[ ! -f "${builder}" ]]; then
    echo "[ERROR] Builder not found: ${builder}" >&2
    echo "        Set PROJECT_ROOT near the top of this file to your COSILab checkout." >&2
    exit 1
fi
if [[ ! -f "${INTENTION_INPUT_JSON}" ]]; then
    echo "[ERROR] Clip manifest not found: ${INTENTION_INPUT_JSON}" >&2
    exit 1
fi

mkdir -p "$(dirname "${INTENTION_ASSIGNMENT_JSON}")"

# Design first, then the caller's flags, which may only cover the rest
# (--mode, --seed, --output, --explain, --no-audio, --skip-validation).
builder_args=(
    --input-json "${INTENTION_INPUT_JSON}"
    --output "${INTENTION_ASSIGNMENT_JSON}"
    --num-clips "${NUM_CLIPS}"
    --personas-per-clip "${PERSONAS_PER_CLIP}"
    --clips-per-persona "${CLIPS_PER_PERSONA}"
    --persona-range "${PERSONA_RANGE}"
    --clip-pool-range "${CLIP_POOL_RANGE}"
    --participant-image-root "${INTENTION_PARTICIPANT_IMAGE_ROOT}"
    --video-media-path-prefix "${INTENTION_VIDEO_MEDIA_PATH_PREFIX}"
    --video-local-path-prefix "${INTENTION_VIDEO_LOCAL_PATH_PREFIX}"
    --audio-media-path-prefix "${INTENTION_AUDIO_MEDIA_PATH_PREFIX}"
    --audio-local-path-prefix "${INTENTION_AUDIO_LOCAL_PATH_PREFIX}"
)

echo "[INFO] project_root       = ${PROJECT_ROOT}"
echo "[INFO] sif_path           = ${SIF_PATH}"
echo "[INFO] builder            = ${builder}"
echo "[INFO] input_json         = ${INTENTION_INPUT_JSON}"
echo "[INFO] output             = ${INTENTION_ASSIGNMENT_JSON}"
echo "[INFO] design (fixed)     = ${NUM_CLIPS} clip(s) from pool ${CLIP_POOL_RANGE}, "\
"${PERSONAS_PER_CLIP} persona(s) per clip, ${CLIPS_PER_PERSONA} clip(s) per persona, "\
"personas ${PERSONA_RANGE}"
echo "[INFO] video_media_prefix = ${INTENTION_VIDEO_MEDIA_PATH_PREFIX}"
echo "[INFO] video_local_prefix = ${INTENTION_VIDEO_LOCAL_PATH_PREFIX}"
echo "[INFO] audio_media_prefix = ${INTENTION_AUDIO_MEDIA_PATH_PREFIX}"
echo "[INFO] audio_local_prefix = ${INTENTION_AUDIO_LOCAL_PATH_PREFIX}"
echo "[INFO] participant_images = ${INTENTION_PARTICIPANT_IMAGE_ROOT}"
echo "[INFO] extra args         = $*"
echo

apptainer exec \
    --bind "${PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    --pwd "${CONTAINER_PROJECT_ROOT}" \
    --env PYTHONPATH="${CONTAINER_SRC_DIR}" \
    "${SIF_PATH}" \
    python -m intention_inference.build_assignment "${builder_args[@]}" "$@"

echo
echo "[INFO] Assignment build finished: ${INTENTION_ASSIGNMENT_JSON}"
