# Shared body for the balanced-design builder. Not runnable on its own: it is
# sourced by a per-environment stub that supplies the SLURM header and
# PROJECT_ROOT, the absolute host path to the COSILab checkout (see
# lib/intention_job.sh for the same pattern).
#
# The builder runs inside a backend's apptainer image, so it uses the image's
# Python rather than whatever the cluster happens to put on PATH. The Python
# itself is standard-library only; the image is here purely to pin the
# interpreter. That makes the assignment reproducible from the recorded inputs
# alone: same manifest + same mode + same seed + same image => same file.
#
# apptainer is invoked directly rather than through srun, so this same script
# works both as an sbatch job and run straight on a login node:
#
#   sbatch job_scripts/intention/build_assignment_daic.sh --mode pa
#   bash   job_scripts/intention/build_assignment_daic.sh --mode pa --explain 2
#
# --mode IS REQUIRED and is not part of the design block. Whether a clip is
# usable depends on what the mode needs to ground the question, so a design
# validated for one mode is not valid for another; the mode is recorded in the
# assignment's __meta__ and the sweep that runs it must match.
#
# THE DESIGN IS FIXED HERE, in the block below. It is deliberately not a
# command-line option, so every assignment built from this repo comes from a
# design that is written down and version-controlled rather than from whatever
# flags one submission happened to carry. Change the design by editing that block
# and rebuilding; the flags that would override it are rejected.
#
# Everything downstream reads the design back out of the assignment JSON's
# __meta__ block -- sweep.sh does exactly that -- so these numbers appear in this
# file only.
#
# The remaining arguments are forwarded to build_assignment.py, so --seed /
# --output / --explain / --no-audio / --skip-validation work here.

set -euo pipefail

if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "[ERROR] PROJECT_ROOT is not set. Submit the environment stub" >&2
    echo "        (job_scripts/intention/build_assignment_daic.sh), not this file." >&2
    exit 1
fi

CONTAINER_PROJECT_ROOT="/workspace"
PACKAGE_SUBDIR="baselines/intention"
# Any image with a Python will do -- the builder imports nothing heavier than the
# standard library -- so the smallest one is used rather than a backend's.
SIF_PATH="${SIF_PATH:-/tudelft.net/staff-umbrella/neon/apptainer/gemma.sif}"

# Same file the inference job and the sweep read, so the assignment is validated
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
            echo "        job_scripts/lib/build_assignment_job.sh:" >&2
            echo "          ${NUM_CLIPS} clip(s), ${PERSONAS_PER_CLIP} persona(s) per clip," >&2
            echo "          ${CLIPS_PER_PERSONA} clip(s) per persona, personas ${PERSONA_RANGE}" >&2
            echo "        Edit that block and rebuild to change the design." >&2
            exit 1
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
    echo "        Set PROJECT_ROOT in the stub to your COSILab checkout." >&2
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
