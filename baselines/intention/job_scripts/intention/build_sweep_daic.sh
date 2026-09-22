#!/bin/bash
#SBATCH --job-name="build_sweep"
#SBATCH --time=1:00:00
#SBATCH --account=testusers
#SBATCH --partition=ewi-insy,all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=END
#SBATCH --output=/home/zli33/linuxhome/slurm_outputs/intention/sweep_%j.out
#SBATCH --error=/home/zli33/linuxhome/slurm_outputs/intention/sweep_%j.err

# Build the prompt-variation sweep: clips x personas x instruction prompts x trials.
#
# The sibling of build_assignment_daic.sh, and it does the same job for a
# different design: it fills in the four media prefixes from
# lib/intention_paths.sh so clip validation resolves against exactly the paths
# inference will use, and runs the builder inside an image so the interpreter is
# pinned.
#
# No --gres: the builder loads no model. It stats media paths and shuffles
# integers, so it is CPU-only, short, and runs unchanged on a login node --
# which is the way to use it, since the artefact is meant to be inspected before
# 15,000 answers are queued behind it:
#
#   bash   job_scripts/intention/build_sweep_daic.sh
#   bash   job_scripts/intention/build_sweep_daic.sh --seed 7
#   sbatch job_scripts/intention/build_sweep_daic.sh
#
# THE DESIGN IS FIXED, in build_sweep.py rather than here: 5 clips, 5 personas,
# the six structural variants, 100 trials. Unlike build_assignment_daic.sh there
# is no design block to edit in this file, because none of those numbers is a
# path and all of them are load-bearing in the same one place.
#
# THIS SCRIPT DOES NOT SUBMIT THE SWEEP. It writes a task spec and a model
# config beside it, and stops. Submitting is one persona.sh run per instruction
# prompt, each reading --trials and --model-config from what this wrote; the
# closing lines below print the shape of it.

set -euo pipefail

# Absolute host path, NOT a container path: everything here runs on the host,
# before apptainer is involved. PROJECT_ROOT is bound to /workspace inside.
PROJECT_ROOT="/home/zli33/linuxhome/projects/COSILab"

CONTAINER_PROJECT_ROOT="/workspace"
PACKAGE_SUBDIR="baselines/intention"

# Run with bash rather than sbatch -- which is how this one is meant to be used
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

# Same file the inference job and persona.sh read, so the sweep is validated
# against exactly the paths inference will use. A prefix that stopped matching
# would otherwise leave every clip unusable here and equally unusable there,
# which is the failure this sharing exists to make impossible.
source "${PROJECT_ROOT}/${PACKAGE_SUBDIR}/job_scripts/lib/intention_paths.sh"

SRC_DIR="${PROJECT_ROOT}/${PACKAGE_SUBDIR}/src"
CONTAINER_SRC_DIR="${CONTAINER_PROJECT_ROOT}/${PACKAGE_SUBDIR}/src"
builder="${SRC_DIR}/intention_inference/build_sweep.py"

# The persona pool and where the sweep lands. Both are paths, so unlike the
# design numbers they belong here beside the rest of the pipeline's paths.
PERSONAS_JSON="${INTENTION_PIPELINE_ROOT}/personas_V2_prolific_ids.json"
SWEEP_SPEC_JSON="${INTENTION_PIPELINE_ROOT}/sweeps/prompt_variation_sweep.json"

for arg in "$@"; do
    case "${arg}" in
        -h|--help)
            echo "Usage: bash $0 [--seed N] [--output PATH] [--trials N]" >&2
            echo "" >&2
            echo "Builds the clip x persona x instruction-prompt x trial sweep spec." >&2
            echo "" >&2
            echo "  --seed N          Chooses the clips and the personas. Default: 0" >&2
            echo "  --output PATH     Default: ${SWEEP_SPEC_JSON}" >&2
            echo "                    The derived model config lands beside it." >&2
            echo "  --trials N        Answers per clip. Default: 100" >&2
            echo "  --temperature F   Sampling temperature for the derived config" >&2
            echo "  --top-p F         Nucleus cutoff for the derived config" >&2
            echo "  --personas-json P Persona pool. Default: ${PERSONAS_JSON}" >&2
            echo "  --mode NAME       Default: pa, which the instruction prompts are" >&2
            echo "                    written for. Only the clip validation uses it." >&2
            echo "  --skip-validation Sample without checking the clips' media resolves" >&2
            echo "" >&2
            echo "The design -- 5 clips, 5 personas, 6 instruction prompts -- is fixed" >&2
            echo "in build_sweep.py and cannot be set on the command line." >&2
            echo "" >&2
            echo "This writes the sweep; it does not submit it." >&2
            exit 0
            ;;
    esac
done

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
if [[ ! -f "${PERSONAS_JSON}" ]]; then
    echo "[ERROR] Persona pool not found: ${PERSONAS_JSON}" >&2
    echo "        Pass --personas-json to point at another one." >&2
    exit 1
fi
if [[ ! -d "${INTENTION_INSTRUCTION_PROMPT_DIR}" ]]; then
    echo "[ERROR] Instruction prompt directory not found:" >&2
    echo "        ${INTENTION_INSTRUCTION_PROMPT_DIR}" >&2
    exit 1
fi

mkdir -p "$(dirname "${SWEEP_SPEC_JSON}")"

# Paths first, then the caller's flags, so --output/--seed/--personas-json can
# override any of them: argparse lets the last value win, and every flag here is
# a path or a knob rather than part of the design.
builder_args=(
    --input-json "${INTENTION_INPUT_JSON}"
    --personas-json "${PERSONAS_JSON}"
    --instruction-prompt-dir "${INTENTION_INSTRUCTION_PROMPT_DIR}"
    --output "${SWEEP_SPEC_JSON}"
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
echo "[INFO] personas_json      = ${PERSONAS_JSON}"
echo "[INFO] instruction_dir    = ${INTENTION_INSTRUCTION_PROMPT_DIR}"
echo "[INFO] output             = ${SWEEP_SPEC_JSON}"
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
    python -m intention_inference.build_sweep "${builder_args[@]}" "$@"

echo
echo "[INFO] Sweep build finished: ${SWEEP_SPEC_JSON}"
echo "[INFO] Nothing has been submitted. To run it, one persona.sh per"
echo "[INFO] instruction prompt named in the spec's __meta__, for example:"
echo "[INFO]"
echo "[INFO]   bash job_scripts/intention/persona.sh --backend qwen7b --mode pa \\"
echo "[INFO]       --task-spec ${SWEEP_SPEC_JSON} \\"
echo "[INFO]       --instruction-prompt ${INTENTION_INSTRUCTION_PROMPT_DIR}/prompt_structural_variant_0.txt \\"
echo "[INFO]       --model-config <the *_model_config.json written beside the spec> \\"
echo "[INFO]       --trials 100 --answers-per-job 5"
echo "[INFO]"
echo "[INFO] Size the jobs before submitting: --answers-per-job counts persona x"
echo "[INFO] clip answers and does not know about --trials, so a 5-answer persona"
echo "[INFO] is really 500 generations."
