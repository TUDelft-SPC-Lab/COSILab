# Shared body for the intention inference job. Not runnable on its own: it is
# sourced by a per-environment stub that supplies the SLURM header and the three
# paths that actually differ between environments.
#
#   PROJECT_ROOT       absolute host path to the COSILab checkout. Bound to
#                      /workspace inside the container.
#   HOME_ROOT          absolute host path to $HOME, bound into the container as
#                      itself.
#   SLURM_OUTPUT_ROOT  must match the #SBATCH --output/--error dir in the stub.
#
# All three are plain host paths, not container paths. Everything under
# /tudelft.net is identical in both DAIC environments and stays in
# lib/intention_paths.sh.
#
# Which model runs is --backend; lib/model_backends.sh maps that name to a
# container image, and the task's model_config.json says which weights load
# inside it. What is asked of it is --mode, which this script only forwards:
# the modes and their prompts live in intention_inference/modes/.
#
# Submit through the stub, for example:
#   sbatch job_scripts/intention/cosilab_daic.sh --backend gemma --mode participant_image
#   sbatch job_scripts/intention/cosilab_daic.sh --backend qwen7b --mode participant_image --index-range 0-250
#   sbatch job_scripts/intention/cosilab_daic.sh --backend gemma --mode participant_image --no-audio --limit 20

set -euo pipefail

for required in PROJECT_ROOT HOME_ROOT SLURM_OUTPUT_ROOT; do
    if [[ -z "${!required:-}" ]]; then
        echo "[ERROR] ${required} is not set. Submit the environment stub" >&2
        echo "        (job_scripts/intention/cosilab_daic.sh), not this file." >&2
        exit 1
    fi
done

CONTAINER_PROJECT_ROOT="/workspace"
PACKAGE_SUBDIR="baselines/intention"

# Umbrella shares bound into the container, each at its own path. A host path
# that is not in this list does not exist inside the container, however plainly
# it exists on the login node -- which is the confusing half of this, because
# `ls` on the host proves nothing about what the job can see.
#
# It matters most for the weights. They are named in the task's model_config.json
# (backends.<name>.model_id) and nothing forces them onto this share. Point
# model_id at a new one and it has to be added here in the same commit.
HOST_BIND_PATHS=(
    /tudelft.net/staff-umbrella/neon      # manifest, media, participant images, results, HF cache
)

source "${PROJECT_ROOT}/${PACKAGE_SUBDIR}/job_scripts/lib/intention_paths.sh"
# Backend -> image. The task itself is model-agnostic; this picks which model runs it.
source "${PROJECT_ROOT}/${PACKAGE_SUBDIR}/job_scripts/lib/model_backends.sh"

# No default: --backend is required. Which model produced a result is the first
# thing anyone asks of it, and it is also a path component, so letting it be
# decided by omission is how a run ends up filed under a model nobody chose.
backend=""
# No default either, for the same reason: --mode is what was asked, and it is a
# path component too. Unlike the backend, this script never needs to know which
# modes exist -- it picks no image from it -- so the list of valid names lives
# only in intention_inference/modes/registry.py and cli.py rejects a bad one.
mode=""
# Empty means "the image the backend declares"; --sif-path overrides it for a
# rebuilt image. There is no weights override: which weights run is
# backends.<name>.model_id in the task's model_config.json.
sif_path_override=""

DATA_ROOT="${INTENTION_DATA_ROOT}"
input_json="${INTENTION_INPUT_JSON}"
output_dir="${INTENTION_OUTPUT_DIR}"
output_json=""
participant_image_root="${INTENTION_PARTICIPANT_IMAGE_ROOT}"
# Empty prompt_config means "whichever one the mode ships with", and is simply
# not passed on. Empty model_config means the one inside the package, resolved
# after argument parsing because it follows PROJECT_ROOT.
prompt_config=""
model_config=""
no_audio=0
limit=""
index_range=""
start_index=""
end_index=""

usage() {
    echo "Usage:" >&2
    echo "  sbatch $0 --backend <name> --mode <name> [options]" >&2
    echo "Options:" >&2
    echo "  --backend NAME                             Model to run the task on. Required." >&2
    echo "                                             Known: $(intention_known_backends | tr '\n' ' ')" >&2
    echo "  --mode NAME                                Task mode: what is asked. Required." >&2
    echo "                                             Names live in modes/registry.py; run with" >&2
    echo "                                             a wrong one to be shown the list." >&2
    echo "  --input-json PATH                          Default: ${input_json}" >&2
    echo "  --output PATH                              Exact output JSON file path" >&2
    echo "  --output-dir PATH                          Default: ${output_dir}" >&2
    echo "                                             Results land in <output-dir>/<backend>/<mode>/" >&2
    echo "  --prompt-config PATH                       Default: the one the mode ships with" >&2
    echo "  --model-config PATH                        Default: packaged model_config.json" >&2
    echo "                                             Weights, token budget, sampling and the" >&2
    echo "                                             frame policy live in that file, not here." >&2
    echo "  --participant-image-root PATH              Default: ${participant_image_root}" >&2
    echo "  --sif-path PATH                            Image override. Default: the backend's own" >&2
    echo "  --data-root PATH                           Default: ${DATA_ROOT}" >&2
    echo "  --no-audio                                 Omit audio inputs" >&2
    echo "  --limit N                                  Cap retained records" >&2
    echo "  --index-range X-Y                          Process zero-based JSON indices X through Y" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            backend="${2:?Missing value for --backend}"
            shift 2
            ;;
        --mode)
            mode="${2:?Missing value for --mode}"
            shift 2
            ;;
        --input-json)
            input_json="${2:?Missing value for --input-json}"
            shift 2
            ;;
        --output)
            output_json="${2:?Missing value for --output}"
            shift 2
            ;;
        --output-dir)
            output_dir="${2:?Missing value for --output-dir}"
            shift 2
            ;;
        --prompt-config)
            prompt_config="${2:?Missing value for --prompt-config}"
            shift 2
            ;;
        --model-config)
            model_config="${2:?Missing value for --model-config}"
            shift 2
            ;;
        --participant-image-root)
            participant_image_root="${2:?Missing value for --participant-image-root}"
            shift 2
            ;;
        --sif-path)
            sif_path_override="${2:?Missing value for --sif-path}"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="${2:?Missing value for --data-root}"
            shift 2
            ;;
        --no-audio)
            no_audio=1
            shift
            ;;
        --limit)
            limit="${2:?Missing value for --limit}"
            shift 2
            ;;
        --index-range)
            index_range="${2:?Missing value for --index-range}"
            shift 2
            ;;
        # Removed in favour of the task's model config. Answered explicitly rather
        # than falling through to "Unknown option", so a command out of shell
        # history or an old wrapper gets a one-line fix.
        --max-new-tokens|--max-video-frames|--enable-thinking|--do-sample|--temperature|--top-p|--top-k)
            echo "[ERROR] $1 is no longer a job option." >&2
            echo "        Set it in the task's model_config.json under" >&2
            echo "        backends.<name>.generation (or .extra), then resubmit." >&2
            exit 1
            ;;
        --model-path|--model)
            echo "[ERROR] $1 is no longer a job option: the weights are part of what a run" >&2
            echo "        is, so they are recorded with the rest of it." >&2
            echo "        Set them in the task's model_config.json under" >&2
            echo "        backends.<name>.model_id, then resubmit." >&2
            exit 1
            ;;
        --project-root)
            echo "[ERROR] --project-root is no longer a job option: the checkout is the" >&2
            echo "        environment, so it is PROJECT_ROOT in the stub that submits this." >&2
            exit 1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            usage
            echo "[ERROR] Unknown option: $1" >&2
            exit 1
            ;;
        *)
            usage
            echo "[ERROR] Unexpected positional argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${backend}" ]]; then
    usage
    echo "[ERROR] --backend is required." >&2
    exit 1
fi

if [[ -z "${mode}" ]]; then
    usage
    echo "[ERROR] --mode is required." >&2
    exit 1
fi

if ! intention_backend_is_known "${backend}"; then
    echo "[ERROR] Unknown backend: ${backend}" >&2
    echo "        Known: $(intention_known_backends | tr '\n' ' ')" >&2
    echo "        Add one in job_scripts/lib/model_backends.sh and models/registry.py." >&2
    exit 1
fi
intention_backend_defaults "${backend}"
SIF_PATH="${sif_path_override:-${BACKEND_SIF_PATH}}"

if [[ -n "${limit}" ]]; then
    case "${limit}" in
        ''|*[!0-9]*)
            echo "[ERROR] Invalid limit: ${limit}" >&2
            exit 1
            ;;
    esac
fi

if [[ -n "${index_range}" ]]; then
    if [[ "${index_range}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        start_index="${BASH_REMATCH[1]}"
        end_index="${BASH_REMATCH[2]}"
        if (( end_index < start_index )); then
            echo "[ERROR] Invalid index range: ${index_range} (end is before start)" >&2
            exit 1
        fi
    else
        echo "[ERROR] Invalid index range: ${index_range}; expected X-Y, for example 0-99" >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if [[ "${input_json}" != /* ]]; then
    input_json="${DATA_ROOT}/${input_json}"
fi

# Re-derive in case --data-root overrode DATA_ROOT above.
intention_set_local_prefixes "${DATA_ROOT}"
VIDEO_LOCAL_PATH_PREFIX="${INTENTION_VIDEO_LOCAL_PATH_PREFIX}"
AUDIO_LOCAL_PATH_PREFIX="${INTENTION_AUDIO_LOCAL_PATH_PREFIX}"

run_output_dir="$(intention_output_dir "${output_dir}" "${backend}" "${mode}")"
if [[ -z "${output_json}" ]]; then
    input_stem="$(basename "${input_json}")"
    input_stem="${input_stem%.json}"
    if [[ -n "${index_range}" ]]; then
        input_stem="${input_stem}_${start_index}-${end_index}"
    fi
    output_json="${run_output_dir}/${input_stem}.json"
fi

if [[ "${output_json}" != /* ]]; then
    output_json="${DATA_ROOT}/${output_json}"
fi

SRC_DIR="${PROJECT_ROOT}/${PACKAGE_SUBDIR}/src"
CONTAINER_SRC_DIR="${CONTAINER_PROJECT_ROOT}/${PACKAGE_SUBDIR}/src"

# Deliberately no default: each mode ships its own prompt config, so empty means
# "let the mode decide" and --prompt-config is simply not passed. That is what
# keeps this script from having to learn where any particular mode's JSON lives.
if [[ -z "${model_config}" ]]; then
    model_config="${SRC_DIR}/intention_inference/model_config.json"
fi

# Only paths under PROJECT_ROOT become /workspace paths; one under /tudelft.net
# is bind-mounted at its own path and keeps working untouched.
container_path() {
    local host_path="$1"
    if [[ "${host_path}" == "${PROJECT_ROOT}"/* ]]; then
        printf '%s\n' "${CONTAINER_PROJECT_ROOT}/${host_path#"${PROJECT_ROOT}/"}"
    else
        printf '%s\n' "${host_path}"
    fi
}
container_prompt_config=""
if [[ -n "${prompt_config}" ]]; then
    container_prompt_config="$(container_path "${prompt_config}")"
fi
container_model_config="$(container_path "${model_config}")"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
if [[ ! -f "${SIF_PATH}" ]]; then
    echo "[ERROR] SIF not found for backend '${backend}': ${SIF_PATH}" >&2
    echo "        Build or copy an inference image with transformers, torch, torchvision," >&2
    echo "        librosa and accelerate, then point job_scripts/lib/model_backends.sh at it." >&2
    exit 1
fi

# The job always runs the checkout's code, never a copy baked into the image.
if [[ ! -f "${SRC_DIR}/intention_inference/cli.py" ]]; then
    echo "[ERROR] intention_inference package not found: ${SRC_DIR}/intention_inference" >&2
    echo "        Set PROJECT_ROOT in the stub to your COSILab checkout." >&2
    exit 1
fi
if [[ ! -f "${SRC_DIR}/models/registry.py" ]]; then
    echo "[ERROR] models package not found: ${SRC_DIR}/models" >&2
    echo "        The backend layer lives beside intention_inference in the same src/." >&2
    exit 1
fi

# prompt_config is skipped when empty: that is the "mode decides" case, and only
# cli.py knows where the mode's own file is.
for required_file in "${prompt_config}" "${model_config}" "${input_json}"; do
    if [[ -z "${required_file}" ]]; then
        continue
    fi
    if [[ ! -f "${required_file}" ]]; then
        echo "[ERROR] Not found: ${required_file}" >&2
        exit 1
    fi
done

# No weights check here: this script no longer knows the path. cli.py reads it
# from the model config and checks it before touching the manifest, so the
# failure is just as early and names the file to fix.

# librosa's kernels are numba-jitted with cache=True, and numba writes that cache
# next to the source file -- which lives inside the read-only SIF. Without a
# writable NUMBA_CACHE_DIR the first librosa.load raises
#   cannot cache function '__o_fold': no locator available for file
#   '.../librosa/core/notation.py'
# and since the conversation-floor mix runs per record, EVERY record is skipped
# as audio_aggregation_failed.
#
# Node-local and per job, because this is a compile artefact rather than data:
# putting it on the shared filesystem would have every job writing the same cache
# entries over NFS. Bound explicitly further down, since TMPDIR is not always a
# path apptainer mounts by default. TRITON_CACHE_DIR is the same story one layer
# down, for a backend whose kernels are JIT-compiled.
NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba_cache_${SLURM_JOB_ID:-$$}"
TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_cache_${SLURM_JOB_ID:-$$}"
mkdir -p "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}"
trap 'rm -rf "${NUMBA_CACHE_DIR}" "${TRITON_CACHE_DIR}"' EXIT

# slurmd already opened this job's own .out/.err before the script started, so
# this only helps the *next* submission. The directory must exist at submit time
# or the job never launches.
mkdir -p "${SLURM_OUTPUT_ROOT}/intention"
mkdir -p "${INTENTION_HF_CACHE}"
mkdir -p "$(dirname "${output_json}")"

echo "[INFO] project_root              = ${PROJECT_ROOT} (commit $(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown))"
echo "[INFO] home_root                 = ${HOME_ROOT}"
echo "[INFO] container_project_root    = ${CONTAINER_PROJECT_ROOT}"
echo "[INFO] src_dir                   = ${SRC_DIR}"
echo "[INFO] container_src_dir         = ${CONTAINER_SRC_DIR}"
echo "[INFO] backend                   = ${backend}"
echo "[INFO] mode                      = ${mode}"
echo "[INFO] sif_path                  = ${SIF_PATH}"
echo "[INFO] model_path                = <backends.${backend}.model_id in the model config>"
echo "[INFO] model_config              = ${model_config}"
echo "[INFO] container_model_config    = ${container_model_config}"
echo "[INFO] prompt_config             = ${prompt_config:-<packaged with mode ${mode}>}"
echo "[INFO] container_prompt_config   = ${container_prompt_config:-<not passed>}"
echo "[INFO] data_root                 = ${DATA_ROOT}"
echo "[INFO] input_json                = ${input_json}"
echo "[INFO] output_dir                = ${run_output_dir}"
echo "[INFO] output_json               = ${output_json}"
echo "[INFO] participant_image_root    = ${participant_image_root}"
echo "[INFO] no_audio                  = ${no_audio}"
echo "[INFO] limit                     = ${limit:-<none>}"
echo "[INFO] index_range               = ${index_range:-<none>}"
echo "[INFO] numba_cache_dir           = ${NUMBA_CACHE_DIR}"
echo "[INFO] triton_cache_dir          = ${TRITON_CACHE_DIR}"
echo "[INFO] video_media_path_prefix   = ${INTENTION_VIDEO_MEDIA_PATH_PREFIX}"
echo "[INFO] video_local_path_prefix   = ${VIDEO_LOCAL_PATH_PREFIX}"
echo "[INFO] audio_media_path_prefix   = ${INTENTION_AUDIO_MEDIA_PATH_PREFIX}"
echo "[INFO] audio_local_path_prefix   = ${AUDIO_LOCAL_PATH_PREFIX}"

python_args=(
    python -m intention_inference
    --backend "${backend}"
    --mode "${mode}"
    --input-json "${input_json}"
    --output "${output_json}"
    --model-config "${container_model_config}"
    --participant-image-root "${participant_image_root}"
    --video-media-path-prefix "${INTENTION_VIDEO_MEDIA_PATH_PREFIX}"
    --video-local-path-prefix "${VIDEO_LOCAL_PATH_PREFIX}"
    --audio-media-path-prefix "${INTENTION_AUDIO_MEDIA_PATH_PREFIX}"
    --audio-local-path-prefix "${AUDIO_LOCAL_PATH_PREFIX}"
)

# Omitted rather than passed empty when unset: an empty --prompt-config would be
# accepted by argparse and then resolve to the working directory.
if [[ -n "${container_prompt_config}" ]]; then
    python_args+=(--prompt-config "${container_prompt_config}")
fi
if [[ "${no_audio}" == "1" ]]; then
    python_args+=(--no-audio)
fi
if [[ -n "${limit}" ]]; then
    python_args+=(--limit "${limit}")
fi
if [[ -n "${index_range}" ]]; then
    python_args+=(--start-index "${start_index}" --end-index "${end_index}")
fi

bind_args=(
    --bind "${PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}"
    --bind "${HOME_ROOT}:${HOME_ROOT}"
    --bind "${NUMBA_CACHE_DIR}:${NUMBA_CACHE_DIR}"
    --bind "${TRITON_CACHE_DIR}:${TRITON_CACHE_DIR}"
)
for host_bind_path in "${HOST_BIND_PATHS[@]}"; do
    # apptainer aborts on a bind source that is not there, which would take down
    # a job that may not even need this share. Warn and carry on: if the run does
    # need it, cli.py says which path it could not find.
    if [[ ! -d "${host_bind_path}" ]]; then
        echo "[WARN] Not mounted on this node, skipping bind: ${host_bind_path}" >&2
        continue
    fi
    bind_args+=(--bind "${host_bind_path}:${host_bind_path}")
    echo "[INFO] bind                      = ${host_bind_path}"
done

apptainer_args=(
    --nv
    ${BACKEND_APPTAINER_ARGS[@]+"${BACKEND_APPTAINER_ARGS[@]}"}
    "${bind_args[@]}"
    --pwd "${CONTAINER_PROJECT_ROOT}"
    --env HF_HOME="${INTENTION_HF_CACHE}"
    --env TRANSFORMERS_CACHE="${INTENTION_HF_CACHE}"
    --env NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR}"
    --env TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
    --env PYTHONPATH="${CONTAINER_SRC_DIR}"
)

# The checkout existing on the host is not enough: PYTHONPATH has to resolve to
# it inside the container, otherwise python silently imports whatever the image
# was built with. Both packages are checked -- models/ is new and just as
# overridable as intention_inference, and a stale copy of it would change which
# model ran without changing anything visible in the log.
echo "[INFO] Verifying the container resolves both packages from the checkout..."
apptainer exec "${apptainer_args[@]}" "${SIF_PATH}" \
    python -c '
import importlib, os, sys
expected = os.path.realpath(sys.argv[1])
for name in ("intention_inference", "models"):
    module = importlib.import_module(name)
    actual = os.path.realpath(os.path.dirname(os.path.dirname(module.__file__)))
    if actual != expected:
        sys.exit(
            f"[ERROR] Container imported {name} from {actual}, not {expected}.\n"
            "        The checkout is not visible inside the container - add a bind mount for it."
        )
    print(f"[INFO] container resolved {name} from {actual}")
' "${CONTAINER_SRC_DIR}"

srun apptainer exec "${apptainer_args[@]}" \
    "${SIF_PATH}" \
    "${python_args[@]}"

echo "[INFO] Intention inference completed on '${backend}': ${output_json}"
