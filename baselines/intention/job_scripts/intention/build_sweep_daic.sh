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

# Build AND submit the prompt-variation sweep for ONE model:
# clips x personas x instruction prompts x trials.
#
#   bash job_scripts/intention/build_sweep_daic.sh --model qwen7b
#   bash job_scripts/intention/build_sweep_daic.sh --model qwen7b --submit
#   bash job_scripts/intention/build_sweep_daic.sh --model gemma --seed 2 --submit
#
# Dry run is the default, as in persona.sh: the first command prints the design,
# the six jobs and where they would write, and queues nothing.
#
# THE DESIGN IS FIXED, in build_sweep.py rather than here: 5 clips, 5 personas,
# the six structural variants, 50 trials. Per model:
#
#     5 x 5 x 6 x 50 = 7,500 answers, as 6 jobs of 1,250
#
# One job per instruction prompt, because an instruction prompt is run-wide -- a
# worker process asks one question -- so a prompt boundary is also a job
# boundary, and a job that straddled one would have to load the model twice.
#
# ONE MODEL PER RUN. --model is the model, and every job this submits runs it;
# another model is another run of this script, at the same seed. That is what
# keeps a model's 7,500 answers together and comparable with the next model's:
# the spec is built once per seed, and each later model REUSES it -- a rebuild
# that disagrees is refused rather than silently giving model B different clips.
#
# --mode is pa and only pa: the instruction prompts being varied are written for
# it.
#
# Submitting needs a login node, so run this with bash. sbatch still works for
# the build alone (it loads no model: it stats media paths and shuffles
# integers), but --submit is refused from inside a job.

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
# standard library -- so the smallest one is used rather than a backend's. It is
# the builder's interpreter only; each job picks its own image from --model.
SIF_PATH="${SIF_PATH:-/tudelft.net/staff-umbrella/neon/apptainer/gemma.sif}"

# Same file the inference job and persona.sh read, so the sweep is validated
# against exactly the paths inference will use. A prefix that stopped matching
# would otherwise leave every clip unusable here and equally unusable there,
# which is the failure this sharing exists to make impossible.
source "${PROJECT_ROOT}/${PACKAGE_SUBDIR}/job_scripts/lib/intention_paths.sh"
source "${PROJECT_ROOT}/${PACKAGE_SUBDIR}/job_scripts/lib/model_backends.sh"

SRC_DIR="${PROJECT_ROOT}/${PACKAGE_SUBDIR}/src"
CONTAINER_SRC_DIR="${CONTAINER_PROJECT_ROOT}/${PACKAGE_SUBDIR}/src"
builder="${SRC_DIR}/intention_inference/build_sweep.py"
launcher="${PROJECT_ROOT}/${PACKAGE_SUBDIR}/job_scripts/intention/persona.sh"

# The persona pool. A path, so unlike the design numbers it belongs here beside
# the rest of the pipeline's paths.
PERSONAS_JSON="${INTENTION_PIPELINE_ROOT}/personas_V2_prolific_ids.json"

model=""
# Seed is parsed here, not just passed through: it names the spec and the results
# directory, so this script has to know it before the builder runs.
seed="1"
mode="pa"
# Counted in GENERATIONS, which is what the design is counted in. persona.sh's
# own --answers-per-job counts persona x clip answers and knows nothing about
# --trials, so this is divided by the trials before being handed over -- see the
# conversion below. 1500 leaves all five personas of a prompt in one job.
answers_per_job="1500"
# Generous on purpose: 1,250 generations is far longer than the worker's 10h
# header was ever sized for.
time_limit="16:00:00"
submit=0
rebuild=0
extra_args=()

usage() {
    echo "Usage: bash $0 --model NAME [options]" >&2
    echo "" >&2
    echo "Builds the clip x persona x instruction-prompt x trial sweep and submits" >&2
    echo "it for one model. Dry run unless --submit is given." >&2
    echo "" >&2
    echo "  --model NAME          Required. Known: $(intention_known_backends | tr '\n' ' ')" >&2
    echo "                        One model per run; every job submitted runs it." >&2
    echo "  --seed N              Chooses the clips and the personas, and names" >&2
    echo "                        the spec and the results dir. Default: ${seed}" >&2
    echo "  --submit              Queue the jobs. Default: dry run" >&2
    echo "  --time HH:MM:SS       Walltime per job. Default: ${time_limit}" >&2
    echo "  --answers-per-job N   Generations per job, converted to persona.sh's" >&2
    echo "                        clip units using the spec's trials." >&2
    echo "                        Default: ${answers_per_job}" >&2
    echo "  --rebuild             Replace an existing spec the rebuild disagrees" >&2
    echo "                        with. Every model at this seed must then be rerun." >&2
    echo "  --personas-json P     Persona pool. Default: ${PERSONAS_JSON}" >&2
    echo "  --mode NAME           pa, and only pa: the instruction prompts being" >&2
    echo "                        varied are written for it." >&2
    echo "  --skip-validation     Sample without checking the clips' media resolves" >&2
    echo "  -- EXTRA...           Forwarded to build_sweep.py (--temperature, --top-p)" >&2
    echo "" >&2
    echo "The design -- 5 clips, 5 personas, 6 instruction prompts, 50 trials -- is" >&2
    echo "fixed in build_sweep.py and cannot be set on the command line." >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--backend)  model="${2:?Missing value for $1}"; shift 2 ;;
        --seed)             seed="${2:?Missing value for --seed}"; shift 2 ;;
        --mode)             mode="${2:?Missing value for --mode}"; shift 2 ;;
        --answers-per-job)  answers_per_job="${2:?Missing value for --answers-per-job}"; shift 2 ;;
        --time)             time_limit="${2:?Missing value for --time}"; shift 2 ;;
        --personas-json)    PERSONAS_JSON="${2:?Missing value for --personas-json}"; shift 2 ;;
        --submit)           submit=1; shift ;;
        --rebuild)          rebuild=1; shift ;;
        --skip-validation)  extra_args+=(--skip-validation); shift ;;
        --trials)
            echo "[ERROR] --trials is part of the design, not an option: it is NUM_TRIALS" >&2
            echo "        in intention_inference/build_sweep.py. A sweep whose trial count" >&2
            echo "        came from whatever flags one submission carried is a sweep nobody" >&2
            echo "        can reproduce from the repo." >&2
            exit 1
            ;;
        --output)
            echo "[ERROR] --output is no longer an option: the spec is named by its seed," >&2
            echo "        under ${INTENTION_SWEEP_DIR}, so that every model at one seed" >&2
            echo "        reads the same design. Use --seed." >&2
            exit 1
            ;;
        --)                 shift; extra_args+=("$@"); break ;;
        -h|--help)          usage; exit 0 ;;
        *)                  usage; echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "${model}" ]]; then
    usage
    echo "[ERROR] --model is required. Which model answered is the first thing anyone" >&2
    echo "        asks of a result, and it is also a path component." >&2
    exit 1
fi
if ! intention_backend_is_known "${model}"; then
    echo "[ERROR] Unknown model: ${model}" >&2
    echo "        Known: $(intention_known_backends | tr '\n' ' ')" >&2
    exit 1
fi
if [[ "${mode}" != "pa" ]]; then
    echo "[ERROR] --mode ${mode} is not supported by this sweep." >&2
    echo "        The instruction prompts it varies are written for pa; asking them" >&2
    echo "        under another mode would measure the mode, not the wording." >&2
    exit 1
fi
case "${seed}" in
    ''|*[!0-9]*) echo "[ERROR] --seed must be a non-negative integer: ${seed}" >&2; exit 1 ;;
esac
case "${answers_per_job}" in
    ''|*[!0-9]*|0) echo "[ERROR] --answers-per-job must be a positive integer" >&2; exit 1 ;;
esac
if [[ ! "${time_limit}" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "[ERROR] --time must look like HH:MM:SS or D-HH:MM:SS: ${time_limit}" >&2
    exit 1
fi
if [[ "${submit}" == "1" && -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[ERROR] --submit needs a login node: persona.sh is a launcher and refuses to" >&2
    echo "        run inside a job. Submit the sweep with bash, or sbatch this script" >&2
    echo "        without --submit to build the spec only." >&2
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
if [[ ! -f "${launcher}" ]]; then
    echo "[ERROR] Launcher not found: ${launcher}" >&2
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

# ---------------------------------------------------------------------------
# Paths. The spec is keyed by seed and shared by every model; only the results
# are per model.
# ---------------------------------------------------------------------------
SWEEP_SPEC_JSON="${INTENTION_SWEEP_DIR}/sweep_seed${seed}.json"
SWEEP_MODEL_CONFIG="${INTENTION_SWEEP_DIR}/sweep_seed${seed}_model_config.json"
# No --audio-mix-dir, deliberately. The conversation-floor mix depends on the
# clip alone, so one cache for the whole seed looks like the obvious saving --
# but aggregate_conversation_floor_audio rewrites its output every run rather
# than reusing an existing file, and these jobs run at the same time. Six jobs
# rewriting the same five WAVs while another reads them is a torn read, and the
# answer it produces looks like an answer.
#
# Left to the default instead: persona_output_dir/_audio_mixes, which is
# per-prompt here because the output directory is, so every job owns its cache
# and still reuses it across its five personas and across resumes. Five clips
# mixed six times is seconds of CPU against 7,500 generations.
RESULTS_DIR="$(intention_sweep_output_dir \
    "${INTENTION_OUTPUT_DIR}" "${model}" "${mode}" "${seed}")"

mkdir -p "${INTENTION_SWEEP_DIR}"

# Paths first, then the caller's flags, so anything in EXTRA can override them:
# argparse lets the last value win, and every flag here is a path or a knob
# rather than part of the design.
builder_args=(
    --input-json "${INTENTION_INPUT_JSON}"
    --personas-json "${PERSONAS_JSON}"
    --instruction-prompt-dir "${INTENTION_INSTRUCTION_PROMPT_DIR}"
    --output "${SWEEP_SPEC_JSON}"
    --seed "${seed}"
    --mode "${mode}"
    --participant-image-root "${INTENTION_PARTICIPANT_IMAGE_ROOT}"
    --video-media-path-prefix "${INTENTION_VIDEO_MEDIA_PATH_PREFIX}"
    --video-local-path-prefix "${INTENTION_VIDEO_LOCAL_PATH_PREFIX}"
    --audio-media-path-prefix "${INTENTION_AUDIO_MEDIA_PATH_PREFIX}"
    --audio-local-path-prefix "${INTENTION_AUDIO_LOCAL_PATH_PREFIX}"
)
[[ "${rebuild}" == "1" ]] && builder_args+=(--rebuild)

echo "[INFO] project_root       = ${PROJECT_ROOT}"
echo "[INFO] sif_path           = ${SIF_PATH} (builder only)"
echo "[INFO] builder            = ${builder}"
echo "[INFO] input_json         = ${INTENTION_INPUT_JSON}"
echo "[INFO] personas_json      = ${PERSONAS_JSON}"
echo "[INFO] instruction_dir    = ${INTENTION_INSTRUCTION_PROMPT_DIR}"
echo "[INFO] sweep_spec         = ${SWEEP_SPEC_JSON}"
echo "[INFO] video_media_prefix = ${INTENTION_VIDEO_MEDIA_PATH_PREFIX}"
echo "[INFO] video_local_prefix = ${INTENTION_VIDEO_LOCAL_PATH_PREFIX}"
echo "[INFO] audio_media_prefix = ${INTENTION_AUDIO_MEDIA_PATH_PREFIX}"
echo "[INFO] audio_local_prefix = ${INTENTION_AUDIO_LOCAL_PATH_PREFIX}"
echo "[INFO] participant_images = ${INTENTION_PARTICIPANT_IMAGE_ROOT}"
echo "[INFO] extra args         = ${extra_args[*]:-<none>}"
echo

apptainer exec \
    --bind "${PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    --pwd "${CONTAINER_PROJECT_ROOT}" \
    --env PYTHONPATH="${CONTAINER_SRC_DIR}" \
    "${SIF_PATH}" \
    python -m intention_inference.build_sweep \
    "${builder_args[@]}" ${extra_args[@]+"${extra_args[@]}"}

# ---------------------------------------------------------------------------
# Read the design back out of the spec that was just built or reused, rather
# than assuming it: a reused spec is the authority on what this model must
# answer, and its trials and model config are what the jobs are given.
# ---------------------------------------------------------------------------
clip_ids=()
persona_ids=()
instruction_prompts=()
trials=""
model_config=""
total_answers=""

describe_text="$(apptainer exec \
    --bind "${PROJECT_ROOT}:${CONTAINER_PROJECT_ROOT}" \
    --bind /tudelft.net/staff-umbrella/neon:/tudelft.net/staff-umbrella/neon \
    --pwd "${CONTAINER_PROJECT_ROOT}" \
    --env PYTHONPATH="${CONTAINER_SRC_DIR}" \
    "${SIF_PATH}" \
    python -m intention_inference.build_sweep --describe "${SWEEP_SPEC_JSON}")" || {
    echo "[ERROR] Could not read the sweep spec: ${SWEEP_SPEC_JSON}" >&2
    exit 1
}

while IFS=$'\t' read -r row_kind field1 field2 field3 field4; do
    case "${row_kind}" in
        META)
            # field1 is the spec's own seed. It came from --seed, but a reused
            # spec is the authority, so a mismatch means the file is not the one
            # this run's paths were built from.
            if [[ "${field1}" != "${seed}" ]]; then
                echo "[ERROR] ${SWEEP_SPEC_JSON} was built with seed ${field1}, not ${seed}." >&2
                exit 1
            fi
            trials="${field2}"
            model_config="${field3}"
            total_answers="${field4}"
            ;;
        CLIP)    clip_ids+=("${field1}") ;;
        PERSONA) persona_ids+=("${field1}") ;;
        PROMPT)  instruction_prompts+=("${field1}") ;;
        '') ;;
        *) echo "[ERROR] Unknown row from --describe: ${row_kind}" >&2; exit 1 ;;
    esac
done <<< "${describe_text}"

if [[ -z "${trials}" || ${#instruction_prompts[@]} -eq 0 ]]; then
    echo "[ERROR] The spec described no trials or no instruction prompts:" >&2
    echo "        ${SWEEP_SPEC_JSON}" >&2
    exit 1
fi
if [[ "${model_config}" != "${SWEEP_MODEL_CONFIG}" ]]; then
    echo "[ERROR] The spec names a model config this script did not derive:" >&2
    echo "        spec says  ${model_config}" >&2
    echo "        expected   ${SWEEP_MODEL_CONFIG}" >&2
    exit 1
fi
if [[ ! -f "${SWEEP_MODEL_CONFIG}" ]]; then
    echo "[ERROR] Sampling model config not found: ${SWEEP_MODEL_CONFIG}" >&2
    echo "        Rebuild the spec: rerun with --rebuild." >&2
    exit 1
fi

# persona.sh counts persona x clip answers and knows nothing about --trials, so
# a budget in generations is converted here. Rounded down, floored at one: a
# budget below one clip's trials still has to submit something, and that job is
# one persona whatever the arithmetic says.
clip_units=$(( answers_per_job / trials ))
(( clip_units < 1 )) && clip_units=1
answers_per_prompt=$(( ${#clip_ids[@]} * ${#persona_ids[@]} * trials ))

echo
echo "[INFO] === sweep seed ${seed}, model ${model} ==="
echo "[INFO] clips     (${#clip_ids[@]}): ${clip_ids[*]}"
echo "[INFO] personas  (${#persona_ids[@]}): ${persona_ids[*]}"
echo "[INFO] prompts   (${#instruction_prompts[@]}):"
for prompt_path in "${instruction_prompts[@]}"; do
    echo "[INFO]   $(basename "${prompt_path}")"
done
echo "[INFO] trials per clip    = ${trials}"
echo "[INFO] model config       = ${SWEEP_MODEL_CONFIG}"
echo "[INFO] results dir        = ${RESULTS_DIR}"
echo "[INFO] walltime per job   = ${time_limit}"
echo "[INFO] answers per job    = ${answers_per_job} generation(s) = ${clip_units} clip-answer(s)"
echo "[INFO] planned            = ${#instruction_prompts[@]} prompt(s) x ${answers_per_prompt} answer(s) = ${total_answers}"
echo "[INFO] submit             = ${submit}"
echo

# The design travels with the answers: a results tree that names its spec by
# path is one broken symlink away from being unreadable, and these two files are
# a few kB against 7,500 answers.
mkdir -p "${RESULTS_DIR}"
cp -f "${SWEEP_SPEC_JSON}" "${SWEEP_MODEL_CONFIG}" "${RESULTS_DIR}/"

# ---------------------------------------------------------------------------
# One persona.sh run per instruction prompt. Packing, the resume skip and sbatch
# stay its job; nothing about Slurm is repeated here.
# ---------------------------------------------------------------------------
for prompt_path in "${instruction_prompts[@]}"; do
    if [[ ! -f "${prompt_path}" ]]; then
        echo "[ERROR] Instruction prompt named by the spec is missing: ${prompt_path}" >&2
        exit 1
    fi
    # Composed here because persona.sh appends the prompt's own subdirectory only
    # when it derives the directory itself; one given with --persona-output-dir is
    # taken verbatim. Without this every variant would write persona_<id>.json
    # over the last one's, and the resume check would then skip what it had just
    # overwritten.
    prompt_output_dir="$(intention_instruction_output_dir "${RESULTS_DIR}" "${prompt_path}")"

    launcher_args=(
        --backend "${model}"
        --mode "${mode}"
        --task-spec "${SWEEP_SPEC_JSON}"
        --instruction-prompt "${prompt_path}"
        --persona-output-dir "${prompt_output_dir}"
        --model-config "${SWEEP_MODEL_CONFIG}"
        --trials "${trials}"
        --answers-per-job "${clip_units}"
        --time "${time_limit}"
    )
    [[ "${submit}" == "1" ]] && launcher_args+=(--submit)

    echo "[INFO] --- $(basename "${prompt_path}") -> ${prompt_output_dir}"
    bash "${launcher}" "${launcher_args[@]}"
    echo
done

echo "[INFO] Sweep spec:   ${SWEEP_SPEC_JSON}"
echo "[INFO] Results:      ${RESULTS_DIR}"
if [[ "${submit}" == "1" ]]; then
    echo "[INFO] Submitted seed ${seed} for ${model}: up to ${total_answers} answer(s)."
    echo "[INFO] Rerun the same command to resume; finished personas are skipped."
    echo "[INFO] The other models run the same design at this seed, for example:"
    echo "[INFO]   bash $0 --model gemma --seed ${seed} --submit"
else
    echo "[INFO] Dry run; nothing was queued. Add --submit to queue the jobs above."
fi
