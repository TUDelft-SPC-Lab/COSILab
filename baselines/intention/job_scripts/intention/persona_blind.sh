#!/bin/bash
# Launch the BLIND control for a persona source: no media, only the persona.
#
# The model gets the persona and a prompt saying a clip exists but is withheld
# (modes/prompt_blind.json), and nothing else. That prompt names no clip, so the
# answer depends on the persona alone -- what this measures is each persona's
# prior, as a distribution over repeated draws:
#
#     one clip per persona  x  TRIALS draws  at  TEMPERATURE
#
# It is a launcher of its own rather than a --mode of persona.sh because almost
# nothing about the run is shared with the media modes: one clip instead of the
# persona's list, sampling on instead of greedy, many trials instead of one, and
# the blind prompt instead of the instruction prompt.
#
# Takes the same persona sources persona.sh does, so the control covers the same
# personas as the run it controls for:
#
#   bash job_scripts/intention/persona_blind.sh \
#       --backend gemma --task-spec /path/to/personas.json
#
#   bash job_scripts/intention/persona_blind.sh \
#       --backend qwen7b --assignment-json /path/to/assignment.json --submit
#
# From the source it derives, into the results directory, the blind spec (each
# persona with the first of its clips, used only as a record id -- no media is
# read or checked) and a sampling model config -- see
# intention_inference/build_blind_spec.py. Dry run is the default;
# rerunning resumes, skipping packs whose persona outputs all exist.

set -euo pipefail

# The design. Fixed here rather than on the command line, and part of the results
# path below, so a run at different settings cannot resume into these files.
TEMPERATURE="0.3"
TRIALS="100"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PACKAGE_ROOT}/job_scripts/lib/intention_paths.sh"
source "${PACKAGE_ROOT}/job_scripts/lib/model_backends.sh"

SRC_DIR="${PACKAGE_ROOT}/src"
job_script="${SCRIPT_DIR}/cosilab_daic.sh"

backend=""
assignment_json=""
task_spec=""
input_json="${INTENTION_INPUT_JSON}"
persona_dir="${INTENTION_PERSONA_DIR}"
persona_output_dir=""
# In personas, not answers: every persona here is exactly TRIALS generations.
personas_per_job="10"
time_limit=""
submit=0
skip_complete=1
max_jobs=""
sleep_seconds="0"
extra_args=()

usage() {
    echo "Usage:" >&2
    printf '  bash %s --backend NAME \\\n' "$0" >&2
    echo "      (--assignment-json PATH | --task-spec PATH) [options]" >&2
    echo "" >&2
    echo "Blind control: one clip per persona, ${TRIALS} draws at temperature ${TEMPERATURE}." >&2
    echo "Options:" >&2
    echo "  --backend NAME          Required. Known: $(intention_known_backends | tr '\n' ' ')" >&2
    echo "  --assignment-json PATH  Balanced persona-to-clip design (any mode's)" >&2
    echo "  --task-spec PATH        Externally provided persona task specification" >&2
    echo "  --personas-per-job N    Personas per GPU job, ${TRIALS} generations each." >&2
    echo "                          Default: ${personas_per_job}" >&2
    echo "  --input-json PATH       Clip manifest. Default: ${input_json}" >&2
    echo "  --persona-dir PATH      Assignment persona_XXXX.txt files. Default: ${persona_dir}" >&2
    echo "  --persona-output-dir P  Default: derived from backend, source and design" >&2
    echo "  --time HH:MM:SS         sbatch walltime override. Default: the worker's" >&2
    echo "                          own #SBATCH --time." >&2
    echo "  --submit                Queue jobs. Default: dry run" >&2
    echo "  --no-skip-complete      Include packs whose outputs already exist" >&2
    echo "  --overwrite             Recompute existing persona outputs" >&2
    echo "  --max-jobs N            Stop after this many non-skipped jobs" >&2
    echo "  --sleep SECONDS         Wait between sbatch calls" >&2
    echo "  --job-script PATH       Internal worker. Default: ${job_script}" >&2
    echo "  -- EXTRA...             Additional options forwarded to every worker job" >&2
    echo "" >&2
    echo "Temperature and trials are fixed at the top of this file." >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)            backend="${2:?Missing value for --backend}"; shift 2 ;;
        --assignment-json)    assignment_json="${2:?Missing value for --assignment-json}"; shift 2 ;;
        --task-spec)          task_spec="${2:?Missing value for --task-spec}"; shift 2 ;;
        --personas-per-job)   personas_per_job="${2:?Missing value for --personas-per-job}"; shift 2 ;;
        --input-json)         input_json="${2:?Missing value for --input-json}"; shift 2 ;;
        --persona-dir)        persona_dir="${2:?Missing value for --persona-dir}"; shift 2 ;;
        --persona-output-dir) persona_output_dir="${2:?Missing value for --persona-output-dir}"; shift 2 ;;
        --time)               time_limit="${2:?Missing value for --time}"; shift 2 ;;
        --job-script)         job_script="${2:?Missing value for --job-script}"; shift 2 ;;
        --max-jobs)           max_jobs="${2:?Missing value for --max-jobs}"; shift 2 ;;
        --sleep)              sleep_seconds="${2:?Missing value for --sleep}"; shift 2 ;;
        --submit)             submit=1; shift ;;
        --no-skip-complete)   skip_complete=0; shift ;;
        --overwrite)          skip_complete=0; extra_args+=(--overwrite); shift ;;
        --mode|--trials|--temperature|--model-config|--instruction-prompt|--prompt-config)
            echo "[ERROR] $1 is not an option here: the blind design is fixed in this" >&2
            echo "        launcher (mode blind, its own prompt, ${TRIALS} trials at" >&2
            echo "        temperature ${TEMPERATURE}). Edit the top of $0 to change it." >&2
            exit 1
            ;;
        --)                   shift; extra_args+=("$@"); break ;;
        -h|--help)            usage; exit 0 ;;
        *)                    usage; echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[ERROR] persona_blind.sh is a login-node launcher; run it with bash, not sbatch." >&2
    exit 1
fi
if [[ -z "${backend}" ]]; then
    usage
    echo "[ERROR] --backend is required." >&2
    exit 1
fi
if ! intention_backend_is_known "${backend}"; then
    echo "[ERROR] Unknown backend: ${backend}" >&2
    echo "        Known: $(intention_known_backends | tr '\n' ' ')" >&2
    exit 1
fi
backend_gpu_count="$(intention_backend_gpu_count "${backend}")"
if [[ -n "${assignment_json}" && -n "${task_spec}" ]]; then
    echo "[ERROR] Choose one persona source: --assignment-json or --task-spec." >&2
    exit 1
fi
if [[ -z "${assignment_json}" && -z "${task_spec}" ]]; then
    echo "[ERROR] One persona source is required: --assignment-json or --task-spec." >&2
    exit 1
fi
case "${personas_per_job}" in
    ''|*[!0-9]*|0) echo "[ERROR] --personas-per-job must be a positive integer" >&2; exit 1 ;;
esac
case "${sleep_seconds}" in
    ''|*[!0-9]*) echo "[ERROR] --sleep must be a non-negative integer" >&2; exit 1 ;;
esac
if [[ -n "${max_jobs}" ]]; then
    case "${max_jobs}" in
        *[!0-9]*|0) echo "[ERROR] --max-jobs must be a positive integer" >&2; exit 1 ;;
    esac
fi
if [[ -n "${time_limit}" && ! "${time_limit}" =~ ^([0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "[ERROR] --time must look like HH:MM:SS or D-HH:MM:SS: ${time_limit}" >&2
    exit 1
fi
if [[ ! -f "${job_script}" ]]; then
    echo "[ERROR] Worker job script not found: ${job_script}" >&2
    exit 1
fi
if [[ ! -f "${input_json}" ]]; then
    echo "[ERROR] Input JSON not found: ${input_json}" >&2
    exit 1
fi

builder_args=(
    python3 -m intention_inference.build_blind_spec
    --input-json "${input_json}"
    --temperature "${TEMPERATURE}"
    --trials "${TRIALS}"
)
if [[ -n "${assignment_json}" ]]; then
    source_kind="assignment"
    source_path="${assignment_json}"
    if [[ ! -f "${assignment_json}" ]]; then
        echo "[ERROR] Assignment JSON not found: ${assignment_json}" >&2
        exit 1
    fi
    if [[ ! -d "${persona_dir}" ]]; then
        echo "[ERROR] Persona directory not found: ${persona_dir}" >&2
        exit 1
    fi
    builder_args+=(--assignment-json "${assignment_json}" --persona-dir "${persona_dir}")
else
    source_kind="spec"
    source_path="${task_spec}"
    if [[ ! -f "${task_spec}" ]]; then
        echo "[ERROR] Task spec not found: ${task_spec}" >&2
        exit 1
    fi
    builder_args+=(--task-spec "${task_spec}")
fi

# Keyed by source and by design. Blind gets its own mode component like any other
# mode; the design leaf keeps a later temperature or trial count from resuming
# into these files and being skipped as already done.
if [[ -z "${persona_output_dir}" ]]; then
    source_stem="$(basename "${source_path}")"
    persona_output_dir="$(intention_output_dir \
        "${INTENTION_OUTPUT_DIR}" "${backend}" "blind")/${source_stem%.json}/temp${TEMPERATURE}_trials${TRIALS}"
fi
# Beside the answers, so the design travels with them. Underscored like the
# worker's _audio_mixes, so neither can be mistaken for a persona's output.
blind_spec="${persona_output_dir}/_blind_spec.json"
blind_model_config="${persona_output_dir}/_model_config.json"
builder_args+=(--spec-output "${blind_spec}" --model-config-output "${blind_model_config}")

PYTHONPATH="${SRC_DIR}" "${builder_args[@]}" || {
    echo "[ERROR] Could not derive the blind spec from: ${source_path}" >&2
    exit 1
}

# Every task has one clip, so persona.sh's answer budget is a persona count here.
plan_text="$(PYTHONPATH="${SRC_DIR}" python3 -m intention_inference.runs.planning \
    --task-spec "${blind_spec}" --answers-per-job "${personas_per_job}")" || {
    echo "[ERROR] Could not plan the blind spec: ${blind_spec}" >&2
    exit 1
}

total_runs=0
packed_jobs=0
chunk_selector_starts=()
chunk_selector_ends=()
chunk_answers=()
chunk_output_stems=()
while IFS=$'\t' read -r row_kind field1 field2 field3 field4; do
    case "${row_kind}" in
        META)
            total_runs="${field2}"
            packed_jobs="${field4}"
            ;;
        CHUNK)
            chunk_selector_starts+=("${field1}")
            chunk_selector_ends+=("${field2}")
            chunk_answers+=("${field3}")
            chunk_output_stems+=("${field4}")
            ;;
        '') ;;
        *) echo "[ERROR] Internal source parser returned an unknown row: ${row_kind}" >&2; exit 1 ;;
    esac
done <<< "${plan_text}"

echo "[INFO] launcher           = ${BASH_SOURCE[0]}"
echo "[INFO] worker             = ${job_script}"
echo "[INFO] backend            = ${backend}"
echo "[INFO] GPUs per job       = ${backend_gpu_count}"
echo "[INFO] mode               = blind"
echo "[INFO] persona source     = ${source_kind}: ${source_path}"
echo "[INFO] blind spec         = ${blind_spec}"
echo "[INFO] model config       = ${blind_model_config} (temperature ${TEMPERATURE})"
echo "[INFO] personas           = ${total_runs}"
echo "[INFO] trials per persona = ${TRIALS}"
echo "[INFO] total generations  = $(( total_runs * TRIALS ))"
echo "[INFO] personas per job   = ${personas_per_job} ($(( personas_per_job * TRIALS )) generations)"
echo "[INFO] walltime           = ${time_limit:-<the worker header #SBATCH --time>}"
echo "[INFO] packed jobs        = ${packed_jobs}"
echo "[INFO] persona output dir = ${persona_output_dir}"
echo "[INFO] submit             = ${submit}"
echo

planned=0
skipped=0
submitted=0
reached_cap=0
for (( chunk = 0; chunk < ${#chunk_selector_starts[@]}; chunk++ )); do
    selector_start="${chunk_selector_starts[chunk]}"
    selector_end="${chunk_selector_ends[chunk]}"
    persona_count="${chunk_answers[chunk]}"

    complete=1
    IFS=',' read -r -a output_stems <<< "${chunk_output_stems[chunk]}"
    for output_stem in "${output_stems[@]}"; do
        if [[ ! -f "${persona_output_dir}/${output_stem}.json" ]]; then
            complete=0
            break
        fi
    done
    if [[ "${skip_complete}" == "1" && "${complete}" == "1" ]]; then
        echo "[SKIP] tasks ${selector_start}-${selector_end}: all outputs exist"
        skipped=$(( skipped + 1 ))
        continue
    fi
    if [[ -n "${max_jobs}" && "${planned}" -ge "${max_jobs}" ]]; then
        reached_cap=1
        break
    fi
    planned=$(( planned + 1 ))

    echo "[PLAN] tasks ${selector_start}-${selector_end}: ${persona_count} persona(s) x ${TRIALS}"

    sbatch_cmd=(
        sbatch
        --export=ALL,INTENTION_PERSONA_JOB=1
        --gres="gpu:nvidia_rtx_pro_6000:${backend_gpu_count}"
        --job-name="int_blind_${backend}_${selector_start}-${selector_end}"
    )
    # Before the script path: an sbatch option after it belongs to the job.
    [[ -n "${time_limit}" ]] && sbatch_cmd+=(--time="${time_limit}")
    sbatch_cmd+=(
        "${job_script}"
        --backend "${backend}"
        --mode blind
        --input-json "${input_json}"
        --task-spec "${blind_spec}"
        --task-range "${selector_start}-${selector_end}"
        --persona-output-dir "${persona_output_dir}"
        --model-config "${blind_model_config}"
        --no-instruction-prompt
        --trials "${TRIALS}"
    )
    (( ${#extra_args[@]} > 0 )) && sbatch_cmd+=("${extra_args[@]}")

    if [[ "${submit}" == "1" ]]; then
        "${sbatch_cmd[@]}"
        submitted=$(( submitted + 1 ))
        (( sleep_seconds > 0 )) && sleep "${sleep_seconds}"
    else
        printf '%q ' "${sbatch_cmd[@]}"
        printf '\n'
    fi
done

echo
echo "[INFO] jobs planned      = ${planned}"
echo "[INFO] jobs skipped      = ${skipped} (already complete)"
if [[ "${submit}" == "1" ]]; then
    echo "[INFO] jobs submitted    = ${submitted}"
else
    echo "[INFO] jobs submitted    = 0 (dry run; add --submit)"
fi
if (( reached_cap == 1 )); then
    echo "[WARN] Stopped at the --max-jobs cap of ${max_jobs}; rerun to continue."
fi
