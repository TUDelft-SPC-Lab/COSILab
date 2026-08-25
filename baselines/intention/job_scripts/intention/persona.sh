#!/bin/bash
# Launch intention inference WITH personas.
#
# This file has one entry point: run it with bash on a login node. It reads
# exactly one persona source, packs complete persona runs into GPU jobs, and
# submits cosilab_daic.sh once per pack. It never performs inference itself.
#
# A source can be either:
#
#   --assignment-json  balanced persona -> manifest-index assignments
#   --task-spec        externally authored persona tasks with clip record ids
#
# A persona run is kept whole because it writes one JSON file. Splitting that
# persona by clip range across concurrent jobs would make several jobs overwrite
# the same file. Instead, --answers-per-job limits the sum of persona x clip
# answers in each job. With 2,000 clips, six answers per clip and a budget of
# 1,200, the source contains 12,000 answers and this launcher creates 10 jobs.
# If each persona has 200 clips, that is exactly six personas per job.
#
# Dry run is the default. Add --submit after checking the printed sbatch lines.
# Rerunning the same command resumes: a pack is skipped when all of its persona
# outputs already exist, and a partially complete pack is safe because the
# inference process skips its finished persona files.
#
# Examples:
#
#   bash job_scripts/intention/persona.sh \
#       --backend qwen7b --mode pa --assignment-json /path/to/assignment.json
#
#   bash job_scripts/intention/persona.sh \
#       --backend qwen7b --mode pa --assignment-json /path/to/assignment.json \
#       --answers-per-job 1200 --submit
#
#   bash job_scripts/intention/persona.sh \
#       --backend gemma --mode pa --task-spec /path/to/personas.json --submit

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PACKAGE_ROOT}/job_scripts/lib/intention_paths.sh"
source "${PACKAGE_ROOT}/job_scripts/lib/model_backends.sh"

SRC_DIR="${PACKAGE_ROOT}/src"
job_script="${SCRIPT_DIR}/cosilab_daic.sh"

backend=""
mode=""
assignment_json=""
task_spec=""
persona_range=""
task_range=""
answers_per_job="1200"
input_json="${INTENTION_INPUT_JSON}"
persona_dir="${INTENTION_PERSONA_DIR}"
persona_output_dir=""
prompt_config=""
audio_mix_dir=""
submit=0
skip_complete=1
max_jobs=""
sleep_seconds="0"
extra_args=()

usage() {
    echo "Usage:" >&2
    printf '  bash %s --backend NAME --mode NAME \\\n' "$0" >&2
    echo "      (--assignment-json PATH | --task-spec PATH) [options]" >&2
    echo "Options:" >&2
    echo "  --backend NAME          Required. Known: $(intention_known_backends | tr '\n' ' ')" >&2
    echo "  --mode NAME             Required. fa | sa | pa" >&2
    echo "  --assignment-json PATH  Balanced persona-to-clip design" >&2
    echo "  --task-spec PATH        Externally provided persona task specification" >&2
    echo "  --answers-per-job N     Maximum persona x clip answers packed into one job" >&2
    echo "                          without splitting a persona. Default: ${answers_per_job}" >&2
    echo "  --persona-range X-Y     Assignment only: submit this persona subset" >&2
    echo "  --task-range X-Y        Spec only: submit these 1-based task positions" >&2
    echo "  --input-json PATH       Clip manifest. Default: ${input_json}" >&2
    echo "  --persona-dir PATH      Assignment persona_XXXX.txt files. Default: ${persona_dir}" >&2
    echo "  --persona-output-dir P  Default: derived from backend, mode and source" >&2
    echo "  --prompt-config PATH    Default: the selected mode's prompt config" >&2
    echo "  --audio-mix-dir PATH    Shared stacked-audio cache" >&2
    echo "  --submit                Queue jobs. Default: dry run" >&2
    echo "  --no-skip-complete      Include packs whose outputs already exist" >&2
    echo "  --overwrite             Recompute existing persona outputs" >&2
    echo "  --max-jobs N            Stop after this many non-skipped jobs" >&2
    echo "  --sleep SECONDS         Wait between sbatch calls" >&2
    echo "  --job-script PATH       Internal worker. Default: ${job_script}" >&2
    echo "  -- EXTRA...             Additional options forwarded to every worker job" >&2
    echo "" >&2
    echo "The assignment/spec owns which clips each persona answers. Create or edit" >&2
    echo "that source to change the clip set; this launcher only packs its complete runs." >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)            backend="${2:?Missing value for --backend}"; shift 2 ;;
        --mode)               mode="${2:?Missing value for --mode}"; shift 2 ;;
        --assignment-json)    assignment_json="${2:?Missing value for --assignment-json}"; shift 2 ;;
        --task-spec)          task_spec="${2:?Missing value for --task-spec}"; shift 2 ;;
        --persona-range)      persona_range="${2:?Missing value for --persona-range}"; shift 2 ;;
        --task-range)         task_range="${2:?Missing value for --task-range}"; shift 2 ;;
        --answers-per-job)    answers_per_job="${2:?Missing value for --answers-per-job}"; shift 2 ;;
        --input-json)         input_json="${2:?Missing value for --input-json}"; shift 2 ;;
        --persona-dir)        persona_dir="${2:?Missing value for --persona-dir}"; shift 2 ;;
        --persona-output-dir) persona_output_dir="${2:?Missing value for --persona-output-dir}"; shift 2 ;;
        --prompt-config)      prompt_config="${2:?Missing value for --prompt-config}"; shift 2 ;;
        --audio-mix-dir)      audio_mix_dir="${2:?Missing value for --audio-mix-dir}"; shift 2 ;;
        --job-script)         job_script="${2:?Missing value for --job-script}"; shift 2 ;;
        --max-jobs)           max_jobs="${2:?Missing value for --max-jobs}"; shift 2 ;;
        --sleep)              sleep_seconds="${2:?Missing value for --sleep}"; shift 2 ;;
        --submit)             submit=1; shift ;;
        --no-skip-complete)   skip_complete=0; shift ;;
        --overwrite)          skip_complete=0; extra_args+=(--overwrite); shift ;;
        --clips|--clips-per-job|--index-range)
            echo "[ERROR] $1 cannot be applied on top of a persona source." >&2
            echo "        The assignment/spec already says which clips each persona answers." >&2
            echo "        Splitting one persona across jobs would create competing writes to" >&2
            echo "        the same output JSON; use --answers-per-job to size whole runs." >&2
            exit 1
            ;;
        --personas-per-job|--tasks-per-job|--runs-per-job)
            echo "[ERROR] $1 was replaced by --answers-per-job." >&2
            echo "        Job cost is the number of persona x clip answers, not just the" >&2
            echo "        number of personas. For six personas with 200 clips each, use 1200." >&2
            exit 1
            ;;
        --)                   shift; extra_args+=("$@"); break ;;
        -h|--help)            usage; exit 0 ;;
        *)                    usage; echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "[ERROR] persona.sh is a login-node launcher; run it with bash, not sbatch." >&2
    exit 1
fi
if [[ -z "${backend}" || -z "${mode}" ]]; then
    usage
    echo "[ERROR] --backend and --mode are both required." >&2
    exit 1
fi
if ! intention_backend_is_known "${backend}"; then
    echo "[ERROR] Unknown backend: ${backend}" >&2
    echo "        Known: $(intention_known_backends | tr '\n' ' ')" >&2
    exit 1
fi
if [[ -n "${assignment_json}" && -n "${task_spec}" ]]; then
    echo "[ERROR] Choose one persona source: --assignment-json or --task-spec." >&2
    exit 1
fi
if [[ -z "${assignment_json}" && -z "${task_spec}" ]]; then
    echo "[ERROR] One persona source is required: --assignment-json or --task-spec." >&2
    exit 1
fi
if [[ -n "${assignment_json}" && -n "${task_range}" ]]; then
    echo "[ERROR] --task-range belongs to --task-spec; use --persona-range here." >&2
    exit 1
fi
if [[ -n "${task_spec}" && -n "${persona_range}" ]]; then
    echo "[ERROR] --persona-range belongs to --assignment-json; use --task-range here." >&2
    exit 1
fi
case "${answers_per_job}" in
    ''|*[!0-9]*|0) echo "[ERROR] --answers-per-job must be a positive integer" >&2; exit 1 ;;
esac
case "${sleep_seconds}" in
    ''|*[!0-9]*) echo "[ERROR] --sleep must be a non-negative integer" >&2; exit 1 ;;
esac
if [[ -n "${max_jobs}" ]]; then
    case "${max_jobs}" in
        *[!0-9]*|0) echo "[ERROR] --max-jobs must be a positive integer" >&2; exit 1 ;;
    esac
fi
if [[ ! -f "${job_script}" ]]; then
    echo "[ERROR] Worker job script not found: ${job_script}" >&2
    exit 1
fi
if [[ ! -f "${input_json}" ]]; then
    echo "[ERROR] Input JSON not found: ${input_json}" >&2
    exit 1
fi

# Read and pack the source through the same schema modules inference uses.
source_kind=""
source_path=""
source_mode=""
planner_args=(
    python3 -m intention_inference.runs.planning
    --answers-per-job "${answers_per_job}"
)

if [[ -n "${assignment_json}" ]]; then
    source_kind="assignment"
    source_path="${assignment_json}"
    if [[ ! -f "${assignment_json}" ]]; then
        echo "[ERROR] Assignment JSON not found: ${assignment_json}" >&2
        exit 1
    fi
    planner_args+=(--assignment-json "${assignment_json}")
    [[ -n "${persona_range}" ]] && planner_args+=(--persona-range "${persona_range}")
else
    source_kind="spec"
    source_path="${task_spec}"
    if [[ ! -f "${task_spec}" ]]; then
        echo "[ERROR] Task spec not found: ${task_spec}" >&2
        exit 1
    fi
    planner_args+=(--task-spec "${task_spec}")
    [[ -n "${task_range}" ]] && planner_args+=(--task-range "${task_range}")
fi

plan_text="$(PYTHONPATH="${SRC_DIR}" "${planner_args[@]}")" || {
    echo "[ERROR] Could not plan persona source: ${source_path}" >&2
    exit 1
}

total_runs=0
total_answers=0
packed_jobs=0
chunk_selector_starts=()
chunk_selector_ends=()
chunk_answers=()
chunk_output_stems=()
while IFS=$'\t' read -r row_kind field1 field2 field3 field4; do
    case "${row_kind}" in
        META)
            source_mode="${field1}"
            [[ "${source_mode}" == "-" ]] && source_mode=""
            total_runs="${field2}"
            total_answers="${field3}"
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

if (( ${#chunk_selector_starts[@]} == 0 )); then
    echo "[ERROR] Persona source selected no runs: ${source_path}" >&2
    exit 1
fi
if [[ -n "${source_mode}" && "${source_mode}" != "${mode}" ]]; then
    echo "[ERROR] --mode ${mode} does not match assignment mode ${source_mode}." >&2
    echo "        Rebuild the assignment for ${mode}, or submit with --mode ${source_mode}." >&2
    exit 1
fi
if [[ "${source_kind}" == "assignment" ]]; then
    if [[ ! -d "${persona_dir}" ]]; then
        echo "[ERROR] Persona directory not found: ${persona_dir}" >&2
        exit 1
    fi
    for chunk_stems in "${chunk_output_stems[@]}"; do
        IFS=',' read -r -a output_stems <<< "${chunk_stems}"
        for output_stem in "${output_stems[@]}"; do
            if [[ ! -f "${persona_dir}/${output_stem}.txt" ]]; then
                echo "[ERROR] Persona prompt not found: ${persona_dir}/${output_stem}.txt" >&2
                exit 1
            fi
        done
    done
fi

if [[ -z "${persona_output_dir}" ]]; then
    if [[ "${source_kind}" == "assignment" ]]; then
        persona_output_dir="$(intention_persona_output_dir \
            "${INTENTION_OUTPUT_DIR}" "${backend}" "${mode}")"
    else
        spec_stem="$(basename "${task_spec}")"
        persona_output_dir="$(intention_spec_output_dir \
            "${INTENTION_OUTPUT_DIR}" "${backend}" "${mode}" "${spec_stem%.json}")"
    fi
fi

echo "[INFO] launcher           = ${BASH_SOURCE[0]}"
echo "[INFO] worker             = ${job_script}"
echo "[INFO] backend            = ${backend}"
echo "[INFO] mode               = ${mode}"
echo "[INFO] persona source     = ${source_kind}: ${source_path}"
echo "[INFO] selected runs      = ${total_runs}"
echo "[INFO] total answers      = ${total_answers}"
echo "[INFO] answers per job    = ${answers_per_job} (whole personas are never split)"
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
    answer_count="${chunk_answers[chunk]}"

    complete=1
    IFS=',' read -r -a output_stems <<< "${chunk_output_stems[chunk]}"
    for output_stem in "${output_stems[@]}"; do
        if [[ ! -f "${persona_output_dir}/${output_stem}.json" ]]; then
            complete=0
            break
        fi
    done
    if [[ "${skip_complete}" == "1" && "${complete}" == "1" ]]; then
        echo "[SKIP] ${source_kind} ${selector_start}-${selector_end}: all outputs exist"
        skipped=$(( skipped + 1 ))
        continue
    fi
    if [[ -n "${max_jobs}" && "${planned}" -ge "${max_jobs}" ]]; then
        reached_cap=1
        break
    fi
    planned=$(( planned + 1 ))

    source_args=()
    if [[ "${source_kind}" == "assignment" ]]; then
        source_args=(
            --persona "${selector_start}-${selector_end}"
            --persona-dir "${persona_dir}"
            --assignment-json "${assignment_json}"
        )
        job_prefix="int_pa"
    else
        source_args=(--task-spec "${task_spec}" --task-range "${selector_start}-${selector_end}")
        job_prefix="int_ps"
    fi

    echo "[PLAN] ${source_kind} ${selector_start}-${selector_end}: ${answer_count} answer(s)"
    if (( answer_count > answers_per_job )); then
        echo "[WARN] One persona has ${answer_count} answers, above the ${answers_per_job} budget;" >&2
        echo "       it remains one job to keep its output atomic." >&2
    fi

    sbatch_cmd=(
        sbatch
        --export=ALL,INTENTION_PERSONA_JOB=1
        --job-name="${job_prefix}_${backend}_${selector_start}-${selector_end}"
        "${job_script}"
        --backend "${backend}"
        --mode "${mode}"
        --input-json "${input_json}"
        "${source_args[@]}"
        --persona-output-dir "${persona_output_dir}"
    )
    [[ -n "${prompt_config}" ]] && sbatch_cmd+=(--prompt-config "${prompt_config}")
    [[ -n "${audio_mix_dir}" ]] && sbatch_cmd+=(--audio-mix-dir "${audio_mix_dir}")
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
