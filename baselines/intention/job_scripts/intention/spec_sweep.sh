#!/bin/bash
# Submit an externally authored persona x clip task spec, tiled by task range.
#
# The sibling of sweep.sh, for the runs the balanced design does not cover.
# sweep.sh submits the design built by build_assignment.py: personas named
# persona_XXXX.txt, one prompt config for the whole sweep, and clips addressed by
# their position in the manifest. Nothing about it can be chosen per persona.
#
# A task spec is the other end. Someone else decides which personas run, what
# each is asked, and which clips each one sees, and writes it down in one JSON.
# Two shapes are accepted.
#
# THE PERSONA FORM -- a bare list, one entry per annotator, and what the
# annotation pipeline emits:
#
#   [
#     {
#       "ID": "5641193817bdbe00122a0f23",
#       "persona_prompt": "You are a 34-year-old nurse ...",
#       "videos": ["3", "12", "44"]
#     }
#   ]
#
#   ID             the annotator's prolific id, and the output name: results land
#                  in <output dir>/persona_5641193817bdbe00122a0f23.json.
#   persona_prompt the persona text itself, inline, folded in ahead of the task
#                  prompt exactly as a persona_XXXX.txt is in the balanced runs.
#   videos         manifest record ids -- ids, not positions, so the spec
#                  survives a manifest that was reordered or appended to.
#
# THE FULL FORM, with relative paths resolved against the spec's own directory:
#
#   {"tasks": [{"task_id": "...", "persona_prompt": "personas/a.txt",
#               "prompt_config": "prompts/p.json", "clips": ["id1", "id2"]}]}
#
# The spec is parsed HERE, on the login node, before anything is queued: a
# malformed spec then costs a second instead of N jobs that die after loading
# weights. That works because intention_inference/runs/spec.py is standard
# library only, so it imports without a container.
#
# DRY RUN BY DEFAULT: add --submit to queue.
#
# EXAMPLES
#
#   bash job_scripts/intention/spec_sweep.sh --backend gemma --mode pa
#   bash job_scripts/intention/spec_sweep.sh --backend gemma --mode pa \
#       --tasks-per-job 25 --submit

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PACKAGE_ROOT}/job_scripts/lib/intention_paths.sh"
source "${PACKAGE_ROOT}/job_scripts/lib/model_backends.sh"

SRC_DIR="${PACKAGE_ROOT}/src"

job_script="${SCRIPT_DIR}/cosilab_daic.sh"
backend=""
mode=""
task_spec="${INTENTION_TASK_SPEC_JSON}"
input_json="${INTENTION_INPUT_JSON}"
tasks_per_job=""
persona_output_dir=""
prompt_config=""
audio_mix_dir=""
submit=0
skip_complete=1
max_jobs=""
sleep_seconds="0"
extra_args=()

usage() {
    echo "Usage: bash $0 --backend <name> --mode <name> [options] [-- <extra job args>]" >&2
    echo "Options:" >&2
    echo "  --backend NAME          Required. Known: $(intention_known_backends | tr '\n' ' ')" >&2
    echo "  --mode NAME             Required." >&2
    echo "  --task-spec PATH        Default: ${task_spec}" >&2
    echo "  --input-json PATH       Default: ${input_json}" >&2
    echo "  --tasks-per-job N       Tile the spec into chunks of N tasks." >&2
    echo "                          Default: one job for the whole spec" >&2
    echo "  --persona-output-dir P  Default: derived from the backend, mode and spec name" >&2
    echo "  --prompt-config PATH    Default: the one the mode ships with" >&2
    echo "  --audio-mix-dir PATH    Share one stacked-audio cache across backends" >&2
    echo "  --submit                Actually queue the jobs. Default: print them" >&2
    echo "  --no-skip-complete      Resubmit even when outputs exist" >&2
    echo "  --max-jobs N            Stop after submitting N jobs this pass" >&2
    echo "  --sleep SECONDS         Wait between submissions" >&2
    echo "  --job-script PATH       Default: ${job_script}" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)            backend="${2:?Missing value for --backend}"; shift 2 ;;
        --mode)               mode="${2:?Missing value for --mode}"; shift 2 ;;
        --task-spec)          task_spec="${2:?Missing value for --task-spec}"; shift 2 ;;
        --input-json)         input_json="${2:?Missing value for --input-json}"; shift 2 ;;
        --tasks-per-job)      tasks_per_job="${2:?Missing value for --tasks-per-job}"; shift 2 ;;
        --persona-output-dir) persona_output_dir="${2:?Missing value for --persona-output-dir}"; shift 2 ;;
        --prompt-config)      prompt_config="${2:?Missing value for --prompt-config}"; shift 2 ;;
        --audio-mix-dir)      audio_mix_dir="${2:?Missing value for --audio-mix-dir}"; shift 2 ;;
        --job-script)         job_script="${2:?Missing value for --job-script}"; shift 2 ;;
        --max-jobs)           max_jobs="${2:?Missing value for --max-jobs}"; shift 2 ;;
        --sleep)              sleep_seconds="${2:?Missing value for --sleep}"; shift 2 ;;
        --submit)             submit=1; shift ;;
        --no-skip-complete)   skip_complete=0; shift ;;
        --overwrite)          skip_complete=0; extra_args+=(--overwrite); shift ;;
        --)                   shift; extra_args+=("$@"); break ;;
        -h|--help)            usage; exit 0 ;;
        *)                    usage; echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
    esac
done

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
for required_file in "${job_script}" "${task_spec}" "${input_json}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "[ERROR] Not found: ${required_file}" >&2
        exit 1
    fi
done

# The login-node preflight. Parses the spec with the same code the job will use,
# so a malformed one is refused here rather than after N jobs have taken places
# in the queue. Needs no container: runs/spec.py is standard library only, and
# keeping it that way is what makes this possible.
echo "[INFO] Validating the spec with runs.spec.load_task_spec ..."
task_count="$(PYTHONPATH="${SRC_DIR}" python3 -c '
import sys
from pathlib import Path
from intention_inference.runs.spec import load_task_spec
spec = load_task_spec(Path(sys.argv[1]))
print(len(spec["tasks"]))
' "${task_spec}")" || {
    echo "[ERROR] The task spec did not parse: ${task_spec}" >&2
    exit 1
}
if (( task_count == 0 )); then
    echo "[ERROR] The task spec defines no tasks: ${task_spec}" >&2
    exit 1
fi

if [[ -n "${tasks_per_job}" ]]; then
    case "${tasks_per_job}" in
        ''|*[!0-9]*|0) echo "[ERROR] --tasks-per-job must be a positive integer" >&2; exit 1 ;;
    esac
else
    tasks_per_job="${task_count}"
fi

spec_stem="$(basename "${task_spec}")"
spec_stem="${spec_stem%.json}"
if [[ -z "${persona_output_dir}" ]]; then
    persona_output_dir="$(intention_spec_output_dir \
        "${INTENTION_OUTPUT_DIR}" "${backend}" "${mode}" "${spec_stem}")"
fi

# Task ids, in file order, so a chunk can be checked for completeness without
# guessing what its files are called. One call, not one per chunk.
mapfile -t task_ids < <(PYTHONPATH="${SRC_DIR}" python3 -c '
import sys
from pathlib import Path
from intention_inference.runs.spec import load_task_spec
for task in load_task_spec(Path(sys.argv[1]))["tasks"]:
    print(task["task_id"])
' "${task_spec}")

chunk_complete() {
    local first="$1" last="$2" n
    for (( n = first; n <= last; n++ )); do
        [[ -f "${persona_output_dir}/${task_ids[n - 1]}.json" ]] || return 1
    done
    return 0
}

echo "[INFO] job_script         = ${job_script}"
echo "[INFO] backend            = ${backend}"
echo "[INFO] mode               = ${mode}"
echo "[INFO] task_spec          = ${task_spec} (${task_count} task(s))"
echo "[INFO] input_json         = ${input_json}"
echo "[INFO] tasks per job      = ${tasks_per_job}"
echo "[INFO] persona_output_dir = ${persona_output_dir}"
echo "[INFO] prompt_config      = ${prompt_config:-<the mode default>}"
echo "[INFO] audio_mix_dir      = ${audio_mix_dir:-<derived>}"
echo "[INFO] skip_complete      = ${skip_complete}"
echo "[INFO] submit             = ${submit}"
echo

planned=0
skipped=0
submitted=0
reached_cap=0
# 1-based and inclusive, because that is how --task-range is written.
for (( chunk_start = 1; chunk_start <= task_count; chunk_start += tasks_per_job )); do
    chunk_end=$(( chunk_start + tasks_per_job - 1 ))
    (( chunk_end > task_count )) && chunk_end="${task_count}"

    if [[ "${skip_complete}" == "1" ]] && chunk_complete "${chunk_start}" "${chunk_end}"; then
        echo "[SKIP] tasks ${chunk_start}-${chunk_end}: all outputs exist"
        skipped=$(( skipped + 1 ))
        continue
    fi
    if [[ -n "${max_jobs}" ]] && (( submitted >= max_jobs )); then
        reached_cap=1
        break
    fi
    planned=$(( planned + 1 ))

    sbatch_cmd=(
        sbatch
        --job-name="int_s${backend}_${chunk_start}-${chunk_end}"
        "${job_script}"
        --backend "${backend}"
        --mode "${mode}"
        --task-spec "${task_spec}"
        --task-range "${chunk_start}-${chunk_end}"
        --input-json "${input_json}"
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
echo "[INFO] chunks planned   = ${planned}"
echo "[INFO] chunks skipped   = ${skipped} (already complete)"
if [[ "${submit}" == "1" ]]; then
    echo "[INFO] jobs submitted   = ${submitted}"
else
    echo "[INFO] jobs submitted   = 0 (dry run; add --submit)"
fi
if (( reached_cap == 1 )); then
    echo "[WARN] Stopped at the --max-jobs cap of ${max_jobs}; rerun to continue where this left off."
fi
