#!/bin/bash
# Run the intention prompt WITHOUT any persona, tiled into one job per clip chunk.
#
# The persona-free counterpart to sweep.sh. With no --persona and no --task-spec
# the task builds a single run from the prompt config's own system prompt and
# writes one result JSON; this script is the launcher for that, with chunking and
# resume on top.
#
# --clips X-Y picks the clips as zero-based indices into the manifest, and the
# range is TILED into one job per --clips-per-job clips (300 by default) rather
# than submitted as one long job. So --clips 0-4000 submits 14 jobs: 0-299,
# 300-599, ..., 3900-4000, the last one short. Pure packing -- it changes the job
# count and the per-job wall clock, never which clips run or what they produce,
# so 0-4000 in one pass and 0-1999 then 2000-4000 give the same results.
#
# Without --clips the job covers the whole manifest and there is nothing to tile
# against: the manifest length is known only inside the job. Pass a range to get
# chunking.
#
# Each chunk writes its own <manifest stem>_<start>-<end>.json, so a chunk whose
# file already exists is skipped rather than resubmitted -- rerun the same
# command after a timeout and it continues where it stopped. --no-skip-complete
# turns that off; forwarding --overwrite turns it off automatically, since that
# asks for the finished chunks to be redone.
#
# DRY RUN BY DEFAULT: add --submit to queue.
#
# EXAMPLES
#
#   bash job_scripts/intention/nopersona.sh --backend gemma --mode pa --clips 0-999
#   bash job_scripts/intention/nopersona.sh --backend gemma --mode pa \
#       --clips 0-3999 --clips-per-job 250 --max-jobs 10 --submit

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PACKAGE_ROOT}/job_scripts/lib/intention_paths.sh"
source "${PACKAGE_ROOT}/job_scripts/lib/model_backends.sh"

job_script="${SCRIPT_DIR}/cosilab_daic.sh"
backend=""
mode=""
input_json="${INTENTION_INPUT_JSON}"
output_dir="${INTENTION_OUTPUT_DIR}"
clips=""
clips_per_job="300"
prompt_config=""
submit=0
skip_complete=1
max_jobs=""
sleep_seconds="0"
extra_args=()

usage() {
    echo "Usage: bash $0 --backend <name> --mode <name> [options] [-- <extra job args>]" >&2
    echo "Options:" >&2
    echo "  --backend NAME        Required. Known: $(intention_known_backends | tr '\n' ' ')" >&2
    echo "  --mode NAME           Required." >&2
    echo "  --clips X-Y           Zero-based manifest indices to run, tiled." >&2
    echo "                        Omit for the whole manifest in one job" >&2
    echo "  --clips-per-job N     Clips per sbatch job. Default: ${clips_per_job}" >&2
    echo "  --input-json PATH     Default: ${input_json}" >&2
    echo "  --output-dir PATH     Default: ${output_dir}" >&2
    echo "  --prompt-config PATH  Default: the one the mode ships with" >&2
    echo "  --submit              Actually queue the jobs. Default: print them" >&2
    echo "  --no-skip-complete    Resubmit chunks whose output already exists" >&2
    echo "  --max-jobs N          Stop after submitting N jobs this pass" >&2
    echo "  --sleep SECONDS       Wait between submissions" >&2
    echo "  --job-script PATH     Default: ${job_script}" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)          backend="${2:?Missing value for --backend}"; shift 2 ;;
        --mode)             mode="${2:?Missing value for --mode}"; shift 2 ;;
        --clips)            clips="${2:?Missing value for --clips}"; shift 2 ;;
        --clips-per-job)    clips_per_job="${2:?Missing value for --clips-per-job}"; shift 2 ;;
        --input-json)       input_json="${2:?Missing value for --input-json}"; shift 2 ;;
        --output-dir)       output_dir="${2:?Missing value for --output-dir}"; shift 2 ;;
        --prompt-config)    prompt_config="${2:?Missing value for --prompt-config}"; shift 2 ;;
        --job-script)       job_script="${2:?Missing value for --job-script}"; shift 2 ;;
        --max-jobs)         max_jobs="${2:?Missing value for --max-jobs}"; shift 2 ;;
        --sleep)            sleep_seconds="${2:?Missing value for --sleep}"; shift 2 ;;
        --submit)           submit=1; shift ;;
        --no-skip-complete) skip_complete=0; shift ;;
        --overwrite)        skip_complete=0; extra_args+=(--overwrite); shift ;;
        --)                 shift; extra_args+=("$@"); break ;;
        -h|--help)          usage; exit 0 ;;
        *)                  usage; echo "[ERROR] Unknown option: $1" >&2; exit 1 ;;
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
if [[ ! -f "${job_script}" ]]; then
    echo "[ERROR] Job script not found: ${job_script}" >&2
    exit 1
fi
case "${clips_per_job}" in
    ''|*[!0-9]*|0) echo "[ERROR] --clips-per-job must be a positive integer" >&2; exit 1 ;;
esac

run_output_dir="$(intention_output_dir "${output_dir}" "${backend}" "${mode}")"
manifest_stem="$(basename "${input_json}")"
manifest_stem="${manifest_stem%.json}"

echo "[INFO] job_script       = ${job_script}"
echo "[INFO] backend          = ${backend}"
echo "[INFO] mode             = ${mode}"
echo "[INFO] input_json       = ${input_json}"
echo "[INFO] output_dir       = ${run_output_dir}"
echo "[INFO] clips            = ${clips:-<whole manifest, one job>}"
echo "[INFO] clips per job    = ${clips_per_job}"
echo "[INFO] skip_complete    = ${skip_complete}"
echo "[INFO] submit           = ${submit}"
echo

submit_chunk() {
    local range="$1" job_name="$2"
    sbatch_cmd=(
        sbatch
        --job-name="${job_name}"
        "${job_script}"
        --backend "${backend}"
        --mode "${mode}"
        --input-json "${input_json}"
        --output-dir "${output_dir}"
    )
    [[ -n "${range}" ]] && sbatch_cmd+=(--index-range "${range}")
    [[ -n "${prompt_config}" ]] && sbatch_cmd+=(--prompt-config "${prompt_config}")
    (( ${#extra_args[@]} > 0 )) && sbatch_cmd+=("${extra_args[@]}")

    if [[ "${submit}" == "1" ]]; then
        "${sbatch_cmd[@]}"
        submitted=$(( submitted + 1 ))
        (( sleep_seconds > 0 )) && sleep "${sleep_seconds}"
    else
        printf '%q ' "${sbatch_cmd[@]}"
        printf '\n'
    fi
}

planned=0
skipped=0
submitted=0
reached_cap=0

if [[ -z "${clips}" ]]; then
    # Nothing to tile against: only the job knows how long the manifest is.
    planned=1
    submit_chunk "" "int_n${backend}_all"
else
    if [[ ! "${clips}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
        echo "[ERROR] Invalid --clips: ${clips}; expected X-Y, for example 0-999" >&2
        exit 1
    fi
    clip_start="${BASH_REMATCH[1]}"
    clip_end="${BASH_REMATCH[2]}"
    if (( clip_end < clip_start )); then
        echo "[ERROR] Invalid --clips: ${clips} (end is before start)" >&2
        exit 1
    fi

    for (( chunk_start = clip_start; chunk_start <= clip_end; chunk_start += clips_per_job )); do
        chunk_end=$(( chunk_start + clips_per_job - 1 ))
        (( chunk_end > clip_end )) && chunk_end="${clip_end}"
        # The name the job derives for this range; see intention_job.sh.
        chunk_output="${run_output_dir}/${manifest_stem}_${chunk_start}-${chunk_end}.json"

        if [[ "${skip_complete}" == "1" && -f "${chunk_output}" ]]; then
            echo "[SKIP] clips ${chunk_start}-${chunk_end}: output exists"
            skipped=$(( skipped + 1 ))
            continue
        fi
        if [[ -n "${max_jobs}" ]] && (( submitted >= max_jobs )); then
            reached_cap=1
            break
        fi
        planned=$(( planned + 1 ))
        submit_chunk "${chunk_start}-${chunk_end}" "int_n${backend}_${chunk_start}-${chunk_end}"
    done
fi

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
