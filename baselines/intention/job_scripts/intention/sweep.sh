#!/bin/bash
# Tile a balanced persona x clip design into one sbatch job per persona chunk.
#
# This is the PERSONA half of the intention runs: every job it submits prepends a
# persona_XXXX.txt observer to the system prompt. For the persona-free prompt use
# nopersona.sh in this directory; for an externally authored spec use
# spec_sweep.sh.
#
# NOT a SLURM job itself: run it on the login node and it submits
# intention/cosilab_daic.sh once per persona chunk.
#
# The only supported design is the balanced one. The full persona x clip cross
# product -- every persona answering every clip -- is not available here or in
# cli.py, which refuses personas without an assignment.
#
# The design numbers are NOT set here. They are fixed in the design block at the
# top of lib/build_assignment_job.sh and baked into the assignment JSON when it
# is built. Build the assignment first:
#
#   sbatch job_scripts/intention/build_assignment_daic.sh --mode pa
#
# That job supplies the media prefixes from lib/intention_paths.sh -- the same
# file this script and the inference body read -- so the assignment is validated
# against exactly the paths inference will use.
#
# THE MODE MUST MATCH THE ASSIGNMENT. A design is only valid for the mode it was
# validated against, because what makes a clip usable depends on what the mode
# needs to ground the question. This script reads the mode back out of the
# assignment's __meta__ and refuses a --mode that disagrees.
#
# DRY RUN BY DEFAULT: it prints the sbatch lines it would run. Add --submit to
# queue them.
#
# EXAMPLES
#
#   # See what a full sweep would submit, without submitting anything.
#   bash job_scripts/intention/sweep.sh --backend qwen7b --mode pa
#
#   # Actually submit it, 20 personas per job, at most 50 jobs this pass.
#   bash job_scripts/intention/sweep.sh --backend qwen7b --mode pa \
#       --personas-per-job 20 --max-jobs 50 --submit
#
#   # Resume: personas whose output already exists are skipped, so rerunning the
#   # same command after a timeout continues where it stopped.
#   bash job_scripts/intention/sweep.sh --backend qwen7b --mode pa --submit

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${PACKAGE_ROOT}/job_scripts/lib/intention_paths.sh"
source "${PACKAGE_ROOT}/job_scripts/lib/model_backends.sh"

job_script="${SCRIPT_DIR}/cosilab_daic.sh"
backend=""
mode=""
assignment_json="${INTENTION_ASSIGNMENT_JSON}"
persona_range=""
personas_per_job="20"
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
    echo "  --mode NAME             Required. Must match the assignment's __meta__.mode" >&2
    echo "  --assignment-json PATH  Default: ${assignment_json}" >&2
    echo "  --persona-range X-Y     Subset of the design's personas. Default: all of them" >&2
    echo "  --personas-per-job N    Personas per sbatch job. Default: ${personas_per_job}" >&2
    echo "  --persona-output-dir P  Default: derived from the backend and mode" >&2
    echo "  --prompt-config PATH    Default: the one the mode ships with" >&2
    echo "  --audio-mix-dir PATH    Share one stacked-audio cache across backends" >&2
    echo "  --submit                Actually queue the jobs. Default: print them" >&2
    echo "  --no-skip-complete      Resubmit chunks whose outputs already exist" >&2
    echo "  --max-jobs N            Stop after submitting N jobs this pass" >&2
    echo "  --sleep SECONDS         Wait between submissions" >&2
    echo "  --job-script PATH       Default: ${job_script}" >&2
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)            backend="${2:?Missing value for --backend}"; shift 2 ;;
        --mode)               mode="${2:?Missing value for --mode}"; shift 2 ;;
        --assignment-json)    assignment_json="${2:?Missing value for --assignment-json}"; shift 2 ;;
        --persona-range)      persona_range="${2:?Missing value for --persona-range}"; shift 2 ;;
        --personas-per-job)   personas_per_job="${2:?Missing value for --personas-per-job}"; shift 2 ;;
        --persona-output-dir) persona_output_dir="${2:?Missing value for --persona-output-dir}"; shift 2 ;;
        --prompt-config)      prompt_config="${2:?Missing value for --prompt-config}"; shift 2 ;;
        --audio-mix-dir)      audio_mix_dir="${2:?Missing value for --audio-mix-dir}"; shift 2 ;;
        --job-script)         job_script="${2:?Missing value for --job-script}"; shift 2 ;;
        --max-jobs)           max_jobs="${2:?Missing value for --max-jobs}"; shift 2 ;;
        --sleep)              sleep_seconds="${2:?Missing value for --sleep}"; shift 2 ;;
        --submit)             submit=1; shift ;;
        --no-skip-complete)   skip_complete=0; shift ;;
        --overwrite)
            # Asking for finished outputs to be redone and then skipping the
            # chunks that have them would do nothing at all.
            skip_complete=0
            extra_args+=(--overwrite)
            shift
            ;;
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
if [[ ! -f "${job_script}" ]]; then
    echo "[ERROR] Job script not found: ${job_script}" >&2
    exit 1
fi
if [[ ! -f "${assignment_json}" ]]; then
    echo "[ERROR] Assignment JSON not found: ${assignment_json}" >&2
    echo "        Build it first:" >&2
    echo "          sbatch job_scripts/intention/build_assignment_daic.sh --mode ${mode}" >&2
    exit 1
fi

# Read a value out of the assignment's __meta__. Written by build_assignment.py
# with json.dumps(indent=2), so a list lands on its own lines; scanning forward
# from the key handles both that and same-line values. Echoes nothing when the
# key is absent, which every caller treats as "unknown".
assignment_meta_ints() {
    local key="$1" count="${2:-1}"
    awk -v key="\"$1\"" -v want="${count}" '
        BEGIN { found = 0; n = 0; out = "" }
        {
            line = $0
            if (!found) {
                pos = index(line, key)
                if (pos == 0) next
                found = 1
                line = substr(line, pos + length(key))
            }
            while (match(line, /[0-9]+/)) {
                out = (n == 0) ? substr(line, RSTART, RLENGTH) : out "-" substr(line, RSTART, RLENGTH)
                line = substr(line, RSTART + RLENGTH)
                if (++n == want) { print out; exit }
            }
        }
    ' "${assignment_json}"
}
assignment_meta_string() {
    awk -v key="\"$1\"" '
        { pos = index($0, key)
          if (pos == 0) next
          rest = substr($0, pos + length(key))
          if (match(rest, /"[^"]*"/)) {
              print substr(rest, RSTART + 1, RLENGTH - 2); exit
          } }
    ' "${assignment_json}"
}

# The design is only valid for the mode it was validated against, so a mismatch
# is refused rather than run: the clips this sweep would ask for were chosen
# under different rules about what makes a clip usable.
assignment_mode="$(assignment_meta_string mode)"
if [[ -n "${assignment_mode}" && "${assignment_mode}" != "${mode}" ]]; then
    echo "[ERROR] --mode ${mode} does not match the assignment, which was built for" >&2
    echo "        mode '${assignment_mode}' (${assignment_json})." >&2
    echo "        A design is only valid for the mode it was validated against." >&2
    echo "        Rebuild it with --mode ${mode}, or sweep with --mode ${assignment_mode}." >&2
    exit 1
fi

# The design lives in the assignment: how many clips each persona annotates and
# which clips they are come from the file and cannot be changed here. What is
# left to choose is how much of it to submit now, so --persona-range narrows the
# personas to a subset, defaulting to all of them rather than to a number
# repeated in this script.
if [[ -z "${persona_range}" ]]; then
    persona_range="$(assignment_meta_ints persona_range 2)"
    if [[ -z "${persona_range}" ]]; then
        echo "[ERROR] Could not read persona_range from ${assignment_json}" >&2
        echo "        Rebuild it, or pass --persona-range X-Y." >&2
        exit 1
    fi
fi
if [[ ! "${persona_range}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    echo "[ERROR] Invalid --persona-range: ${persona_range}; expected X-Y" >&2
    exit 1
fi
persona_start="${BASH_REMATCH[1]}"
persona_end="${BASH_REMATCH[2]}"
if (( persona_end < persona_start )); then
    echo "[ERROR] Invalid --persona-range: ${persona_range} (end is before start)" >&2
    exit 1
fi
case "${personas_per_job}" in
    ''|*[!0-9]*|0) echo "[ERROR] --personas-per-job must be a positive integer" >&2; exit 1 ;;
esac

if [[ -z "${persona_output_dir}" ]]; then
    persona_output_dir="$(intention_persona_output_dir \
        "${INTENTION_OUTPUT_DIR}" "${backend}" "${mode}")"
fi

# Every persona in the chunk has to have finished, or the chunk is resubmitted.
# The job itself then skips the individual personas that are already done, so a
# partially finished chunk costs only what is left of it.
chunk_complete() {
    local first="$1" last="$2" n output_file
    for (( n = first; n <= last; n++ )); do
        output_file="${persona_output_dir}/persona_$(printf '%04d' "${n}").json"
        [[ -f "${output_file}" ]] || return 1
    done
    return 0
}

echo "[INFO] job_script         = ${job_script}"
echo "[INFO] backend            = ${backend}"
echo "[INFO] mode               = ${mode} (assignment: ${assignment_mode:-<unrecorded>})"
echo "[INFO] assignment_json    = ${assignment_json}"
echo "[INFO] persona range      = ${persona_start}-${persona_end}"
echo "[INFO] personas per job   = ${personas_per_job}"
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
for (( chunk_start = persona_start; chunk_start <= persona_end; chunk_start += personas_per_job )); do
    chunk_end=$(( chunk_start + personas_per_job - 1 ))
    (( chunk_end > persona_end )) && chunk_end="${persona_end}"

    if [[ "${skip_complete}" == "1" ]] && chunk_complete "${chunk_start}" "${chunk_end}"; then
        echo "[SKIP] personas ${chunk_start}-${chunk_end}: all outputs exist"
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
        --job-name="int_p${backend}_${chunk_start}-${chunk_end}"
        "${job_script}"
        --backend "${backend}"
        --mode "${mode}"
        --persona "${chunk_start}-${chunk_end}"
        --assignment-json "${assignment_json}"
        # Passed explicitly rather than left to the job body's derivation, so the
        # directory the jobs write to is always the one chunk_complete() reads.
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
