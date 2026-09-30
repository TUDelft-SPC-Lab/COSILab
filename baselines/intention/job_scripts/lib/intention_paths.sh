# Single source of truth for where the intention baseline's data lives.
#
# Sourced by every script in job_scripts/intention/.
#
# The media prefixes matter most. intention_inference/media.py rewrites each
# covfee URL in the manifest by stripping the media prefix and joining the rest
# onto the local prefix; if a prefix is missing or does not match, the rewrite
# silently does not fire and every clip is reported as video_not_found. So a run
# that skipped everything is usually a prefix that stopped matching, not missing
# data.
#
# Everything here is under /tudelft.net, which is identical in both DAIC
# environments, so nothing in this file is environment-specific.

# Where the clip media lives. The two annotation_* folders below hang off it.
INTENTION_DATA_ROOT="/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed/benchmark_tasks/benchmark_1"
INTENTION_PIPELINE_ROOT="/tudelft.net/staff-umbrella/neon/cosilab_project/B1_pipeline"

INTENTION_VIDEO_MEDIA_PATH_PREFIX="https://covfee.ewi.tudelft.nl/P8wPkLamHiAMOvb29g9h3AFy8tXACT1e/video_segs"
INTENTION_AUDIO_MEDIA_PATH_PREFIX="https://covfee.ewi.tudelft.nl/P8wPkLamHiAMOvb29g9h3AFy8tXACT1e/audio_segs_normalized"

INTENTION_INPUT_JSON="${INTENTION_PIPELINE_ROOT}/annotation_clips.json"
INTENTION_OUTPUT_DIR="${INTENTION_PIPELINE_ROOT}/model_responses"
INTENTION_PARTICIPANT_IMAGE_ROOT="${INTENTION_PIPELINE_ROOT}/participant_imgs"

# The persona experiment. persona_NNNN.txt files, the balanced design built from
# them, and the externally authored spec the annotation pipeline emits.
#
# The assignment is built by intention/build_assignment_daic.sh and read by
# persona.sh; nothing in this repo writes the task spec -- it arrives from the
# annotation pipeline and is read by persona.sh too.
INTENTION_PERSONA_DIR="${INTENTION_PIPELINE_ROOT}/persona_prompts"
INTENTION_ASSIGNMENT_JSON="${INTENTION_PIPELINE_ROOT}/ingroup_assignment.json"
INTENTION_TASK_SPEC_JSON="${INTENTION_PIPELINE_ROOT}/personas_prolific_ids.json"

# The prompt structure-and-phrasing experiment. An instruction prompt is one
# whole user prompt as plain text, and it replaces whatever the prompt config
# would have asked. Every run has one: prompt_original_revised.txt is the
# baseline the structural variants in this directory are measured against, and
# --instruction-prompt swaps in another.
#
# The variants themselves are prompt_structural_variant_<N>.txt in the same
# directory, and are named on the command line by path like any other file.
INTENTION_INSTRUCTION_PROMPT_DIR="${INTENTION_PIPELINE_ROOT}/prompt_structure_and_phrasing_variants"
INTENTION_DEFAULT_INSTRUCTION_PROMPT="${INTENTION_INSTRUCTION_PROMPT_DIR}/prompt_original_revised.txt"

# Where the prompt-variation sweep's own artefacts live: sweep_seed<N>.json, the
# sampling model config derived beside it, and the shared audio-mix cache.
#
# Model-independent on purpose. Every model at one seed reads the same spec from
# here, which is what makes "the same clips and personas for every model"
# something enforced rather than hoped for -- see build_sweep.py, which rebuilds
# the spec on each submission and refuses when the rebuild would differ.
INTENTION_SWEEP_DIR="${INTENTION_PIPELINE_ROOT}/sweeps"

# The HF cache is a write target, so it is here rather than inlined: a job that
# cannot write it re-downloads weights it already has.
INTENTION_HF_CACHE="/tudelft.net/staff-umbrella/neon/zonghuan/.cache/huggingface"

# The local media roots hang off the data root, which --data-root can override at
# run time. Call this again after parsing arguments to pick up an override.
#
# Only the prefix is swapped: everything the manifest URL carries after the media
# prefix -- the seg_* folder and the filename -- is joined on verbatim. That is
# why the two trees may name their segment folders differently (video uses
# seg_134500_134530, audio uses seg_13-45-00_13-45-30) without anything here
# knowing about it: whichever form the manifest holds is the form that is looked
# for on disk.
intention_set_local_prefixes() {
    local data_root="${1:-${INTENTION_DATA_ROOT}}"
    INTENTION_VIDEO_LOCAL_PATH_PREFIX="${data_root}/annotation_video"
    INTENTION_AUDIO_LOCAL_PATH_PREFIX="${data_root}/annotation_audio"
}
intention_set_local_prefixes

# intention_output_dir <output_dir> <backend> <mode>
#
# Results are filed per backend and then per mode. Two runs differing only in
# which model answered, or only in what was asked, produce the same file name, so
# without this the second silently overwrites the first.
#
# NOTE: this is two levels deeper than the old flat tree. Runs made before the
# backend split still sit directly in ${INTENTION_OUTPUT_DIR}, and runs made
# between the backend split and the mode split sit one level in; nothing was
# moved, but a notebook globbing model_responses/*.json will not see new runs
# until its pattern is updated.
intention_output_dir() {
    printf '%s\n' "$1/$2/$3"
}

# intention_persona_output_dir <output_dir> <backend> <mode>
#
# Where a balanced-design sweep writes its persona_XXXX.json. Its own level under
# the backend/mode path, so a sweep of several hundred persona files never mixes
# with the handful of plain runs beside it.
intention_persona_output_dir() {
    printf '%s\n' "$(intention_output_dir "$1" "$2" "$3")/personas"
}

# intention_spec_output_dir <output_dir> <backend> <mode> <spec stem>
#
# Where a task-spec run writes its <task_id>.json. A separate tree from the
# personas above, and keyed by the spec: spec output is named by task id, and the
# persona form of a spec names its tasks persona_<prolific id>, which landing
# among the design's own persona_NNNN.json would be indistinguishable from it.
intention_spec_output_dir() {
    printf '%s\n' "$(intention_output_dir "$1" "$2" "$3")/spec_runs/$4"
}

# intention_sweep_output_dir <output_dir> <backend> <mode> <seed>
#
# Where the prompt-variation sweep writes its results. The sibling of the two
# above, and its own level for the same reason: a sweep is 30 persona files per
# model spread over six instruction-prompt subdirectories, and mixing that into
# personas/ or spec_runs/ would make a sweep indistinguishable from the runs
# beside it.
#
# Keyed by seed rather than by the spec's file name, because the seed is what
# the design *is*: two sweeps differing in seed answer different clips, and
# filing them together would put two designs' answers in one tree.
#
# The per-prompt leaf is still intention_instruction_output_dir's, applied by the
# caller on top of this.
intention_sweep_output_dir() {
    printf '%s\n' "$(intention_output_dir "$1" "$2" "$3")/sweep_seed$4"
}

# intention_instruction_output_dir <dir> <instruction prompt file, or empty>
#
# The instruction prompt changes what was asked, so it changes where results are
# filed -- the same reason --mode is a path component. Without this, a sweep
# under variant 1 writes persona_<id>.json straight over the one the baseline
# produced, and the resume check then *skips* the run it just overwrote, so the
# damage is silent.
#
# Empty second argument means --no-instruction-prompt, and the directory is
# returned as-is: that run asked the prompt config's own question, which is what
# the mode component already says.
intention_instruction_output_dir() {
    local dir="$1" instruction_file="$2" stem
    if [[ -z "${instruction_file}" ]]; then
        printf '%s\n' "${dir}"
        return
    fi
    stem="$(basename "${instruction_file}")"
    printf '%s\n' "${dir}/${stem%.txt}"
}
