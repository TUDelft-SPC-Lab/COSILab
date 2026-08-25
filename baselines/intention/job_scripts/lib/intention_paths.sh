# Single source of truth for where the intention baseline's data lives.
#
# Sourced by lib/intention_job.sh (via the env stub in job_scripts/intention/).
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
# sweep.sh; nothing in this repo writes the task spec -- it arrives from the
# annotation pipeline and is read by spec_sweep.sh.
INTENTION_PERSONA_DIR="${INTENTION_PIPELINE_ROOT}/persona_prompts"
INTENTION_ASSIGNMENT_JSON="${INTENTION_PIPELINE_ROOT}/ingroup_assignment.json"
INTENTION_TASK_SPEC_JSON="${INTENTION_PIPELINE_ROOT}/personas_prolific_ids.json"

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
