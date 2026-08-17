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

INTENTION_DATA_ROOT="/tudelft.net/staff-umbrella/neon/ingroup_dataset"
INTENTION_PIPELINE_ROOT="/tudelft.net/staff-umbrella/neon/B1_pipeline"

INTENTION_VIDEO_MEDIA_PATH_PREFIX="https://covfee.ewi.tudelft.nl/P8wPkLamHiAMOvb29g9h3AFy8tXACT1e/video_segs"
INTENTION_AUDIO_MEDIA_PATH_PREFIX="https://covfee.ewi.tudelft.nl/P8wPkLamHiAMOvb29g9h3AFy8tXACT1e/audio_segs_normalized"

INTENTION_INPUT_JSON="${INTENTION_PIPELINE_ROOT}/annotation_clips.json"
INTENTION_OUTPUT_DIR="${INTENTION_PIPELINE_ROOT}/model_responses"
INTENTION_PARTICIPANT_IMAGE_ROOT="${INTENTION_PIPELINE_ROOT}/participant_imgs"

# The HF cache is a write target, so it is here rather than inlined: a job that
# cannot write it re-downloads weights it already has.
INTENTION_HF_CACHE="/tudelft.net/staff-umbrella/neon/zonghuan/.cache/huggingface"

# The local media roots hang off the data root, which --data-root can override at
# run time. Call this again after parsing arguments to pick up an override.
intention_set_local_prefixes() {
    local data_root="${1:-${INTENTION_DATA_ROOT}}"
    INTENTION_VIDEO_LOCAL_PATH_PREFIX="${data_root}/for_annotation/annotation_video"
    INTENTION_AUDIO_LOCAL_PATH_PREFIX="${data_root}/for_annotation/annotation_audio"
}
intention_set_local_prefixes

# intention_output_dir <output_dir> <backend>
#
# Results are filed per backend. Two backends run over the same manifest produce
# the same file name, so without this the second silently overwrites the first.
#
# NOTE: this is one level deeper than the old flat tree. Runs made before the
# backend split still sit directly in ${INTENTION_OUTPUT_DIR}; nothing was moved,
# but a notebook globbing model_responses/*.json will not see new runs until its
# pattern is updated.
intention_output_dir() {
    printf '%s\n' "$1/$2"
}
