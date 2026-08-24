# Which model backends the intention job can run, and the one thing each needs
# from the cluster that Python cannot work out: its container image.
#
# Sourced by the job body (lib/intention_job.sh), so a backend name is validated
# at submit time -- a typo fails in a second instead of after hours in the queue.
#
# The name itself must match a key in models/registry.py BACKEND_MODULES.
# Nothing else about a backend is described here. Which weights run inside the
# image is set in the task's model_config.json under backends.<name>.model_id,
# because the weights are part of what a run *is* and belong next to the frame
# budget and the token cap rather than in a shell file no result records. What a
# backend can do (thinking mode, audio, video) stays in the backend class.
#
# The images are prebuilt artefacts on the share; neither is built from a .def in
# this repo.

# Every backend this cluster can run, one per line.
intention_known_backends() {
    printf '%s\n' gemma qwen7b qwen3omni30b
}

intention_backend_is_known() {
    local candidate="$1" known
    while read -r known; do
        [[ "${candidate}" == "${known}" ]] && return 0
    done < <(intention_known_backends)
    return 1
}

# intention_backend_defaults <backend>
# Sets BACKEND_SIF_PATH, and BACKEND_APPTAINER_ARGS for the rare backend that
# needs a container flag of its own; returns 1 when the name is unknown, so
# callers can print their own error.
intention_backend_defaults() {
    # Reset first: the array is expanded unquoted-but-guarded by the job body, and
    # a leftover value from an earlier call would silently apply to the next
    # backend.
    BACKEND_APPTAINER_ARGS=()
    case "$1" in
        gemma)
            BACKEND_SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/gemma.sif"
            ;;
        qwen7b)
            # Its own image, not the gemma one: Qwen2.5-Omni needs a transformers
            # carrying Qwen2_5OmniForConditionalGeneration plus the audio stack
            # (av, librosa, audioread), and pinning that separately means a
            # transformers bump for one model cannot silently change what the
            # other one produces.
            #
            # This image is also where qwen_omni_utils comes from. It is NOT
            # vendored in this repo -- no model code is -- so if the image lacks
            # it, models/qwen/engine.py says so and QWEN_OMNI_UTILS_PATH is the
            # way to supply it from outside.
            #
            # Before the first run, confirm the image can reach both:
            #
            #   apptainer exec \
            #     /tudelft.net/staff-umbrella/neon/apptainer/qwen2.5-omni-inference.sif \
            #     python -c "import qwen_omni_utils, av, librosa, transformers;
            #                from transformers import Qwen2_5OmniForConditionalGeneration;
            #                print(transformers.__version__)"
            BACKEND_SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/qwen2.5-omni-inference.sif"
            ;;
        qwen3omni30b)
            # Its own image again, not the qwen7b one: Qwen3OmniMoe* landed in
            # transformers well after the 4.52 that image carries, and pinning
            # the two separately is what keeps a transformers bump for one
            # generation from silently changing what the other produces.
            #
            # qwen_omni_utils serves both generations and is likewise NOT
            # vendored in this repo. QWEN_OMNI_UTILS_PATH supplies it from
            # outside if the image lacks it.
            #
            # Before the first run, confirm the image can reach all of it:
            #
            #   apptainer exec \
            #     /tudelft.net/staff-umbrella/neon/apptainer/qwen3-omni-inference.sif \
            #     python -c "import qwen_omni_utils, av, librosa, transformers;
            #                from transformers import Qwen3OmniMoeForConditionalGeneration;
            #                print(transformers.__version__)"
            #
            # 30B in bfloat16 is a much bigger residency than qwen7b, so the
            # first run is also where to check that the default --time and the
            # single GPU in the stub are still enough; both are overridable on
            # the sbatch line without editing the stub.
            BACKEND_SIF_PATH="/tudelft.net/staff-umbrella/neon/apptainer/qwen3-omni.sif"
            ;;
        *)
            return 1
            ;;
    esac
}
