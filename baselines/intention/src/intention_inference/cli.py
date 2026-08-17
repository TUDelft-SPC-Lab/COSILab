"""Command line and order of operations for intention inference.

This module owns neither of the two things that vary. Which model answers is
``--backend`` and the backend API in ``models/`` (see
``models.base.BaseMultimodalModel``), with ``model_config.json`` saying which
weights it loads under what decoding parameters. What is asked, and how the
question is grounded, is ``--mode`` and the task modes in ``modes/`` (see
``modes.base.BaseTaskMode``), each of which ships its own prompt config. The two
are independent: any mode runs on any backend.

What is left here is manifest loading, path resolution and the sequence -- which
is chosen so that anything cheap that can fail does so before the weights are
paid for. The work itself is in ``engine.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from models.config import load_model_config, validate_capabilities
from models.registry import available_backends, backend_capabilities, load_model

from . import engine
from .manifest import load_manifest_records
from .modes.registry import available_modes, load_mode
from .prompt import load_prompt_config
from .records import RecordContext

# This task's own weights, decoding parameters and frame policy. There is no
# matching prompt-config constant: that one belongs to the mode, since which
# question is asked is exactly what a mode is.
DEFAULT_MODEL_CONFIG_PATH = Path(__file__).resolve().with_name("model_config.json")
DEFAULT_PARTICIPANT_IMAGE_ROOT = Path(
    "/tudelft.net/staff-umbrella/neon/B1_pipeline/participant_imgs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multimodal ingroup intention inference from a flat JSON manifest."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=available_backends(),
        help=(
            "Model backend to run the task on. Required and not defaulted: it decides "
            "which weights run and where the results are filed, so a run should never "
            "be attributed to a model chosen by omission."
        ),
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=available_modes(),
        help=(
            "Task mode: which variant of the question is asked, and what grounds it. "
            "Required for the same reason as --backend -- it decides what was asked and "
            "where the results are filed, so it should never be settled by omission."
        ),
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="JSON file containing ~1 record per inference item.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path to the output JSON file. Defaults to "
            "ingroup_results/<backend>/<mode>/<input stem>.json beside the working directory."
        ),
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=None,
        help="Prompt config JSON. Defaults to the one the chosen --mode ships with.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL_CONFIG_PATH,
        help=(
            "Model and decoding parameters JSON (weights, token budget, sampling, frame "
            "policy, backend options). Defaults to model_config.json alongside this "
            "package. These are not CLI flags: edit the file and resubmit."
        ),
    )
    parser.add_argument(
        "--media-path-prefix",
        default=None,
        help=(
            "Source prefix to replace in each record's media path, for example "
            "a URL/media prefix stored in the JSON."
        ),
    )
    parser.add_argument(
        "--local-path-prefix",
        type=Path,
        default=None,
        help=(
            "Local filesystem prefix that replaces --media-path-prefix in each "
            "record's media path before loading the media."
        ),
    )
    parser.add_argument(
        "--video-media-path-prefix",
        default=None,
        help="Optional video-specific source prefix override.",
    )
    parser.add_argument(
        "--video-local-path-prefix",
        type=Path,
        default=None,
        help="Optional video-specific local prefix override.",
    )
    parser.add_argument(
        "--audio-media-path-prefix",
        default=None,
        help="Optional audio-specific source prefix override.",
    )
    parser.add_argument(
        "--audio-local-path-prefix",
        type=Path,
        default=None,
        help="Optional audio-specific local prefix override.",
    )
    parser.add_argument(
        "--id-key",
        default="id",
        help="Record key used as the stable result id. Dotted paths are supported.",
    )
    parser.add_argument(
        "--participant-image-root",
        type=Path,
        default=DEFAULT_PARTICIPANT_IMAGE_ROOT,
        help=(
            "Folder containing participant_<n>.png files used to identify the "
            "indicated participant. Used by the participant_image mode."
        ),
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=None,
        help="Optional root used to resolve relative media paths.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional override for the system prompt from the prompt config.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Run video-only inference by omitting separate audio inputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of retained records to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First zero-based manifest index to consider.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Last zero-based manifest index to consider, inclusive.",
    )
    parser.add_argument(
        "--exclude-video-substring",
        action="append",
        default=[],
        help="Skip records whose resolved video path contains this substring. Repeatable.",
    )
    parser.add_argument(
        "--exclude-audio-substring",
        action="append",
        default=[],
        help="Skip records whose resolved audio path contains this substring. Repeatable.",
    )
    return parser.parse_args()


def default_output_path(args: argparse.Namespace, input_json_path: Path) -> Path:
    """Where results land when --output is not given.

    Backend and mode are both directory components, for one reason: two runs that
    differ only in which model answered, or only in what was asked, produce the
    same file name, and without this the second would silently overwrite the
    first. Mode sits inside backend so the existing <backend>/ level keeps the
    meaning it already has.
    """
    stem = input_json_path.stem
    if args.start_index or args.end_index is not None:
        stem = f"{stem}_{args.start_index}-{args.end_index}"
    return Path("ingroup_results") / args.backend / args.mode / f"{stem}.json"


def resolve_optional_path(value: Path | None) -> Path | None:
    return None if value is None else value.expanduser().resolve()


def main() -> None:
    args = parse_args()

    # First, before any file is opened: an unfinished mode refuses here, so it
    # costs a second rather than a queued job that dies after loading weights.
    # It also decides the prompt config below.
    mode = load_mode(args.mode)

    # Resolved and checked next, before anything is loaded or prepared: a
    # parameter the backend cannot honour is refused rather than silently
    # dropped, because a run that quietly ignored one is indistinguishable
    # afterwards from a run that never asked for it.
    model_config_path = args.model_config.expanduser().resolve()
    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_config = load_model_config(model_config_path, args.backend)
    validate_capabilities(model_config, backend_capabilities(args.backend))

    # The weights are named only in the config, so a typo there would otherwise
    # surface as a transformers error after every record had been prepared and
    # every audio mix written. Only for an absolute path: a bare model_id is a
    # hub id, which only the backend can resolve.
    if model_config.model_id is not None and Path(model_config.model_id).is_absolute():
        weights_path = Path(model_config.model_id)
        if not weights_path.is_dir():
            # Two very different causes, one symptom, and the second is the
            # confusing one: the path can be plainly there on the login node and
            # still be missing here, because only the shares in HOST_BIND_PATHS
            # (job_scripts/lib/intention_job.sh) are bound into the container.
            raise FileNotFoundError(
                f"{model_config_path}: backends.{args.backend}.model_id points at "
                f"{weights_path}, which is not a directory here. Either the path is wrong "
                f"-- fix it there, the weights are not a command-line flag -- or it exists "
                f"on the host but is not bound into the container, in which case add its "
                f"share to HOST_BIND_PATHS in job_scripts/lib/intention_job.sh."
            )

    if args.start_index < 0:
        raise ValueError(f"--start-index must be non-negative: {args.start_index}")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError(
            f"--end-index must be greater than or equal to --start-index: "
            f"{args.end_index} < {args.start_index}"
        )

    input_json_path = args.input_json.expanduser().resolve()
    prompt_config_path = (
        mode.prompt_config_path if args.prompt_config is None else args.prompt_config
    ).expanduser().resolve()
    output_path = (
        default_output_path(args, input_json_path)
        if args.output is None
        else args.output
    ).expanduser().resolve()
    media_root = resolve_optional_path(args.media_root)
    local_path_prefix = resolve_optional_path(args.local_path_prefix)
    video_local_path_prefix = resolve_optional_path(args.video_local_path_prefix)
    audio_local_path_prefix = resolve_optional_path(args.audio_local_path_prefix)
    participant_image_root = args.participant_image_root.expanduser().resolve()

    if not input_json_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")
    if not prompt_config_path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {prompt_config_path}")

    prompt_config = load_prompt_config(prompt_config_path)
    # Handed over before any record is prepared: a mode that labels its own parts
    # keeps those templates in the prompt config, and needs them while building
    # turns as well as while rendering the prompt.
    mode.configure(prompt_config)
    system_prompt = (
        args.system_prompt.strip()
        if isinstance(args.system_prompt, str) and args.system_prompt.strip()
        else prompt_config["system_prompt"]
    )

    all_records = load_manifest_records(input_json_path)
    end_index = len(all_records) - 1 if args.end_index is None else args.end_index
    selected_records = [
        (record_index, record)
        for record_index, record in enumerate(all_records)
        if args.start_index <= record_index <= end_index
    ]
    print(f"[INFO] Loaded {len(all_records)} record(s) from {input_json_path}")
    print(
        f"[INFO] Selected manifest index range: "
        f"{args.start_index}-{end_index} ({len(selected_records)} record(s))"
    )
    print(f"[INFO] Backend: {args.backend}")
    print(f"[INFO] Mode: {mode.name}")
    print(f"[INFO] Model config: {model_config_path}")
    print(f"[INFO] Prompt config: {prompt_config_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Media path prefix: {args.media_path_prefix}")
    print(f"[INFO] Local path prefix: {local_path_prefix}")
    print(f"[INFO] Video media path prefix: {args.video_media_path_prefix}")
    print(f"[INFO] Video local path prefix: {video_local_path_prefix}")
    print(f"[INFO] Audio media path prefix: {args.audio_media_path_prefix}")
    print(f"[INFO] Audio local path prefix: {audio_local_path_prefix}")
    print(f"[INFO] Participant image root: {participant_image_root}")
    print(f"[INFO] No audio: {args.no_audio}")
    print(
        f"[INFO] Frame policy: fps={model_config.video_fps} "
        f"min={model_config.min_video_frames} max={model_config.max_video_frames}"
    )

    aggregated_audio_dir = output_path.parent / "_audio_mixes" / output_path.stem
    ctx = RecordContext(
        id_key=args.id_key,
        manifest_dir=input_json_path.parent,
        media_root=media_root,
        media_path_prefix=args.media_path_prefix,
        local_path_prefix=local_path_prefix,
        video_media_path_prefix=args.video_media_path_prefix,
        video_local_path_prefix=video_local_path_prefix,
        audio_media_path_prefix=args.audio_media_path_prefix,
        audio_local_path_prefix=audio_local_path_prefix,
        participant_image_root=participant_image_root,
        no_audio=args.no_audio,
        aggregated_audio_dir=aggregated_audio_dir,
        exclude_video_substrings=args.exclude_video_substring,
        exclude_audio_substrings=args.exclude_audio_substring,
        user_prompt_template=prompt_config["user_prompt_template"],
    )
    prepared = engine.prepare_records(
        mode=mode,
        ctx=ctx,
        selected_records=selected_records,
        limit=args.limit,
    )
    if not prepared.kept:
        print("[WARN] Nothing to process. Exiting.")
        return

    # Loaded only once every record has been prepared, so a manifest or media
    # problem surfaces before paying for the weights.
    model = load_model(args.backend, model_config.model_id, **model_config.load_kwargs)

    results = engine.run_inference_pass(
        prepared=prepared,
        mode=mode,
        system_prompt=system_prompt,
        no_audio=args.no_audio,
        model=model,
        model_config=model_config,
    )

    summary = engine.build_summary(
        args=args,
        backend=args.backend,
        mode=mode.name,
        model_config=model_config,
        input_json_path=input_json_path,
        prompt_config_path=prompt_config_path,
        record_count=len(all_records),
        selected_record_count=len(selected_records),
        end_index=end_index,
        prepared=prepared,
        results=results,
        aggregated_audio_dir=aggregated_audio_dir,
    )
    engine.write_results(
        output_path,
        summary=summary,
        prepared=prepared,
        results=results,
    )
