from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoModelForMultimodalLM, AutoProcessor

from .gemma import combine_system_and_user_prompt, infer_turn, select_video_num_frames

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/scratch/zli33/models/GemmaE4B")
from .manifest import get_nested_value, load_manifest_records
from .prompt import load_prompt_config
from .records import prepare_record

DEFAULT_PROMPT_CONFIG_PATH = Path(__file__).resolve().with_name("prompt_ingroup.json")
DEFAULT_PARTICIPANT_IMAGE_ROOT = Path(
    "/tudelft.net/staff-umbrella/neon/B1_pipeline/participant_imgs"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Gemma 4 ingroup inference from a flat JSON manifest."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Local Gemma model directory or Hugging Face model id.",
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
        default=Path("ingroup_results.json"),
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=DEFAULT_PROMPT_CONFIG_PATH,
        help="Prompt config JSON. Defaults to prompt_ingroup.json alongside this package.",
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
            "indicated participant."
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
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Gemma thinking mode in the chat template when supported.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling. By default generation is deterministic.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=32,
        help="Maximum frames to sample per video.",
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


def main() -> None:
    args = parse_args()

    if args.start_index < 0:
        raise ValueError(f"--start-index must be non-negative: {args.start_index}")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError(
            f"--end-index must be greater than or equal to --start-index: "
            f"{args.end_index} < {args.start_index}"
        )

    input_json_path = args.input_json.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    prompt_config_path = args.prompt_config.expanduser().resolve()
    media_root = None if args.media_root is None else args.media_root.expanduser().resolve()
    local_path_prefix = (
        None if args.local_path_prefix is None else args.local_path_prefix.expanduser().resolve()
    )
    video_local_path_prefix = (
        None
        if args.video_local_path_prefix is None
        else args.video_local_path_prefix.expanduser().resolve()
    )
    audio_local_path_prefix = (
        None
        if args.audio_local_path_prefix is None
        else args.audio_local_path_prefix.expanduser().resolve()
    )
    participant_image_root = args.participant_image_root.expanduser().resolve()

    if not input_json_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")
    if not prompt_config_path.is_file():
        raise FileNotFoundError(f"Prompt config not found: {prompt_config_path}")

    prompt_config = load_prompt_config(prompt_config_path)
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
    print(f"[INFO] Prompt config: {prompt_config_path}")
    print(f"[INFO] Media path prefix: {args.media_path_prefix}")
    print(f"[INFO] Local path prefix: {local_path_prefix}")
    print(f"[INFO] Video media path prefix: {args.video_media_path_prefix}")
    print(f"[INFO] Video local path prefix: {video_local_path_prefix}")
    print(f"[INFO] Audio media path prefix: {args.audio_media_path_prefix}")
    print(f"[INFO] Audio local path prefix: {audio_local_path_prefix}")
    print(f"[INFO] Participant image root: {participant_image_root}")
    print(f"[INFO] No audio: {args.no_audio}")
    print(f"[INFO] Max video frames: {args.max_video_frames}")

    kept_records: list[dict[str, Any]] = []
    skipped_records: list[dict[str, Any]] = []
    skip_counter: Counter[str] = Counter()
    manifest_dir = input_json_path.parent
    aggregated_audio_dir = output_path.parent / "_audio_mixes" / output_path.stem

    for record_index, record in selected_records:
        prepared, skip_reason = prepare_record(
            record=record,
            record_index=record_index,
            id_key=args.id_key,
            manifest_dir=manifest_dir,
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
        if prepared is None:
            skip_reason = skip_reason or "filtered_out"
            skip_counter[skip_reason] += 1
            skipped_records.append(
                {
                    "record_index": record_index,
                    "record_id": str(get_nested_value(record, args.id_key) or record_index),
                    "skip_reason": skip_reason,
                }
            )
            continue
        kept_records.append(prepared)
        if args.limit is not None and len(kept_records) >= args.limit:
            break

    print(
        f"[INFO] Retained {len(kept_records)} record(s); "
        f"skipped {sum(skip_counter.values())} before inference."
    )
    if skip_counter:
        for reason, count in sorted(skip_counter.items()):
            print(f"[INFO]   skip {reason}: {count}")
    if not kept_records:
        print("[WARN] Nothing to process. Exiting.")
        return

    print(f"[INFO] Loading Gemma model: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    results: list[dict[str, Any]] = []
    total_records = len(kept_records)
    for processed_index, item in enumerate(kept_records, start=1):
        print(
            f"[{processed_index}/{total_records}] "
            f"{item['record_id']} -> {Path(item['video_path']).name}",
            flush=True,
        )
        video_num_frames = select_video_num_frames(
            video_path=item["video_path"],
            max_video_frames=args.max_video_frames,
        )
        text_prompt = combine_system_and_user_prompt(
            system_prompt=system_prompt,
            user_prompt=item["user_prompt"],
        )

        user_content: list[dict[str, Any]] = []
        user_content.append({"type": "image", "image": item["participant_image_path"]})
        if not args.no_audio:
            for audio_path in item["audio_paths"]:
                user_content.append({"type": "audio", "audio": audio_path})
        user_content.extend(
            [
                {
                    "type": "video",
                    "video": item["video_path"],
                    "num_frames": video_num_frames,
                },
                {"type": "text", "text": text_prompt},
            ]
        )

        try:
            response = infer_turn(
                model=model,
                processor=processor,
                messages=[{"role": "user", "content": user_content}],
                max_new_tokens=args.max_new_tokens,
                enable_thinking=args.enable_thinking,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        except Exception as exc:
            response = f"[ERROR] {exc}"
            print(f"  [WARN] Error: {exc}", flush=True)

        results.append(
            {
                "record_index": item["record_index"],
                "record_id": item["record_id"],
                "source_video_path": item["source_video_path"],
                "rewritten_video_path": item["rewritten_video_path"],
                "video_path": item["video_path"],
                "participant_image_path": item["participant_image_path"],
                "speaker_ids": item["speaker_ids"],
                "participant_speaker_id": item["participant_speaker_id"],
                "conversation_floor_speaker_ids": item["conversation_floor_speaker_ids"],
                "source_audio_paths": item["source_audio_paths"],
                "rewritten_audio_paths": item["rewritten_audio_paths"],
                "audio_paths": item["audio_paths"],
                "participant_audio_path": item["participant_audio_path"],
                "conversation_floor_audio_paths": item["conversation_floor_audio_paths"],
                "aggregated_conversation_floor_audio_path": (
                    item["aggregated_conversation_floor_audio_path"]
                ),
                "audio_warnings": item["audio_warnings"],
                "system": system_prompt,
                "user": item["user_prompt"],
                "assistant": response,
            }
        )

    error_count = sum(1 for item in results if str(item["assistant"]).startswith("[ERROR]"))
    summary = {
        "input_json": str(input_json_path),
        "prompt_config": str(prompt_config_path),
        "record_count": len(all_records),
        "selected_record_count": len(selected_records),
        "start_index": args.start_index,
        "end_index": end_index,
        "retained_count": len(kept_records),
        "skipped_count": len(skipped_records),
        "processed_count": len(results),
        "error_count": error_count,
        "no_audio": args.no_audio,
        "max_video_frames": args.max_video_frames,
        "aggregated_audio_dir": None if args.no_audio else str(aggregated_audio_dir),
        "skip_reasons": dict(sorted(skip_counter.items())),
    }

    output_payload = {
        "__summary__": summary,
        "__skipped__": skipped_records,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Results saved to {output_path}")
