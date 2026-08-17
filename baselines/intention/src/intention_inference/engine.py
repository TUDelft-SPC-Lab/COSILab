"""Everything the task does once the command line has been read.

Split from ``cli.py`` so that file holds only the argument surface and the order
of operations. The order matters more than it looks: preparing records is cheap
and fails for boring reasons -- a manifest path that does not resolve, a clip
that is not on this share -- so it all happens before the weights are loaded.

The model is reached only through ``models.base.BaseMultimodalModel``. Nothing
here imports torch or transformers, and nothing here knows which backend is
running: a turn is a list of :class:`~models.base.MediaPart` values, and what
becomes of them is the backend's business.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.base import ChatMessage, MediaPart, combine_system_and_user_prompt
from models.config import ModelRunConfig, describe_model_config

from .manifest import get_nested_value
from .records import prepare_record

__all__ = [
    "PreparedRecords",
    "build_summary",
    "prepare_records",
    "run_inference_pass",
    "write_results",
]


@dataclass
class PreparedRecords:
    """What survived preparation, and an account of what did not."""

    kept: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    skip_counter: Counter[str] = field(default_factory=Counter)


def prepare_records(
    *,
    args,
    selected_records: list[tuple[int, dict[str, Any]]],
    manifest_dir: Path,
    media_root: Path | None,
    local_path_prefix: Path | None,
    video_local_path_prefix: Path | None,
    audio_local_path_prefix: Path | None,
    participant_image_root: Path,
    aggregated_audio_dir: Path,
    user_prompt_template: str,
) -> PreparedRecords:
    """Resolve every selected record to media that exists on this filesystem.

    A record that cannot be resolved is counted and recorded rather than raised:
    one clip missing from a share should not take down a shard of several
    hundred. The counts are what the summary reports, and what tells you
    afterwards whether a run was thin because the data was thin or because the
    share was not mounted.
    """
    prepared = PreparedRecords()

    for record_index, record in selected_records:
        record_result, skip_reason = prepare_record(
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
            user_prompt_template=user_prompt_template,
        )
        if record_result is None:
            skip_reason = skip_reason or "filtered_out"
            prepared.skip_counter[skip_reason] += 1
            prepared.skipped.append(
                {
                    "record_index": record_index,
                    "record_id": str(get_nested_value(record, args.id_key) or record_index),
                    "skip_reason": skip_reason,
                }
            )
            continue

        prepared.kept.append(record_result)
        if args.limit is not None and len(prepared.kept) >= args.limit:
            break

    print(
        f"[INFO] Retained {len(prepared.kept)} record(s); "
        f"skipped {sum(prepared.skip_counter.values())} before inference."
    )
    for reason, count in sorted(prepared.skip_counter.items()):
        print(f"[INFO]   skip {reason}: {count}")

    return prepared


def build_turn(
    item: dict[str, Any],
    *,
    system_prompt: str,
    no_audio: bool,
    model,
    model_config: ModelRunConfig,
) -> ChatMessage:
    """One user turn: who to look at, what was said, the clip, the question.

    The order is the order the model sees the placeholders in, so it is part of
    the prompt rather than a detail of how the list was built.
    """
    content: list[MediaPart] = [MediaPart.image(item["participant_image_path"])]
    if not no_audio:
        content.extend(MediaPart.audio(path) for path in item["audio_paths"])
    content.append(
        # The frame-sampling policy belongs to the run, not the backend: passing
        # all three values means gemma and qwen7b see the same frames of the same
        # clip, and a difference between them is the model.
        model.prepare_video_part(
            item["video_path"],
            model_config.max_video_frames,
            fps=model_config.video_fps,
            min_frames=model_config.min_video_frames,
        )
    )
    content.append(
        MediaPart.text_part(
            combine_system_and_user_prompt(
                system_prompt=system_prompt,
                user_prompt=item["user_prompt"],
            )
        )
    )
    return ChatMessage("user", content)


def run_inference_pass(
    *,
    prepared: PreparedRecords,
    system_prompt: str,
    no_audio: bool,
    model,
    model_config: ModelRunConfig,
) -> list[dict[str, Any]]:
    """One turn per retained record, in manifest order."""
    results: list[dict[str, Any]] = []
    total_records = len(prepared.kept)

    for processed_index, item in enumerate(prepared.kept, start=1):
        print(
            f"[{processed_index}/{total_records}] "
            f"{item['record_id']} -> {Path(item['video_path']).name}",
            flush=True,
        )
        try:
            turn = build_turn(
                item,
                system_prompt=system_prompt,
                no_audio=no_audio,
                model=model,
                model_config=model_config,
            )
            response = model.generate([turn], model_config.generation)
        except Exception as exc:
            # Recorded, not raised: a clip the model chokes on should cost that
            # clip, not the rest of the shard. build_turn is inside the try for
            # the same reason -- decoding a corrupt video fails here too.
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

    return results


def build_summary(
    *,
    args,
    backend: str,
    model_config: ModelRunConfig,
    input_json_path: Path,
    prompt_config_path: Path,
    record_count: int,
    selected_record_count: int,
    end_index: int,
    prepared: PreparedRecords,
    results: list[dict[str, Any]],
    aggregated_audio_dir: Path,
) -> dict[str, Any]:
    """What this run was, recorded beside its answers.

    ``describe_model_config`` contributes the weights, the decoding parameters
    and the frame policy, so a result file answers "what produced this" without
    anyone having to find the sbatch line that made it.
    """
    error_count = sum(1 for item in results if str(item["assistant"]).startswith("[ERROR]"))
    return {
        "input_json": str(input_json_path),
        "prompt_config": str(prompt_config_path),
        "backend": backend,
        "record_count": record_count,
        "selected_record_count": selected_record_count,
        "start_index": args.start_index,
        "end_index": end_index,
        "retained_count": len(prepared.kept),
        "skipped_count": len(prepared.skipped),
        "processed_count": len(results),
        "error_count": error_count,
        "no_audio": args.no_audio,
        **describe_model_config(model_config),
        "aggregated_audio_dir": None if args.no_audio else str(aggregated_audio_dir),
        "skip_reasons": dict(sorted(prepared.skip_counter.items())),
    }


def write_results(
    output_path: Path,
    *,
    summary: dict[str, Any],
    prepared: PreparedRecords,
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "__summary__": summary,
        "__skipped__": prepared.skipped,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Results saved to {output_path}")
