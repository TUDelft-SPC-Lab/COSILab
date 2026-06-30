from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .audio import aggregate_conversation_floor_audio, select_audio_speakers
from .manifest import get_nested_value
from .media import (
    effective_local_prefix,
    effective_media_prefix,
    path_matches_any,
    resolve_prefixed_media_path,
    safe_filename_part,
)
from .prompt import render_prompt


def resolve_participant_image_path(
    record: Mapping[str, Any],
    participant_image_root: Path,
) -> Path | None:
    participant = record.get("participant")
    if not isinstance(participant, int) or participant <= 0:
        return None
    return participant_image_root / f"participant_{participant}.png"


def prepare_record(
    *,
    record: dict[str, Any],
    record_index: int,
    id_key: str,
    manifest_dir: Path,
    media_root: Path | None,
    media_path_prefix: str | None,
    local_path_prefix: Path | None,
    video_media_path_prefix: str | None,
    video_local_path_prefix: Path | None,
    audio_media_path_prefix: str | None,
    audio_local_path_prefix: Path | None,
    participant_image_root: Path,
    no_audio: bool,
    aggregated_audio_dir: Path,
    exclude_video_substrings: Sequence[str],
    exclude_audio_substrings: Sequence[str],
    user_prompt_template: str,
) -> tuple[dict[str, Any] | None, str | None]:
    record_id = get_nested_value(record, id_key)
    if record_id is None:
        record_id = record.get("id", record_index)

    resolved_video_media_path_prefix = effective_media_prefix(
        video_media_path_prefix, media_path_prefix
    )
    resolved_video_local_path_prefix = effective_local_prefix(
        video_local_path_prefix, local_path_prefix
    )
    source_video_path = record.get("video")
    video_path, rewritten_video_path = resolve_prefixed_media_path(
        source_video_path,
        manifest_dir=manifest_dir,
        media_root=media_root,
        media_path_prefix=resolved_video_media_path_prefix,
        local_path_prefix=resolved_video_local_path_prefix,
    )
    if video_path is None:
        return None, "missing_video_path"
    if path_matches_any(video_path, exclude_video_substrings):
        return None, "excluded_video"
    if not video_path.exists():
        return None, "video_not_found"

    participant_image_path = resolve_participant_image_path(record, participant_image_root)
    if participant_image_path is None:
        return None, "missing_participant_image"
    if not participant_image_path.exists():
        return None, "participant_image_not_found"

    source_audio_paths: list[str] = []
    rewritten_audio_paths: list[str] = []
    audio_paths: list[str] = []
    participant_speaker_id: int | None = None
    conversation_floor_speaker_ids: list[int] = []
    participant_audio_path: str | None = None
    conversation_floor_audio_paths: list[str] = []
    aggregated_conversation_floor_audio_path: str | None = None
    audio_warnings: list[str] = []
    if not no_audio:
        resolved_audio_media_path_prefix = effective_media_prefix(
            audio_media_path_prefix, media_path_prefix
        )
        resolved_audio_local_path_prefix = effective_local_prefix(
            audio_local_path_prefix, local_path_prefix
        )
        audio_entries = record.get("audios")
        if not isinstance(audio_entries, Sequence) or isinstance(
            audio_entries, (str, bytes, bytearray)
        ):
            return None, "invalid_audio_list"

        speaker_selection = select_audio_speakers(record)
        if speaker_selection is None:
            return None, "invalid_speaker_selection"

        participant_speaker_id, conversation_floor_speaker_ids = speaker_selection

        def resolve_audio_for_speaker(
            speaker_id: int,
        ) -> tuple[str, str, Path] | tuple[None, None, None]:
            audio_index = speaker_id - 1
            if audio_index < 0 or audio_index >= len(audio_entries):
                return None, None, None

            source_audio_path = audio_entries[audio_index]
            resolved_audio_path, rewritten_audio_path = resolve_prefixed_media_path(
                source_audio_path,
                manifest_dir=manifest_dir,
                media_root=media_root,
                media_path_prefix=resolved_audio_media_path_prefix,
                local_path_prefix=resolved_audio_local_path_prefix,
            )
            if resolved_audio_path is None:
                return None, None, None
            if path_matches_any(resolved_audio_path, exclude_audio_substrings):
                return None, None, None
            if not resolved_audio_path.exists():
                return None, None, None

            return (
                str(source_audio_path),
                "" if rewritten_audio_path is None else str(rewritten_audio_path),
                resolved_audio_path,
            )

        resolved_participant_audio = resolve_audio_for_speaker(participant_speaker_id)
        if resolved_participant_audio == (None, None, None):
            return None, "participant_audio_not_found"

        source_participant_audio, rewritten_participant_audio, participant_audio = (
            resolved_participant_audio
        )
        source_audio_paths.append(str(source_participant_audio))
        rewritten_audio_paths.append(str(rewritten_participant_audio))
        participant_audio_path = str(participant_audio)

        resolved_conversation_floor_paths: list[Path] = []
        for speaker_id in conversation_floor_speaker_ids:
            resolved_floor_audio = resolve_audio_for_speaker(speaker_id)
            if resolved_floor_audio == (None, None, None):
                return None, "conversation_floor_audio_not_found"
            source_audio_path, rewritten_audio_path, resolved_audio_path = resolved_floor_audio
            source_audio_paths.append(str(source_audio_path))
            rewritten_audio_paths.append(str(rewritten_audio_path))
            conversation_floor_audio_paths.append(str(resolved_audio_path))
            resolved_conversation_floor_paths.append(resolved_audio_path)

        aggregate_audio_name = (
            f"{record_index:06d}_{safe_filename_part(record_id)}_conversation_floor.wav"
        )
        aggregate_audio_path = aggregated_audio_dir / aggregate_audio_name
        try:
            audio_warnings = aggregate_conversation_floor_audio(
                participant_audio_path=participant_audio,
                conversation_floor_audio_paths=resolved_conversation_floor_paths,
                output_audio_path=aggregate_audio_path,
                record_id=str(record_id),
            )
        except Exception as exc:
            print(
                f"[WARN] record {record_id}: failed to aggregate conversation-floor audio: {exc}",
                flush=True,
            )
            return None, "audio_aggregation_failed"

        aggregated_conversation_floor_audio_path = str(aggregate_audio_path)
        audio_paths = [str(participant_audio), str(aggregate_audio_path)]

    return (
        {
            "record_index": record_index,
            "record_id": str(record_id),
            "source_video_path": None if source_video_path is None else str(source_video_path),
            "rewritten_video_path": rewritten_video_path,
            "video_path": str(video_path),
            "participant_image_path": str(participant_image_path),
            "speaker_ids": [] if no_audio else [
                participant_speaker_id,
                *conversation_floor_speaker_ids,
            ],
            "participant_speaker_id": participant_speaker_id,
            "conversation_floor_speaker_ids": conversation_floor_speaker_ids,
            "source_audio_paths": source_audio_paths,
            "rewritten_audio_paths": rewritten_audio_paths,
            "audio_paths": audio_paths,
            "participant_audio_path": participant_audio_path,
            "conversation_floor_audio_paths": conversation_floor_audio_paths,
            "aggregated_conversation_floor_audio_path": (
                aggregated_conversation_floor_audio_path
            ),
            "audio_warnings": audio_warnings,
            "user_prompt": render_prompt(user_prompt_template, record),
            "source_record": record,
        },
        None,
    )
