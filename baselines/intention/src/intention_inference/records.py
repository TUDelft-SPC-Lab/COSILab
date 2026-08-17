"""Resolving one manifest record to media that exists on this filesystem.

What lives here is what every task mode needs regardless of what it asks: the
clip, and the two audio tracks the task feeds alongside it. The mode-specific
part -- which visual reference grounds the question, and how the prompt is
rendered -- is in ``modes/``, and a mode composes these helpers into its own
``prepare_record``.

Nothing here raises for missing data. A record that cannot be resolved comes back
as ``(None, "<reason>")`` and the caller counts it: one clip missing from a share
should not take down a shard of several hundred.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import aggregate_conversation_floor_audio, select_audio_speakers
from .media import (
    effective_local_prefix,
    effective_media_prefix,
    path_matches_any,
    resolve_prefixed_media_path,
    safe_filename_part,
)

__all__ = [
    "RecordContext",
    "resolve_audio_bundle",
    "resolve_video_media",
]


@dataclass(frozen=True)
class RecordContext:
    """Everything record preparation needs that is not the record itself.

    Built once in ``cli.main`` and passed down unchanged. It exists because the
    alternative is threading fifteen keyword arguments through every mode hook,
    which makes the hooks unimplementable without copying the list.
    """

    id_key: str
    manifest_dir: Path
    media_root: Path | None
    media_path_prefix: str | None
    local_path_prefix: Path | None
    video_media_path_prefix: str | None
    video_local_path_prefix: Path | None
    audio_media_path_prefix: str | None
    audio_local_path_prefix: Path | None
    participant_image_root: Path
    no_audio: bool
    aggregated_audio_dir: Path
    exclude_video_substrings: Sequence[str]
    exclude_audio_substrings: Sequence[str]
    user_prompt_template: str


def resolve_video_media(
    record: Mapping[str, Any],
    ctx: RecordContext,
) -> tuple[dict[str, Any], str | None]:
    """The clip for one record, rewritten from its manifest path to a local one."""
    resolved_media_prefix = effective_media_prefix(
        ctx.video_media_path_prefix, ctx.media_path_prefix
    )
    resolved_local_prefix = effective_local_prefix(
        ctx.video_local_path_prefix, ctx.local_path_prefix
    )
    source_video_path = record.get("video")
    video_path, rewritten_video_path = resolve_prefixed_media_path(
        source_video_path,
        manifest_dir=ctx.manifest_dir,
        media_root=ctx.media_root,
        media_path_prefix=resolved_media_prefix,
        local_path_prefix=resolved_local_prefix,
    )
    if video_path is None:
        return {}, "missing_video_path"
    if path_matches_any(video_path, ctx.exclude_video_substrings):
        return {}, "excluded_video"
    if not video_path.exists():
        return {}, "video_not_found"

    return (
        {
            "source_video_path": None if source_video_path is None else str(source_video_path),
            "rewritten_video_path": rewritten_video_path,
            "video_path": str(video_path),
        },
        None,
    )


def resolve_audio_bundle(
    record: Mapping[str, Any],
    *,
    record_index: int,
    record_id: Any,
    ctx: RecordContext,
) -> tuple[dict[str, Any], str | None]:
    """The participant's own track plus a single mixed conversation-floor track.

    Audio 2 is deliberately one file rather than one part per speaker: the task
    asks about the indicated participant, and the others are context, so they are
    summed into one soundtrack of the same duration. Which speakers those are is
    ``select_audio_speakers``; the mixing and its length tolerance is
    ``aggregate_conversation_floor_audio``.
    """
    empty: dict[str, Any] = {
        "speaker_ids": [],
        "participant_speaker_id": None,
        "conversation_floor_speaker_ids": [],
        "source_audio_paths": [],
        "rewritten_audio_paths": [],
        "audio_paths": [],
        "participant_audio_path": None,
        "conversation_floor_audio_paths": [],
        "aggregated_conversation_floor_audio_path": None,
        "audio_warnings": [],
    }
    if ctx.no_audio:
        return empty, None

    resolved_media_prefix = effective_media_prefix(
        ctx.audio_media_path_prefix, ctx.media_path_prefix
    )
    resolved_local_prefix = effective_local_prefix(
        ctx.audio_local_path_prefix, ctx.local_path_prefix
    )
    audio_entries = record.get("audios")
    if not isinstance(audio_entries, Sequence) or isinstance(
        audio_entries, (str, bytes, bytearray)
    ):
        return {}, "invalid_audio_list"

    speaker_selection = select_audio_speakers(record)
    if speaker_selection is None:
        return {}, "invalid_speaker_selection"

    participant_speaker_id, conversation_floor_speaker_ids = speaker_selection

    def resolve_audio_for_speaker(
        speaker_id: int,
    ) -> tuple[str, str, Path] | tuple[None, None, None]:
        # Speaker id to track is positional: the manifest's "audios" list is in
        # speaker order, so speaker 1 is audios[0].
        audio_index = speaker_id - 1
        if audio_index < 0 or audio_index >= len(audio_entries):
            return None, None, None

        source_audio_path = audio_entries[audio_index]
        resolved_audio_path, rewritten_audio_path = resolve_prefixed_media_path(
            source_audio_path,
            manifest_dir=ctx.manifest_dir,
            media_root=ctx.media_root,
            media_path_prefix=resolved_media_prefix,
            local_path_prefix=resolved_local_prefix,
        )
        if resolved_audio_path is None:
            return None, None, None
        if path_matches_any(resolved_audio_path, ctx.exclude_audio_substrings):
            return None, None, None
        if not resolved_audio_path.exists():
            return None, None, None

        return (
            str(source_audio_path),
            "" if rewritten_audio_path is None else str(rewritten_audio_path),
            resolved_audio_path,
        )

    source_audio_paths: list[str] = []
    rewritten_audio_paths: list[str] = []
    conversation_floor_audio_paths: list[str] = []

    resolved_participant_audio = resolve_audio_for_speaker(participant_speaker_id)
    if resolved_participant_audio == (None, None, None):
        return {}, "participant_audio_not_found"

    source_participant_audio, rewritten_participant_audio, participant_audio = (
        resolved_participant_audio
    )
    source_audio_paths.append(str(source_participant_audio))
    rewritten_audio_paths.append(str(rewritten_participant_audio))

    resolved_conversation_floor_paths: list[Path] = []
    for speaker_id in conversation_floor_speaker_ids:
        resolved_floor_audio = resolve_audio_for_speaker(speaker_id)
        if resolved_floor_audio == (None, None, None):
            return {}, "conversation_floor_audio_not_found"
        source_audio_path, rewritten_audio_path, resolved_audio_path = resolved_floor_audio
        source_audio_paths.append(str(source_audio_path))
        rewritten_audio_paths.append(str(rewritten_audio_path))
        conversation_floor_audio_paths.append(str(resolved_audio_path))
        resolved_conversation_floor_paths.append(resolved_audio_path)

    aggregate_audio_name = (
        f"{record_index:06d}_{safe_filename_part(record_id)}_conversation_floor.wav"
    )
    aggregate_audio_path = ctx.aggregated_audio_dir / aggregate_audio_name
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
        return {}, "audio_aggregation_failed"

    return (
        {
            "speaker_ids": [participant_speaker_id, *conversation_floor_speaker_ids],
            "participant_speaker_id": participant_speaker_id,
            "conversation_floor_speaker_ids": conversation_floor_speaker_ids,
            "source_audio_paths": source_audio_paths,
            "rewritten_audio_paths": rewritten_audio_paths,
            # The order is the order the turn presents them in: own track first,
            # floor mix second, which is what the prompt calls audio1 and audio2.
            "audio_paths": [str(participant_audio), str(aggregate_audio_path)],
            "participant_audio_path": str(participant_audio),
            "conversation_floor_audio_paths": conversation_floor_audio_paths,
            "aggregated_conversation_floor_audio_path": str(aggregate_audio_path),
            "audio_warnings": audio_warnings,
        },
        None,
    )
