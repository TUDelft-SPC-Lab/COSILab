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
    "SpeakerTracks",
    "resolve_audio_bundle",
    "resolve_per_speaker_audio",
    "resolve_speaker_tracks",
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


@dataclass(frozen=True)
class SpeakerTracks:
    """Every speaker's own audio track, before any mode decides what to do with it.

    ``participant`` is the person the question is about; ``conversation_floor``
    is everyone else in their group, in the order the record listed them.
    ``goa`` is the whole group -- the participant included -- sorted by id, which
    is what a mode presenting people symmetrically should iterate.
    """

    participant_id: int
    conversation_floor_ids: list[int]
    paths: dict[int, Path]
    source_paths: dict[int, str]
    rewritten_paths: dict[int, str]

    @property
    def goa_ids(self) -> list[int]:
        return sorted({self.participant_id, *self.conversation_floor_ids})


def resolve_speaker_tracks(
    record: Mapping[str, Any],
    ctx: RecordContext,
) -> tuple[SpeakerTracks | None, str | None]:
    """Resolve one local audio file per speaker in the group.

    Shared by every mode, because which track belongs to whom is a property of
    the record rather than of the question. What differs between modes is only
    what they then do with the tracks: hand them over one by one, or stack some
    or all of them into a single soundtrack.
    """
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
        return None, "invalid_audio_list"

    speaker_selection = select_audio_speakers(record)
    if speaker_selection is None:
        return None, "invalid_speaker_selection"

    participant_id, conversation_floor_ids = speaker_selection

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

    paths: dict[int, Path] = {}
    source_paths: dict[int, str] = {}
    rewritten_paths: dict[int, str] = {}

    resolved_participant_audio = resolve_audio_for_speaker(participant_id)
    if resolved_participant_audio == (None, None, None):
        return None, "participant_audio_not_found"
    source_paths[participant_id], rewritten_paths[participant_id], paths[participant_id] = (
        resolved_participant_audio
    )

    for speaker_id in conversation_floor_ids:
        resolved_floor_audio = resolve_audio_for_speaker(speaker_id)
        if resolved_floor_audio == (None, None, None):
            return None, "conversation_floor_audio_not_found"
        source_paths[speaker_id], rewritten_paths[speaker_id], paths[speaker_id] = (
            resolved_floor_audio
        )

    return (
        SpeakerTracks(
            participant_id=participant_id,
            conversation_floor_ids=conversation_floor_ids,
            paths=paths,
            source_paths=source_paths,
            rewritten_paths=rewritten_paths,
        ),
        None,
    )


def resolve_per_speaker_audio(
    record: Mapping[str, Any],
    ctx: RecordContext,
) -> tuple[dict[str, Any], str | None]:
    """One track per person, nothing stacked.

    The group is reported in ascending id order rather than participant-first:
    a mode that presents people symmetrically and only names the one of interest
    at the end would give the answer away by ordering.
    """
    empty: dict[str, Any] = {
        "goa_speaker_ids": [],
        "participant_speaker_id": None,
        "conversation_floor_speaker_ids": [],
        "source_audio_paths": [],
        "rewritten_audio_paths": [],
        "audio_paths": [],
        "speaker_audio_paths": {},
        "participant_audio_path": None,
        "audio_warnings": [],
    }
    if ctx.no_audio:
        return empty, None

    tracks, skip_reason = resolve_speaker_tracks(record, ctx)
    if tracks is None:
        return {}, skip_reason

    goa_ids = tracks.goa_ids
    return (
        {
            "goa_speaker_ids": goa_ids,
            "participant_speaker_id": tracks.participant_id,
            "conversation_floor_speaker_ids": tracks.conversation_floor_ids,
            "source_audio_paths": [tracks.source_paths[i] for i in goa_ids],
            "rewritten_audio_paths": [tracks.rewritten_paths[i] for i in goa_ids],
            "audio_paths": [str(tracks.paths[i]) for i in goa_ids],
            # The id -> track map, so a result file says which part was whose
            # without anyone having to re-derive it from the ordering.
            "speaker_audio_paths": {str(i): str(tracks.paths[i]) for i in goa_ids},
            "participant_audio_path": str(tracks.paths[tracks.participant_id]),
            "audio_warnings": [],
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
    summed into one soundtrack of the same duration. The mixing and its length
    tolerance is ``aggregate_conversation_floor_audio``.
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

    tracks, skip_reason = resolve_speaker_tracks(record, ctx)
    if tracks is None:
        return {}, skip_reason

    participant_speaker_id = tracks.participant_id
    conversation_floor_speaker_ids = tracks.conversation_floor_ids
    participant_audio = tracks.paths[participant_speaker_id]

    # Participant first, then the floor in record order: this is the order the
    # pre-mode implementation wrote these lists in, and result files already
    # carry it.
    ordered_ids = [participant_speaker_id, *conversation_floor_speaker_ids]
    source_audio_paths = [tracks.source_paths[i] for i in ordered_ids]
    rewritten_audio_paths = [tracks.rewritten_paths[i] for i in ordered_ids]
    resolved_conversation_floor_paths = [
        tracks.paths[i] for i in conversation_floor_speaker_ids
    ]
    conversation_floor_audio_paths = [
        str(path) for path in resolved_conversation_floor_paths
    ]

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
