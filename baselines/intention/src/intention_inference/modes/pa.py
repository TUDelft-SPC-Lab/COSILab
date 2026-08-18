"""PA -- partially stacked audio.

The middle of the three audio conditions. The person of interest keeps their own
soundtrack, and everyone else in their group is summed into a second one: two
audio parts, however many people are in the group. SA stacks all of them into
one, FA stacks none of them.

Grounding is a single crop -- ``participant_<n>.png`` -- and the model has to
find that person in the clip itself. No identifier is passed: the participant's
number selects the file and the audio track, and then stays out of the prompt,
so the grounding is entirely facial, and the other speakers are anonymous by
construction because their tracks arrive summed.

This is the task exactly as it ran before modes existed, and it is kept that way
on purpose: results already collected under it stay comparable. In particular
the prompt's "<image>", "<audio1>" and "<audio2>" markers are literal text that
resolve to nothing -- the real tokens are all emitted ahead of the text block --
so the binding rests on part order. FA does not repeat that; PA keeps it rather
than silently becoming a different condition.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ChatMessage, MediaPart

from ..records import RecordContext
from .base import BaseTaskMode

__all__ = ["PartiallyStackedAudioMode"]


class PartiallyStackedAudioMode(BaseTaskMode):
    name = "pa"
    prompt_config_path = Path(__file__).resolve().with_name("prompt_pa.json")

    def resolve_reference_media(
        self,
        record: Mapping[str, Any],
        ctx: RecordContext,
    ) -> tuple[dict[str, Any], str | None]:
        participant = record.get("participant")
        if not isinstance(participant, int) or participant <= 0:
            return {}, "missing_participant_image"

        image_path = ctx.participant_image_root / f"participant_{participant}.png"
        if not image_path.exists():
            return {}, "participant_image_not_found"

        return {"participant_image_path": str(image_path)}, None

    def build_turn(
        self,
        item: Mapping[str, Any],
        *,
        system_prompt: str,
        no_audio: bool,
        model,
        model_config,
    ) -> ChatMessage:
        content: list[MediaPart] = [MediaPart.image(item["participant_image_path"])]
        if not no_audio:
            content.extend(MediaPart.audio(path) for path in item["audio_paths"])
        content.append(self.video_part(item, model, model_config))
        content.append(self.text_part(item, system_prompt))
        return ChatMessage("user", content)

    # Reported for every record, in this order. It is the order the pre-mode
    # implementation wrote them in, and result files already collected under this
    # mode carry it, so leave it alone.
    RESULT_KEYS = (
        "participant_image_path",
        "speaker_ids",
        "participant_speaker_id",
        "conversation_floor_speaker_ids",
        "source_audio_paths",
        "rewritten_audio_paths",
        "audio_paths",
        "participant_audio_path",
        "conversation_floor_audio_paths",
        "aggregated_conversation_floor_audio_path",
        "audio_warnings",
    )

    def result_fields(self, item: Mapping[str, Any]) -> dict[str, Any]:
        return {key: item[key] for key in self.RESULT_KEYS}
