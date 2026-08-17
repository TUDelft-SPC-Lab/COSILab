"""Grounding the question with a reference photo of the indicated participant.

The model is shown one crop -- ``participant_<n>.png`` -- and has to find that
person in the clip itself. No identifier is passed: the participant's number
selects the file and the audio track, and then stays out of the prompt, so the
grounding is entirely facial. The conversation-floor speakers are anonymous by
construction, since their tracks arrive summed into one soundtrack.

This is the task as it was run before modes existed, moved unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ChatMessage, MediaPart

from ..records import RecordContext
from .base import BaseTaskMode

__all__ = ["ParticipantImageMode"]


class ParticipantImageMode(BaseTaskMode):
    name = "participant_image"
    prompt_config_path = Path(__file__).resolve().with_name("prompt_participant_image.json")

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

    def result_fields(self, item: Mapping[str, Any]) -> dict[str, Any]:
        return {"participant_image_path": item["participant_image_path"]}
