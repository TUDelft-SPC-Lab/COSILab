"""BLIND -- no media at all: the prior, not a perception.

The control condition for every other mode. The model gets the persona (as the
system prompt) and a text prompt saying a clip exists but is withheld, and
nothing else: no video, no audio, no participant crop. Whatever it answers is
what it would have said about *any* clip at such an event, so the distance
between a blind answer and a pa/fa answer for the same persona and clip is what
the media actually contributed.

Two consequences worth knowing before reading results:

  * The prompt names no clip, so every record a persona answers gets the same
    turn. Under the packaged do_sample=false that is one answer copied N times.
    job_scripts/intention/persona_blind.sh therefore runs one clip per persona
    with sampling on and --trials for the distribution.
  * The clip is still resolved, though never shown. That keeps the retained
    clip set identical to the media modes' -- a clip missing from the share is
    skipped here exactly as it is there -- and keeps the video path columns in
    the result rows, so blind rows join to pa/fa rows on the same keys.

Any assignment built for another mode is valid here: blind needs a subset of
what every mode needs. persona_blind.sh accepts one accordingly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ChatMessage

from ..records import RecordContext, resolve_video_media
from .base import BaseTaskMode

__all__ = ["BlindMode"]


class BlindMode(BaseTaskMode):
    name = "blind"
    prompt_config_path = Path(__file__).resolve().with_name("prompt_blind.json")
    # The baseline instruction prompt asks the model to watch and listen, which
    # this mode cannot honestly ask. Its own prompt config carries the question
    # instead, so the default instruction prompt is dropped rather than applied;
    # one passed explicitly is still checked (see instruction_prompt.py).
    uses_default_instruction_prompt = False

    def prepare_record(
        self,
        record: dict[str, Any],
        *,
        record_index: int,
        record_id: Any,
        ctx: RecordContext,
    ) -> tuple[dict[str, Any] | None, str | None]:
        # Overridden rather than inherited: the default composition resolves the
        # audio bundle, which for pa *writes* a conversation-floor mix per clip.
        # Paying for that, and skipping clips whose audio is missing, would make
        # a condition that uses no audio depend on it.
        video_fields, skip_reason = resolve_video_media(record, ctx)
        if skip_reason is not None:
            return None, skip_reason

        item: dict[str, Any] = {
            "record_index": record_index,
            "record_id": str(record_id),
            **video_fields,
            "source_record": record,
        }
        item["user_prompt"] = self.render_user_prompt(
            ctx.user_prompt_template, record, item
        )
        return item, None

    def resolve_reference_media(
        self,
        record: Mapping[str, Any],
        ctx: RecordContext,
    ) -> tuple[dict[str, Any], str | None]:
        # Nothing grounds the question: there is no one to point at.
        return {}, None

    def build_turn(
        self,
        item: Mapping[str, Any],
        *,
        system_prompt: str,
        no_audio: bool,
        model,
        model_config,
    ) -> ChatMessage:
        # Text only. no_audio is moot: there was never any audio to omit.
        return ChatMessage("user", [self.text_part(item, system_prompt)])

    def result_fields(self, item: Mapping[str, Any]) -> dict[str, Any]:
        # The participant the answer is nominally about, for joining against
        # pa/fa rows; the record itself is not in the result row.
        return {"participant": item["source_record"].get("participant")}
