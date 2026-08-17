"""FA -- flattened audio.

Nothing is stacked. Every person in the group of analysis gets their own audio
part and their own gallery image, each introduced by a text part naming their
id, and only after all of that -- and the clip -- is the person of interest
named and the question asked.

Two things about that shape are deliberate and easy to undo by accident:

* **The group is presented in ascending id order, not participant-first.** The
  point of withholding the person of interest until the end is that the model
  reads the group symmetrically; ordering the parts by who matters would give the
  answer away before the question. ``records.resolve_per_speaker_audio`` sorts,
  and this module iterates ``goa_speaker_ids`` as it comes.
* **Ids are bound by interleaved text, not by position.** A text part naming the
  speaker precedes each media part, so the association survives regardless of how
  a backend lays out its placeholders. This is the thing PA gets wrong -- its
  "<audio1>"/"<audio2>" markers are inert text sitting after every media token --
  and the reason not to copy PA's prompt shape here.

The labels themselves live in the prompt config, not in this file, so the exact
wording of a run stays diffable.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ChatMessage, MediaPart

from ..prompt import render_prompt
from ..records import RecordContext, resolve_per_speaker_audio, resolve_video_media
from .base import BaseTaskMode

__all__ = ["FlattenedAudioMode"]

DEFAULT_PREAMBLE = (
    "You will be shown, for each person in a conversation group, their own audio "
    "and a photograph of them, each labelled with their participant number, "
    "followed by a video of the group."
)
DEFAULT_AUDIO_LABEL = "Audio of participant {speaker_id}:"
DEFAULT_IMAGE_LABEL = "Photograph of participant {speaker_id}:"
DEFAULT_VIDEO_LABEL = "Video of the group:"


class FlattenedAudioMode(BaseTaskMode):
    name = "fa"
    prompt_config_path = Path(__file__).resolve().with_name("prompt_fa.json")

    def _label(self, key: str, default: str, **values: Any) -> str:
        template = self.prompt_config.get(key, default)
        if not isinstance(template, str):
            raise ValueError(
                f"{self.prompt_config_path}: {key!r} must be a string, "
                f"got {type(template).__name__}."
            )
        return template.format(**values)

    def resolve_reference_media(
        self,
        record: Mapping[str, Any],
        ctx: RecordContext,
    ) -> tuple[dict[str, Any], str | None]:
        """A gallery image for every person in the group, not just the one asked about."""
        participant = record.get("participant")
        if not isinstance(participant, int) or participant <= 0:
            return {}, "missing_participant_image"

        conversation_floor = record.get("conversation_floor") or []
        if not isinstance(conversation_floor, (list, tuple)):
            return {}, "invalid_speaker_selection"
        if any(not isinstance(i, int) or i <= 0 for i in conversation_floor):
            return {}, "invalid_speaker_selection"

        # The group of analysis is the participant plus their floor. Derived
        # rather than read from a field because the manifest has no group of its
        # own, and conversation_floor may or may not already list the
        # participant -- the union settles it either way.
        goa_ids = sorted({participant, *conversation_floor})

        image_paths: dict[str, str] = {}
        for speaker_id in goa_ids:
            image_path = ctx.participant_image_root / f"participant_{speaker_id}.png"
            if not image_path.exists():
                # Distinct from PA's reason: here a record is dropped because
                # *someone* in the group has no photograph, which is worth
                # telling apart from the person of interest having none.
                return {}, "goa_participant_image_not_found"
            image_paths[str(speaker_id)] = str(image_path)

        return (
            {
                "goa_image_ids": goa_ids,
                "speaker_image_paths": image_paths,
                "participant_image_path": image_paths[str(participant)],
            },
            None,
        )

    def prepare_record(
        self,
        record: dict[str, Any],
        *,
        record_index: int,
        record_id: Any,
        ctx: RecordContext,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """As the default, but with per-speaker audio instead of a floor mix.

        Overridden rather than hooked because the difference is which audio
        helper runs, and that is the whole point of this mode.
        """
        video_fields, skip_reason = resolve_video_media(record, ctx)
        if skip_reason is not None:
            return None, skip_reason

        reference_fields, skip_reason = self.resolve_reference_media(record, ctx)
        if skip_reason is not None:
            return None, skip_reason

        audio_fields, skip_reason = resolve_per_speaker_audio(record, ctx)
        if skip_reason is not None:
            return None, skip_reason

        item: dict[str, Any] = {
            "record_index": record_index,
            "record_id": str(record_id),
            **video_fields,
            **reference_fields,
            **audio_fields,
            "source_record": record,
        }
        item["user_prompt"] = self.render_user_prompt(
            ctx.user_prompt_template, record, item
        )
        return item, None

    def render_user_prompt(
        self,
        template: str,
        record: Mapping[str, Any],
        item: Mapping[str, Any],
    ) -> str:
        """As the default, plus the group ids the record does not carry as a scalar.

        ``conversation_floor`` is a list, and the flattener only keeps scalars,
        so ``{conversation_floor}`` would otherwise render as itself. ``{goa_ids}``
        and ``{other_ids}`` are the readable forms this mode's prompt wants.
        """
        goa_ids = item.get("goa_image_ids") or []
        participant = record.get("participant")
        other_ids = [i for i in goa_ids if i != participant]
        extra = {
            "goa_ids": ", ".join(str(i) for i in goa_ids),
            "other_ids": ", ".join(str(i) for i in other_ids),
            "goa_size": len(goa_ids),
        }
        return render_prompt(template, {**record, **extra})

    def build_turn(
        self,
        item: Mapping[str, Any],
        *,
        system_prompt: str,
        no_audio: bool,
        model,
        model_config,
    ) -> ChatMessage:
        content: list[MediaPart] = []

        preamble = self.prompt_config.get("preamble", DEFAULT_PREAMBLE)
        if isinstance(preamble, str) and preamble.strip():
            content.append(MediaPart.text_part(preamble.strip()))

        if not no_audio:
            for speaker_id in item["goa_speaker_ids"]:
                content.append(
                    MediaPart.text_part(
                        self._label("audio_label", DEFAULT_AUDIO_LABEL, speaker_id=speaker_id)
                    )
                )
                content.append(
                    MediaPart.audio(item["speaker_audio_paths"][str(speaker_id)])
                )

        for speaker_id in item["goa_image_ids"]:
            content.append(
                MediaPart.text_part(
                    self._label("image_label", DEFAULT_IMAGE_LABEL, speaker_id=speaker_id)
                )
            )
            content.append(MediaPart.image(item["speaker_image_paths"][str(speaker_id)]))

        video_label = self.prompt_config.get("video_label", DEFAULT_VIDEO_LABEL)
        if isinstance(video_label, str) and video_label.strip():
            content.append(MediaPart.text_part(video_label.strip()))
        content.append(self.video_part(item, model, model_config))

        # Last, and only here, is the person of interest named.
        content.append(self.text_part(item, system_prompt))
        return ChatMessage("user", content)

    def result_fields(self, item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "goa_speaker_ids": item["goa_speaker_ids"],
            "goa_image_ids": item["goa_image_ids"],
            "speaker_audio_paths": item["speaker_audio_paths"],
            "speaker_image_paths": item["speaker_image_paths"],
            "participant_image_path": item["participant_image_path"],
        }
