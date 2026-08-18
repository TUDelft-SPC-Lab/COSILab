"""FA -- flattened audio.

Nothing is stacked. Every person in the group of analysis gets their own audio
part and their own gallery image, and every one of those parts is introduced by
text naming whose it is.

The turn reads as one continuous instruction with media embedded in it: the clip
first, then the person of interest with their own voice and face, then each of
the others as an ``[id, audio, image]`` group, then the question. The person of
interest is named up front -- the model is told who to watch before it is shown
anyone else -- and the gallery images are what let it find those people in the
top-view video.

**Ids are bound by interleaved text, not by position.** A text fragment naming
the speaker precedes each media part, so the association survives regardless of
how a backend lays out its placeholders. This is the thing PA gets wrong -- its
"<audio1>"/"<audio2>" markers are inert text sitting after every media token --
and the reason not to copy PA's prompt shape here.

Every fragment lives in the prompt config rather than in this file, so the exact
wording of a run stays diffable, and ``_segments`` is the single place the order
is decided: ``build_turn`` and the prompt recorded in the results are both
rendered from it, so they cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ChatMessage, MediaPart, combine_system_and_user_prompt

from ..prompt import SafeFormatDict, render_prompt
from ..records import RecordContext, resolve_per_speaker_audio, resolve_video_media
from .base import BaseTaskMode

__all__ = ["FlattenedAudioMode"]

DEFAULT_VIDEO_INTRO = (
    "Watch the 30-second video clip and listen to the provided audio. "
    "The following is a video that shows the conversation from top view:"
)
DEFAULT_POA_AUDIO_INTRO = (
    "This is the participant that you would observe: participant {participant}, "
    "and this is their audio in the conversation:"
)
DEFAULT_POA_IMAGE_INTRO = "This is an image of them:"
DEFAULT_POA_INTRO_VIDEO_ONLY = (
    "This is the participant that you would observe: participant {participant}. "
    "This is an image of them:"
)
DEFAULT_OTHERS_INTRO = (
    "These are all other participants that are within the same conversation as "
    "participant {participant}:"
)
DEFAULT_OTHER_OPEN = "[participant {speaker_id}, audio:"
DEFAULT_OTHER_BETWEEN = ", image:"
DEFAULT_OTHER_CLOSE = "],"


class FlattenedAudioMode(BaseTaskMode):
    name = "fa"
    prompt_config_path = Path(__file__).resolve().with_name("prompt_fa.json")

    def _fragment(self, key: str, default: str, **values: Any) -> str:
        """One text fragment from the prompt config.

        Formatted the same forgiving way as the main template: an unknown
        placeholder is left as itself rather than raising, so a fragment someone
        edits mid-experiment cannot take down a queued job.
        """
        template = self.prompt_config.get(key, default)
        if not isinstance(template, str):
            raise ValueError(
                f"{self.prompt_config_path}: {key!r} must be a string, "
                f"got {type(template).__name__}."
            )
        return template.format_map(SafeFormatDict(values))

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
                # Carried explicitly rather than read back off the audio fields,
                # which are empty under --no-audio: who the question is about is
                # not a property of whether audio was loaded.
                "poa_id": participant,
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
        # The closing block is rendered first because the composed prompt embeds
        # it, and the composed prompt is what the result file records.
        item["closing_prompt"] = self._render_closing(
            ctx.user_prompt_template, record, item
        )
        item["user_prompt"] = self.render_user_prompt(
            ctx.user_prompt_template, record, item
        )
        return item, None

    def _render_closing(
        self,
        template: str,
        record: Mapping[str, Any],
        item: Mapping[str, Any],
    ) -> str:
        """The final block, with the group ids the record does not carry as scalars.

        ``conversation_floor`` is a list, and the flattener keeps only scalars,
        so ``{conversation_floor}`` would otherwise render as itself.
        """
        goa_ids = item.get("goa_image_ids") or []
        poa_id = item.get("poa_id")
        other_ids = [i for i in goa_ids if i != poa_id]
        extra = {
            "goa_ids": ", ".join(str(i) for i in goa_ids),
            "other_ids": ", ".join(str(i) for i in other_ids),
            "goa_size": len(goa_ids),
        }
        return render_prompt(template, {**record, **extra})

    def _segments(
        self,
        item: Mapping[str, Any],
    ) -> list[tuple[str, Any, str | None]]:
        """The turn as ordered ``(kind, value, marker)`` steps.

        The single source of the order. ``build_turn`` turns these into media
        parts and the recorded prompt turns them into text, so what a result file
        says was asked is what was asked.

        Whether audio is present is read off the item rather than passed in:
        ``--no-audio`` leaves ``speaker_audio_paths`` empty, and deriving it here
        means the two renderings cannot disagree about it.
        """
        poa_id = item["poa_id"]
        audio_paths = item.get("speaker_audio_paths") or {}
        image_paths = item["speaker_image_paths"]
        no_audio = not audio_paths
        other_ids = [i for i in item["goa_image_ids"] if i != poa_id]

        segments: list[tuple[str, Any, str | None]] = []

        def text(key: str, default: str, **values: Any) -> None:
            rendered = self._fragment(key, default, **values).strip()
            if rendered:
                segments.append(("text", rendered, None))

        text("video_intro", DEFAULT_VIDEO_INTRO, participant=poa_id)
        segments.append(("video", None, "<video>"))

        if no_audio:
            text(
                "poa_intro_video_only",
                DEFAULT_POA_INTRO_VIDEO_ONLY,
                participant=poa_id,
                speaker_id=poa_id,
            )
        else:
            text(
                "poa_audio_intro",
                DEFAULT_POA_AUDIO_INTRO,
                participant=poa_id,
                speaker_id=poa_id,
            )
            segments.append(("audio", audio_paths[str(poa_id)], f"<audio {poa_id}>"))
            text(
                "poa_image_intro",
                DEFAULT_POA_IMAGE_INTRO,
                participant=poa_id,
                speaker_id=poa_id,
            )
        segments.append(("image", image_paths[str(poa_id)], f"<gallery_image {poa_id}>"))

        if other_ids:
            text("others_intro", DEFAULT_OTHERS_INTRO, participant=poa_id)
            for speaker_id in other_ids:
                text(
                    "other_open",
                    DEFAULT_OTHER_OPEN,
                    participant=poa_id,
                    speaker_id=speaker_id,
                )
                if not no_audio:
                    segments.append(
                        ("audio", audio_paths[str(speaker_id)], f"<audio {speaker_id}>")
                    )
                    text(
                        "other_between",
                        DEFAULT_OTHER_BETWEEN,
                        participant=poa_id,
                        speaker_id=speaker_id,
                    )
                segments.append(
                    (
                        "image",
                        image_paths[str(speaker_id)],
                        f"<gallery_image {speaker_id}>",
                    )
                )
                text(
                    "other_close",
                    DEFAULT_OTHER_CLOSE,
                    participant=poa_id,
                    speaker_id=speaker_id,
                )

        segments.append(("text", item["closing_prompt"], None))
        return segments

    def render_user_prompt(
        self,
        template: str,
        record: Mapping[str, Any],
        item: Mapping[str, Any],
    ) -> str:
        """The whole prompt as prose, with each media part shown as a marker.

        What lands in the result file's ``user`` field. Unlike PA, where the
        prompt is one block, FA's is spread across a dozen text parts, so
        recording only the last of them would record almost nothing.
        """
        return " ".join(
            value if kind == "text" else str(marker)
            for kind, value, marker in self._segments(item)
        )

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
        system_prompt_pending = True

        for kind, value, _marker in self._segments(item):
            if kind == "text":
                text = value
                if system_prompt_pending:
                    # Folded into the opening instruction rather than the closing
                    # one: this prompt starts by telling the model what it is
                    # about to be shown, so the system prompt belongs there.
                    text = combine_system_and_user_prompt(
                        system_prompt=system_prompt, user_prompt=text
                    )
                    system_prompt_pending = False
                content.append(MediaPart.text_part(text))
            elif kind == "video":
                content.append(self.video_part(item, model, model_config))
            elif kind == "audio":
                content.append(MediaPart.audio(value))
            elif kind == "image":
                content.append(MediaPart.image(value))
            else:
                raise ValueError(f"Unknown segment kind: {kind!r}")

        return ChatMessage("user", content)

    # Reported for every record: who was asked about, who else was in the group,
    # and which file was whose. There is no mixed track and no
    # "conversation_floor_audio_paths" here -- nothing is stacked, so the id-to-
    # path maps are the whole story.
    RESULT_KEYS = (
        "poa_id",
        "participant_image_path",
        "goa_image_ids",
        "speaker_image_paths",
        "goa_speaker_ids",
        "participant_speaker_id",
        "conversation_floor_speaker_ids",
        "speaker_audio_paths",
        "participant_audio_path",
        "source_audio_paths",
        "rewritten_audio_paths",
        "audio_paths",
        "audio_warnings",
    )

    def result_fields(self, item: Mapping[str, Any]) -> dict[str, Any]:
        return {key: item[key] for key in self.RESULT_KEYS}
