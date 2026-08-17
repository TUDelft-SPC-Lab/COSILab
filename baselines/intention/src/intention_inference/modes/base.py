"""The interface every task mode implements.

A mode answers three questions the task itself has no opinion about:

    what grounds the question    resolve_reference_media()
    how the prompt is worded     render_user_prompt()
    what the model actually sees  build_turn()

``prepare_record`` below is the default composition -- clip, reference media,
audio, prompt -- and most modes will want it. It is a plain method rather than a
module function so that a mode whose shape does not fit can override it outright
and compose ``records.resolve_video_media`` / ``records.resolve_audio_bundle``
itself. That escape hatch is the point: the second mode's design is not settled,
and a seam that only bends where today's mode happens to bend is not a seam.

Standard library plus ``models.base`` only. A mode names media by path and never
decodes it, so importing this must not pull in torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ChatMessage, MediaPart, combine_system_and_user_prompt

from ..prompt import render_prompt
from ..records import RecordContext, resolve_audio_bundle, resolve_video_media

__all__ = ["BaseTaskMode"]


class BaseTaskMode(ABC):
    """One way of asking the intention question."""

    name: str = "base"
    # The prompt config this mode ships with. --prompt-config overrides it; the
    # mode owning a default is what lets the job script stop knowing where any
    # particular JSON lives.
    prompt_config_path: Path

    def prepare_record(
        self,
        record: dict[str, Any],
        *,
        record_index: int,
        record_id: Any,
        ctx: RecordContext,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Resolve one record to media on this filesystem, or say why not.

        Returns ``(item, None)`` or ``(None, skip_reason)``. The skip reason is
        counted and recorded by the caller rather than raised.
        """
        video_fields, skip_reason = resolve_video_media(record, ctx)
        if skip_reason is not None:
            return None, skip_reason

        reference_fields, skip_reason = self.resolve_reference_media(record, ctx)
        if skip_reason is not None:
            return None, skip_reason

        audio_fields, skip_reason = resolve_audio_bundle(
            record,
            record_index=record_index,
            record_id=record_id,
            ctx=ctx,
        )
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

    @abstractmethod
    def resolve_reference_media(
        self,
        record: Mapping[str, Any],
        ctx: RecordContext,
    ) -> tuple[dict[str, Any], str | None]:
        """Whatever grounds "which person are we talking about" for this mode.

        Returns the item fields it contributes, plus a skip reason when the
        record cannot be grounded.
        """

    def render_user_prompt(
        self,
        template: str,
        record: Mapping[str, Any],
        item: Mapping[str, Any],
    ) -> str:
        """Fill the mode's template. Default: the record's own fields.

        ``item`` is passed so a mode can render values it just resolved -- a
        speaker id, a gallery position -- that are not in the record verbatim.
        """
        return render_prompt(template, record)

    @abstractmethod
    def build_turn(
        self,
        item: Mapping[str, Any],
        *,
        system_prompt: str,
        no_audio: bool,
        model,
        model_config,
    ) -> ChatMessage:
        """The one user turn for this record.

        Part order is the order the model meets the placeholders in, so it is
        part of the prompt rather than a detail of how a list was built.
        """

    def result_fields(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """Mode-specific keys for the result row, beside the shared ones."""
        return {}

    def text_part(self, item: Mapping[str, Any], system_prompt: str) -> MediaPart:
        """The trailing text part, shared by every mode that has one."""
        return MediaPart.text_part(
            combine_system_and_user_prompt(
                system_prompt=system_prompt,
                user_prompt=item["user_prompt"],
            )
        )

    def video_part(self, item: Mapping[str, Any], model, model_config) -> MediaPart:
        """The clip, sampled by the run's frame policy rather than the backend's.

        Passing all three values means every backend sees the same frames of the
        same clip, so a difference between two models is the model.
        """
        return model.prepare_video_part(
            item["video_path"],
            model_config.max_video_frames,
            fps=model_config.video_fps,
            min_frames=model_config.min_video_frames,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
