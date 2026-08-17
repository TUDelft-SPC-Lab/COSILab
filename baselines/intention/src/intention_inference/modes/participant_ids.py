"""PLACEHOLDER: grounding the question by participant id rather than by face.

The idea is that people are named to the model instead of merely shown to it --
"the indicated participant is 3, the others on the floor are 1 and 4" -- which
needs some visible carrier for those numbers: an id-labelled gallery of all
participants, or a clip with the ids drawn on, or both. None of that is decided,
so nothing here is implemented and selecting this mode fails immediately.

It is registered in ``modes/registry.py`` anyway, so ``--mode participant_ids``
gets the message below instead of an argparse "invalid choice" that reads like a
typo.

To finish it:

  1. Decide what carries the ids and write it into resolve_reference_media().
     ``ctx`` already carries participant_image_root and every media prefix; a new
     input root means a new field on ``records.RecordContext``.
  2. Write prompt_participant_ids.json. Unlike the participant_image prompt, it
     can name people: ``{participant}`` and ``{conversation_floor}`` are on the
     record, so ``render_prompt`` fills them with no extra work. Override
     render_user_prompt() only for values that are not on the record verbatim --
     ``item`` is passed to it for exactly that.
  3. Lay out build_turn(). Part order is what the model reads, so decide it
     against the prompt's wording rather than copying participant_image's.
  4. Have result_fields() return whatever a result file needs to be reproducible
     -- at minimum the ids that were shown.

Note the flaw worth not reproducing: the participant_image prompt refers to
"<image>", "<audio1>" and "<audio2>" as literal text, but the real media tokens
are emitted at the positions of the parts, all of them ahead of the text block.
Those markers resolve to nothing and the binding survives on part order alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ChatMessage

from ..records import RecordContext
from .base import BaseTaskMode

__all__ = ["ParticipantIdsMode"]

_UNFINISHED = (
    "The 'participant_ids' task mode is a placeholder and cannot run yet.\n"
    "        Decide what carries the ids, then fill in, in "
    "intention_inference/modes/participant_ids.py:\n"
    "          - resolve_reference_media()  what grounds the ids visually\n"
    "          - render_user_prompt()       only if ids are not on the record verbatim\n"
    "          - build_turn()               the part order the prompt describes\n"
    "          - result_fields()            what a result file needs to be reproducible\n"
    "        and write modes/prompt_participant_ids.json.\n"
    "        Until then, run with --mode participant_image."
)


class ParticipantIdsMode(BaseTaskMode):
    name = "participant_ids"
    prompt_config_path = Path(__file__).resolve().with_name("prompt_participant_ids.json")

    def __init__(self) -> None:
        # Raised at construction, which cli.main does before it reads the model
        # config or the manifest -- so this costs a second, not a queued job that
        # dies after loading weights.
        raise NotImplementedError(_UNFINISHED)

    def resolve_reference_media(
        self,
        record: Mapping[str, Any],
        ctx: RecordContext,
    ) -> tuple[dict[str, Any], str | None]:
        raise NotImplementedError(_UNFINISHED)

    def build_turn(
        self,
        item: Mapping[str, Any],
        *,
        system_prompt: str,
        no_audio: bool,
        model,
        model_config,
    ) -> ChatMessage:
        raise NotImplementedError(_UNFINISHED)
