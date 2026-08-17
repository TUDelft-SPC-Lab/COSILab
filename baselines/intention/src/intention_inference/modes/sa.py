"""SA -- stacked audio. PLACEHOLDER, not implemented.

The far end of the audio axis: *everyone* in the group, the person of interest
included, summed into a single soundtrack. One audio part regardless of group
size, so no voice can be attributed to anyone from the audio alone. Grounding
stays a single crop of the person of interest, as in PA.

    SA   1 audio part    everyone stacked
    PA   2 audio parts   person of interest, and everyone else stacked
    FA   N audio parts   one per person, ids attached

Selecting it raises, so it fails in a second rather than after weights load.

To finish it, the shape is PA with one line moved:

  1. ``resolve_reference_media`` is PA's verbatim -- a single
     ``participant_<n>.png``. Subclass ``PartiallyStackedAudioMode`` or copy the
     six lines; do not invent a third way of finding that file.
  2. The audio is the only real work. ``records.resolve_speaker_tracks`` already
     returns every speaker's own track plus ``goa_ids``, so what is missing is a
     mix over the *whole* group rather than over the floor.
     ``audio.aggregate_conversation_floor_audio`` takes the participant's track
     as the length reference and mixes only the paths it is handed, so passing
     it ``[participant, *floor]`` produces exactly this -- but its parameter
     names, its output filename (``..._conversation_floor.wav``) and its warning
     text all say "conversation floor". Generalise those names when you use it;
     do not leave SA writing files that claim to be something else.
  3. ``build_turn`` is PA's with one audio part instead of two.
  4. Write ``prompt_sa.json``. It cannot reuse PA's wording: PA's prompt tells
     the model that audio 1 is the participant's own voice, which is precisely
     the information SA withholds. The prompt has to say that the soundtrack is
     everyone at once and that voices are not separable.
  5. Decide whether the participant's own track is mixed at unit gain like the
     others. It is one voice among N here, unlike PA where it stands alone, and
     that is a real experimental choice rather than an implementation detail.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from models.base import ChatMessage

from ..records import RecordContext
from .base import BaseTaskMode

__all__ = ["StackedAudioMode"]

_UNFINISHED = (
    "The 'sa' (stacked audio) task mode is a placeholder and cannot run yet.\n"
    "        It needs, in intention_inference/modes/sa.py:\n"
    "          - resolve_reference_media()  PA's single participant crop, unchanged\n"
    "          - build_turn()               PA's layout with ONE audio part\n"
    "          - a whole-group mix          records.resolve_speaker_tracks gives every\n"
    "                                       track; aggregate_conversation_floor_audio\n"
    "                                       mixes what it is handed, but its names and\n"
    "                                       output filename say 'conversation floor'\n"
    "                                       and must be generalised first\n"
    "        and modes/prompt_sa.json, which cannot reuse PA's wording: PA tells the\n"
    "        model that audio 1 is the participant's own voice, and that is exactly\n"
    "        what SA withholds.\n"
    "        Until then, run with --mode pa or --mode fa."
)


class StackedAudioMode(BaseTaskMode):
    name = "sa"
    prompt_config_path = Path(__file__).resolve().with_name("prompt_sa.json")

    def __init__(self) -> None:
        # Raised at construction, which cli.main does before it reads the model
        # config or the manifest.
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
