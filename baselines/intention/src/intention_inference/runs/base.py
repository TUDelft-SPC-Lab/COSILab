"""What every selector is handed, what every selector must answer, and what two of them share.

The contract is deliberately small: a selector reads its own JSON and says which runs
exist (:class:`RunPlan`), which manifest records those runs need, where the audio
mixes belong, and the handful of facts about itself that end up in the result
file (:class:`SummaryFacts`). Everything else -- resolving paths, preparing
records, loading the model, running the inference, writing the output -- is the
engine's, identical in all three selectors.

Both persona selectors answer the record-selection and audio-mix questions the same
way, so :class:`PersonaRunsSelector` answers them once and ``assignment.py`` and
``spec.py`` are left holding only what actually differs between the two.

Standard library only: ``spec`` is imported by a login-node preflight that has no
container and no ML stack, and it reaches this module.
"""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "NothingToDo",
    "PersonaRunsSelector",
    "RunPlan",
    "SummaryFacts",
    "RunContext",
    "RunSelector",
]


class NothingToDo(Exception):
    """There is legitimately nothing to run, and that is not an error.

    Carries the message to print, because the two cases say different things:
    an empty ``--task-range`` is the short last chunk of a tiled submission, and
    a fully finished set of outputs is a resumed sweep that has caught up.
    """


@dataclass(frozen=True)
class RunContext:
    """Everything resolved from the command line that a selector may read.

    Built once by the engine, before any selector runs. ``args`` is here so a selector can
    reach its own flags without the engine having to know which those are; treat
    it as read-only, and resist adding derived per-selector values to this class --
    that is how the selectors grew into each other in the first place.
    """

    args: argparse.Namespace
    input_json_path: Path
    # --output. Only the single-run selector writes there; the persona selectors name
    # their outputs after the persona or the task.
    output_path: Path
    prompt_config_path: Path
    prompt_config: dict[str, str]
    # The --system-prompt override already applied, so a selector never re-derives it.
    system_prompt: str
    persona_output_dir: Path | None
    all_records: list[dict[str, Any]]


@dataclass(frozen=True)
class SummaryFacts:
    """The only ``__summary__`` values that depend on which selector ran.

    Every field defaults to what a selector that does not own it reports, so a selector
    sets one or two and says nothing about the rest. The engine writes them into
    the summary literal, which keeps the key order fixed across all three selectors.
    """

    task_spec: str | None = None
    assignment_json: str | None = None
    assignment_seed: Any = None
    # None in the persona selectors: the clips come from a design or a spec, not from
    # a manifest range, and reporting 0 there would read as "started at the top".
    start_index: int | None = None
    end_index: int | None = None


@dataclass(frozen=True)
class RunPlan:
    """The runs a selector produced, plus what it wants recorded about itself.

    Each run is the dict contract the engine consumes: ``persona_path``,
    ``system_prompt``, ``output_path`` and ``clip_indices`` are required;
    ``task_id``, ``clip_ids``, ``user_prompt_template``, ``prompt_config`` and
    ``prompt_config_path`` are optional and read with ``.get()``.

    ``clip_indices`` is ``None`` in the single-run selector, which is how the
    engine tells "this run takes every prepared record" from "this run takes
    exactly these clips".
    """

    runs: list[dict[str, Any]]
    facts: SummaryFacts = field(default_factory=SummaryFacts)


class RunSelector(ABC):
    """One way of deciding which personas run over which clips."""

    #: How the selector names itself in log lines and error messages.
    name: str

    def validate(self, args: argparse.Namespace) -> None:
        """Refuse flag combinations this selector cannot honour.

        Runs before anything is loaded, so a contradictory command line costs a
        second rather than a place in the queue. Rules that need to see more than
        one selector live in ``select_run_plan`` instead.
        """

    @abstractmethod
    def build_plan(self, ctx: RunContext) -> RunPlan:
        """Read this selector's JSON and expand it into one run per persona.

        Raises :class:`NothingToDo` when the selector legitimately has no work.
        """

    @abstractmethod
    def select_records(
        self, ctx: RunContext, runs: Sequence[Mapping[str, Any]]
    ) -> list[tuple[int, dict[str, Any]]]:
        """Which manifest records the still-pending runs need, as (index, record).

        Called after finished runs have been dropped, so a resumed sweep prepares
        only what is left rather than re-mixing the audio of everything.
        """

    @abstractmethod
    def default_audio_mix_dir(self, ctx: RunContext) -> Path:
        """Where conversation-floor mixes go when --audio-mix-dir is absent."""


class PersonaRunsSelector(RunSelector):
    """The half the two persona selectors have in common.

    A balanced assignment and an external spec disagree about which file says who
    runs what, and about nothing else: both end up with per-persona runs carrying
    explicit clip indices, so both select records and cache audio mixes the same
    way.
    """

    def select_records(
        self, ctx: RunContext, runs: Sequence[Mapping[str, Any]]
    ) -> list[tuple[int, dict[str, Any]]]:
        needed_indices = sorted({i for run in runs for i in run["clip_indices"]})
        out_of_range = [i for i in needed_indices if not 0 <= i < len(ctx.all_records)]
        if out_of_range:
            raise ValueError(
                f"The {self.name} references manifest indices outside {ctx.input_json_path} "
                f"(0-{len(ctx.all_records) - 1}): {out_of_range[:10]}"
            )
        selected_records = [(i, ctx.all_records[i]) for i in needed_indices]
        print(
            f"[INFO] Selected {len(selected_records)} record(s) from the {self.name} "
            f"(union over {len(runs)} pending run(s))"
        )
        return selected_records

    def default_audio_mix_dir(self, ctx: RunContext) -> Path:
        # Persona-independent so the mixes are written once and reused across
        # every persona in the sweep, and across resumed jobs. Both selectors pick
        # clips globally rather than by range, so there is nothing to key on
        # beyond the manifest, and every job wants the same cache anyway.
        return ctx.persona_output_dir / "_audio_mixes" / ctx.input_json_path.stem
