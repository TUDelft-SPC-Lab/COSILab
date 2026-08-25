"""No persona at all: one run over a contiguous slice of the manifest.

The selector that existed before personas did, and still the one to use for a plain
sweep of the model over clips: no persona is folded in, the clips are whatever
``--start-index``/``--end-index``/``--limit`` select, and everything lands in the
single file ``--output`` names. Driven by ``cosilab_daic.sh``.

The other two selectors refuse those three flags, because there the clips are chosen
by a design or a spec and a range on top could only silently drop some of them.
Here they are the entire selection.

Standard library only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from intention_inference.runs.base import (
    RunPlan,
    SummaryFacts,
    RunContext,
    RunSelector,
)

__all__ = ["SingleRun", "build_single_run"]


def build_single_run(*, system_prompt: str, output_path: Path) -> list[dict[str, Any]]:
    """The one run this selector has: no persona, no per-run clip list.

    ``clip_indices`` is None rather than a list, which is how the engine tells
    "this run takes every record that was prepared" from "this run takes exactly
    these clips".
    """
    return [
        {
            "persona_path": None,
            "system_prompt": system_prompt,
            "output_path": output_path,
            "clip_indices": None,
        }
    ]


class SingleRun(RunSelector):
    """The default: no ``--assignment-json``, no ``--task-spec``, no persona."""

    name = "single run"

    def build_plan(self, ctx: RunContext) -> RunPlan:
        end_index = (
            len(ctx.all_records) - 1 if ctx.args.end_index is None else ctx.args.end_index
        )
        return RunPlan(
            runs=build_single_run(
                system_prompt=ctx.system_prompt,
                output_path=ctx.output_path,
            ),
            facts=SummaryFacts(start_index=ctx.args.start_index, end_index=end_index),
        )

    def select_records(
        self, ctx: RunContext, runs: Sequence[Mapping[str, Any]]
    ) -> list[tuple[int, dict[str, Any]]]:
        end_index = (
            len(ctx.all_records) - 1 if ctx.args.end_index is None else ctx.args.end_index
        )
        selected_records = [
            (record_index, record)
            for record_index, record in enumerate(ctx.all_records)
            if ctx.args.start_index <= record_index <= end_index
        ]
        print(
            f"[INFO] Selected manifest index range: "
            f"{ctx.args.start_index}-{end_index} ({len(selected_records)} record(s))"
        )
        return selected_records

    def default_audio_mix_dir(self, ctx: RunContext) -> Path:
        # Keyed on the manifest, exactly like the persona selectors, and NOT on the
        # output stem. A mix depends only on the record's own audio, so keying it
        # on the output name -- which carries the clip range -- gave every chunk
        # of a tiled submission its own private copy of clips it shares with its
        # neighbours: _audio_mixes/annotation_clips_0-250,
        # _audio_mixes/annotation_clips_0-700 and so on, each re-mixing the same
        # overlapping records. One directory per manifest instead, which the file
        # names already support: they carry the global manifest index and the
        # record id, so they are the same in every run.
        return ctx.output_path.parent / "_audio_mixes" / ctx.input_json_path.stem
