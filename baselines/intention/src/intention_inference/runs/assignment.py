"""The balanced design: personas and clips fixed in advance by build_assignment.py.

This is the consumer side of ``build_assignment.py``: it loads the JSON that
file writes, maps ``persona_0007.txt`` to the clips persona 0007 is responsible
for, and turns a list of persona files into one inference run each.

What makes this selector itself, rather than the spec selector next door: the personas
are files named ``persona_NNNN.txt`` passed on the command line one flag at a
time, the clips are manifest *positions* chosen globally so every clip gets the
same number of observers, and neither can be varied per persona -- that balance
is the whole point of the design. A run that needs per-persona freedom is a task
spec, not an assignment.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from models.base import combine_system_and_user_prompt
from intention_inference.runs.base import (
    PersonaRunsSelector,
    RunPlan,
    SummaryFacts,
    RunContext,
)

__all__ = [
    "PERSONA_STEM_PATTERN",
    "AssignmentRuns",
    "build_persona_runs",
    "load_assignment",
    "persona_assignment_key",
]


PERSONA_STEM_PATTERN = re.compile(r"^persona_(\d+)$")


def load_assignment(assignment_json_path: Path) -> dict[str, Any]:
    """Load the persona -> clip-index assignment written by build_assignment.py."""
    with assignment_json_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError(f"Assignment JSON must be an object: {assignment_json_path}")

    persona_to_clips = payload.get("persona_to_clips")
    if not isinstance(persona_to_clips, Mapping) or not persona_to_clips:
        raise ValueError(
            f"{assignment_json_path} must contain a non-empty 'persona_to_clips' object."
        )

    normalized: dict[str, list[int]] = {}
    for persona_key, clip_indices in persona_to_clips.items():
        if not isinstance(clip_indices, Sequence) or isinstance(
            clip_indices, (str, bytes, bytearray)
        ):
            raise ValueError(
                f"{assignment_json_path}: persona {persona_key} must map to a list of ints."
            )
        indices: list[int] = []
        for clip_index in clip_indices:
            if not isinstance(clip_index, int) or isinstance(clip_index, bool):
                raise ValueError(
                    f"{assignment_json_path}: persona {persona_key} has a non-integer "
                    f"clip index: {clip_index!r}"
                )
            indices.append(clip_index)
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"{assignment_json_path}: persona {persona_key} has duplicate clip indices."
            )
        normalized[str(persona_key)] = sorted(indices)

    meta = payload.get("__meta__")
    return {
        "persona_to_clips": normalized,
        "seed": meta.get("seed") if isinstance(meta, Mapping) else None,
    }


def persona_assignment_key(persona_path: Path) -> str:
    """Map persona_0007.txt -> "0007", the key build_assignment.py writes."""
    match = PERSONA_STEM_PATTERN.match(persona_path.stem)
    if match is None:
        raise ValueError(
            f"Persona prompt {persona_path} does not follow the persona_<number>.txt naming "
            "required by --assignment-json."
        )
    return match.group(1)


def build_persona_runs(
    *,
    persona_prompt_paths: Sequence[Path],
    persona_output_dir: Path,
    base_system_prompt: str,
    assignment: Mapping[str, list[int]],
) -> list[dict[str, Any]]:
    """Expand the requested personas into one inference run each.

    Each run carries its own effective system prompt and output path, plus
    ``clip_indices``: the manifest indices that persona is responsible for under
    the balanced design. That set is fixed globally, so the output is a single
    ``persona_XXXX.json``.
    """
    runs: list[dict[str, Any]] = []
    for persona_prompt_path in persona_prompt_paths:
        persona_path = persona_prompt_path.expanduser().resolve()
        if not persona_path.is_file():
            raise FileNotFoundError(f"Persona prompt not found: {persona_path}")
        persona_text = persona_path.read_text(encoding="utf-8").strip()

        persona_key = persona_assignment_key(persona_path)
        if persona_key not in assignment:
            raise KeyError(
                f"Persona {persona_key} ({persona_path}) is not present in the assignment "
                "JSON. Rebuild the assignment with a --persona-range that covers it."
            )
        output_path = persona_output_dir / f"{persona_path.stem}.json"
        clip_indices = list(assignment[persona_key])

        runs.append(
            {
                "persona_path": persona_path,
                # Persona first so it establishes the observer, then the task prompt.
                "system_prompt": combine_system_and_user_prompt(
                    system_prompt=persona_text,
                    user_prompt=base_system_prompt,
                ),
                "output_path": output_path,
                "clip_indices": clip_indices,
            }
        )
    return runs


class AssignmentRuns(PersonaRunsSelector):
    """``--assignment-json`` plus one ``--persona-prompt`` per persona."""

    name = "assignment"

    def validate(self, args: argparse.Namespace) -> None:
        # The assignment owns the clip dimension; a range or cap on top of it
        # would silently drop assigned clips and break the 6-per-clip balance.
        if args.start_index != 0 or args.end_index is not None:
            raise ValueError(
                "--start-index/--end-index cannot be combined with --assignment-json; "
                "the assignment already selects the clips."
            )
        if args.limit is not None:
            raise ValueError("--limit cannot be combined with --assignment-json.")
        if not args.persona_prompt:
            raise ValueError("--assignment-json requires at least one --persona-prompt.")
        if not args.assignment_json.expanduser().resolve().is_file():
            raise FileNotFoundError(
                f"Assignment JSON not found: {args.assignment_json.expanduser().resolve()}"
            )
        if args.persona_output_dir is None:
            raise ValueError("--persona-output-dir is required when --persona-prompt is given.")

    def build_plan(self, ctx: RunContext) -> RunPlan:
        assignment_json_path = ctx.args.assignment_json.expanduser().resolve()
        loaded = load_assignment(assignment_json_path)
        print(f"[INFO] Assignment JSON: {assignment_json_path} (seed={loaded['seed']})")

        runs = build_persona_runs(
            persona_prompt_paths=ctx.args.persona_prompt,
            persona_output_dir=ctx.persona_output_dir,
            base_system_prompt=ctx.system_prompt,
            assignment=loaded["persona_to_clips"],
        )
        print(f"[INFO] Persona output dir: {ctx.persona_output_dir}")
        print(f"[INFO] Personas requested: {len(runs)}")

        return RunPlan(
            runs=runs,
            facts=SummaryFacts(
                assignment_json=str(assignment_json_path),
                assignment_seed=loaded["seed"],
            ),
        )

