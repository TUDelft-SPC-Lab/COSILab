"""The three ways a run can be specified, one module each.

NOT to be confused with ``intention_inference/modes/``, which is the other axis.
A run is decided along two of them, and they are orthogonal:

    modes/  pa | fa | sa            WHAT is asked -- how much of the group's
                                    audio is stacked, and how the prompt is worded
    runs/   single | assignment |   WHO asks it, over WHICH clips
            spec

Nothing here inspects the mode and nothing there inspects the selector. Because
"mode" already means the first of those, this package says *selector* for the
second throughout, even though the code it was ported from called both "mode".

``cli.py`` performs one job -- prepare clips, fold a system prompt in
front of each, run the model, write the answers -- and everything in this package
is about the one question it cannot answer itself: which personas run, and which
clips does each of them see.

    assignment.py  the balanced design built by build_assignment.py: personas
                   named persona_NNNN.txt, clips by manifest position.
    spec.py        an externally authored spec, e.g. personas_prolific_ids.json:
                   the annotation pipeline decides, clips by record id.
    single.py      no persona at all, a contiguous slice of the manifest.

The first two are nearly the same job read from two different files, so what they
share lives in ``base.py`` and each module here holds only its own differences:
how its JSON is read, which flags it accepts, and what it records about itself in
the result file.

Standard library only, so the login-node preflight in spec_sweep.sh can
import ``spec`` without a container.
"""

from __future__ import annotations

import argparse

from intention_inference.runs.assignment import AssignmentRuns
from intention_inference.runs.base import (
    NothingToDo,
    RunPlan,
    SummaryFacts,
    RunContext,
    RunSelector,
)
from intention_inference.runs.single import SingleRun
from intention_inference.runs.spec import SpecRuns

__all__ = [
    "AssignmentRuns",
    "NothingToDo",
    "RunPlan",
    "SingleRun",
    "SpecRuns",
    "SummaryFacts",
    "RunContext",
    "RunSelector",
    "select_run_plan",
]


def select_run_plan(args: argparse.Namespace) -> RunSelector:
    """Pick the selector the flags ask for, and refuse combinations that name two.

    Only the rules that need to see more than one selector live here; everything a
    selector can check on its own is in that selector's ``validate``.
    """
    if args.task_spec is not None:
        if args.assignment_json is not None:
            raise ValueError(
                "--task-spec cannot be combined with --assignment-json: they are two "
                "different ways of choosing the same thing. Pick one."
            )
        return SpecRuns()

    # Checked here rather than in SpecRuns so it is refused in every other selector
    # too, which is what the flag means: a range over tasks a spec does not
    # define is not a range at all.
    if args.task_range is not None:
        raise ValueError("--task-range requires --task-spec.")

    if args.assignment_json is not None:
        return AssignmentRuns()

    if args.persona_prompt:
        # Personas only ever run the balanced design built by
        # build_assignment.py. Running every persona over every clip is
        # not supported: it is the cross product the design exists to avoid.
        raise ValueError(
            "--persona-prompt requires --assignment-json. Build one with "
            "job_scripts/intention/build_assignment_daic.sh."
        )

    return SingleRun()
