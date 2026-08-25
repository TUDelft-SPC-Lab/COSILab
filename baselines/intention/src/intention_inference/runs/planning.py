"""Plan atomic persona runs into bounded inference jobs.

The shell launcher owns Slurm; this module owns the two JSON schemas it already
uses at inference time. Keeping source parsing here prevents the launcher from
growing a second, subtly different assignment/spec reader.

Output is tab-separated for a Bash caller:

    META   <source mode or empty>   <runs>   <answers>   <chunks>
    CHUNK  <first selector>         <last selector>      <answers>  <output stems,comma-separated>

Selectors are persona numbers for an assignment and 1-based task positions for
a spec. Runs remain whole, so no two jobs write the same persona output.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from intention_inference.runs.assignment import load_assignment
from intention_inference.runs.spec import load_task_spec


@dataclass(frozen=True)
class AtomicRun:
    selector: int
    output_stem: str
    answers: int


def parse_range(raw: str | None, *, label: str, minimum: int) -> tuple[int, int] | None:
    if raw is None:
        return None
    parts = raw.split("-")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(
            f"{label} must look like X-Y, for example {minimum}-{minimum + 9}: {raw}"
        )
    start, end = map(int, parts)
    if start < minimum:
        raise ValueError(f"{label} starts below {minimum}: {raw}")
    if end < start:
        raise ValueError(f"{label} ends before it starts: {raw}")
    return start, end


def assignment_runs(
    path: Path, requested: str | None
) -> tuple[str | None, list[AtomicRun]]:
    loaded = load_assignment(path)
    by_number: dict[int, Sequence[int]] = {}
    for key, clips in loaded["persona_to_clips"].items():
        if not key.isdigit():
            raise ValueError(f"{path}: persona key is not numeric: {key!r}")
        number = int(key)
        expected_key = f"{number:04d}"
        if key != expected_key:
            raise ValueError(
                f"{path}: persona key {key!r} must match persona_{expected_key}.txt"
            )
        if number in by_number:
            raise ValueError(f"{path}: duplicate numeric persona key: {key!r}")
        if not clips:
            raise ValueError(f"{path}: persona {key} has no clips")
        by_number[number] = clips

    numbers = sorted(by_number)
    selected_range = parse_range(requested, label="--persona-range", minimum=0)
    start, end = selected_range or (numbers[0], numbers[-1])
    selected = [number for number in numbers if start <= number <= end]
    expected = list(range(start, end + 1))
    if selected != expected:
        missing = sorted(set(expected) - set(selected))
        raise ValueError(
            f"{path}: persona range {start}-{end} is not contiguous; missing {missing[:10]}"
        )

    runs = [
        AtomicRun(
            selector=number,
            output_stem=f"persona_{number:04d}",
            answers=len(by_number[number]),
        )
        for number in selected
    ]
    mode = loaded["mode"] if isinstance(loaded["mode"], str) else None
    return mode, runs


def spec_runs(path: Path, requested: str | None) -> tuple[None, list[AtomicRun]]:
    tasks = load_task_spec(path)["tasks"]
    selected_range = parse_range(requested, label="--task-range", minimum=1)
    start, end = selected_range or (1, len(tasks))
    if start > len(tasks):
        raise ValueError(
            f"--task-range starts at {start}, but {path} has {len(tasks)} task(s)"
        )
    end = min(end, len(tasks))
    return None, [
        AtomicRun(
            selector=position,
            output_stem=tasks[position - 1]["task_id"],
            answers=len(tasks[position - 1]["clip_ids"]),
        )
        for position in range(start, end + 1)
    ]


def pack_runs(runs: Sequence[AtomicRun], answers_per_job: int) -> list[list[AtomicRun]]:
    if answers_per_job < 1:
        raise ValueError(f"--answers-per-job must be positive: {answers_per_job}")
    chunks: list[list[AtomicRun]] = []
    current: list[AtomicRun] = []
    current_answers = 0
    for run in runs:
        if current and current_answers + run.answers > answers_per_job:
            chunks.append(current)
            current = []
            current_answers = 0
        current.append(run)
        current_answers += run.answers
    if current:
        chunks.append(current)
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--assignment-json", type=Path)
    source.add_argument("--task-spec", type=Path)
    parser.add_argument("--persona-range")
    parser.add_argument("--task-range")
    parser.add_argument("--answers-per-job", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.assignment_json is not None:
        if args.task_range is not None:
            raise ValueError("--task-range requires --task-spec")
        mode, runs = assignment_runs(args.assignment_json.resolve(), args.persona_range)
    else:
        if args.persona_range is not None:
            raise ValueError("--persona-range requires --assignment-json")
        mode, runs = spec_runs(args.task_spec.resolve(), args.task_range)

    if not runs:
        raise ValueError("Persona source selected no runs")
    chunks = pack_runs(runs, args.answers_per_job)
    print(
        "\t".join(
            [
                "META",
                mode or "-",
                str(len(runs)),
                str(sum(run.answers for run in runs)),
                str(len(chunks)),
            ]
        )
    )
    for chunk in chunks:
        print(
            "\t".join(
                [
                    "CHUNK",
                    str(chunk[0].selector),
                    str(chunk[-1].selector),
                    str(sum(run.answers for run in chunk)),
                    ",".join(run.output_stem for run in chunk),
                ]
            )
        )


if __name__ == "__main__":
    main()
