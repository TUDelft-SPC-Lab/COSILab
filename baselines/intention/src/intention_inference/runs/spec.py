"""An externally authored persona x clip task spec, e.g. personas_prolific_ids.json.

The sibling of ``assignment.py``, for the runs the balanced design does not
cover. ``assignment.py`` reads what ``build_assignment.py`` wrote: a fixed
design, personas named ``persona_NNNN.txt`` under one directory, one prompt
config for the whole run, and clips addressed by their position in the manifest.

A task spec is the opposite end: someone else decides which personas run, what
each of them is asked, and which clips each one sees. Each entry names its own
persona text, optionally its own prompt config, and its clips by manifest record
id rather than by position -- ids survive a manifest that gets reordered or
appended to, which a spec written by hand weeks earlier does not otherwise.

What it deliberately does NOT change is the prompting. A run built here carries
the same system prompt fold as a persona run does -- persona first, then the
task prompt -- and the same ``render_prompt`` template. Only the parsing differs.

Two file shapes are accepted, and both normalise to the same internal task, so
everything downstream of :func:`load_task_spec` is unaware of which was used.

The full form, with relative paths resolved against the spec's own directory:

    {
      "__meta__": {"name": "pilot_alpha"},
      "tasks": [
        {
          "task_id": "alpha_p0007",
          "persona_prompt": "personas/alpha_0007.txt",
          "prompt_config": "prompts/prompt_alpha.json",
          "clips": ["vid3_seg12_p2", "vid3_seg44_p1"]
        }
      ]
    }

And the persona form -- a bare list, one entry per annotator, persona text
inline -- which is what the annotation pipeline emits:

    [
      {
        "ID": "5641193817bdbe00122a0f23",
        "persona_prompt": "You are a 34-year-old nurse ...",
        "videos": ["3", "12", "44"]
      }
    ]

``ID`` is the annotator's prolific id, and becomes the output name:
``persona_5641193817bdbe00122a0f23.json``, one file per persona.
``persona_prompt`` is the text folded in ahead of the task prompt -- inline
here, unlike the full form above, where the same key names a file.  ``videos``
are manifest record ids. An optional ``prompt_config`` is honoured here too,
though the persona form does not usually carry one.

Nothing heavier than the standard library is imported here, deliberately:
persona.sh reads the spec through :func:`load_task_spec` on the login
node, so that a malformed spec fails at submit time rather than after N jobs have
taken places in the queue -- and there is no container and no ML stack there.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from models.base import combine_system_and_user_prompt
from intention_inference.manifest import get_nested_value
from intention_inference.runs.base import (
    NothingToDo,
    PersonaRunsSelector,
    RunPlan,
    SummaryFacts,
    RunContext,
)
from intention_inference.prompt import load_prompt_config

__all__ = [
    "TASK_ID_PATTERN",
    "SpecRuns",
    "build_record_index_by_id",
    "build_spec_runs",
    "load_task_spec",
]


# The task id becomes a file name, so it may not carry a separator or anything
# else that would land the output outside --persona-output-dir.
TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _resolve_spec_path(raw: Any, *, spec_dir: Path, spec_path: Path, field: str, task_id: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{spec_path}: task {task_id!r} field {field!r} must be a non-empty string.")
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        # Against the spec's own directory rather than the working directory, so
        # a spec and the files it names travel together and mean the same thing
        # from the login node and from inside the container.
        path = spec_dir / path
    return path.resolve()


def _parse_clip_ids(
    raw_clips: Any, *, field: str, spec_path: Path, label: str
) -> list[str]:
    """Validate one task's clip list. Shared by both spec shapes.

    The two shapes name this field differently ('clips' and 'videos') but mean
    exactly the same thing, so the rules live here once.

    A repeated clip id is dropped, keeping the first occurrence, and reported on
    stderr. The alternative readings are worse: refusing the spec stops a whole
    sweep over a generator artefact, and honouring the repeat would run the same
    persona on the same clip twice and write both answers into one result file.
    """
    if not isinstance(raw_clips, Sequence) or isinstance(raw_clips, (str, bytes, bytearray)):
        raise ValueError(f"{spec_path}: {label} must map {field!r} to a list of record ids.")
    if not raw_clips:
        raise ValueError(f"{spec_path}: {label} has an empty {field!r} list.")
    clip_ids: list[str] = []
    seen: set[str] = set()
    duplicates: list[str] = []
    for clip_id in raw_clips:
        if isinstance(clip_id, bool) or not isinstance(clip_id, (str, int)):
            raise ValueError(
                f"{spec_path}: {label} has a clip id in {field!r} that is neither a string "
                f"nor an integer: {clip_id!r}"
            )
        text_id = str(clip_id)
        if text_id in seen:
            duplicates.append(text_id)
            continue
        seen.add(text_id)
        clip_ids.append(text_id)
    if duplicates:
        # stderr, so it survives the sweep's preflight -- which captures stdout to
        # read the task listing -- and lands in the job log next to the run it
        # describes.
        shown = sorted(set(duplicates))
        print(
            f"[WARN] {spec_path}: {label} lists {len(duplicates)} clip(s) more than once in "
            f"{field!r}; each is run once. Repeated: {shown[:10]}"
            f"{' ...' if len(shown) > 10 else ''}",
            file=sys.stderr,
        )
    return clip_ids


def _parse_full_task(
    raw_task: Mapping[str, Any], *, position: int, spec_dir: Path, spec_path: Path
) -> tuple[dict[str, Any], str]:
    """One entry of the ``{"tasks": [...]}`` form."""
    task_id = raw_task.get("task_id")
    if not isinstance(task_id, str) or not TASK_ID_PATTERN.match(task_id):
        raise ValueError(
            f"{spec_path}: tasks[{position}] needs a 'task_id' matching "
            f"{TASK_ID_PATTERN.pattern} -- it is used as the output file name. Got: {task_id!r}"
        )
    label = f"task {task_id!r}"

    persona_prompt = raw_task.get("persona_prompt")
    persona_text = raw_task.get("persona_text")
    if (persona_prompt is None) == (persona_text is None):
        raise ValueError(
            f"{spec_path}: {label} needs exactly one of 'persona_prompt' (a path) "
            "or 'persona_text' (inline)."
        )
    persona_path: Path | None = None
    if persona_prompt is not None:
        persona_path = _resolve_spec_path(
            persona_prompt,
            spec_dir=spec_dir,
            spec_path=spec_path,
            field="persona_prompt",
            task_id=task_id,
        )
    else:
        if not isinstance(persona_text, str) or not persona_text.strip():
            raise ValueError(
                f"{spec_path}: {label} field 'persona_text' must be a non-empty string."
            )
        persona_text = persona_text.strip()

    prompt_config_path: Path | None = None
    if raw_task.get("prompt_config") is not None:
        prompt_config_path = _resolve_spec_path(
            raw_task["prompt_config"],
            spec_dir=spec_dir,
            spec_path=spec_path,
            field="prompt_config",
            task_id=task_id,
        )

    return (
        {
            "task_id": task_id,
            "persona_path": persona_path,
            "persona_text": None if persona_path is not None else persona_text,
            "prompt_config_path": prompt_config_path,
            "clip_ids": _parse_clip_ids(
                raw_task.get("clips"), field="clips", spec_path=spec_path, label=label
            ),
        },
        label,
    )


def _parse_persona_task(
    raw_task: Mapping[str, Any], *, position: int, spec_dir: Path, spec_path: Path
) -> tuple[dict[str, Any], str]:
    """One entry of the bare-list persona form."""
    raw_id = raw_task.get("ID")
    if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
        raise ValueError(
            f"{spec_path}: entry [{position}] needs an 'ID' -- the annotator's prolific "
            f"id, which becomes the output file name. Got: {raw_id!r}"
        )
    persona_id = str(raw_id).strip()
    # It ends up in a file name, so it may not carry a separator or anything else
    # that would land the output outside --persona-output-dir. Prolific ids are
    # hex already; this catches a generator that put something else in the field.
    if not TASK_ID_PATTERN.match(persona_id):
        raise ValueError(
            f"{spec_path}: entry [{position}] has an 'ID' that cannot be a file name "
            f"(must match {TASK_ID_PATTERN.pattern}): {raw_id!r}"
        )

    # Prefixed to match what the balanced design writes, so one reader handles
    # persona_*.json from either kind of run.
    task_id = f"persona_{persona_id}"
    label = f"ID {persona_id!r}"

    persona = raw_task.get("persona_prompt")
    if not isinstance(persona, str) or not persona.strip():
        raise ValueError(
            f"{spec_path}: {label} has an empty or missing 'persona_prompt'. It is the "
            "persona prompt text, folded in ahead of the task prompt -- give it text, or "
            "drop the entry. An empty string is refused rather than silently run as a "
            "no-persona control, because in a persona spec it is almost always a "
            "generator bug."
        )

    prompt_config_path: Path | None = None
    if raw_task.get("prompt_config") is not None:
        prompt_config_path = _resolve_spec_path(
            raw_task["prompt_config"],
            spec_dir=spec_dir,
            spec_path=spec_path,
            field="prompt_config",
            task_id=task_id,
        )

    return (
        {
            "task_id": task_id,
            "persona_path": None,
            "persona_text": persona.strip(),
            "prompt_config_path": prompt_config_path,
            "clip_ids": _parse_clip_ids(
                raw_task.get("videos"), field="videos", spec_path=spec_path, label=label
            ),
        },
        label,
    )


def load_task_spec(spec_path: Path) -> dict[str, Any]:
    """Load and validate an external task spec, in either accepted shape.

    Returns ``{"name": str | None, "tasks": [...]}`` where each task is
    ``{"task_id", "persona_path", "persona_text", "prompt_config_path", "clip_ids"}``.
    Persona and prompt-config files are resolved but not read here; that happens
    in :func:`build_spec_runs`, so every path problem in the spec surfaces
    together rather than one file at a time.
    """
    with spec_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    # A bare list is the persona form; an object with 'tasks' is the full one.
    # Dispatch on the shape rather than on a version field: the two are already
    # unambiguous, and a spec nobody has to remember to label is one fewer thing
    # to get wrong.
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        raw_tasks: Sequence[Any] = payload
        parse_task = _parse_persona_task
        name: Any = None
        container = "the spec"
    elif isinstance(payload, Mapping):
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes, bytearray)):
            raise ValueError(f"{spec_path} must contain a 'tasks' array.")
        parse_task = _parse_full_task
        meta = payload.get("__meta__")
        name = meta.get("name") if isinstance(meta, Mapping) else None
        container = "'tasks'"
    else:
        raise ValueError(
            f"Task spec must be either a JSON array of persona entries "
            f"({{ID, persona_prompt, videos}}) or an object with a 'tasks' array: "
            f"{spec_path}"
        )

    if not raw_tasks:
        raise ValueError(f"{spec_path}: {container} is empty.")

    spec_dir = spec_path.parent
    tasks: list[dict[str, Any]] = []
    seen_task_ids: dict[str, tuple[int, str]] = {}

    for position, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, Mapping):
            raise ValueError(f"{spec_path}: entry [{position}] is not an object.")

        task, label = parse_task(
            raw_task, position=position, spec_dir=spec_dir, spec_path=spec_path
        )
        task_id = task["task_id"]
        if task_id in seen_task_ids:
            first_position, first_label = seen_task_ids[task_id]
            raise ValueError(
                f"{spec_path}: {label} at entry [{position}] and {first_label} at entry "
                f"[{first_position}] both write {task_id}.json; the second would overwrite "
                "the first."
            )
        seen_task_ids[task_id] = (position, label)
        tasks.append(task)

    return {"name": name if isinstance(name, str) else None, "tasks": tasks}




def build_record_index_by_id(records: Sequence[Mapping[str, Any]], id_key: str) -> dict[str, int]:
    """Map each manifest record's id to its position.

    A spec addresses clips by id; everything downstream -- record preparation,
    the audio-mix file names, the results -- is keyed by position, so this is the
    one translation between them. A duplicate id is refused rather than resolved
    to whichever came first: the spec would then silently mean a clip its author
    did not pick.
    """
    index_by_id: dict[str, int] = {}
    for position, record in enumerate(records):
        raw_id = get_nested_value(record, id_key)
        if raw_id is None:
            continue
        record_id = str(raw_id)
        if record_id in index_by_id:
            raise ValueError(
                f"Manifest has a duplicate {id_key}={record_id!r} at positions "
                f"{index_by_id[record_id]} and {position}; a task spec cannot address it "
                "unambiguously."
            )
        index_by_id[record_id] = position
    return index_by_id


def build_spec_runs(
    *,
    tasks: Sequence[Mapping[str, Any]],
    record_index_by_id: Mapping[str, int],
    persona_output_dir: Path,
    default_prompt_config: Mapping[str, str],
    default_prompt_config_path: Path,
    load_prompt_config: Callable[[Path], dict[str, str]],
) -> list[dict[str, Any]]:
    """Expand spec tasks into one inference run each.

    Yields the same run contract ``build_persona_runs`` does -- ``persona_path``,
    ``system_prompt``, ``output_path``, ``clip_indices`` -- so the caller's
    already-exists skip, record selection and inference loop are shared with the
    assignment path. The extra keys carry what is specific to a spec: the task
    id, the clip ids behind the indices, and the prompt config this task runs
    under, which may differ per task.

    Prompt configs are cached by path, so a spec where 500 tasks share two
    configs reads two files.
    """
    prompt_config_cache: dict[Path, dict[str, str]] = {}
    runs: list[dict[str, Any]] = []

    for task in tasks:
        task_id = task["task_id"]

        persona_path: Path | None = task["persona_path"]
        if persona_path is not None:
            if not persona_path.is_file():
                raise FileNotFoundError(
                    f"Task {task_id!r}: persona prompt not found: {persona_path}"
                )
            persona_text = persona_path.read_text(encoding="utf-8").strip()
        else:
            persona_text = task["persona_text"]

        prompt_config_path: Path | None = task["prompt_config_path"]
        if prompt_config_path is None:
            prompt_config = default_prompt_config
            effective_prompt_config_path = default_prompt_config_path
        else:
            if not prompt_config_path.is_file():
                raise FileNotFoundError(
                    f"Task {task_id!r}: prompt config not found: {prompt_config_path}"
                )
            if prompt_config_path not in prompt_config_cache:
                prompt_config_cache[prompt_config_path] = load_prompt_config(prompt_config_path)
            prompt_config = prompt_config_cache[prompt_config_path]
            effective_prompt_config_path = prompt_config_path

        clip_ids: list[str] = list(task["clip_ids"])
        unknown = [clip_id for clip_id in clip_ids if clip_id not in record_index_by_id]
        if unknown:
            # A hand-authored spec that names a clip the manifest does not have
            # is a mistake in the spec, not a clip that went bad: dropping it
            # quietly would produce a short result file nobody would question.
            raise KeyError(
                f"Task {task_id!r} references {len(unknown)} clip id(s) that are not in the "
                f"manifest: {unknown[:10]}"
            )
        clip_indices = [record_index_by_id[clip_id] for clip_id in clip_ids]

        runs.append(
            {
                "task_id": task_id,
                "persona_path": persona_path,
                # Persona first so it establishes the observer, then the task
                # prompt -- the same fold the assignment path uses.
                "system_prompt": combine_system_and_user_prompt(
                    system_prompt=persona_text,
                    user_prompt=prompt_config["system_prompt"],
                ),
                "user_prompt_template": prompt_config["user_prompt_template"],
                # The whole config, not just the template: a mode is handed its
                # prompt config through configure() and may keep more of it than
                # the user template -- fa reads its per-speaker labels from
                # there. A run that brought its own config has to be able to
                # re-configure the mode, so the dict travels with the run.
                "prompt_config": dict(prompt_config),
                "prompt_config_path": effective_prompt_config_path,
                "output_path": persona_output_dir / f"{task_id}.json",
                "clip_indices": clip_indices,
                "clip_ids": clip_ids,
            }
        )

    return runs


class SpecRuns(PersonaRunsSelector):
    """``--task-spec``, optionally sliced by ``--task-range``."""

    name = "task spec"

    def __init__(self) -> None:
        # Parsed in validate, used in build_plan.
        self.task_range: tuple[int, int] | None = None

    def validate(self, args: argparse.Namespace) -> None:
        # The spec owns both dimensions -- which personas run and which clips
        # each of them sees -- so it is exclusive with any range or cap that
        # would silently drop clips a task asked for.
        if args.persona_prompt:
            raise ValueError(
                "--task-spec cannot be combined with --persona-prompt; the spec names "
                "each task's persona itself."
            )
        if args.start_index != 0 or args.end_index is not None:
            raise ValueError(
                "--start-index/--end-index cannot be combined with --task-spec; "
                "the spec already selects the clips."
            )
        if args.limit is not None:
            raise ValueError("--limit cannot be combined with --task-spec.")
        if args.persona_output_dir is None:
            raise ValueError("--task-spec requires --persona-output-dir.")
        if not args.task_spec.expanduser().resolve().is_file():
            raise FileNotFoundError(
                f"Task spec not found: {args.task_spec.expanduser().resolve()}"
            )

        if args.task_range is not None:
            match = re.match(r"^([0-9]+)-([0-9]+)$", args.task_range)
            if match is None:
                raise ValueError(
                    f"--task-range must look like X-Y, for example 1-50: {args.task_range}"
                )
            task_start, task_end = int(match.group(1)), int(match.group(2))
            if task_start < 1:
                raise ValueError(
                    f"--task-range is 1-based, so X must be at least 1: {args.task_range}"
                )
            if task_end < task_start:
                raise ValueError(f"--task-range ends before it starts: {args.task_range}")
            self.task_range = (task_start, task_end)

    def build_plan(self, ctx: RunContext) -> RunPlan:
        spec_path = ctx.args.task_spec.expanduser().resolve()
        spec = load_task_spec(spec_path)
        spec_tasks = spec["tasks"]
        print(
            f"[INFO] Task spec: {spec_path} "
            f"(name={spec['name'] or '<unnamed>'}, {len(spec_tasks)} task(s))"
        )

        if self.task_range is not None:
            # 1-based and inclusive, so the slice reads the way the range is
            # written. Out of range is not an error: the last chunk of a tiled
            # submission is short by construction.
            spec_tasks = spec_tasks[self.task_range[0] - 1 : self.task_range[1]]
            print(
                f"[INFO] Task range: {self.task_range[0]}-{self.task_range[1]} "
                f"({len(spec_tasks)} task(s))"
            )
            if not spec_tasks:
                raise NothingToDo("No tasks in that range; nothing to do.")

        runs = build_spec_runs(
            tasks=spec_tasks,
            record_index_by_id=build_record_index_by_id(ctx.all_records, ctx.args.id_key),
            persona_output_dir=ctx.persona_output_dir,
            default_prompt_config={
                **ctx.prompt_config,
                # --system-prompt overrides the config's, and a task that brings
                # no config of its own should see the same prompt a persona-free
                # run would.
                "system_prompt": ctx.system_prompt,
            },
            default_prompt_config_path=ctx.prompt_config_path,
            load_prompt_config=load_prompt_config,
        )
        print(f"[INFO] Task output dir: {ctx.persona_output_dir}")

        return RunPlan(runs=runs, facts=SummaryFacts(task_spec=str(spec_path)))
