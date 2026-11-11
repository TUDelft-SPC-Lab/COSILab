#!/usr/bin/env python3
"""Derive the blind control's task spec and sampling model config from a persona source.

A blind turn names no clip (modes/blind.py), so every clip a persona was
assigned asks it the same question. Running the source's full clip list would
pay for N identical prompts and file the draws under N clip ids that mean
nothing. This writes the shape the question actually has instead: one task per
persona, ONE clip each, and the repeats come from ``--trials`` on that clip.

The clip is still needed, though never shown -- the worker resolves a record to
build a row, and a clip whose video is missing is skipped, which here would cost
the persona every one of its trials. So the first clip in the persona's own list
whose video resolves is the one kept, checked with the same resolver inference
uses. A persona with no such clip is dropped and reported, not run empty.

Persona text is always inlined. A spec's paths are resolved inside the worker's
container, where only /tudelft.net keeps its host path; text cannot go stale.

A task's own ``prompt_config`` is dropped: blind asks its own prompt config's
question, and a task that kept another would quietly not be blind.

Writes two files and prints nothing but a one-line summary on stderr, so the
launcher can plan packs with runs/planning.py exactly as for any other spec.

Standard library only: persona_blind.sh runs this on the login node.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from intention_inference.build_sweep import (  # noqa: E402
    DEFAULT_MODEL_CONFIG_PATH,
    build_sampling_model_config,
)
from intention_inference.manifest import get_nested_value, load_manifest_records  # noqa: E402
from intention_inference.records import RecordContext, resolve_video_media  # noqa: E402
from intention_inference.runs.assignment import load_assignment  # noqa: E402
from intention_inference.runs.spec import build_record_index_by_id, load_task_spec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--assignment-json", type=Path)
    source.add_argument("--task-spec", type=Path)
    parser.add_argument(
        "--persona-dir",
        type=Path,
        help="Assignment only: where persona_NNNN.txt live.",
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Clip manifest.")
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--video-media-path-prefix", required=True)
    parser.add_argument("--video-local-path-prefix", type=Path, required=True)
    parser.add_argument("--base-model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--trials", type=int, required=True, help="Recorded in the spec only.")
    parser.add_argument("--spec-output", type=Path, required=True)
    parser.add_argument("--model-config-output", type=Path, required=True)
    return parser.parse_args()


def video_context(args: argparse.Namespace) -> RecordContext:
    """Enough of a RecordContext for resolve_video_media; the rest is unused."""
    return RecordContext(
        id_key=args.id_key,
        manifest_dir=args.input_json.resolve().parent,
        media_root=None,
        media_path_prefix=None,
        local_path_prefix=None,
        video_media_path_prefix=args.video_media_path_prefix,
        video_local_path_prefix=args.video_local_path_prefix,
        audio_media_path_prefix=None,
        audio_local_path_prefix=None,
        participant_image_root=Path("/nonexistent"),
        no_audio=True,
        aggregated_audio_dir=Path("/nonexistent"),
        exclude_video_substrings=[],
        exclude_audio_substrings=[],
        user_prompt_template="",
    )


def assignment_personas(args: argparse.Namespace) -> list[dict[str, Any]]:
    """(task_id, persona text, candidate clip indices) per assignment persona."""
    if args.persona_dir is None:
        raise ValueError("--assignment-json needs --persona-dir.")
    loaded = load_assignment(args.assignment_json.resolve())
    personas = []
    for key, clip_indices in sorted(loaded["persona_to_clips"].items()):
        persona_path = args.persona_dir / f"persona_{key}.txt"
        if not persona_path.is_file():
            raise FileNotFoundError(f"Persona prompt not found: {persona_path}")
        personas.append(
            {
                "task_id": f"persona_{key}",
                "persona_text": persona_path.read_text(encoding="utf-8").strip(),
                "clip_indices": list(clip_indices),
                "dropped_prompt_config": False,
            }
        )
    return personas


def spec_personas(
    args: argparse.Namespace, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The same, for either shape of task spec."""
    index_by_id = build_record_index_by_id(records, args.id_key)
    personas = []
    for task in load_task_spec(args.task_spec.resolve())["tasks"]:
        if task["persona_path"] is not None:
            persona_text = task["persona_path"].read_text(encoding="utf-8").strip()
        else:
            persona_text = task["persona_text"]
        unknown = [clip_id for clip_id in task["clip_ids"] if clip_id not in index_by_id]
        if unknown:
            raise KeyError(
                f"Task {task['task_id']!r} references clip id(s) not in the manifest: "
                f"{unknown[:10]}"
            )
        personas.append(
            {
                "task_id": task["task_id"],
                "persona_text": persona_text,
                "clip_indices": [index_by_id[clip_id] for clip_id in task["clip_ids"]],
                "dropped_prompt_config": task["prompt_config_path"] is not None,
            }
        )
    return personas


def first_usable_clip_id(
    clip_indices: list[int],
    records: list[dict[str, Any]],
    ctx: RecordContext,
) -> str | None:
    for record_index in clip_indices:
        record = records[record_index]
        record_id = get_nested_value(record, ctx.id_key)
        if record_id is None:
            continue
        _, skip_reason = resolve_video_media(record, ctx)
        if skip_reason is None:
            return str(record_id)
    return None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError(f"--trials must be at least 1: {args.trials}")

    records = load_manifest_records(args.input_json.resolve())
    if args.assignment_json is not None:
        source = args.assignment_json.resolve()
        personas = assignment_personas(args)
    else:
        source = args.task_spec.resolve()
        personas = spec_personas(args, records)

    ctx = video_context(args)
    tasks: list[dict[str, Any]] = []
    dropped: list[str] = []
    prompt_configs_dropped: list[str] = []
    for persona in personas:
        clip_id = first_usable_clip_id(persona["clip_indices"], records, ctx)
        if clip_id is None:
            dropped.append(persona["task_id"])
            continue
        if persona["dropped_prompt_config"]:
            prompt_configs_dropped.append(persona["task_id"])
        tasks.append(
            {
                "task_id": persona["task_id"],
                "persona_text": persona["persona_text"],
                "clips": [clip_id],
            }
        )

    if dropped:
        print(
            f"[WARN] {len(dropped)} persona(s) have no clip whose video resolves and are "
            f"not run: {dropped[:10]}{' ...' if len(dropped) > 10 else ''}",
            file=sys.stderr,
        )
    if prompt_configs_dropped:
        print(
            f"[WARN] {len(prompt_configs_dropped)} task(s) named their own prompt_config; "
            "blind asks its own question, so it was dropped.",
            file=sys.stderr,
        )
    if not tasks:
        raise ValueError(f"No persona in {source} has a usable clip; nothing to run.")

    model_config = build_sampling_model_config(
        base_path=args.base_model_config.resolve(),
        temperature=args.temperature,
        top_p=None,
    )
    # The helper names build_sweep.py as the generator; this file is ours.
    model_config["_comment"][0] = (
        "GENERATED by intention_inference/build_blind_spec.py. Edit that script, or the"
    )
    write_json(args.model_config_output, model_config)
    write_json(
        args.spec_output,
        {
            "__meta__": {
                "name": f"blind_{source.stem}",
                "source": str(source),
                "temperature": args.temperature,
                "trials": args.trials,
                "dropped_personas": dropped,
            },
            "tasks": tasks,
        },
    )
    print(
        f"[INFO] Blind spec: {len(tasks)} persona(s), one clip each, from {source}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
