"""Everything the task does once the command line has been read.

Split from ``cli.py`` so that file holds only the argument surface and the order
of operations. The order matters more than it looks: preparing records is cheap
and fails for boring reasons -- a manifest path that does not resolve, a clip
that is not on this share -- so it all happens before the weights are loaded.

Neither of the two things that vary is decided here. The model is reached only
through ``models.base.BaseMultimodalModel``, and what is asked of it only through
``modes.base.BaseTaskMode``: this module drives the loop, counts what was
skipped, and writes the answers down. A turn is a list of
:class:`~models.base.MediaPart` values built by the mode, and what becomes of
them is the backend's business.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.config import ModelRunConfig, describe_model_config

from .manifest import get_nested_value
from .records import RecordContext
from .runs.base import NothingToDo, RunPlan, SummaryFacts

__all__ = [
    "PreparedRecords",
    "build_summary",
    "drop_finished_runs",
    "execute_runs",
    "prepare_records",
    "run_inference_pass",
    "write_results",
]


@dataclass
class PreparedRecords:
    """What survived preparation, and an account of what did not."""

    kept: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    skip_counter: Counter[str] = field(default_factory=Counter)


def prepare_records(
    *,
    mode,
    ctx: RecordContext,
    selected_records: list[tuple[int, dict[str, Any]]],
    limit: int | None,
) -> PreparedRecords:
    """Resolve every selected record to media that exists on this filesystem.

    A record that cannot be resolved is counted and recorded rather than raised:
    one clip missing from a share should not take down a shard of several
    hundred. The counts are what the summary reports, and what tells you
    afterwards whether a run was thin because the data was thin or because the
    share was not mounted.
    """
    prepared = PreparedRecords()

    for record_index, record in selected_records:
        record_id = get_nested_value(record, ctx.id_key)
        if record_id is None:
            record_id = record.get("id", record_index)

        record_result, skip_reason = mode.prepare_record(
            record,
            record_index=record_index,
            record_id=record_id,
            ctx=ctx,
        )
        if record_result is None:
            skip_reason = skip_reason or "filtered_out"
            prepared.skip_counter[skip_reason] += 1
            prepared.skipped.append(
                {
                    "record_index": record_index,
                    "record_id": str(record_id),
                    "skip_reason": skip_reason,
                }
            )
            continue

        prepared.kept.append(record_result)
        if limit is not None and len(prepared.kept) >= limit:
            break

    print(
        f"[INFO] Retained {len(prepared.kept)} record(s); "
        f"skipped {sum(prepared.skip_counter.values())} before inference."
    )
    for reason, count in sorted(prepared.skip_counter.items()):
        print(f"[INFO]   skip {reason}: {count}")

    return prepared


def drop_finished_runs(plan: RunPlan, *, overwrite: bool) -> RunPlan:
    """Drop runs whose output is already on disk.

    Before record preparation and the model load, so resubmitting a timed-out
    sweep resumes instead of restarting, and before record selection, so a resume
    prepares only the clips the remaining runs actually need rather than
    re-mixing the audio of everything.
    """
    if not overwrite:
        pending_runs: list[dict[str, Any]] = []
        for run in plan.runs:
            if run["output_path"].is_file():
                print(f"[INFO] Skipping {run['output_path'].name}: output already exists")
                continue
            pending_runs.append(run)
        plan = RunPlan(runs=pending_runs, facts=plan.facts)
    if not plan.runs:
        raise NothingToDo("Every requested output already exists. Pass --overwrite to redo them.")
    return plan


def run_inference_pass(
    *,
    records: list[dict[str, Any]],
    mode,
    system_prompt: str,
    no_audio: bool,
    model,
    model_config: ModelRunConfig,
) -> list[dict[str, Any]]:
    """One turn per retained record, in manifest order."""
    results: list[dict[str, Any]] = []
    total_records = len(records)

    for processed_index, item in enumerate(records, start=1):
        print(
            f"[{processed_index}/{total_records}] "
            f"{item['record_id']} -> {Path(item['video_path']).name}",
            flush=True,
        )
        try:
            turn = mode.build_turn(
                item,
                system_prompt=system_prompt,
                no_audio=no_audio,
                model=model,
                model_config=model_config,
            )
            response = model.generate([turn], model_config.generation)
        except Exception as exc:
            # Recorded, not raised: a clip the model chokes on should cost that
            # clip, not the rest of the shard. build_turn is inside the try for
            # the same reason -- decoding a corrupt video fails here too.
            response = f"[ERROR] {exc}"
            print(f"  [WARN] Error: {exc}", flush=True)

        # Everything past the clip is the mode's to report. This module must not
        # name an audio field: how many audio parts there are, and whether any of
        # them is a mix, is precisely what differs between modes -- naming
        # "speaker_ids" here is what made a flattened-audio run die after it had
        # already paid for the inference.
        try:
            mode_fields = mode.result_fields(item)
        except Exception as exc:
            # Reported, not raised, for the same reason a bad clip is: a mode's
            # bookkeeping bug should cost one record's metadata, not a shard of
            # several hundred that has already been generated.
            print(f"  [WARN] {mode.name}.result_fields failed: {exc}", flush=True)
            mode_fields = {"result_fields_error": f"{type(exc).__name__}: {exc}"}

        results.append(
            {
                "record_index": item["record_index"],
                "record_id": item["record_id"],
                "source_video_path": item["source_video_path"],
                "rewritten_video_path": item["rewritten_video_path"],
                "video_path": item["video_path"],
                **mode_fields,
                "system": system_prompt,
                "user": item["user_prompt"],
                "assistant": response,
            }
        )

    return results


def build_summary(
    *,
    args,
    backend: str,
    mode: str,
    model_config: ModelRunConfig,
    input_json_path: Path,
    prompt_config_path: Path,
    record_count: int,
    selected_record_count: int,
    prepared: PreparedRecords,
    results: list[dict[str, Any]],
    aggregated_audio_dir: Path,
    facts: SummaryFacts,
    run: Mapping[str, Any],
    missing_clip_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """What this run was, recorded beside its answers.

    ``describe_model_config`` contributes the weights, the decoding parameters
    and the frame policy, so a result file answers "what produced this" without
    anyone having to find the sbatch line that made it.

    One literal with a fixed key order for every selector: a field a selector
    does not own is written as None rather than omitted, so two result files can
    be diffed without the keys shifting under each other.
    """
    error_count = sum(1 for item in results if str(item["assistant"]).startswith("[ERROR]"))
    persona_path = run.get("persona_path")
    clip_indices = run.get("clip_indices")
    return {
        "input_json": str(input_json_path),
        "prompt_config": str(run.get("prompt_config_path") or prompt_config_path),
        "backend": backend,
        "mode": mode,
        "persona_prompt": None if persona_path is None else str(persona_path),
        "task_spec": facts.task_spec,
        "task_id": run.get("task_id"),
        "assignment_json": facts.assignment_json,
        "assignment_seed": facts.assignment_seed,
        "record_count": record_count,
        "selected_record_count": selected_record_count,
        # None in the persona selectors: the clips come from a design or a spec,
        # not from a manifest range, and reporting 0 would read as "from the top".
        "start_index": facts.start_index,
        "end_index": facts.end_index,
        "assigned_clip_indices": None if clip_indices is None else list(clip_indices),
        "assigned_clip_ids": run.get("clip_ids"),
        # Clips this run was assigned that failed preparation. Non-empty means the
        # design is short here, which is worth seeing rather than inferring from a
        # count that does not add up.
        "missing_clip_indices": list(missing_clip_indices),
        "retained_count": len(prepared.kept),
        "skipped_count": len(prepared.skipped),
        "processed_count": len(results),
        "error_count": error_count,
        "no_audio": args.no_audio,
        **describe_model_config(model_config),
        # Reported only if something was actually written there. Whether a mode
        # stacks any audio is the mode's business, and a summary that names a
        # directory the run never created is a summary that misleads: fa mixes
        # nothing, so it has no mixes directory.
        "aggregated_audio_dir": (
            str(aggregated_audio_dir) if aggregated_audio_dir.is_dir() else None
        ),
        "skip_reasons": dict(sorted(prepared.skip_counter.items())),
    }


def execute_runs(
    *,
    args,
    backend: str,
    mode,
    plan: RunPlan,
    prepared: PreparedRecords,
    model,
    model_config: ModelRunConfig,
    input_json_path: Path,
    prompt_config: Mapping[str, Any],
    prompt_config_path: Path,
    record_count: int,
    selected_record_count: int,
    aggregated_audio_dir: Path,
) -> None:
    """Run each plan entry over its own clips and write its result file.

    One file per run, written as soon as that run finishes, so a job that times
    out keeps whatever it already completed and a resubmission picks up from
    there rather than starting over.
    """
    records_by_index = {item["record_index"]: item for item in prepared.kept}
    total_runs = len(plan.runs)

    for run_index, run in enumerate(plan.runs, start=1):
        run_output_path = run["output_path"]
        persona_path = run["persona_path"]
        clip_indices = run["clip_indices"]
        # A spec task names itself; a persona run is named by its file. Either way
        # this is what the log lines below identify a run by, and a spec task may
        # have no persona file at all -- its persona text is inline.
        run_label = run.get("task_id") or (
            run_output_path.stem if persona_path is None else persona_path.stem
        )

        if clip_indices is None:
            run_records = prepared.kept
            missing_clip_indices: list[int] = []
        else:
            # Pre-filtered by build_assignment.py, so a miss here means a clip
            # went bad after the assignment was built. Log it and carry on rather
            # than failing the whole job.
            run_records = [records_by_index[i] for i in clip_indices if i in records_by_index]
            missing_clip_indices = [i for i in clip_indices if i not in records_by_index]
            if missing_clip_indices:
                print(
                    f"[WARN] {run_label}: {len(missing_clip_indices)} assigned clip(s) "
                    f"failed preparation and will be missing: {missing_clip_indices}",
                    flush=True,
                )

        print(
            f"[INFO] Run {run_index}/{total_runs}: {run_label} "
            f"({len(run_records)} clip(s)) -> {run_output_path}",
            flush=True,
        )

        # A run that brought its own prompt config gets the mode configured for
        # it, and the default restored afterwards. configure() holds more than
        # the user template -- fa keeps its per-speaker labels there and reads
        # them while building turns -- so leaving a task's config in place would
        # silently apply it to every later run in this process.
        run_prompt_config = run.get("prompt_config")
        if run_prompt_config is not None:
            mode.configure(run_prompt_config)
        try:
            # Records are prepared once, under the run-wide template, because the
            # expensive half of preparation -- path resolution and any audio
            # mixing -- does not depend on the prompt. A run with its own config
            # only needs the text re-rendered, which is why every prepared record
            # keeps its source record.
            #
            # Through the mode, not render_prompt directly: render_user_prompt is
            # where a mode fills in values it resolved itself, so going around it
            # would drop exactly those for the tasks that brought their own
            # wording.
            run_template = run.get("user_prompt_template")
            if run_template is not None and run_template != prompt_config["user_prompt_template"]:
                run_records = [
                    {
                        **item,
                        "user_prompt": mode.render_user_prompt(
                            run_template, item["source_record"], item
                        ),
                    }
                    for item in run_records
                ]

            results = run_inference_pass(
                records=run_records,
                mode=mode,
                system_prompt=run["system_prompt"],
                no_audio=args.no_audio,
                model=model,
                model_config=model_config,
            )
        finally:
            if run_prompt_config is not None:
                mode.configure(prompt_config)

        summary = build_summary(
            args=args,
            backend=backend,
            mode=mode.name,
            model_config=model_config,
            input_json_path=input_json_path,
            prompt_config_path=prompt_config_path,
            record_count=record_count,
            selected_record_count=selected_record_count,
            prepared=prepared,
            results=results,
            aggregated_audio_dir=aggregated_audio_dir,
            facts=plan.facts,
            run=run,
            missing_clip_indices=missing_clip_indices,
        )
        write_results(
            run_output_path,
            summary=summary,
            prepared=prepared,
            results=results,
        )


def write_results(
    output_path: Path,
    *,
    summary: dict[str, Any],
    prepared: PreparedRecords,
    results: list[dict[str, Any]],
) -> None:
    payload = {
        "__summary__": summary,
        "__skipped__": prepared.skipped,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Results saved to {output_path}")
