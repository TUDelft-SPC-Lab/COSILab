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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from models.config import ModelRunConfig, describe_model_config

from .manifest import get_nested_value
from .records import RecordContext

__all__ = [
    "PreparedRecords",
    "build_summary",
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


def run_inference_pass(
    *,
    prepared: PreparedRecords,
    mode,
    system_prompt: str,
    no_audio: bool,
    model,
    model_config: ModelRunConfig,
) -> list[dict[str, Any]]:
    """One turn per retained record, in manifest order."""
    results: list[dict[str, Any]] = []
    total_records = len(prepared.kept)

    for processed_index, item in enumerate(prepared.kept, start=1):
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
    end_index: int,
    prepared: PreparedRecords,
    results: list[dict[str, Any]],
    aggregated_audio_dir: Path,
) -> dict[str, Any]:
    """What this run was, recorded beside its answers.

    ``describe_model_config`` contributes the weights, the decoding parameters
    and the frame policy, so a result file answers "what produced this" without
    anyone having to find the sbatch line that made it.
    """
    error_count = sum(1 for item in results if str(item["assistant"]).startswith("[ERROR]"))
    return {
        "input_json": str(input_json_path),
        "prompt_config": str(prompt_config_path),
        "backend": backend,
        "mode": mode,
        "record_count": record_count,
        "selected_record_count": selected_record_count,
        "start_index": args.start_index,
        "end_index": end_index,
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
