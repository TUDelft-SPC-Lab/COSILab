#!/usr/bin/env python3
"""Build a balanced persona x clip assignment for the persona sweep.

Instead of running the full persona x clip cross product, the sweep runs a
balanced incomplete design: a random subset of clips is drawn from the manifest,
and each persona annotates exactly ``--clips-per-persona`` of them while each
selected clip is annotated by exactly ``--personas-per-clip`` personas.

Everything is derived from ``--seed``, so rerunning with the same seed and the
same manifest reproduces the assignment byte for byte.

Candidate clips are validated with the same checks inference uses -- the clip,
whatever the mode needs to ground the question, and one audio track per speaker --
so a sampled clip cannot silently drop out at inference time and leave itself
under-annotated. Validation is deliberately ``exists()``-cheap and never mixes
audio: see ``clip_is_usable``.

Because the middle check belongs to the mode, ``--mode`` is required and the
design is only valid for that mode.

This script is stdlib-only on purpose: it imports ``records`` and ``modes``
rather than ``models.registry``, so it needs no torch, transformers or librosa.

DO NOT RUN THIS DIRECTLY unless you pass all four --video-*/--audio-* prefix
flags. A missing pair leaves the manifest's covfee URLs unrewritten, so every
clip is reported unusable. Submit one of the job scripts instead, which fills the
prefixes in from job_scripts/lib/intention_paths.sh -- the same file inference
reads -- and pins the interpreter by running inside the backend's apptainer image::

    sbatch job_scripts/intention/build_assignment_daic.sh
    bash   job_scripts/intention/build_assignment_daic.sh --explain 2

Those scripts also FIX the design. ``--num-clips``, ``--personas-per-clip``,
``--clips-per-persona``, ``--persona-range`` and ``--clip-pool-range`` are set
from the design block at the top of job_scripts/lib/build_assignment_job.sh
(1000 clips, 6 personas per clip, 6 clips per persona, personas 1-1000) and
passing them on the command line is rejected there, so the design changes by
editing that block and rebuilding. The defaults below match it, but the job
script is what actually decides. Every other flag -- ``--seed``, ``--output``,
``--explain``, ``--no-audio``, ``--skip-validation`` -- is forwarded normally,
as is ``--mode``, which the sweep must then match.

EXAMPLES (via the job scripts)

  # The fixed design, as declared in lib/build_assignment_job.sh.
  sbatch job_scripts/intention/build_assignment_daic.sh

  # A different reproducible draw of that same design, written somewhere else.
  sbatch job_scripts/intention/build_assignment_daic.sh \
      --seed 7 --output /path/to/other.json

  # Debug path resolution: show where the first 2 records' media is looked for,
  # then exit. Start here if validation rejects everything.
  bash job_scripts/intention/build_assignment_daic.sh --explain 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# src/intention_inference/build_assignment.py -> src/, which is what PYTHONPATH
# points at inside the apptainer bind mount. Must run before any first-party
# import: this file is invoked by absolute path, so src/ is not on sys.path by
# default. Deliberately no models.registry import anywhere below -- this stays
# runnable with the standard library alone.
_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from intention_inference.manifest import (  # noqa: E402
    get_nested_value,
    load_manifest_records,
)
from intention_inference.media import (  # noqa: E402
    effective_local_prefix,
    effective_media_prefix,
    resolve_prefixed_media_path,
)
from intention_inference.modes.registry import available_modes, load_mode  # noqa: E402
from intention_inference.records import (  # noqa: E402
    RecordContext,
    resolve_speaker_tracks,
    resolve_video_media,
)

DEFAULT_PARTICIPANT_IMAGE_ROOT = Path(
    "/tudelft.net/staff-umbrella/neon/cosilab_project/B1_pipeline/participant_imgs"
)


def none_or_str(value: Any) -> str | None:
    return None if value is None else str(value)


def parse_inclusive_range(raw: str, label: str) -> tuple[int, int]:
    parts = raw.split("-")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise argparse.ArgumentTypeError(f"{label} must look like X-Y, for example 0-4000: {raw}")
    start, end = int(parts[0]), int(parts[1])
    if end < start:
        raise argparse.ArgumentTypeError(f"{label} end is before start: {raw}")
    return start, end


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced persona x clip assignment JSON for the persona sweep."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=available_modes(),
        help=(
            "Task mode the design is built for. Required, and not incidental: whether a "
            "clip is usable depends on what the mode needs to ground the question, so a "
            "design validated for one mode is not valid for another. Recorded in the "
            "output's __meta__."
        ),
    )
    parser.add_argument("--input-json", type=Path, required=True, help="Clip manifest JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Assignment JSON to write.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed controlling clip sampling and pairing. Default: 42",
    )
    parser.add_argument(
        "--clip-pool-range",
        default="0-4000",
        help="Inclusive manifest index range to sample clips from. Default: 0-4000",
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=1000,
        help="How many clips to sample from the pool. Default: 1000",
    )
    parser.add_argument(
        "--persona-range",
        default="1-1000",
        help="Inclusive persona numbers to assign clips to. Default: 1-1000",
    )
    parser.add_argument(
        "--clips-per-persona",
        type=int,
        default=6,
        help="Clips each persona annotates. Default: 6",
    )
    parser.add_argument(
        "--personas-per-clip",
        type=int,
        default=6,
        help="Personas annotating each clip. Default: 6",
    )
    parser.add_argument(
        "--persona-digits",
        type=int,
        default=4,
        help="Zero-padding width of persona keys, matching persona_%%04d.txt. Default: 4",
    )

    # Media resolution, mirroring cosilab_daic.sh so validation matches runtime.
    parser.add_argument("--media-path-prefix", default=None)
    parser.add_argument("--local-path-prefix", type=Path, default=None)
    parser.add_argument("--video-media-path-prefix", default=None)
    parser.add_argument("--video-local-path-prefix", type=Path, default=None)
    parser.add_argument("--audio-media-path-prefix", default=None)
    parser.add_argument("--audio-local-path-prefix", type=Path, default=None)
    parser.add_argument("--media-root", type=Path, default=None)
    parser.add_argument(
        "--participant-image-root",
        type=Path,
        default=DEFAULT_PARTICIPANT_IMAGE_ROOT,
    )
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--exclude-video-substring", action="append", default=[])
    parser.add_argument("--exclude-audio-substring", action="append", default=[])
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio validation; pair with --no-audio inference.",
    )
    parser.add_argument(
        "--explain",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Print where the first N pool records' media were looked for, then exit. Use this "
            "to debug prefix flags without waiting for a full validation pass."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "Sample from the raw index range without touching the filesystem. Only for "
            "exercising the design math off-cluster; the 6-per-clip balance may not hold."
        ),
    )
    return parser.parse_args()


def describe_clip_resolution(
    *,
    record: dict[str, Any],
    record_index: int,
    mode,
    ctx: RecordContext,
    args: argparse.Namespace,
    manifest_dir: Path,
) -> list[str]:
    """Explain, path by path, where this record's media was looked for.

    Used to make a zero-valid-candidates failure self-diagnosing: the usual cause
    is a media/local prefix pair that does not match the manifest, which silently
    leaves the raw URL in place and resolves it against the manifest directory.
    """
    lines = [f"  record {record_index} (id={get_nested_value(record, args.id_key)!r}):"]

    def explain(label: str, raw: Any, media_prefix: str | None, local_prefix: Path | None) -> None:
        lines.append(f"    {label} raw       = {raw!r}")
        if not isinstance(raw, str) or not raw.strip():
            lines.append(f"    {label} -> record field is missing or not a string")
            return
        resolved_media_prefix = effective_media_prefix(media_prefix, args.media_path_prefix)
        resolved_local_prefix = effective_local_prefix(local_prefix, args.local_path_prefix)
        if not resolved_media_prefix:
            lines.append(f"    {label} !! no media path prefix given, so no rewrite was attempted")
        elif not raw.startswith(resolved_media_prefix):
            lines.append(f"    {label} !! does NOT start with {resolved_media_prefix!r}")
            lines.append(f"    {label}    so the prefix rewrite never fired; fix the prefix flag")
        else:
            lines.append(f"    {label} prefix    = {resolved_media_prefix!r} (matched)")
            lines.append(f"    {label} local     = {resolved_local_prefix}")
        try:
            resolved, _ = resolve_prefixed_media_path(
                raw,
                manifest_dir=manifest_dir,
                media_root=args.media_root,
                media_path_prefix=resolved_media_prefix,
                local_path_prefix=resolved_local_prefix,
            )
        except ValueError as exc:
            lines.append(f"    {label} -> {exc}")
            return
        exists = resolved is not None and resolved.exists()
        lines.append(f"    {label} looked at = {resolved}")
        lines.append(f"    {label} exists    = {exists}")

    explain("video", record.get("video"), args.video_media_path_prefix, args.video_local_path_prefix)

    # Whatever grounds "which person are we talking about" belongs to the mode,
    # so it is asked rather than reimplemented here -- pa wants a participant
    # image, another mode may want nothing at all.
    reference_fields, reference_skip = mode.resolve_reference_media(record, ctx)
    lines.append(f"    reference mode    = {mode.name}")
    if reference_skip is not None:
        lines.append(f"    reference !! {reference_skip}")
    else:
        for key, value in reference_fields.items():
            lines.append(f"    reference {key} = {value}")

    if not args.no_audio:
        audios = record.get("audios")
        if not isinstance(audios, Sequence) or isinstance(audios, (str, bytes, bytearray)):
            lines.append(f"    audio !! 'audios' is not a list: {audios!r}")
        else:
            # Every speaker's track, not a selected subset: which of them a mode
            # stacks is the mode's business, but all of them have to resolve.
            for speaker_id, raw in enumerate(audios, start=1):
                explain(
                    f"audio[s{speaker_id}]",
                    raw,
                    args.audio_media_path_prefix,
                    args.audio_local_path_prefix,
                )
            _, audio_skip = resolve_speaker_tracks(record, ctx)
            if audio_skip is not None:
                lines.append(f"    audio !! resolve_speaker_tracks: {audio_skip}")
    return lines


def report_resolution_failures(
    *,
    all_records: list[dict[str, Any]],
    pool_start: int,
    pool_end: int,
    mode,
    args: argparse.Namespace,
    manifest_dir: Path,
    sample_size: int,
) -> None:
    print("")
    print("[DIAG] Nothing validated. Here is where the first record(s) were looked for.")
    print("[DIAG] A '!!' line marks the check that most likely broke.")
    for record_index in range(pool_start, min(pool_start + sample_size, pool_end + 1)):
        for line in describe_clip_resolution(
            record=all_records[record_index],
            record_index=record_index,
            mode=mode,
            ctx=validation_context(args, manifest_dir),
            args=args,
            manifest_dir=manifest_dir,
        ):
            print(f"[DIAG] {line}")
    print("")


def validation_context(args: argparse.Namespace, manifest_dir: Path) -> RecordContext:
    """A RecordContext for validation only.

    ``aggregated_audio_dir`` and ``user_prompt_template`` are placeholders: the
    checks below never mix audio and never render a prompt, which is the whole
    point -- see collect_valid_clip_indices.
    """
    return RecordContext(
        id_key=args.id_key,
        manifest_dir=manifest_dir,
        media_root=args.media_root,
        media_path_prefix=args.media_path_prefix,
        local_path_prefix=args.local_path_prefix,
        video_media_path_prefix=args.video_media_path_prefix,
        video_local_path_prefix=args.video_local_path_prefix,
        audio_media_path_prefix=args.audio_media_path_prefix,
        audio_local_path_prefix=args.audio_local_path_prefix,
        participant_image_root=args.participant_image_root,
        no_audio=args.no_audio,
        aggregated_audio_dir=Path("/nonexistent"),
        exclude_video_substrings=args.exclude_video_substring,
        exclude_audio_substrings=args.exclude_audio_substring,
        user_prompt_template="",
    )


def clip_is_usable(
    record: dict[str, Any],
    *,
    mode,
    ctx: RecordContext,
) -> str | None:
    """None if inference could run this clip, else the reason it could not.

    Three cheap checks, and cheap is the requirement rather than a nicety: this
    runs over the whole candidate pool to decide what to sample, so it must cost
    path resolution and ``exists()`` and nothing more. In particular it does NOT
    call ``resolve_audio_bundle`` -- that runs the stacked-audio mix, and mixing
    every candidate to decide whether to sample it would cost more than the sweep
    being planned.

    The middle check is the mode's, because what grounds the question differs
    between modes and so does whether a given record can be used at all. A design
    built for one mode is therefore only valid for that mode, which is why
    ``--mode`` is required here and recorded in the output.
    """
    _, skip_reason = resolve_video_media(record, ctx)
    if skip_reason is not None:
        return skip_reason

    _, skip_reason = mode.resolve_reference_media(record, ctx)
    if skip_reason is not None:
        return skip_reason

    if not ctx.no_audio:
        _, skip_reason = resolve_speaker_tracks(record, ctx)
        if skip_reason is not None:
            return skip_reason

    return None


def collect_valid_clip_indices(
    *,
    all_records: list[dict[str, Any]],
    pool_start: int,
    pool_end: int,
    mode,
    args: argparse.Namespace,
    manifest_dir: Path,
) -> tuple[list[int], Counter[str]]:
    """Return the pool indices whose media all resolve, plus a tally of rejections."""
    ctx = validation_context(args, manifest_dir)
    valid: list[int] = []
    reasons: Counter[str] = Counter()
    for record_index in range(pool_start, pool_end + 1):
        skip_reason = clip_is_usable(all_records[record_index], mode=mode, ctx=ctx)
        if skip_reason is not None:
            reasons[skip_reason] += 1
            continue
        valid.append(record_index)
    return valid, reasons


def build_design(
    *,
    rng: random.Random,
    clips: list[int],
    num_personas: int,
    clips_per_persona: int,
) -> list[list[int]]:
    """Assign ``clips_per_persona`` distinct clips to each persona, balanced on both sides.

    Each round is an independent shuffle of the clip list (repeated
    ``num_personas / len(clips)`` times so it is exactly one clip per persona),
    so every clip is handed out the same number of times per round and therefore
    exactly ``clips_per_persona * num_personas / len(clips)`` times overall. The
    only thing a shuffle can get wrong is giving one persona the same clip twice;
    that is repaired by swapping *within* the round, which leaves the round's
    multiset untouched and so preserves clip balance.

    Returns one clip list per persona, indexed by persona position.
    """
    if not clips or num_personas % len(clips) != 0:
        raise ValueError(
            "Round-permutation construction needs the persona count to be a multiple of the "
            f"clip count; got {len(clips)} clip(s) for {num_personas} persona(s)."
        )
    repeats = num_personas // len(clips)
    if clips_per_persona * repeats > len(clips):
        raise ValueError(
            f"Cannot give each persona {clips_per_persona} distinct clips out of "
            f"{len(clips)} clip(s) at {repeats} repeat(s) per round."
        )

    rounds: list[list[int]] = []
    for _ in range(clips_per_persona):
        round_clips = clips * repeats
        rng.shuffle(round_clips)
        rounds.append(round_clips)

    for round_index in range(1, clips_per_persona):
        current = rounds[round_index]
        for persona_position in range(num_personas):
            prior = {rounds[k][persona_position] for k in range(round_index)}
            if current[persona_position] not in prior:
                continue
            # rng.sample keeps the search order deterministic for a given seed.
            for other in rng.sample(range(num_personas), num_personas):
                if other == persona_position:
                    continue
                prior_other = {rounds[k][other] for k in range(round_index)}
                if current[other] in prior or current[persona_position] in prior_other:
                    continue
                current[persona_position], current[other] = (
                    current[other],
                    current[persona_position],
                )
                break
            else:
                raise RuntimeError(
                    f"Could not repair a duplicate clip for persona position {persona_position} "
                    f"in round {round_index}. Try a different --seed."
                )

    return [
        sorted(rounds[r][persona_position] for r in range(clips_per_persona))
        for persona_position in range(num_personas)
    ]


def verify_design(
    *,
    persona_to_clips: dict[str, list[int]],
    clips: list[int],
    clips_per_persona: int,
    personas_per_clip: int,
) -> Counter[int]:
    """Assert both degree conditions hold; return the per-clip annotation counts."""
    clip_counts: Counter[int] = Counter()
    for persona_key, assigned in persona_to_clips.items():
        if len(assigned) != clips_per_persona:
            raise AssertionError(
                f"persona {persona_key} has {len(assigned)} clip(s), expected {clips_per_persona}"
            )
        if len(set(assigned)) != len(assigned):
            raise AssertionError(f"persona {persona_key} has duplicate clips: {assigned}")
        clip_counts.update(assigned)

    unexpected = set(clip_counts) - set(clips)
    if unexpected:
        raise AssertionError(f"Assignment references unselected clips: {sorted(unexpected)[:10]}")

    off_balance = [clip for clip in clips if clip_counts[clip] != personas_per_clip]
    if off_balance:
        raise AssertionError(
            f"{len(off_balance)} clip(s) are not annotated exactly {personas_per_clip} time(s); "
            f"first offenders: {off_balance[:10]}"
        )
    return clip_counts


def main() -> None:
    args = parse_args()

    # First, before any file is opened: an unfinished mode refuses here. A design
    # is only valid for the mode it was validated against, so this is not an
    # incidental parameter -- it is recorded in the output's __meta__.
    mode = load_mode(args.mode)

    if args.num_clips < 1:
        raise ValueError(f"--num-clips must be at least 1: {args.num_clips}")
    if args.clips_per_persona < 1:
        raise ValueError(f"--clips-per-persona must be at least 1: {args.clips_per_persona}")
    if args.personas_per_clip < 1:
        raise ValueError(f"--personas-per-clip must be at least 1: {args.personas_per_clip}")

    pool_start, pool_end = parse_inclusive_range(args.clip_pool_range, "--clip-pool-range")
    persona_start, persona_end = parse_inclusive_range(args.persona_range, "--persona-range")
    num_personas = persona_end - persona_start + 1

    if num_personas * args.clips_per_persona != args.num_clips * args.personas_per_clip:
        raise ValueError(
            "The design is not balanceable: "
            f"{num_personas} persona(s) x {args.clips_per_persona} clip(s) = "
            f"{num_personas * args.clips_per_persona} pairs, but "
            f"{args.num_clips} clip(s) x {args.personas_per_clip} persona(s) = "
            f"{args.num_clips * args.personas_per_clip} pairs."
        )

    input_json_path = args.input_json.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_json_path.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")

    all_records = load_manifest_records(input_json_path)
    if pool_start >= len(all_records):
        raise ValueError(
            f"--clip-pool-range starts at {pool_start} but {input_json_path} only has "
            f"{len(all_records)} record(s)."
        )
    clamped_pool_end = min(pool_end, len(all_records) - 1)

    print(f"[INFO] input_json          = {input_json_path}")
    print(f"[INFO] manifest records    = {len(all_records)}")
    print(f"[INFO] clip pool range     = {pool_start}-{clamped_pool_end} (requested {args.clip_pool_range})")
    print(f"[INFO] persona range       = {persona_start}-{persona_end} ({num_personas} personas)")
    print(f"[INFO] clips per persona   = {args.clips_per_persona}")
    print(f"[INFO] personas per clip   = {args.personas_per_clip}")
    print(f"[INFO] num clips           = {args.num_clips}")
    print(f"[INFO] seed                = {args.seed}")

    if args.explain > 0:
        report_resolution_failures(
            all_records=all_records,
            pool_start=pool_start,
            pool_end=clamped_pool_end,
            mode=mode,
            args=args,
            manifest_dir=input_json_path.parent,
            sample_size=args.explain,
        )
        return

    if args.skip_validation:
        print("[WARN] --skip-validation: sampling without checking that media exists.")
        candidates = list(range(pool_start, clamped_pool_end + 1))
        skip_reasons: Counter[str] = Counter()
    else:
        candidates, skip_reasons = collect_valid_clip_indices(
            all_records=all_records,
            pool_start=pool_start,
            pool_end=clamped_pool_end,
            mode=mode,
            args=args,
            manifest_dir=input_json_path.parent,
        )
        print(
            f"[INFO] valid candidates    = {len(candidates)} of "
            f"{clamped_pool_end - pool_start + 1} in the pool"
        )
        for reason, count in sorted(skip_reasons.items()):
            print(f"[INFO]   rejected {reason}: {count}")

    if len(candidates) < args.num_clips:
        if not args.skip_validation:
            report_resolution_failures(
                all_records=all_records,
                pool_start=pool_start,
                pool_end=clamped_pool_end,
                mode=mode,
                args=args,
                manifest_dir=input_json_path.parent,
                sample_size=2 if candidates else 3,
            )
        raise ValueError(
            f"Only {len(candidates)} usable clip(s) in {pool_start}-{clamped_pool_end}, "
            f"but --num-clips is {args.num_clips}. The [DIAG] block above shows where the "
            "media was looked for; the usual cause is a --video-media-path-prefix / "
            "--video-local-path-prefix pair that does not match the manifest. Otherwise widen "
            "--clip-pool-range or lower --num-clips."
        )

    rng = random.Random(args.seed)
    clips = sorted(rng.sample(candidates, args.num_clips))

    persona_numbers = list(range(persona_start, persona_end + 1))
    design = build_design(
        rng=rng,
        clips=clips,
        num_personas=num_personas,
        clips_per_persona=args.clips_per_persona,
    )
    persona_to_clips = {
        f"{persona_number:0{args.persona_digits}d}": assigned
        for persona_number, assigned in zip(persona_numbers, design)
    }

    clip_counts = verify_design(
        persona_to_clips=persona_to_clips,
        clips=clips,
        clips_per_persona=args.clips_per_persona,
        personas_per_clip=args.personas_per_clip,
    )
    print(
        f"[INFO] verified: {len(persona_to_clips)} persona(s) x {args.clips_per_persona} clip(s) "
        f"= {len(persona_to_clips) * args.clips_per_persona} inference(s); "
        f"every one of {len(clips)} clip(s) annotated exactly {args.personas_per_clip} time(s)"
    )
    print(f"[INFO] clips-per-persona histogram: {{{args.clips_per_persona}: {len(persona_to_clips)}}}")
    print(f"[INFO] personas-per-clip histogram: {dict(Counter(clip_counts.values()))}")

    clip_to_personas: dict[str, list[int]] = {str(clip): [] for clip in clips}
    for persona_number, assigned in zip(persona_numbers, design):
        for clip in assigned:
            clip_to_personas[str(clip)].append(persona_number)
    for personas in clip_to_personas.values():
        personas.sort()

    payload = {
        "__meta__": {
            "seed": args.seed,
            # The design is only valid for the mode it was validated against.
            "mode": args.mode,
            "input_json": str(input_json_path),
            "manifest_record_count": len(all_records),
            "clip_pool_range": [pool_start, clamped_pool_end],
            "clip_pool_requested_range": args.clip_pool_range,
            "clip_pool_valid_count": len(candidates),
            "num_clips": args.num_clips,
            "num_personas": num_personas,
            "persona_range": [persona_start, persona_end],
            "clips_per_persona": args.clips_per_persona,
            "personas_per_clip": args.personas_per_clip,
            "total_inferences": num_personas * args.clips_per_persona,
            "no_audio": args.no_audio,
            "skip_validation": args.skip_validation,
            "excluded_clip_reasons": dict(sorted(skip_reasons.items())),
            # Recorded so the validation this assignment was built against can be
            # reproduced exactly, and so a mismatch with what inference reads is
            # visible in the file rather than only in a job log.
            "media_resolution": {
                "media_path_prefix": args.media_path_prefix,
                "local_path_prefix": none_or_str(args.local_path_prefix),
                "video_media_path_prefix": args.video_media_path_prefix,
                "video_local_path_prefix": none_or_str(args.video_local_path_prefix),
                "audio_media_path_prefix": args.audio_media_path_prefix,
                "audio_local_path_prefix": none_or_str(args.audio_local_path_prefix),
                "media_root": none_or_str(args.media_root),
                "participant_image_root": none_or_str(args.participant_image_root),
                "id_key": args.id_key,
                "exclude_video_substring": list(args.exclude_video_substring),
                "exclude_audio_substring": list(args.exclude_audio_substring),
            },
        },
        "clips": clips,
        "persona_to_clips": persona_to_clips,
        # Written for inspection and downstream analysis; the pipeline never reads it back.
        "clip_to_personas": clip_to_personas,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[INFO] Assignment saved to {output_path}")

    sample_key = next(iter(persona_to_clips))
    sample_clip = clips[0]
    print(f"[INFO] example persona {sample_key} -> {persona_to_clips[sample_key]}")
    print(f"[INFO] example clip {sample_clip} -> personas {clip_to_personas[str(sample_clip)]}")
    if not args.skip_validation:
        sample_record = all_records[sample_clip]
        print(
            f"[INFO] example clip {sample_clip} id = "
            f"{get_nested_value(sample_record, args.id_key)}"
        )


if __name__ == "__main__":
    main()
