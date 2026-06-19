#!/usr/bin/env python3
"""Extract clips from redacted channel audio using clip-meta time mapping.

According to each channel's `master_1330_1602_tl_chXX_clip_meta.json`, the
master_trim clip spans:
  absolute clip start: 13:30:00.000
  absolute clip end:   16:02:00.000

More generally, this script reads `clip_start_abs_clock` from the matching
clip-meta JSON and converts user-provided absolute daytime timestamps into
times relative to the redacted audio file before extraction.

Output filenames include both:
  - the requested absolute daytime range
  - the derived relative-in-audio range
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

TIME_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<ms>\d{1,3}))?$")


def parse_time_to_seconds(value: str) -> float:
    match = TIME_RE.fullmatch(value.strip())
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid time '{value}'. Use HH:MM:SS or HH:MM:SS.mmm"
        )
    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    millis = int((match.group("ms") or "0").ljust(3, "0"))
    if minutes >= 60 or seconds >= 60:
        raise argparse.ArgumentTypeError(
            f"Invalid time '{value}'. Minutes and seconds must be < 60."
        )
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def normalize_channels(raw_channels: list[str]) -> list[str]:
    channels: list[str] = []
    for raw in raw_channels:
        for piece in raw.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if not piece.isdigit():
                raise argparse.ArgumentTypeError(
                    f"Invalid channel '{piece}'. Use numbers like 1, 02, 32."
                )
            channel_num = int(piece)
            if not 1 <= channel_num <= 32:
                raise argparse.ArgumentTypeError(
                    f"Invalid channel '{piece}'. Expected range 1-32."
                )
            channels.append(f"{channel_num:02d}")
    if not channels:
        raise argparse.ArgumentTypeError("At least one channel is required.")
    return list(dict.fromkeys(channels))


def format_seconds_for_name(value: float) -> str:
    total_ms = round(value * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    seconds, millis = divmod(rem_ms, 1000)
    return f"{hours:02d}-{minutes:02d}-{seconds:02d}p{millis:03d}"


def build_output_prefix(channel: str, participant: str | None) -> str:
    prefix = f"ch{channel}"
    if participant is not None:
        prefix += f"_p{participant}"
    return prefix


def build_output_name(
    channel: str,
    variant_name: str,
    start: float,
    end: float,
    suffix: str,
    participant: str | None = None,
) -> str:
    return (
        f"{build_output_prefix(channel, participant)}_{variant_name}_"
        f"{format_seconds_for_name(start)}_to_{format_seconds_for_name(end)}{suffix}"
    )


def build_output_name_with_absolute(
    channel: str,
    variant_name: str,
    start_relative: float,
    end_relative: float,
    start_absolute_label: str,
    end_absolute_label: str,
    suffix: str,
    participant: str | None = None,
) -> str:
    return (
        f"{build_output_prefix(channel, participant)}_{variant_name}_"
        f"abs_{start_absolute_label}_to_{end_absolute_label}_"
        f"rel_{format_seconds_for_name(start_relative)}_to_{format_seconds_for_name(end_relative)}"
        f"{suffix}"
    )


def format_clock_label(value: str) -> str:
    return value.replace(":", "-").replace(".", "p")


def normalize_csv_row(row: dict[str, str]) -> dict[str, str]:
    return {key.strip(): value.strip() for key, value in row.items() if key is not None}


def load_channel_participants(mapping_path: Path) -> dict[str, str]:
    channel_to_participant: dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = normalize_csv_row(raw_row)
            participant_id = row.get("Participant ID")
            channel = row.get("microphone Channels")
            if not participant_id or not channel:
                raise RuntimeError(
                    "Participant mapping must contain 'Participant ID' and "
                    f"'microphone Channels' columns: {mapping_path}"
                )
            channel_to_participant[f"{int(channel):02d}"] = f"{int(participant_id):02d}"
    return channel_to_participant


def load_clip_start_seconds(meta_path: Path) -> float:
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    clock_value = meta.get("clip_start_abs_clock")
    if not isinstance(clock_value, str):
        raise RuntimeError(f"Missing clip_start_abs_clock in {meta_path}")
    return parse_time_to_seconds(clock_value)


def find_meta_path(meta_root: Path, channel: str) -> Path:
    filename = "master_1330_1602_tl_ch%s_clip_meta.json" % channel
    candidates = [
        meta_root / filename,
        meta_root / ("ch%s" % channel) / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def probe_duration_seconds(audio_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {audio_path}")
    return float(result.stdout.strip())


def run_ffmpeg_extract(
    input_path: Path, output_path: Path, start: float, end: float, overwrite: bool
) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_path),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or "ffmpeg failed"
        raise RuntimeError(f"ffmpeg failed for {input_path}: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract clips from redacted per-channel audio using HH:MM:SS.mmm times."
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        required=True,
        help="Channel list, e.g. --channels 1 2 3 or --channels 01,02,03",
    )
    parser.add_argument("--start", required=True, help="Start time: HH:MM:SS or HH:MM:SS.mmm")
    parser.add_argument("--end", required=True, help="End time: HH:MM:SS or HH:MM:SS.mmm")
    parser.add_argument(
        "--time-mode",
        choices=("absolute", "relative"),
        default="absolute",
        help=(
            "Interpret --start/--end as absolute session clock time or as time "
            "relative to the redacted audio. Default: absolute"
        ),
    )
    parser.add_argument(
        "--input-root",
        default="outputs/pii_redaction_v2/master_trim_norm/UFX_01",
        help="Root directory containing ch01..ch32",
    )
    parser.add_argument(
        "--meta-root",
        default=None,
        help=(
            "Directory containing master_1330_1602_tl_chXX_clip_meta.json files. "
            "Defaults to --input-root."
        ),
    )
    parser.add_argument(
        "--filename",
        default="audio_redacted_lowpass_1250hz.wav",
        help="Input filename inside each channel directory",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/extracted_redacted_clips",
        help="Where extracted clips will be written",
    )
    parser.add_argument(
        "--participant-map",
        default=None,
        help=(
            "Optional CSV containing 'Participant ID' and 'microphone Channels' "
            "columns. When provided, output filenames include chXX_pYY."
        ),
    )
    parser.add_argument(
        "--suffix",
        default=".wav",
        help="Output file suffix, default: .wav",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    channels = normalize_channels(args.channels)
    start_input = parse_time_to_seconds(args.start)
    end_input = parse_time_to_seconds(args.end)
    if end_input <= start_input:
        print("--end must be greater than --start", file=sys.stderr)
        return 2

    input_root = Path(args.input_root)
    meta_root = Path(args.meta_root) if args.meta_root else input_root
    output_dir = Path(args.output_dir)
    variant_name = Path(args.filename).stem
    channel_participants = (
        load_channel_participants(Path(args.participant_map))
        if args.participant_map
        else {}
    )

    missing_inputs: list[Path] = []
    missing_meta: list[Path] = []
    missing_participants: list[str] = []
    plans: list[tuple[Path, Path, float, float]] = []
    for channel in channels:
        participant = None
        if args.participant_map:
            participant = channel_participants.get(channel)
            if participant is None:
                missing_participants.append(channel)
                continue

        input_path = input_root / f"ch{channel}" / args.filename
        if not input_path.exists():
            missing_inputs.append(input_path)
            continue

        start = start_input
        end = end_input
        if args.time_mode == "absolute":
            meta_path = find_meta_path(meta_root, channel)
            if not meta_path.exists():
                missing_meta.append(meta_path)
                continue
            clip_start_abs = load_clip_start_seconds(meta_path)
            start = start_input - clip_start_abs
            end = end_input - clip_start_abs

        if start < 0 or end < 0:
            print(
                f"Requested range is before clip start for ch{channel}: "
                f"{args.start} - {args.end}",
                file=sys.stderr,
            )
            return 2

        duration = probe_duration_seconds(input_path)
        if end > duration:
            print(
                f"Requested range exceeds audio duration for ch{channel}: "
                f"end={end:.3f}s duration={duration:.3f}s",
                file=sys.stderr,
            )
            return 2

        if args.time_mode == "absolute":
            output_name = build_output_name_with_absolute(
                channel=channel,
                variant_name=variant_name,
                start_relative=start,
                end_relative=end,
                start_absolute_label=format_clock_label(args.start),
                end_absolute_label=format_clock_label(args.end),
                suffix=args.suffix,
                participant=participant,
            )
        else:
            output_name = build_output_name(
                channel, variant_name, start, end, args.suffix, participant=participant
            )

        output_path = output_dir / output_name
        plans.append((input_path, output_path, start, end))

    if missing_inputs:
        print("Missing input files:", file=sys.stderr)
        for path in missing_inputs:
            print(f"  {path}", file=sys.stderr)
        return 1

    if missing_meta:
        print("Missing meta files:", file=sys.stderr)
        for path in missing_meta:
            print(f"  {path}", file=sys.stderr)
        return 1

    if missing_participants:
        print("Missing participant mapping for channels:", file=sys.stderr)
        for channel in missing_participants:
            print(f"  ch{channel}", file=sys.stderr)
        return 1

    print("Planned extractions:")
    for input_path, output_path, start, end in plans:
        print(f"  {input_path} -> {output_path}")

    for input_path, output_path, start, end in plans:
        run_ffmpeg_extract(input_path, output_path, start, end, args.overwrite)

    print(f"Done. Wrote {len(plans)} clip(s) to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
