"""Generate a CSV aligning every frame of a split section video to its timecode.

A section is first cut with `extract_segment_from_video.py` (e.g. 35 min, as in
`batch_extract_segments.py`) using nominal 60 fps start/end timecodes. The footage is really
59.94 fps (60000/1001), so a 35 min section holds round(35 * 60 * 60 * 1000/1001) = 125874
physical frames, not 126000. That section is then split into consecutive segments of
`--frames-per-segment` frames (18000 by default), the last one taking the remainder: for 35 min,
six segments of 18000 frames and a seventh of 17874.

Each output row is (segment, frame, timecode):
  * segment  - 1-based segment number.
  * frame    - 0-based frame index within that segment.
  * timecode - nominal 60 fps "HH:MM:SS:FF" of that frame, i.e. the value that
               `--start-time --use-timecode` would need to land on it. Physical frame `g` of
               the section is `g / 59.94` s after the start, which is `g * 1001/1000` nominal
               frames, so one nominal FF value is skipped roughly every 1000 frames.
"""

from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

import click

from video_postprocess.timecode import VideoTimecode
from video_postprocess.video_segments import FPS_60, parse_time_str_to_frames

FPS_59_94 = Fraction(60000, 1001)
DEFAULT_FRAMES_PER_SEGMENT = 18000


def physical_frame_count(nominal_frame_count: int, framerate: Fraction) -> int:
    """Number of physical frames recorded at `framerate` while the nominal 60 fps clock
    advances by `nominal_frame_count` frames."""
    return round(nominal_frame_count * framerate / FPS_60)


def segment_lengths(total_frames: int, frames_per_segment: int) -> list[int]:
    full_segments, remainder = divmod(total_frames, frames_per_segment)
    return [frames_per_segment] * full_segments + ([remainder] if remainder else [])


def generate_segment_timecode_csv(
    start_time: str,
    end_time: str,
    output_csv: Path,
    frames_per_segment: int = DEFAULT_FRAMES_PER_SEGMENT,
    framerate: Fraction = FPS_59_94,
) -> list[int]:
    """Write the CSV and return the per-segment frame counts."""
    start_nominal = parse_time_str_to_frames(start_time, FPS_60)
    end_nominal = parse_time_str_to_frames(end_time, FPS_60)
    assert start_nominal is not None and end_nominal is not None
    if end_nominal <= start_nominal:
        error_msg = f"end-time {end_time} must be after start-time {start_time}."
        raise ValueError(error_msg)

    total_frames = physical_frame_count(end_nominal - start_nominal, framerate)
    lengths = segment_lengths(total_frames, frames_per_segment)
    nominal_per_physical = Fraction(FPS_60) / framerate

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "frame", "timecode"])
        section_frame = 0
        for segment, length in enumerate(lengths, 1):
            for frame in range(length):
                nominal_frame = start_nominal + round(section_frame * nominal_per_physical)
                timecode = VideoTimecode.from_total_frames(nominal_frame, FPS_60)
                writer.writerow([segment, frame, timecode.to_ffmpeg_format()])
                section_frame += 1

    return lengths


@click.command()
@click.option(
    "--start-time",
    type=str,
    required=True,
    help="Start timecode of the section, as passed to extract_segment_from_video.py (HH:MM:SS:FF or HH:MM:SS).",
)
@click.option(
    "--end-time",
    type=str,
    required=True,
    help="End timecode of the section, as passed to extract_segment_from_video.py (HH:MM:SS:FF or HH:MM:SS).",
)
@click.option(
    "--output-csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Path of the CSV to write.",
)
@click.option(
    "--frames-per-segment",
    type=click.IntRange(min=1),
    default=DEFAULT_FRAMES_PER_SEGMENT,
    show_default=True,
    help="Frames in each segment; the last segment takes the remainder.",
)
def main(start_time: str, end_time: str, output_csv: Path, frames_per_segment: int) -> None:
    lengths = generate_segment_timecode_csv(start_time, end_time, output_csv, frames_per_segment)
    print(f"Wrote {sum(lengths)} frames in {len(lengths)} segments to '{output_csv}'.")
    for segment, length in enumerate(lengths, 1):
        print(f"  segment {segment}: {length} frames")


if __name__ == "__main__":
    main()
