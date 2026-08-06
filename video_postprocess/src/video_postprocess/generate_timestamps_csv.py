from __future__ import annotations

import csv
from fractions import Fraction
from pathlib import Path

import click
from tqdm import tqdm

from video_postprocess.timecode import format_exact_seconds
from video_postprocess.utils import get_camera_to_process
from video_postprocess.video_segments import (
    VideoFileInfo,
    collect_camera_video_infos,
    timecode_at_local_frame,
)

FRACTION_DIGITS = 6


def write_timestamps_csv_for_file(info: VideoFileInfo) -> None:
    """Write a sibling CSV next to `info.path` with one row per frame of that file: its embedded
    (camera) timecode and the corresponding world (real elapsed) timestamp.

    Both columns describe the same instant, computed from the file's own start timecode advanced
    by the file's actual (physical) framerate rather than the nominal 60 fps the timecode's FF
    field counts against -- this is the same 59.94 fps correction `video_segments` applies when
    locating frames, just expressed per-frame instead of per-segment.
    """
    csv_path = info.path.with_suffix(".csv")
    start_seconds = info.start_timecode.to_exact_seconds()
    frame_duration = Fraction(1, 1) / info.framerate

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_timecode", "world_timestamp"])
        for local_frame in tqdm(
            range(info.frame_count), desc=info.path.name, unit="Frame", leave=False
        ):
            original_timecode = timecode_at_local_frame(info, local_frame)
            world_seconds = start_seconds + local_frame * frame_duration
            writer.writerow(
                [
                    original_timecode.to_ffmpeg_format(),
                    format_exact_seconds(world_seconds, FRACTION_DIGITS),
                ]
            )


def generate_timestamps_csv(source_directory: Path) -> None:
    camera_to_process = get_camera_to_process()

    camera_directories = [
        camera_dir
        for camera_dir in sorted(source_directory.iterdir())
        if camera_dir.is_dir() and (camera_to_process is None or camera_dir.name == camera_to_process)
    ]

    for camera_directory in tqdm(camera_directories, desc="Timestamp CSVs", unit="Camera"):
        infos = collect_camera_video_infos(camera_directory)
        if not infos:
            print(f"No videos found for camera '{camera_directory.name}', skipping.")
            continue

        for info in infos:
            write_timestamps_csv_for_file(info)


@click.command()
@click.option(
    "--source-directory",
    type=click.Path(file_okay=False, dir_okay=True, writable=False, path_type=Path),
    help=(
        "Directory that contains one subdirectory per camera, each holding "
        "that camera's video file(s). This is expected to be a "
        "raw_sensor_data/gopro_data/<mingle folder>"
    ),
    required=True,
)
def main(source_directory: Path) -> None:
    generate_timestamps_csv(source_directory=source_directory)


if __name__ == "__main__":
    main()
