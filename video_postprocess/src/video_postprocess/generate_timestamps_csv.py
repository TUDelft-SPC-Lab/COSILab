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


def write_timestamps_csv_for_file(info: VideoFileInfo, world_start_seconds: Fraction) -> None:
    """Write a sibling CSV next to `info.path` with one row per frame of that file: its embedded
    (camera) timecode and the corresponding world (real elapsed) timestamp.

    `original_timecode` is this file's own embedded clock, advanced from its own start timecode --
    it does not know about earlier files. `world_timestamp` instead starts from
    `world_start_seconds`, the real elapsed time already accumulated over the camera's earlier
    files (see `generate_timestamps_csv`), so it stays correct across a multi-file recording even
    though a camera's true (e.g. 59.94 fps) capture rate drifts against the nominal 60 fps its
    embedded timecode counts against -- the same cross-file correction `video_segments` applies
    via `physical_frames_before` when locating frames.
    """
    csv_path = info.path.with_suffix(".csv")
    frame_duration = Fraction(1, 1) / info.framerate

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_timecode", "world_timestamp"])
        for local_frame in tqdm(
            range(info.frame_count), desc=info.path.name, unit="Frame", leave=False
        ):
            original_timecode = timecode_at_local_frame(info, local_frame)
            world_seconds = world_start_seconds + local_frame * frame_duration
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

        # Real elapsed time accumulates from the first file's own embedded start timecode
        world_seconds = infos[0].start_timecode.to_exact_seconds()
        for info in infos:
            write_timestamps_csv_for_file(info, world_seconds)
            world_seconds += Fraction(info.frame_count, 1) / info.framerate


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
