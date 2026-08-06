from __future__ import annotations

import csv
from pathlib import Path

import click
from tqdm import tqdm

from video_postprocess.timecode import timecode_ff_to_microseconds
from video_postprocess.utils import get_camera_to_process
from video_postprocess.video_segments import (
    VideoFileInfo,
    collect_camera_video_infos,
    timecode_at_local_frame,
    world_timecode_at_local_frame,
)


def write_timestamps_csv_for_file(info: VideoFileInfo) -> None:
    """Write a sibling CSV next to `info.path` with one row per frame of that file: the timecode
    ffmpeg actually reaches when seeking there, and the `--start-time --use-timecode` value that
    would produce that seek.

    `original_timecode` is this file's own embedded clock, advanced from its own start timecode by
    the frame's true elapsed time -- i.e. the corrected time `split_video_into_frames` actually
    passes to ffmpeg's `-ss` when targeting this frame. `world_timecode` is the inverse: the
    nominal (60 fps, drift-uncorrected) timecode a caller would need to pass as `--start-time` to
    land on this exact frame, undoing the same cross-file `physical_frames_before` correction
    `video_segments.correct_for_fps_59_94` applies when locating frames. `world_timestamp` is the
    same value as `world_timecode`, with its `FF` frame field converted to microseconds.
    """
    csv_path = info.path.with_suffix(".csv")

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["original_timecode", "world_timecode", "world_timestamp"])
        for local_frame in tqdm(
            range(info.frame_count), desc=info.path.name, unit="Frame", leave=False
        ):
            original_timecode = timecode_at_local_frame(info, local_frame)
            world_timecode = world_timecode_at_local_frame(info, local_frame)
            world_microseconds = round(timecode_ff_to_microseconds(world_timecode.frames, world_timecode.fps))
            world_timestamp = (
                f"{world_timecode.hours:02}:{world_timecode.minutes:02}:"
                f"{world_timecode.seconds:02}.{world_microseconds:06}"
            )
            writer.writerow(
                [
                    original_timecode.to_ffmpeg_format(),
                    world_timecode.to_ffmpeg_format(),
                    world_timestamp,
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
