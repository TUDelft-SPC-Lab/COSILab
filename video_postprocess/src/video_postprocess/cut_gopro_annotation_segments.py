"""Cut GoPro annotation segments from concatenated camera videos."""

from datetime import datetime, timedelta
from pathlib import Path

import click

from video_postprocess.extract_segment_from_video import (
    extract_segment_from_video,
    str_to_timedelta,
)


def fmt_time(value: datetime) -> str:
    return value.strftime("%H:%M:%S")


def fmt_segment(value: datetime) -> str:
    return value.strftime("%H%M%S")


def extract_segments(
    source_directory: Path,
    target_directory: Path,
    start: str,
    end: str,
) -> None:
    current = datetime.strptime(start, "%H:%M:%S")
    stop = datetime.strptime(end, "%H:%M:%S")
    step = timedelta(seconds=30)

    while current < stop:
        next_time = min(current + step, stop)
        segment_name = (
            f"seg_{fmt_segment(current)}_{fmt_segment(next_time)}"
        )
        segment_directory = target_directory / segment_name
        segment_directory.mkdir(parents=True, exist_ok=True)

        print(
            f"Extracting {fmt_time(current)}-{fmt_time(next_time)} "
            f"from {source_directory} into {segment_directory}",
            flush=True,
        )
        extract_segment_from_video(
            source_directory=source_directory,
            target_directory=segment_directory,
            start_time=str_to_timedelta(fmt_time(current)),
            end_time=str_to_timedelta(fmt_time(next_time)),
            use_timecode=True,
            every_n_frames=None,
        )
        current = next_time


@click.command()
@click.option(
    "--group-06-10-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    required=True,
)
@click.option(
    "--group-01-05-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    required=True,
)
@click.option(
    "--target-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    required=True,
)
def main(
    group_06_10_directory: Path,
    group_01_05_directory: Path,
    target_directory: Path,
) -> None:
    target_directory.mkdir(parents=True, exist_ok=True)
    extract_segments(
        group_06_10_directory,
        target_directory,
        "13:45:00",
        "14:20:00",
    )
    extract_segments(
        group_01_05_directory,
        target_directory,
        "14:52:00",
        "15:27:00",
    )


if __name__ == "__main__":
    main()
