#!/usr/bin/env python3
import tempfile
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path

import click
from ffmpeg import FFmpeg, Progress
from tqdm import tqdm

from video_postprocess.timecode import VideoTimecode
from video_postprocess.utils import get_num_threads
from video_postprocess.video_utils import get_video_duration_in_seconds


def concat_videos(
    source_directory: Path,
    target_directory: Path,
) -> None:
    """
    For each camera directory in the source directory,
    concatenate all videos into a single video per camera.
    Save the result in the target directory with the
    same name as the camera directory.
    """
    target_directory.mkdir(parents=True, exist_ok=True)

    num_threads = get_num_threads()

    for source_camera_folder in tqdm(
        sorted(source_directory.iterdir()), desc="Video concat", unit="Camera"
    ):
        if not source_camera_folder.is_dir():
            continue

        # GoPro videos are named with capital letters, including the extension
        source_camera_video_paths: list[Path] = list(
            source_camera_folder.glob("*.mp4")
        )
        source_camera_video_paths += source_camera_folder.glob("*.MP4")
        source_camera_video_paths = sorted(source_camera_video_paths)

        if source_camera_video_paths == []:
            print(f"No videos found in {source_camera_folder}")
            continue

        target_camera_video_path = target_directory / (
            source_camera_folder.name + ".mp4"
        )

        timecode = VideoTimecode.from_video(
            video_path=source_camera_video_paths[0]
        ).to_ffmpeg_format()

        with tempfile.NamedTemporaryFile(
            suffix=".txt", dir=target_directory
        ) as temp_file:
            concat_txt = Path(temp_file.name)

            output_video_duration = 0

            with concat_txt.open("w") as f:
                for source_camera_video_path in source_camera_video_paths:
                    f.write(f"file '{source_camera_video_path.resolve()}'\n")
                    output_video_duration += get_video_duration_in_seconds(
                        source_camera_video_path
                    )

            # fmt: off
            ffmpeg = (
                FFmpeg()
                .option("y") # Overwrite the output file if it exists
                .option("threads", num_threads)
                .option("f", "concat")
                .option("safe", "0")
                .input(str(concat_txt))
                .output(
                    str(target_camera_video_path),
                    timecode=timecode,
                    c = "copy"
                )
            )
            # fmt: on

            with tqdm(desc="Concat", unit="s", leave=False, total=100) as pbar:
                # The execute() below spawns a new thread using
                # the concurrent.futures.ThreadPoolExecutor
                # we need a shared variable to be able to track
                # the progress inside the on_progress callback
                prev_frame_idx_shared = Value("f", 0)

                def thread_initializer(args: Synchronized) -> None:
                    global sval  # noqa: PLW0603
                    sval = args

                @ffmpeg.on("progress")
                def on_progress(progress: Progress) -> None:
                    # Frames does not work with concat, so we use time instead.
                    # The time is converted to a percentage of
                    # the total output video duration for better progress report.
                    time_in_seconds = progress.time.total_seconds()
                    pbar.update(
                        int(
                            (
                                (time_in_seconds - prev_frame_idx_shared.value)  # noqa: B023
                                / output_video_duration  # noqa:B023
                            )
                            * 100
                        )
                    )
                    prev_frame_idx_shared.value = time_in_seconds  # noqa: B023

                ffmpeg.execute(
                    initializer=thread_initializer,
                    initargs=(prev_frame_idx_shared,),
                )


@click.command()
@click.option(
    "--source-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    help="Directory that contains all input data files",
    required=True,
)
@click.option(
    "--target-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path to output directory",
    required=True,
)
def main(
    source_directory: Path,
    target_directory: Path,
) -> None:
    concat_videos(
        source_directory=source_directory,
        target_directory=target_directory,
    )


if __name__ == "__main__":
    main()
