from datetime import timedelta
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path

import click
from ffmpeg import FFmpeg, Progress
from tqdm import tqdm

from video_postprocess.timecode import (
    VideoTimecode,
)
from video_postprocess.utils import get_camera_to_process, get_num_threads
from video_postprocess.video_utils import (
    get_video_duration_in_seconds,
    get_video_framerate,
)


def str_to_timedelta(time_str: str | None) -> timedelta | None:
    """Convert a time string in HH:MM:SS, MM:SS or SS format to a timedelta object."""
    if time_str is None:
        return None

    parts = list(map(int, time_str.split(":")))
    if len(parts) == 2:
        parts = [0, parts[0], parts[1]]
    elif len(parts) == 1:
        parts = [0, 0, parts[0]]
    elif len(parts) != 3:
        error_msg = "Time string must be in HH:MM:SS, MM:SS or SS format."
        raise ValueError(error_msg)
    return timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])


def split_video_into_frames(
    source_directory: Path,
    target_directory: Path,
    start_time: timedelta | None = None,
    end_time: timedelta | None = None,
    use_timecode: bool = False,
    every_n_frames: int | None = None,
) -> None:
    camera_to_process = get_camera_to_process()

    if camera_to_process is None:
        source_camera_video_paths = sorted(source_directory.glob("*.mp4"))
    else:
        source_camera_video_paths = [
            source_directory / (camera_to_process + ".mp4")
        ]

    target_directory.mkdir(parents=True, exist_ok=True)

    num_threads = get_num_threads()

    for source_camera_video_path in tqdm(
        source_camera_video_paths, desc="Video to frames", unit="Camera"
    ):
        camera_target_directory = (
            target_directory / source_camera_video_path.stem
        )
        camera_target_directory.mkdir(exist_ok=True)

        output_options = {}
        if every_n_frames is not None:
            output_options["vf"] = f"select=not(mod(n\\,{every_n_frames}))"
            output_options["vsync"] = "vfr"

        # fmt: off
        ffmpeg = (
            FFmpeg()
            .option("y") # Overwrite the output file if it exists
            .option("threads", num_threads)
            .input(str(source_camera_video_path))
            .output(
                str(camera_target_directory / "%09d.png"),
                **output_options,
            )
        )
        # fmt: on

        if use_timecode is True:
            if start_time is None and end_time is None:
                print(
                    "--use-timecode ignored, start-time and end-time were not provided."
                )
            else:
                timecode = VideoTimecode.from_video(
                    source_camera_video_path
                ).to_timedelta()
                if start_time is not None:
                    start_time_i = start_time - timecode
                if end_time is not None:
                    end_time_i = end_time - timecode
        else:
            start_time_i = start_time
            end_time_i = end_time

        framerate = get_video_framerate(video_path=source_camera_video_path)
        video_duration = get_video_duration_in_seconds(source_camera_video_path)
        total_frames = int(video_duration * framerate)
        num_frames_to_process = total_frames

        if start_time_i is not None:
            if start_time_i.total_seconds() < 0:
                print(
                    f"Start time '{start_time_i}' is negative for video '{source_camera_video_path.name}', skipping."
                )
                continue
            ffmpeg = ffmpeg.option("ss", str(start_time_i))
            num_frames_to_process -= int(
                start_time_i.total_seconds() * framerate
            )

        if end_time_i is not None:
            if end_time_i.total_seconds() < 0:
                print(
                    f"End time '{end_time_i}' is negative for video '{source_camera_video_path.name}', skipping."
                )
                continue
            if end_time_i.total_seconds() > video_duration:
                print(
                    f"End time '{end_time_i}' exceeds video duration '{video_duration}' for video '{source_camera_video_path.name}', skipping."
                )
                continue
            ffmpeg = ffmpeg.option("to", str(end_time_i))
            num_frames_to_process -= total_frames - int(
                end_time_i.total_seconds() * framerate
            )

        if every_n_frames is not None:
            num_frames_to_process = round(
                num_frames_to_process / every_n_frames
            )

        with tqdm(
            desc="Splitting",
            unit="Frame",
            leave=False,
            total=num_frames_to_process,
        ) as pbar:
            # The execute() below spawns a new thread using the concurrent.futures.ThreadPoolExecutor
            # we need a shared variable to be able to track the progress inside the on_progress callback
            prev_frame_idx_shared = Value("i", 0)

            def thread_initializer(args: Synchronized) -> None:
                global sval  # noqa: PLW0603
                sval = args

            @ffmpeg.on("progress")
            def on_progress(progress: Progress) -> None:
                pbar.update(progress.frame - prev_frame_idx_shared.value)  # noqa: B023
                prev_frame_idx_shared.value = progress.frame  # noqa: B023

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
@click.option(
    "--start-time",
    type=str,
    default=None,
    help="Start time in format HH:MM:SS, MM:SS or SS",
)
@click.option(
    "--end-time",
    type=str,
    default=None,
    help="End time in format HH:MM:SS, MM:SS or SS",
)
@click.option(
    "--use-timecode",
    is_flag=True,
    help="Whether the start-time and end-time should be interpreted as timecodes from the video",
)
@click.option(
    "--every-n-frames",
    type=int,
    default=None,
    help="If specified, only every Nth frame will be extracted from the video",
)
def main(
    source_directory: Path,
    target_directory: Path,
    start_time: str | None,
    end_time: str | None,
    use_timecode: bool,
    every_n_frames: int | None,
) -> None:
    split_video_into_frames(
        source_directory=source_directory,
        target_directory=target_directory,
        start_time=str_to_timedelta(start_time),
        end_time=str_to_timedelta(end_time),
        use_timecode=use_timecode,
        every_n_frames=every_n_frames,
    )


if __name__ == "__main__":
    main()
