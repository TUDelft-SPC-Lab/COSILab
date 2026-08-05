from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import click
from ffmpeg import FFmpeg, Progress
from tqdm import tqdm

from video_postprocess.utils import get_camera_to_process, get_num_threads
from video_postprocess.video_segments import (
    VideoFileInfo,
    collect_camera_video_infos,
    resolve_end_position,
    resolve_start_position,
)


def compute_segment_frame_count(
    local_start: int,
    local_end: int,
    every_n_frames: int | None,
    phase_offset: int = 0,
) -> int:
    """Count frames selected by `select=not(mod(n+phase_offset\\,every_n_frames))` for n in
    [0, total). `phase_offset` carries the sampling stride's phase across file boundaries, so a
    multi-file segment keeps sampling every Nth frame of the whole requested range rather than
    restarting the stride at 0 for each file."""
    total = local_end - local_start
    if total <= 0:
        return 0
    if every_n_frames is None:
        return total
    first_match = (-phase_offset) % every_n_frames
    if first_match >= total:
        return 0
    return (total - 1 - first_match) // every_n_frames + 1


def extract_frames_from_file(
    video_path: Path,
    camera_target_directory: Path,
    local_start: int,
    local_end: int,
    framerate: Fraction | int | float,
    start_number: int,
    every_n_frames: int | None,
    phase_offset: int,
    num_threads: int,
    pbar: tqdm,
) -> int:
    """Extract PNG frames for [local_start, local_end) of a single file, numbered continuously
    from `start_number` so multiple files of the same camera concatenate into one sequence.

    `phase_offset` is the sampling stride's phase carried over from earlier files of the same
    camera (see `compute_segment_frame_count`), so `every_n_frames` samples evenly across the
    whole requested range instead of restarting at each file boundary."""
    frame_count = compute_segment_frame_count(local_start, local_end, every_n_frames, phase_offset)
    if frame_count <= 0:
        return 0

    output_options: dict[str, str | int] = {
        "start_number": start_number,
        "frames:v": frame_count,
    }
    if every_n_frames is not None:
        offset_term = f"+{phase_offset}" if phase_offset else ""
        output_options["vf"] = f"select=not(mod(n{offset_term}\\,{every_n_frames}))"
        output_options["vsync"] = "vfr"

    # fmt: off
    ffmpeg = (
        FFmpeg()
        .option("y")  # Overwrite the output file if it exists
        .option("threads", num_threads)
        .option("ss", f"{float(local_start / framerate):.6f}")
        .input(str(video_path))
        .output(
            str(camera_target_directory / "%09d.png"),
            **output_options,
        )
    )
    # fmt: on

    prev_frame_idx = [0]

    @ffmpeg.on("progress")
    def on_progress(progress: Progress) -> None:
        pbar.update(progress.frame - prev_frame_idx[0])
        prev_frame_idx[0] = progress.frame

    ffmpeg.execute()

    return frame_count


def split_camera_into_frames(
    camera_directory: Path,
    target_directory: Path,
    start_time: str | None,
    end_time: str | None,
    use_timecode: bool,
    every_n_frames: int | None,
    num_threads: int,
) -> None:
    infos = collect_camera_video_infos(camera_directory)
    if not infos:
        print(f"No videos found for camera '{camera_directory.name}', skipping.")
        return

    start = resolve_start_position(infos, start_time, use_timecode)
    end = resolve_end_position(infos, end_time, use_timecode)

    if (start.file_index, start.local_frame) >= (end.file_index, end.local_frame):
        print(f"Requested section for camera '{camera_directory.name}' is empty, skipping.")
        return

    camera_target_directory = target_directory / camera_directory.name
    camera_target_directory.mkdir(parents=True, exist_ok=True)

    # phase_offset carries the every_n_frames stride's phase across file boundaries: it's the
    # number of (pre-sampling) frames already consumed by earlier segments of this camera, mod
    # every_n_frames, so the sampling grid stays anchored to the start of the whole requested
    # range instead of restarting at 0 for each file.
    segments: list[tuple[VideoFileInfo, int, int, int]] = []
    raw_offset = 0
    for file_index in range(start.file_index, end.file_index + 1):
        info = infos[file_index]
        segment_start = start.local_frame if file_index == start.file_index else 0
        segment_end = end.local_frame if file_index == end.file_index else info.frame_count
        if segment_start >= segment_end:
            continue
        phase_offset = raw_offset % every_n_frames if every_n_frames else 0
        segments.append((info, segment_start, segment_end, phase_offset))
        raw_offset += segment_end - segment_start

    total_frames = sum(
        compute_segment_frame_count(segment_start, segment_end, every_n_frames, phase_offset)
        for _, segment_start, segment_end, phase_offset in segments
    )

    with tqdm(
        desc="Splitting", unit="Frame", leave=False, total=total_frames
    ) as pbar:
        frame_counter = 0
        for info, segment_start, segment_end, phase_offset in segments:
            frame_counter += extract_frames_from_file(
                video_path=info.path,
                camera_target_directory=camera_target_directory,
                local_start=segment_start,
                local_end=segment_end,
                framerate=info.framerate,
                start_number=frame_counter,
                every_n_frames=every_n_frames,
                phase_offset=phase_offset,
                num_threads=num_threads,
                pbar=pbar,
            )


def split_video_into_frames(
    source_directory: Path,
    target_directory: Path,
    start_time: str | None = None,
    end_time: str | None = None,
    use_timecode: bool = False,
    every_n_frames: int | None = None,
) -> None:
    camera_to_process = get_camera_to_process()

    target_directory.mkdir(parents=True, exist_ok=True)

    camera_directories = [
        camera_dir
        for camera_dir in sorted(source_directory.iterdir())
        if camera_dir.is_dir() and (camera_to_process is None or camera_dir.name == camera_to_process)
    ]

    num_threads = get_num_threads()

    for camera_directory in tqdm(
        camera_directories, desc="Video to frames", unit="Camera"
    ):
        split_camera_into_frames(
            camera_directory=camera_directory,
            target_directory=target_directory,
            start_time=start_time,
            end_time=end_time,
            use_timecode=use_timecode,
            every_n_frames=every_n_frames,
            num_threads=num_threads,
        )


@click.command()
@click.option(
    "--source-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    help=(
        "Directory that contains one subdirectory per camera, each holding "
        "that camera's video file(s). This is expected to be a "
        "raw_sensor_data/gopro_data/<mingle folder>"
    ),
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
    help="Start time in format HH:MM:SS:FF, HH:MM:SS, MM:SS or SS",
)
@click.option(
    "--end-time",
    type=str,
    default=None,
    help="End time in format HH:MM:SS:FF, HH:MM:SS, MM:SS or SS",
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
        start_time=start_time,
        end_time=end_time,
        use_timecode=use_timecode,
        every_n_frames=every_n_frames,
    )


if __name__ == "__main__":
    main()
