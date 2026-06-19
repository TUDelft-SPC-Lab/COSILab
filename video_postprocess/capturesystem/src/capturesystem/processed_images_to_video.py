#!/usr/bin/env python3
import os
import shutil
from enum import Enum
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from tempfile import TemporaryDirectory

import click
from ffmpeg import FFmpeg, Progress
from tqdm import tqdm

from capturesystem.timecode import VideoTimecode, timecode_microseconds_to_ff
from capturesystem.utils import (
    SimplePath,
    get_camera_to_process,
    get_num_threads,
)


class VideoCodecs(str, Enum):
    H264_CPU = "libx264"
    H265_CPU = "libx265"

    H264_GPU = "h264_nvenc"
    H265_GPU = "hevc_nvenc"


def convert_png_to_mp4(
    source_directory: Path,
    target_directory: Path,
    tmp_directory: Path | None = None,
    frame_suffix: str = ".png",
    frame_rate: float = 60,
    video_codec: VideoCodecs = VideoCodecs.H265_GPU,
    crf: int = 18,
    delete_source_directory: bool = False,
    rename_images: bool = False,
) -> None:
    timecode = None
    target_directory.mkdir(parents=True, exist_ok=True)
    if tmp_directory is None:
        tmp_directory = target_directory

    camera_to_process = get_camera_to_process()

    if camera_to_process is None:
        source_camera_directories = sorted(source_directory.iterdir())
    else:
        source_camera_directories = [source_directory / camera_to_process]

    if rename_images:

        def PathContext(prefix: str) -> SimplePath:  # noqa: ARG001
            return SimplePath(source_camera_directory)
    else:

        def PathContext(prefix: str) -> TemporaryDirectory:
            return TemporaryDirectory(dir=tmp_directory, prefix=prefix)

    for source_camera_directory in tqdm(
        source_camera_directories, desc="Image to video", unit="Camera"
    ):
        if not source_camera_directory.is_dir():
            continue

        with PathContext(
            prefix=source_camera_directory.name + "_tmp_for_mp4_"
        ) as tmp_dir:
            tmp_png_directory = Path(tmp_dir)
            frame_idx = 0
            for frame_path in sorted(source_camera_directory.iterdir()):
                if frame_path.is_file() and frame_path.suffix == frame_suffix:
                    target_frame = (
                        tmp_png_directory / f"{frame_idx:09d}{frame_suffix}"
                    )
                    if rename_images:
                        frame_path.rename(target_frame)
                    else:
                        target_frame.hardlink_to(frame_path)
                    frame_idx += 1
                    if timecode is None:
                        if frame_rate != int(frame_rate):
                            # Have a look at drop frames and whatnot in ffmpeg if you want to implement this
                            error_msg = "Only integer frame rates are supported"
                            raise NotImplementedError(error_msg)

                        # Timecode format is MMDDYYYYHHMMSSFFFFFF
                        # For example: 04162024103931800001 -> 04/16/2024 10:39:31:800001
                        timecode = frame_path.stem[
                            len("MMDDYYYY") :
                        ]  # Remove the MMDDYYYY
                        frame_num = timecode_microseconds_to_ff(
                            int(timecode[6:]), frame_rate
                        )
                        timecode = VideoTimecode(
                            hours=int(timecode[:2]),
                            minutes=int(timecode[2:4]),
                            seconds=int(timecode[4:6]),
                            frames=frame_num,
                            fps=frame_rate,
                        ).to_ffmpeg_format()

            assert timecode is not None, "Could not find any frames to encode"

            # NB. the nvidia codecs ignore the crf flag, so we set the compression via the quantization qp flag.
            # This crf is actually better because it allows the qp to change dynamically per frame but the GPU encoding
            # is much faster around an order of magnitude than the CPU encoding
            if crf == 0:
                # Lossless encoding of the images
                if video_codec in [VideoCodecs.H264_GPU, VideoCodecs.H265_GPU]:
                    compression_kwargs = {"qp": 0}
                elif video_codec == VideoCodecs.H265_CPU:
                    compression_kwargs = {"x265-params": "lossless=1"}
                    if os.getenv("SLURM_JOB_NAME") is not None:
                        print(
                            "--- Number of threads has no effect with CPU encoding ---"
                        )
                else:
                    compression_kwargs = {"crf": 0}
            else:
                if video_codec in [VideoCodecs.H264_GPU, VideoCodecs.H265_GPU]:
                    # With a qp of crf + 7 we get roughly the same file sizes as with GPU encoding
                    # as we do with CPU encoding
                    compression_kwargs = {"qp": min(crf + 7, 51)}
                else:
                    # A crf of 17 or 18 gives perceptually lossless compression
                    compression_kwargs = {"crf": crf}
                    if os.getenv("SLURM_JOB_NAME") is not None:
                        print(
                            "--- Number of threads has no effect with CPU encoding ---"
                        )

            # fmt: off
            ffmpeg = (
                FFmpeg()
                .option("y") # Overwrite the output file if it exists
                .option("threads", get_num_threads())
                .input(str(tmp_png_directory / f'%09d{frame_suffix}'), framerate=frame_rate)
                .output(
                    str(target_directory / (source_camera_directory.name + ".mp4")),
                    pix_fmt='yuv420p',
                    timecode=timecode,
                    vcodec=video_codec.value,
                    **compression_kwargs,
                )
            )
            # fmt: on

            if video_codec in [VideoCodecs.H264_GPU, VideoCodecs.H265_GPU]:
                ffmpeg = ffmpeg.option("hwaccel", "cuda").option(
                    "hwaccel_output_format", "cuda"
                )

            with tqdm(
                total=frame_idx, desc="Encoding", unit="Frame", leave=False
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

        if delete_source_directory:
            shutil.rmtree(source_camera_directory)


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
    "--frame-suffix",
    default=".png",
    type=str,
    help="The suffix of the image files in the source-directory",
)
@click.option(
    "--frame-rate",
    default=60,
    type=float,
    help="The frame rate of the output video",
)
@click.option(
    "--video-codec",
    default=VideoCodecs.H265_GPU,
    type=click.Choice(VideoCodecs),
    help="The video codec to use",
)
@click.option(
    "--crf",
    default=18,
    type=int,
    help="The encoding compression quality 0: lossless, 51: maximum compression",
)
@click.option(
    "--tmp-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path to where temporary data will be created",
)
@click.option("--delete-source-directory", is_flag=True)
@click.option("--rename-images", is_flag=True)
def main(
    source_directory: Path,
    target_directory: Path,
    frame_suffix: str,
    frame_rate: float,
    video_codec: VideoCodecs,
    crf: int,
    delete_source_directory: bool,
    tmp_directory: Path | None,
    rename_images: bool,
) -> None:
    convert_png_to_mp4(
        source_directory=source_directory,
        target_directory=target_directory,
        frame_suffix=frame_suffix,
        frame_rate=frame_rate,
        video_codec=video_codec,
        crf=crf,
        tmp_directory=tmp_directory,
        delete_source_directory=delete_source_directory,
        rename_images=rename_images,
    )


if __name__ == "__main__":
    main()
