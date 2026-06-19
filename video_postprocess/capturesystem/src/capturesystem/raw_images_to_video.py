#!/usr/bin/env python3
from pathlib import Path
from tempfile import TemporaryDirectory

import click

from capturesystem.processed_images_to_video import (
    VideoCodecs,
    convert_png_to_mp4,
)
from capturesystem.raw_images_to_processed_images import convert_dng_to_png
from capturesystem.utils import SimplePath


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
    "--png-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path to output directory for the png files",
)
@click.option(
    "--tmp-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path to where temporary data will be created",
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
    "--frame-rate",
    default=60,
    type=float,
    help="The frame rate of the output video",
)
def main(
    source_directory: Path,
    target_directory: Path,
    png_directory: Path | None,
    tmp_directory: Path | None,
    video_codec: VideoCodecs,
    crf: int,
    frame_rate: float,
) -> None:
    target_directory.mkdir(exist_ok=True, parents=True)
    if png_directory is None:
        if tmp_directory is None:
            tmp_directory = target_directory
        path_manager = TemporaryDirectory(
            dir=tmp_directory, prefix=source_directory.name + "_png_"
        )
    else:
        path_manager = SimplePath(png_directory)

    with path_manager as tmp_dir:
        png_directory = Path(tmp_dir)
        convert_dng_to_png(source_directory, png_directory)
        convert_png_to_mp4(
            png_directory,
            target_directory,
            video_codec=video_codec,
            crf=crf,
            tmp_directory=tmp_directory,
            frame_rate=frame_rate,
        )


if __name__ == "__main__":
    main()
