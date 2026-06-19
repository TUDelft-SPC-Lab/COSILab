#!/usr/bin/env python3
import subprocess
from pathlib import Path

import click
from tqdm import tqdm

from capturesystem.video_utils import get_video_duration_in_seconds


def fix_truncated_video(
    source_directory: Path,
    target_directory: Path,
    docker_untrunc: bool = False,
) -> None:
    """
    For each camera directory in the source directory, check if the last video was truncated because the GoPro battery
    died. If so, fix the video and save the result in target directory.
    """
    target_directory.mkdir(parents=True, exist_ok=True)

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

        if len(source_camera_video_paths) == 1:
            print(
                f"Cannot fix truncated video if there is only one video in the folder {source_camera_folder}"
            )

        try:
            # Check if the last video is truncated
            get_video_duration_in_seconds(source_camera_video_paths[-1])
        except subprocess.CalledProcessError as e:
            # If it was, use a good video (the first one) to fix it by copying the metadata from it
            if "moov atom not found" in e.stdout.decode():
                # Install from https://github.com/anthwlock/untrunc
                if docker_untrunc:
                    cmd = [
                        "docker",
                        "run",
                        "--rm",
                        "-v",
                        f"{source_camera_folder}:/mnt",
                        "untrunc:latest",
                        f"/mnt/{source_camera_video_paths[0].name}",
                        f"/mnt/{source_camera_video_paths[-1].name}",
                    ]
                else:
                    cmd = [
                        "untrunc-anthwlock",
                        str(source_camera_video_paths[0]),
                        str(source_camera_video_paths[-1]),
                    ]
                subprocess.run(cmd, check=True)

                video_stem = source_camera_video_paths[-1].stem
                video_suffix = source_camera_video_paths[-1].suffix
                fixed_video_path = source_camera_video_paths[-1].parent / (
                    f"{video_stem}{video_suffix}_fixed{video_suffix}"
                )
                target_camera_folder = (
                    target_directory / source_camera_folder.name
                )
                target_camera_folder.mkdir(parents=True, exist_ok=True)

                fixed_video_path.rename(
                    target_camera_folder / (video_stem + video_suffix)
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
    fix_truncated_video(
        source_directory=source_directory,
        target_directory=target_directory,
    )


if __name__ == "__main__":
    main()
