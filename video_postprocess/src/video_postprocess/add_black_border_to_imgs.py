#!/usr/bin/env python3
from pathlib import Path

import click
import numpy as np
from PIL import Image
from tqdm import tqdm


def add_black_border_to_image(
    image_path: Path, border_width: int, border_height: int, output_path: Path
) -> None:
    rgb_black = (0, 0, 0)
    with Image.open(image_path) as img:
        pixels = np.array(img)
        width, height = img.size

        pixels[:, :border_width, 0:3] = rgb_black
        pixels[:border_height, :, 0:3] = rgb_black
        pixels[:, width - border_width :, 0:3] = rgb_black
        pixels[height - border_height :, :, 0:3] = rgb_black

        out_img = Image.fromarray(pixels)
        out_img.save(output_path)


def process_folders_with_border(
    source_directory: Path,
    target_directory: Path,
    border_width: int,
    border_height: int,
) -> None:
    """
    For each camera directory in the source directory, copy all images to the target directory,
    replacing the border with black pixels.
    """
    target_directory.mkdir(parents=True, exist_ok=True)

    for source_camera_folder in tqdm(
        sorted(source_directory.iterdir()), desc="Add border", unit="Camera"
    ):
        if not source_camera_folder.is_dir():
            continue

        target_camera_folder = target_directory / source_camera_folder.name
        target_camera_folder.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(source_camera_folder.glob("*.png"))
        if not image_paths:
            print(f"No images found in {source_camera_folder}")
            continue

        for image_path in tqdm(
            image_paths,
            desc=f"{source_camera_folder.name}",
            unit="img",
            leave=False,
        ):
            output_path = target_camera_folder / image_path.name
            add_black_border_to_image(
                image_path, border_width, border_height, output_path
            )


@click.command()
@click.option(
    "--source-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    help="Directory that contains all input camera folders",
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
    "--border-width",
    type=int,
    help="Width of the border to replace with black pixels",
    required=True,
)
@click.option(
    "--border-height",
    type=int,
    help="Height of the border to replace with black pixels",
    required=True,
)
def main(
    source_directory: Path,
    target_directory: Path,
    border_width: int,
    border_height: int,
) -> None:
    process_folders_with_border(
        source_directory=source_directory,
        target_directory=target_directory,
        border_width=border_width,
        border_height=border_height,
    )


if __name__ == "__main__":
    main()
