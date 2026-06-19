#!/usr/bin/env python3
import json
from enum import Enum
from pathlib import Path
from typing import Any

import click

from capturesystem.camera_calibration.easymocap import (
    save_extrinsic_easymocap,
    save_intrinsic_easymocap,
)
from capturesystem.camera_calibration.idiap import (
    save_extrinsic_idiap,
    save_intrinsic_idiap,
)


class CameraFormat(str, Enum):
    EASYMOCAP = "easymocap"
    IDIAP = "idiap"


def load_data(
    calibrator_file: Path, cameras_names: None | str
) -> list[dict[str, Any]]:
    with calibrator_file.open() as f:
        cam_data = json.load(f)
    cam_data = cam_data["Calibration"]["cameras"]

    if (
        cam_data[0]["model"]["polymorphic_name"]
        != "libCalib::CameraModelOpenCV"
    ):
        error_msg = f"Expected 'libCalib::CameraModelOpenCV', got '{cam_data[0]['model']['polymorphic_name']}'"
        raise RuntimeError(error_msg)

    if cameras_names is None:
        print(
            f"Getting camera names from the folders that are present in {calibrator_file.parent}"
        )
        cameras_names_processed = [
            f.name
            for f in sorted(calibrator_file.parent.iterdir())
            if f.is_dir()
        ]
    else:
        cameras_names_processed = cameras_names.split(" ")

    for i, name in enumerate(cameras_names_processed):
        cam_data[i]["name"] = name

    return cam_data


def save_intrinsic(
    cam_data: list[dict[str, Any]],
    camera_format: CameraFormat,
    output_directory: Path,
) -> None:
    if camera_format == CameraFormat.EASYMOCAP:
        save_intrinsic_easymocap(cam_data, output_directory)
    elif camera_format == CameraFormat.IDIAP:
        save_intrinsic_idiap(cam_data, output_directory)
    else:
        error_msg = f"Unknown camera format {camera_format}"
        raise RuntimeError(error_msg)


def save_extrinsic(
    all_cam_data: list[dict[str, Any]],
    camera_format: CameraFormat,
    output_directory: Path,
) -> None:
    if camera_format == CameraFormat.EASYMOCAP:
        save_extrinsic_easymocap(all_cam_data, output_directory)
    elif camera_format == CameraFormat.IDIAP:
        save_extrinsic_idiap(all_cam_data, output_directory)
    else:
        error_msg = f"Unknown camera format {camera_format}"
        raise RuntimeError(error_msg)


@click.command()
@click.option(
    "--calibrator-file",
    type=click.Path(
        file_okay=True, dir_okay=False, writable=False, path_type=Path
    ),
    help="Directory that contains all input data files",
    required=True,
)
@click.option(
    "--camera-format",
    default=CameraFormat.EASYMOCAP,
    type=click.Choice(CameraFormat),
    help="The output format to use",
)
@click.option(
    "--output-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=True, path_type=Path
    ),
    help="Path to output directory",
    required=True,
)
@click.option(
    "--cameras-names",
    type=str,
    help="The names of cameras",
    required=False,
)
def main(
    calibrator_file: Path,
    camera_format: CameraFormat,
    output_directory: Path,
    cameras_names: None | str,
) -> None:
    """
    Convert a json calibrator file with OpenCV camera parameters as defined in
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    into camera parameter files that are compatible with EasyMocap, idiap, etc.

    The converted files are stored in the --output-directory
    """
    data = load_data(calibrator_file, cameras_names)

    output_directory.mkdir(parents=True, exist_ok=True)

    save_intrinsic(data, camera_format, output_directory)
    save_extrinsic(data, camera_format, output_directory)


if __name__ == "__main__":
    main()
