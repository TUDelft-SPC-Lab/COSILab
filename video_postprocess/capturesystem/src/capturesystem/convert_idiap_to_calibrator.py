#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any

import click
import numpy as np
from scipy.spatial.transform import Rotation as R

from capturesystem.camera_calibration.idiap import (
    convert_extrinsics_to_calibrator,
)


def load_json(calibrator_file: Path) -> dict[str, Any]:
    with calibrator_file.open() as f:
        return json.load(f)


def move_origin_to_first_camera(
    rvec1: np.ndarray,
    tvec1: np.ndarray,
    rvec2: np.ndarray,
    tvec2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform extrinsics of camera2 so that camera1 is at origin.
    In other words, transform the extrinsics of camera2 to the coordinate system of camera1.

    @param rvec1: rotation vector of the first camera (in radians)
    @param tvec1: translation vector of the first camera (in meters)
    @param rvec2: rotation vector of the second camera (in radians)
    @param tvec2: translation vector of the second camera (in meters)
    @rtype: transformed rotation and translation vectors
    """
    rot1 = R.from_rotvec(rvec1).as_matrix()
    rot2 = R.from_rotvec(rvec2).as_matrix()

    # Rotation matrices are orthogonal, so the transpose is the inverse.
    rot1_inv = rot1.T

    rot2_new = rot1_inv @ rot2
    t2_new = rot1_inv @ (tvec2 - tvec1)

    rvec2_new = R.from_matrix(rot2_new).as_rotvec()

    return rvec2_new, t2_new


def convert_translation_from_cm_to_m(translation: np.ndarray) -> np.ndarray:
    """
    Convert the translation from centimeters to meters.

    @param translation: translation vector in centimeters
    @rtype: new translation in meters
    """
    return translation / 100.0


def zero_t_and_rot() -> dict[str, Any]:
    """
    Create a dict with translation and rotation set to zero.

    @return: new extrinsics with zero translation and rotation
    """
    return {
        "rotation": {"rx": 0.0, "ry": 0.0, "rz": 0.0},
        "translation": {"x": 0.0, "y": 0.0, "z": 0.0},
    }


def format_floats(obj: dict, precision: int = 2) -> dict:
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: format_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [format_floats(v) for v in obj]
    return obj


@click.command()
@click.option(
    "--idiap-extrinsics-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    help="Directory that contains all input data files",
    required=True,
)
@click.option(
    "--calibrator-intrinsics-directory",
    type=click.Path(
        file_okay=False, dir_okay=True, writable=False, path_type=Path
    ),
    help="Directory that contains all input data files",
    required=True,
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
    idiap_extrinsics_directory: Path,
    calibrator_intrinsics_directory: Path,
    output_directory: Path,
    cameras_names: None | str,
) -> None:
    """
    Append idiap extrinsic parameters to a calibrator file that contains intrinsic parameters.
    Read a save the data as json files.

    @param idiap_extrinsics_directory: directory that contains all idiap extrinsic files.
        The directory structure should be camera_name/extrinsic.json
    @param calibrator_intrinsics_directory: directory that contains all calibrator intrinsic files.
        The files should be flat on the directory as camera_name.json
    @param output_directory: directory where the combined files are stored.
        The files are stored as camera_name.json
    @param cameras_names: if given, only convert the cameras with these names. The names should be
        separated by spaces, e.g. "cam1 cam2 cam3"
    @rtype: None
    """

    output_directory.mkdir(parents=True, exist_ok=True)

    if cameras_names is not None:
        cameras_names = cameras_names.split(" ")

    first_cam_tvec: np.ndarray | None = None
    first_cam_rvec: np.ndarray | None = None

    for idiap_extri_file, calib_intri_file in zip(
        sorted(idiap_extrinsics_directory.iterdir()),
        sorted(calibrator_intrinsics_directory.iterdir()),
        strict=False,
    ):
        if (
            cameras_names is not None
            and calib_intri_file.name not in cameras_names
        ):
            continue

        assert (
            idiap_extri_file.name in calib_intri_file.name
            or calib_intri_file.name in idiap_extri_file.name
        ), (
            f"File names do not match: {idiap_extri_file.name} and {calib_intri_file.name}"
        )

        calib_data = load_json(calib_intri_file)
        idiap_extri = load_json(idiap_extri_file / "extrinsic.json")

        rvec = np.array(idiap_extri["rvec"]).squeeze()
        tvec = convert_translation_from_cm_to_m(
            np.array(idiap_extri["tvec"]).squeeze()
        )

        if first_cam_tvec is None:
            first_cam_tvec = tvec
            first_cam_rvec = rvec

            calib_extri = zero_t_and_rot()
        else:
            rvec, tvec = move_origin_to_first_camera(
                first_cam_rvec, first_cam_tvec, rvec, tvec
            )
            idiap_extri["rvec"] = rvec[:, np.newaxis].tolist()
            idiap_extri["tvec"] = tvec[:, np.newaxis].tolist()
            calib_extri = convert_extrinsics_to_calibrator(idiap_extri)

        print(f"{calib_intri_file.stem}: {format_floats(calib_extri)}")

        calib_data["Calibration"]["cameras"][0]["transform"] = calib_extri

        output_file = output_directory / calib_intri_file.name
        if output_file.exists():
            error_msg = f"File already exist, will not overwrite {output_file}"
            raise RuntimeError(error_msg)

        with (output_file).open("w") as f:
            json.dump(calib_data, f, indent=2)


if __name__ == "__main__":
    main()
