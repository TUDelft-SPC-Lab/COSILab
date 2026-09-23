#!/usr/bin/env python3
import json
import pprint
from pathlib import Path
from typing import Any

import click
import numpy as np
from scipy.spatial.transform import RigidTransform as RT
from scipy.spatial.transform import Rotation as R


def load_json(calibrator_file: Path) -> dict[str, Any]:
    with calibrator_file.open() as f:
        return json.load(f)


def make_m1_relative_to_m2(
    rvec1: np.ndarray,
    tvec1: np.ndarray,
    rvec2: np.ndarray,
    tvec2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    m1 = RT.from_components(rotation=R.from_rotvec(rvec1), translation=tvec1)
    m2 = RT.from_components(rotation=R.from_rotvec(rvec2), translation=tvec2)

    new_m2 = m2 * m1

    return (new_m2.rotation.as_rotvec(), new_m2.translation)


def save_extrinsic_idiap(
    rvec: np.ndarray, tvec: np.ndarray, output_file: Path
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    idiap_data = {}

    rvec = rvec.tolist()
    tvec = tvec.tolist()

    idiap_data["rvec"] = [[rvec[0]], [rvec[1]], [rvec[2]]]

    idiap_data["tvec"] = [[tvec[0]], [tvec[1]], [tvec[2]]]

    with output_file.open(mode="w") as f:
        pretty_json_str = pprint.pformat(
            idiap_data, compact=False, sort_dicts=False
        ).replace("'", '"')
        f.write(pretty_json_str)


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
    "--extrincsics-cam1",
    type=click.Path(dir_okay=False, writable=False, path_type=Path),
    help="Directory that contains all input data files",
    required=True,
)
@click.option(
    "--extrincsics-cam2",
    type=click.Path(dir_okay=False, writable=False, path_type=Path),
    help="Directory that contains all input data files",
    required=True,
)
@click.option(
    "--extrincsics-cam2-out",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Directory that contains all input data files",
    required=True,
)
@click.option(
    "--swap-y",
    is_flag=True,
    help="Swap the y-axis of the extrinsic parameters of camera 2 before making it relative to camera 1",
)
def main(
    extrincsics_cam1: Path,
    extrincsics_cam2: Path,
    extrincsics_cam2_out: Path,
    swap_y: bool,
) -> None:
    """Script to make the extrinsic parameters of camera 2 relative to camera 1.
    For example, if camera 0 is the reference camera (the origin), and camera 1 extrinsics are defined with
    respect to camera 0. The in a second calibration setup, camera 2 extrinsics are defined with respect to camera 1,
    where camera 1 is the origin.
    This script will make the extrinsics of camera 2 relative to camera 0. For that it needs:
     * The extrinsics of camera 1 (rvec1, tvec1) with respect to camera 0, when camera 0 is the origin.
     * The extrinsics of camera 2 (rvec2, tvec2) with respect to camera 1, when camera 1 is the origin.
    """

    if extrincsics_cam2_out.exists():
        error_msg = (
            f"File already exist, will not overwrite {extrincsics_cam2_out}"
        )
        raise RuntimeError(error_msg)

    idiap_extri_cam1 = load_json(extrincsics_cam1)
    idiap_extri_cam2 = load_json(extrincsics_cam2)

    rvec1 = np.array(idiap_extri_cam1["rvec"]).squeeze()
    tvec1 = np.array(idiap_extri_cam1["tvec"]).squeeze()

    rvec2 = np.array(idiap_extri_cam2["rvec"]).squeeze()
    tvec2 = np.array(idiap_extri_cam2["tvec"]).squeeze()

    print("tvec2:\n\t", tvec2)
    print("rvec2:\n\t", rvec2)
    rvec2 = R.from_rotvec(rvec2).as_rotvec()
    print("rvec2 simplified:\n\t", rvec2)

    if swap_y:
        # This is equivalent to doing minus the y of the translation and
        # minus the x, z of the rotation vector.
        # However, this formulation is more general.
        flip_y = np.diag([1, -1, 1])
        tvec2 = flip_y @ tvec2
        mvec2 = R.from_rotvec(rvec2).as_matrix()
        mvec2 = flip_y @ mvec2 @ flip_y
        rvec2 = R.from_matrix(mvec2).as_rotvec()

    new_rvec2, new_tvec2 = make_m1_relative_to_m2(
        rvec1,
        tvec1,
        rvec2,
        tvec2,
    )

    print(f"New tvec2:\n\t{new_tvec2}")
    print(f"New rvec2:\n\t{new_rvec2}")

    save_extrinsic_idiap(new_rvec2, new_tvec2, extrincsics_cam2_out)

    (extrincsics_cam2.parent / "intrinsic.json").copy(
        extrincsics_cam2_out.parent / "intrinsic.json"
    )


if __name__ == "__main__":
    # for old_p, new_p in zip(
    #     sorted(Path("/home/era/code/NEON/ingroup-calib-003/idiap").iterdir()),
    #     sorted(Path("/home/era/code/NEON/ingroup-calib-004/idiap").iterdir()),
    #     strict=True,
    # ):
    #     old_p_e = old_p / "extrinsic.json"
    #     new_p_e = new_p / "extrinsic.json"
    #     print(f"Processing {old_p_e} -> {new_p_e}")
    #     idiap_extri_cam1 = load_json(old_p_e)
    #     idiap_extri_cam2 = load_json(new_p_e)

    #     print("Before:")
    #     print(format_floats(idiap_extri_cam1))

    #     print("After:")
    #     print(format_floats(idiap_extri_cam2))
    #     print("\n")

    main()
