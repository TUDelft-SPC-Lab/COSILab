import pprint
from pathlib import Path
from typing import Any


def save_intrinsic_idiap(
    all_cam_data: list[dict[str, Any]], output_directory: Path
) -> None:
    for cam_data in all_cam_data:
        name: str = cam_data["name"]

        intri_file = output_directory / name / "intrinsic.json"
        if intri_file.exists():
            error_msg = f"File already exist, will not overwrite {intri_file}"
            raise RuntimeError(error_msg)

        intri_file.parent.mkdir(parents=True, exist_ok=True)

        idiap_data = {}

        intrinsic = cam_data["model"]["ptr_wrapper"]["data"]["parameters"]
        f = intrinsic["f"]["val"]
        cx = intrinsic["cx"]["val"]
        ar = intrinsic["ar"]["val"]
        cy = intrinsic["cy"]["val"]

        idiap_data["intrinsic"] = [
            [f, 0.0, cx],
            [0.0, f * ar, cy],
            [0.0, 0.0, 1.0],
        ]

        k1 = intrinsic["k1"]["val"]
        k2 = intrinsic["k2"]["val"]
        p1 = intrinsic["p1"]["val"]
        p2 = intrinsic["p2"]["val"]
        k3 = intrinsic["k3"]["val"]

        idiap_data["distortion_coefficients"] = [k1, k2, p1, p2, k3]

        with intri_file.open(mode="w") as f:
            pretty_json_str = pprint.pformat(
                idiap_data, compact=False, sort_dicts=False
            ).replace("'", '"')
            f.write(pretty_json_str)


def save_extrinsic_idiap(
    all_cam_data: list[dict[str, Any]], output_directory: Path
) -> None:
    for cam_data in all_cam_data:
        name: str = cam_data["name"]

        extri_file = output_directory / name / "extrinsic.json"
        if extri_file.exists():
            error_msg = f"File already exist, will not overwrite {extri_file}"
            raise RuntimeError(error_msg)

        extri_file.parent.mkdir(parents=True, exist_ok=True)

        idiap_data = {}

        extrinsic = cam_data["transform"]

        rx = extrinsic["rotation"]["rx"]
        ry = extrinsic["rotation"]["ry"]
        rz = extrinsic["rotation"]["rz"]

        idiap_data["rvec"] = [[rx], [ry], [rz]]

        tx = extrinsic["translation"]["x"]
        ty = extrinsic["translation"]["y"]
        tz = extrinsic["translation"]["z"]

        idiap_data["tvec"] = [[tx], [ty], [tz]]

        with extri_file.open(mode="w") as f:
            pretty_json_str = pprint.pformat(
                idiap_data, compact=False, sort_dicts=False
            ).replace("'", '"')
            f.write(pretty_json_str)


def convert_extrinsics_to_calibrator(
    idiap_data: dict[str, Any],
) -> dict[str, Any]:
    rvec = idiap_data["rvec"]
    tvec = idiap_data["tvec"]

    # Convert to calibrator format
    extrinsic = {}

    extrinsic["rotation"] = {
        "rx": rvec[0][0],
        "ry": rvec[1][0],
        "rz": rvec[2][0],
    }

    extrinsic["translation"] = {
        "x": tvec[0][0],
        "y": tvec[1][0],
        "z": tvec[2][0],
    }

    return extrinsic
