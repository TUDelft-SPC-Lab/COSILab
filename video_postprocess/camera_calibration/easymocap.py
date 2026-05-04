import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def indent_write(data, f: io.TextIOWrapper) -> None:
    string_f = io.StringIO()
    yaml.dump(data, string_f)
    string_f.seek(0)
    f.write("  " + string_f.read())


def save_intrinsic_easymocap(
    all_cam_data: list[dict[str, Any]], output_directory: Path
) -> None:
    intri_file = output_directory / "intri.yml"
    if intri_file.exists():
        errror_msg = f"File already exist, will not overwrite {intri_file}"
        raise RuntimeError(errror_msg)

    with intri_file.open(mode="w") as f:
        f.write("%YAML:1.0\n")
        f.write("---\n")
        f.write("names:\n")
        for cam_data in all_cam_data:
            f.write(f'  - "{cam_data["name"]}"\n')

        for cam_data in all_cam_data:
            name = cam_data["name"]

            # Intrinsic matrix
            f.write(f"K_{name}: !!opencv-matrix\n")
            indent_write({"rows": 3}, f)
            indent_write({"cols": 3}, f)
            indent_write({"dt": "d"}, f)

            intrinsic = cam_data["model"]["ptr_wrapper"]["data"]["parameters"]
            f_val = intrinsic["f"]["val"]
            cx = intrinsic["cx"]["val"]
            ar = intrinsic["ar"]["val"]
            cy = intrinsic["cy"]["val"]

            f.write(
                f"  data: [{f_val}, 0.000, {cx}, 0.000, {f_val * ar}, {cy}, "
                "0.000, 0.000, 1.000]\n"
            )

            # Distortion coefficients
            f.write(f"dist_{name}: !!opencv-matrix\n")
            indent_write({"rows": 1}, f)
            indent_write({"cols": 5}, f)
            indent_write({"dt": "d"}, f)

            k1 = intrinsic["k1"]["val"]
            k2 = intrinsic["k2"]["val"]
            p1 = intrinsic["p1"]["val"]
            p2 = intrinsic["p2"]["val"]
            k3 = intrinsic["k3"]["val"]

            f.write(f"  data: [{k1}, {k2}, {p1}, {p2}, {k3}]\n")


def save_extrinsic_easymocap(
    all_cam_data: list[dict[str, Any]], output_directory: Path
) -> None:
    extri_file = output_directory / "extri.yml"
    if extri_file.exists():
        error_msg = f"File already exist, will not overwrite {extri_file}"
        raise RuntimeError(error_msg)

    with extri_file.open(mode="w") as f:
        f.write("%YAML:1.0\n")
        f.write("---\n")
        f.write("names:\n")
        for cam_data in all_cam_data:
            f.write(f'  - "{cam_data["name"]}"\n')

        for cam_data in all_cam_data:
            name = cam_data["name"]

            # Rotation vector
            f.write(f"R_{name}: !!opencv-matrix\n")
            indent_write({"rows": 3}, f)
            indent_write({"cols": 1}, f)
            indent_write({"dt": "d"}, f)
            extrinsic = cam_data["transform"]

            rx = extrinsic["rotation"]["rx"]
            ry = extrinsic["rotation"]["ry"]
            rz = extrinsic["rotation"]["rz"]

            f.write(f"  data: [{rx}, {ry}, {rz}]\n")

            # Rotation matrix
            rvec = np.array([rx, ry, rz], dtype=float)
            rot_m = cv2.Rodrigues(rvec)[0]
            f.write(f"Rot_{name}: !!opencv-matrix\n")
            indent_write({"rows": 3}, f)
            indent_write({"cols": 3}, f)
            indent_write({"dt": "d"}, f)
            f.write(
                "  data: [{}]\n".format(
                    ", ".join([f"{i:.6f}" for i in rot_m.reshape(-1)])
                )
            )

            # Translation vector
            f.write(f"T_{name}: !!opencv-matrix\n")
            indent_write({"rows": 3}, f)
            indent_write({"cols": 1}, f)
            indent_write({"dt": "d"}, f)

            tx = extrinsic["translation"]["x"]
            ty = extrinsic["translation"]["y"]
            tz = extrinsic["translation"]["z"]

            f.write(f"  data: [{tx}, {ty}, {tz}]\n")
