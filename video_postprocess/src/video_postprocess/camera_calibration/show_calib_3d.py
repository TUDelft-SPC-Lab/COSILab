from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import cv2
import fastplotlib as fpl
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

GLOBAL_ROTATION = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
GLOBAL_TRANSLATION = np.array([0.0, 0.0, 3.5], dtype=float)
# GLOBAL_ROTATION = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
# GLOBAL_TRANSLATION = np.array([1.0, 1.0, 0.0], dtype=float)

# Unflip the camera text.
TEXT_UPRIGHT_ROTATION = R.from_euler("zyx", [90, 90, 0], degrees=True).as_quat()


class OpenCVYamlLoader(yaml.SafeLoader):
    """YAML loader with support for OpenCV matrix tags."""


def _opencv_matrix_constructor(
    loader: OpenCVYamlLoader, node: yaml.nodes.MappingNode
) -> np.ndarray:
    data = loader.construct_mapping(node, deep=True)
    rows = int(data["rows"])
    cols = int(data["cols"])
    values = np.asarray(data["data"], dtype=float)
    return values.reshape(rows, cols)


OpenCVYamlLoader.add_constructor(
    "tag:yaml.org,2002:opencv-matrix", _opencv_matrix_constructor
)


def _load_yaml_with_opencv(path: Path) -> dict[str, Any]:
    text = path.read_text()
    filtered_lines = [
        line for line in text.splitlines() if not line.startswith("%YAML")
    ]
    filtered = "\n".join(filtered_lines)
    return yaml.load(filtered, Loader=OpenCVYamlLoader) or {}


def _to_vec3(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        error_msg = f"Expected 3 values, got {arr.size}"
        raise ValueError(error_msg)
    return arr


def _extract_rvec_tvec(
    extrinsic_data: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    # extrinsic_data["rvec"][2][0] = -extrinsic_data["rvec"][2][0]
    # extrinsic_data["tvec"][2][0] = -extrinsic_data["tvec"][2][0]
    return _to_vec3(extrinsic_data["rvec"]), _to_vec3(extrinsic_data["tvec"])


def _load_extrinsic_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open() as f:
            data = json.load(f)
    elif suffix in {".yml", ".yaml"}:
        data = _load_yaml_with_opencv(path)
    else:
        error_msg = f"Unsupported extrinsic file type: {path}"
        raise ValueError(error_msg)

    return _extract_rvec_tvec(data)


def _find_camera_files(camera_dir: Path) -> tuple[Path, Path] | None:
    intrinsic_candidates = [
        camera_dir / "intrinsic.yml",
        camera_dir / "intrinsic.yaml",
        camera_dir / "intrinsic.json",
        camera_dir / "intri.yml",
        camera_dir / "intri.yaml",
    ]
    extrinsic_candidates = [
        camera_dir / "extrinsic.yml",
        camera_dir / "extrinsic.yaml",
        camera_dir / "extrinsic.json",
        camera_dir / "extri.yml",
        camera_dir / "extri.yaml",
    ]

    intrinsic = next((p for p in intrinsic_candidates if p.exists()), None)
    extrinsic = next((p for p in extrinsic_candidates if p.exists()), None)

    if intrinsic is None or extrinsic is None:
        return None
    return intrinsic, extrinsic


def _add_camera_axes(
    subplot: fpl.layouts.Subplot,
    origin: np.ndarray,
    rotation_matrix: np.ndarray,
    axis_length: float,
    axis_radius: float,
) -> None:
    # Use world-space line thickness so axis_radius acts like geometric radius.
    line_thickness = max(axis_radius * 2.0, 1e-6)

    axis_specs = [
        (rotation_matrix[:, 0], "#ff4d4d"),
        (rotation_matrix[:, 1], "#52c41a"),
        (rotation_matrix[:, 2], "#4096ff"),
    ]

    for axis_dir, color in axis_specs:
        axis_end = origin + axis_dir * axis_length
        axis_points = np.vstack([origin, axis_end])
        subplot.add_line(
            axis_points,
            thickness=float(line_thickness),
            colors=color,
            uniform_color=True,
            size_space="world",
        )


def _add_world_axis_labels(
    subplot: fpl.layouts.Subplot, axis_length: float
) -> None:
    origin = np.zeros(3, dtype=float)
    axis_specs = [
        (np.array([1.0, 0.0, 0.0], dtype=float), "X", "#ff4d4d"),
        (np.array([0.0, 1.0, 0.0], dtype=float), "Y", "#52c41a"),
        (np.array([0.0, 0.0, 1.0], dtype=float), "Z", "#4096ff"),
    ]

    for axis_dir, label, color in axis_specs:
        label_pos = origin + axis_dir * axis_length * 1.15
        subplot.add_text(
            label,
            font_size=max(axis_length * 0.25, 0.01),
            face_color=color,
            outline_color="black",
            outline_thickness=0.2,
            screen_space=False,
            offset=tuple(float(x) for x in label_pos),
            anchor="middle-center",
        )


def _add_translated_flat_grid(
    subplot: fpl.layouts.Subplot,
    z_height: float,
    half_extent: float = 5.0,
    step: float = 0.1,
) -> None:
    positions = np.arange(-half_extent, half_extent + 0.5 * step, step)

    for v in positions:
        x_line = np.array(
            [[-half_extent, v, z_height], [half_extent, v, z_height]],
            dtype=float,
        )
        y_line = np.array(
            [[v, -half_extent, z_height], [v, half_extent, z_height]],
            dtype=float,
        )

        subplot.add_line(
            x_line,
            thickness=0.002,
            colors="#666666",
            uniform_color=True,
            size_space="world",
        )
        subplot.add_line(
            y_line,
            thickness=0.002,
            colors="#666666",
            uniform_color=True,
            size_space="world",
        )


def _collect_cameras(
    data_folders: tuple[Path, ...],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    cameras: list[tuple[str, np.ndarray, np.ndarray]] = []

    for data_folder in data_folders:
        for camera_dir in sorted(data_folder.iterdir()):
            if not camera_dir.is_dir():
                continue

            files = _find_camera_files(camera_dir)
            if files is None:
                continue

            _, extrinsic_path = files
            try:
                rvec, tvec = _load_extrinsic_file(extrinsic_path)
            except (OSError, ValueError, KeyError, yaml.YAMLError) as exc:
                click.echo(
                    f"Skipping {camera_dir}: failed to parse extrinsics ({exc})"
                )
                continue

            label = f"{data_folder.name}/{camera_dir.name}"
            cameras.append((label, rvec, tvec))

    return cameras


@click.command()
@click.argument(
    "data_folders",
    nargs=-1,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--axis-length",
    default=0.25,
    show_default=True,
    type=float,
    help="Length of each axis cylinder.",
)
@click.option(
    "--axis-radius",
    default=0.01,
    show_default=True,
    type=float,
    help="Radius of each axis cylinder.",
)
def main(
    data_folders: tuple[Path, ...], axis_length: float, axis_radius: float
) -> None:
    if not data_folders:
        error_msg = "Provide at least one data folder."
        raise click.UsageError(error_msg)

    cameras = _collect_cameras(data_folders)
    if not cameras:
        error_msg = (
            "No camera folders with intrinsic+extrinsic files were found."
        )
        raise click.ClickException(error_msg)

    fig = fpl.Figure(cameras="3d", size=(1000, 760))
    subplot = fig[0, 0]
    subplot.axes.grids.xy.visible = True
    _add_translated_flat_grid(subplot, z_height=3.5)
    _add_world_axis_labels(subplot, axis_length=axis_length)

    for label, rvec, tvec in cameras:
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        origin = tvec.astype(float)
        transformed_rotation = GLOBAL_ROTATION @ rotation_matrix
        transformed_origin = GLOBAL_ROTATION @ origin + GLOBAL_TRANSLATION

        _add_camera_axes(
            subplot,
            transformed_origin,
            transformed_rotation,
            axis_length=axis_length,
            axis_radius=axis_radius,
        )

        label_offset = transformed_origin + (
            transformed_rotation @ np.array([0.0, 0.0, axis_length * 0.7])
        )
        camera_label = subplot.add_text(
            label,
            font_size=max(axis_length * 0.2, 0.01),
            face_color="white",
            outline_color="black",
            outline_thickness=0.2,
            screen_space=False,
            offset=tuple(float(x) for x in label_offset),
            anchor="bottom-left",
        )
        camera_label.rotation = TEXT_UPRIGHT_ROTATION

    subplot.camera.set_state(
        {
            "position": (1.0, 2.0, 2.0),
            "reference_up": (0.0, 0.0, 1.0),
        }
    )
    subplot.camera.show_pos((0.5, 2.0, 2.0), up=(0.0, 0.0, 1.0))

    fig.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
