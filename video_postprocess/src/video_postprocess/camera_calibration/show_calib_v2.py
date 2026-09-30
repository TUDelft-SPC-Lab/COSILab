import json
from pathlib import Path

import click
import cv2
import fastplotlib as fpl
import numpy as np
from scipy.spatial.transform import Rotation as R

GLOBAL_ROTATION = R.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
GLOBAL_TRANSLATION = np.array([0.0, 0.0, 3.5], dtype=float)
# GLOBAL_ROTATION = R.from_euler("xyz", [0, 0, 0], degrees=True).as_matrix()
# GLOBAL_TRANSLATION = np.array([1.0, 1.0, 0.0], dtype=float)

# Hardcode a camera viewpoint here (paste the dict printed on space-press) to
# set it as the starting view. Leave as None to use the default view.
STARTING_CAMERA_STATE = {
    "position": [
        5.7512796283431085,
        3.2245257974149664,
        4.218728715388076
    ],
    "rotation": [
        0.47366075453516204,
        0.47282452471392267,
        0.5271645412385363,
        0.5236220057474603
    ],
    "scale": [
        1.0,
        1.0,
        1.0
    ],
    "reference_up": [
        0.0,
        0.0,
        1.0
    ],
    "fov": 50.0,
    "width": 5.694629669189453,
    "height": 7.5956726169201225,
    "depth": 9.508053877617346,
    "zoom": 0.75,
    "maintain_aspect": True,
    "depth_range": None
}

# Unflip the camera text.
TEXT_UPRIGHT_ROTATION = R.from_euler("zyx", [90, 90, 0], degrees=True).as_quat()

def camera_pose_from_extrinsics(rvec, tvec):
    """
    Convert OpenCV extrinsics (rvec, tvec) to camera center
    and orientation in world coordinates.
    """
    R, _ = cv2.Rodrigues(rvec)

    # camera center in world coordinates
    C = -R.T @ tvec.reshape(3)  # This is good, changing it meshes up top view

    # camera-to-world rotation
    Rcw = R.T  # This is good, changing it meshes up top view

    return C, Rcw


def create_camera_axes(C, Rcw, scale=0.1):
    """
    Create line segments representing camera axes.
    """
    x_axis = C + scale * Rcw[:, 0]
    y_axis = C + scale * Rcw[:, 1]
    z_axis = C + scale * Rcw[:, 2]

    return [
        (np.vstack([C, x_axis]), "red"),
        (np.vstack([C, y_axis]), "green"),
        (np.vstack([C, z_axis]), "blue"),
    ]


def _load_extrinsic_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as f:
        data = json.load(f)

    rvec = np.asarray(data["rvec"], dtype=float).reshape(3)
    tvec = np.asarray(data["tvec"], dtype=float).reshape(3)
    return rvec, tvec


def _load_extrinsics_from_folders(
    data_folders: tuple[Path, ...],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    extrinsics: list[tuple[str, np.ndarray, np.ndarray]] = []

    for data_folder in data_folders:
        for camera_dir in sorted(data_folder.iterdir()):
            if not camera_dir.is_dir():
                continue

            extrinsic_file = camera_dir / "extrinsic.json"
            if not extrinsic_file.exists():
                continue

            try:
                rvec, tvec = _load_extrinsic_file(extrinsic_file)
            except (OSError, ValueError, KeyError) as exc:
                click.echo(
                    f"Skipping {camera_dir}: failed to parse extrinsics ({exc})"
                )
                continue

            label = f"{data_folder.name}/{camera_dir.name}"
            extrinsics.append((label, rvec, tvec))

    return extrinsics


@click.command()
@click.argument(
    "data_folders",
    nargs=-1,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--axis-scale", default=0.2, show_default=True, type=float)
def main(data_folders: tuple[Path, ...], axis_scale: float) -> None:
    if not data_folders:
        error_msg = "Provide at least one data folder."
        raise click.UsageError(error_msg)

    extrinsics = _load_extrinsics_from_folders(data_folders)
    if not extrinsics:
        error_msg = "No camera extrinsics were found in the provided folders."
        raise click.ClickException(error_msg)

    fig = fpl.Figure(cameras="3d")
    subplot = fig[0, 0]
    subplot.axes.grids.xy.visible = True

    camera_centers = []

    for label, rvec, tvec in extrinsics:
        C, Rcw = camera_pose_from_extrinsics(rvec, tvec)

        Rcw = GLOBAL_ROTATION @ Rcw
        C = GLOBAL_ROTATION @ C + GLOBAL_TRANSLATION

        camera_centers.append(C)

        for pts, color in create_camera_axes(C, Rcw, scale=axis_scale):
            subplot.add_line(data=pts, colors=color, thickness=4)

        label_pos = C + np.array([axis_scale * 0.2, axis_scale * 0.2, 0.0])
        camera_label = subplot.add_text(
            label,
            font_size=max(axis_scale * 0.2, 0.01),
            face_color="white",
            outline_color="black",
            outline_thickness=0.2,
            screen_space=False,
            offset=tuple(float(x) for x in label_pos),
            anchor="bottom-left",
        )
        camera_label.rotation = TEXT_UPRIGHT_ROTATION

    camera_centers_array = np.asarray(camera_centers)
    subplot.add_scatter(
        data=camera_centers_array,
        colors="yellow",
        sizes=10,
    )

    if STARTING_CAMERA_STATE is not None:
        subplot.camera.set_state(STARTING_CAMERA_STATE)

    def _on_key_down(event) -> None:
        if event.key != " ":
            return
        state = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in subplot.camera.get_state().items()
        }
        click.echo("STARTING_CAMERA_STATE = " + json.dumps(state, indent=4))

    fig.renderer.add_event_handler(_on_key_down, "key_down")

    fig.show()
    fpl.loop.run()


if __name__ == "__main__":
    main()
