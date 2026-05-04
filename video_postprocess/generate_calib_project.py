import json
from copy import deepcopy
from pathlib import Path

import click
import cv2


@click.command()
@click.argument(
    "input-directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--calib-template",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path(__file__).parent / "calib-project-template.json",
    help="The JSON file template for the calibration project.",
)
@click.option(
    "--output-filename",
    type=click.Path(exists=False, dir_okay=False),
    help="The filename of the generated calibration project.",
)
def main(
    input_directory: Path, calib_template: Path, output_filename: Path | None
) -> None:
    """Generate a Camera Calibrator project JSON file from images in the specified directory.

    INPUT_DIRECTORY is the path to the folder containing the camera folders.
    """
    with calib_template.open() as f:
        calib_project_data = json.load(f)

    first_camera_data = calib_project_data["calibration"]["cameras"][0]
    other_camera_data_template = deepcopy(
        calib_project_data["calibration"]["cameras"][1]
    )
    poses_data = calib_project_data["calibration"]["poses"][0]

    calib_project_data["calibration"]["cameras"].pop(1)
    calib_project_data["calibration"]["poses"] = []

    first_camera = True
    for camera_folder in sorted(input_directory.iterdir()):
        if camera_folder.is_dir():
            for i, image_file in enumerate(sorted(camera_folder.iterdir())):
                if image_file.is_file():
                    if i == 0:
                        img = cv2.imread(image_file)
                        img_height, img_width = img.shape[:2]

                        if first_camera:
                            _first_camera_data = first_camera_data["model"][
                                "ptr_wrapper"
                            ]["data"]
                            _first_camera_data["CameraModelCRT"][
                                "CameraModelBase"
                            ]["imageSize"] = {
                                "width": img_width,
                                "height": img_height,
                            }
                            _first_camera_data["parameters"]["cx"]["val"] = (
                                img_width - 1
                            ) / 2
                            _first_camera_data["parameters"]["cy"]["val"] = (
                                img_height - 1
                            ) / 2
                        else:
                            other_camera_data = deepcopy(
                                other_camera_data_template
                            )
                            _other_camera_data = other_camera_data["model"][
                                "ptr_wrapper"
                            ]["data"]
                            _other_camera_data["CameraModelCRT"][
                                "CameraModelBase"
                            ]["imageSize"] = {
                                "width": img_width,
                                "height": img_height,
                            }
                            _other_camera_data["parameters"]["cx"]["val"] = (
                                img_width - 1
                            ) / 2
                            _other_camera_data["parameters"]["cy"]["val"] = (
                                img_height - 1
                            ) / 2

                            calib_project_data["calibration"]["cameras"].append(
                                other_camera_data
                            )

                    if first_camera:
                        calib_project_data["fileInfo"].append([])
                        calib_project_data["calibration"]["poses"].append(
                            poses_data
                        )

                    calib_project_data["fileInfo"][i].append(
                        {
                            "filePath": image_file.as_posix(),
                            "status": 0,
                            "included": True,
                        }
                    )
            first_camera = False

    if output_filename is None:
        output_filename = input_directory / "calib-project.json"

    if output_filename.exists():
        confirm = (
            input(f"{output_filename} already exists. Overwrite? [y/N]: ")
            .strip()
            .lower()
        )
        if confirm != "y":
            print("Aborted.")
            return

    with open(output_filename, "w") as f:
        json.dump(calib_project_data, f, indent=4)

    print(f"Calibration project saved to {output_filename}")


if __name__ == "__main__":
    main()
