import json
from pathlib import Path

import cv2 as cv
import numpy as np
import typer
from tqdm import tqdm


def calib_point(img_points: np.ndarray) -> list:
    c_points = []
    for i, point in enumerate(img_points):
        c_points.append(
            {
                "point": {"x": float(point[0][0]), "y": float(point[0][1])},
                "id": i,
            }
        )
    return c_points


def cv_detections_to_calib(img_points, camera_id: int, pose_id: int) -> dict:
    return {
        "featurePoints": calib_point(img_points),
        "cameraId": camera_id,
        "poseId": pose_id,
        "targetId": 0,
    }


def main(
    calib_file: Path,
    output_file: Path,
    camera_nums: tuple[int, int] | None = None,
    images_path: Path | None = None,
    image_prefix: str = "camera_",
) -> None:
    # termination criteria for the subpixel refinement
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    img_points = []

    with calib_file.open() as f:
        json_data = json.load(f)

    file_info = json_data["fileInfo"]

    if camera_nums is None:
        camera_nums = []
        for single_camera_info in file_info[0]:
            camera_id = int(
                Path(single_camera_info["filePath"]).parent.name.split("_")[1]
            )
            camera_nums.append(camera_id)

    if images_path is None:
        all_images = []
        valid_camera_nums = set()

        for i, single_camera_info in enumerate(file_info[0]):
            camera_id = int(
                Path(single_camera_info["filePath"]).parent.name.split("_")[1]
            )
            if camera_id in camera_nums:
                valid_camera_nums.add(i)
                all_images.append([])

        for single_info in file_info:
            for i, single_camera_info in enumerate(single_info):
                if i in valid_camera_nums:
                    all_images[i].append(single_camera_info["filePath"])
    else:
        all_images = []
        for camera_num in camera_nums:
            all_images.append(
                sorted(
                    images_path.glob(f"{image_prefix}{camera_num:02d}/*.png")
                )
            )

    num_points = 0
    total_images = len(all_images[0])
    for i in tqdm(range(total_images)):
        detected_camera_ids: set[int] = set()

        for camera_id, _camera_num in enumerate(camera_nums):
            file_info[i][camera_id]["included"] = (
                False  # File not included by default
            )
            file_info[i][camera_id]["status"] = (
                2  # 2 means checkerboard not detected
            )

        for camera_id, _camera_num in enumerate(camera_nums):
            if camera_id - len(detected_camera_ids) >= len(camera_nums) - 1:
                # If N-1 cameras failed, the last one cannot produce a valid
                # multi-camera detection for this pose.
                break

            img = cv.imread(all_images[camera_id][i])
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            size = (5, 3)  # rows and columns of the checkerboard
            ret, corners = cv.findChessboardCorners(gray, size, None)

            # If found, add object points, image points (after refining them)
            if ret:
                # Refine the corner positions small [3,3] neighborhood because the board is small
                corners2 = cv.cornerSubPix(
                    gray, corners, (3, 3), (-1, -1), criteria
                )

                img_points.append(
                    cv_detections_to_calib(
                        corners2, camera_id=camera_id, pose_id=i
                    )
                )
                num_points += 1
                detected_camera_ids.add(camera_id)

                file_info[i][camera_id]["status"] = (
                    1  # 1 means checkerboard detected
                )

        # Keep detections only when they appear in at least two cameras.
        if len(detected_camera_ids) >= 2:
            for camera_id in detected_camera_ids:
                file_info[i][camera_id]["included"] = True

        # if num_points >= 10:
        #     break

    json_data["detections"] = img_points

    if output_file.exists():
        # Ask to overwrite
        overwrite = input(
            f"Output file {output_file} already exists. Overwrite? (y/n): "
        )
        if overwrite.lower() != "y":
            print("Aborting.")
            return

    with output_file.open("w") as f:
        json.dump(json_data, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
