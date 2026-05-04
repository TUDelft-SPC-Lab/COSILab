import subprocess
from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any
import numpy as np
from tqdm import tqdm

IMAGE_WIDTH = 960
IMAGE_HEIGHT = 540
SKELETON = [
    [0, 1],
    [0, 2],
    [2, 3],
    [2, 6],
    [3, 4],
    [4, 5],
    [6, 7],
    [7, 8],
    [2, 9],
    [9, 10],
    [10, 11],
    [11, 15],
    [2, 12],
    [12, 13],
    [13, 14],
    [14, 16],
]
KEYPOINT_NAMES = [
    "head",
    "nose",
    "neck",
    "rightShoulder",
    "rightElbow",
    "rightWrist",
    "leftShoulder",
    "leftElbow",
    "leftWrist",
    "rightHip",
    "rightKnee",
    "rightAnkle",
    "leftHip",
    "leftKnee",
    "leftAnkle",
    "rightFoot",
    "leftFoot",
]
TRAIN_CAMERAS = ["cam2", "cam4", "cam6"]
TEST_CAMERAS = ["cam8"]

def remove_border_keypoints(
    keypoints: np.ndarray,
    valid_keypoints_mask: np.ndarray,
    image_width: int,
    image_height: int,
    border_percentage: float = 0.025,  # Ignore the points that are on the 2.5% borders of the image
) -> tuple[np.ndarray, np.ndarray]:
    keypoints = keypoints.copy()
    valid_keypoints_mask = valid_keypoints_mask.copy()
    valid_keypoints_mask = valid_keypoints_mask.reshape(-1, 2)

    num_valid = np.sum(valid_keypoints_mask)

    x_min = image_width * border_percentage
    x_max = image_width * (1 - border_percentage)
    y_min = image_height * border_percentage
    y_max = image_height * (1 - border_percentage)

    valid_keypoints_mask[keypoints[:, 0] < x_min, :] = False
    valid_keypoints_mask[keypoints[:, 0] > x_max, :] = False
    valid_keypoints_mask[keypoints[:, 1] < y_min, :] = False
    valid_keypoints_mask[keypoints[:, 1] > y_max, :] = False

    keypoints[keypoints[:, 0] < x_min] = 0.0
    keypoints[keypoints[:, 0] > x_max] = 0.0
    keypoints[keypoints[:, 1] < y_min] = 0.0
    keypoints[keypoints[:, 1] > y_max] = 0.0

    valid_keypoints_mask = valid_keypoints_mask.flatten()

    return keypoints, valid_keypoints_mask


def parse_keypoints(
    keypoints: list[float], occluded: list[float], image_width: int, image_height: int
) -> tuple[list[float], np.ndarray, np.ndarray]:
    keypoints_np = np.array(keypoints)
    valid_keypoints_mask = keypoints_np != None
    keypoints_np[np.logical_not(valid_keypoints_mask)] = 0
    keypoints_np = keypoints_np.astype(np.float32)
    keypoints_np = keypoints_np.reshape(-1, 2)
    keypoints_np[:, 0] *= image_width
    keypoints_np[:, 1] *= image_height

    # Z axis for coco is 2 for visible, 1 for occluded, 0 for not annotated
    # occluded is 1 for occluded, 0 for visible, None for not annotated
    occluded_np = np.array(occluded)
    occluded_np[occluded_np == 0] = 2
    occluded_np[occluded_np == None] = 0

    keypoints_np = np.concatenate([keypoints_np, occluded_np[:, np.newaxis]], axis=1)

    keypoints_np, valid_keypoints_mask = remove_border_keypoints(
        keypoints_np, valid_keypoints_mask, image_width, image_height
    )
    return keypoints_np.flatten().tolist(), keypoints_np, valid_keypoints_mask


def get_bbox(
    keypoints_np: np.ndarray,
    valid_keypoints_mask: np.ndarray,
    image_width: int,
    image_height: int,
    border_percentage: float = 0.05, # Percetange to add to the bounding box so that it is not too tight
) -> tuple[float, float, float, float]:
    keypoints_valid = keypoints_np[:, :2].flatten()[valid_keypoints_mask]

    if keypoints_valid.size == 0:
        return (0, 0, 0, 0)

    keypoints_valid = keypoints_valid.reshape(-1, 2)
    x_min = np.min(keypoints_valid[:, 0]) 
    x_max = np.max(keypoints_valid[:, 0])
    y_min = np.min(keypoints_valid[:, 1])
    y_max = np.max(keypoints_valid[:, 1])

    box_width = x_max - x_min
    box_height = y_max - y_min

    x_min = np.clip(x_min - border_percentage * box_width, 0, image_width)
    x_max = np.clip(x_max + border_percentage * box_width, 0, image_width)
    y_min = np.clip(y_min - border_percentage * box_height, 0, image_height)
    y_max = np.clip(y_max + border_percentage * box_height, 0, image_height)

    return (x_min, y_min, x_max - x_min, y_max - y_min)

def exit_if_file_exists_or_folder_not_empty(file_or_folder: Path):
    if file_or_folder.exists():
        if file_or_folder.is_dir():
            if len(list(file_or_folder.iterdir())) > 0:
                print(f"{file_or_folder} is not empty. Will not overwrite.")
                exit(-1)
        else:
            print(f"{file_or_folder} already exists. Will not overwrite.")
            exit(-1)


def main():
    parser = ArgumentParser()
    parser.add_argument("--netid", type=str)
    args = parser.parse_args()

    raw_conflab_path = Path("/tudelft.net/staff-bulk/ewi/insy/SPCDataSets/conflab-mm/")
    if not raw_conflab_path.exists():
        staff_bulk_path = Path("/tudelft.net/staff-bulk")
        staff_bulk_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            "sshfs",
            "-o",
            "ro",
            f"{args.netid}@sftp.tudelft.nl:/staff-bulk/",
            staff_bulk_path,
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError("Failed to mount staff-bulk")

    stuff_umbrella_path = Path("/tudelft.net/staff-umbrella")
    processed_conflab_path = stuff_umbrella_path / "neon/experiments/VIT003/"
    if not processed_conflab_path.exists():
        stuff_umbrella_path.mkdir(exist_ok=True, parents=True)
        cmd = [
            "sshfs",
            f"{args.netid}@sftp.tudelft.nl:/staff-umbrella",
            stuff_umbrella_path,
        ]
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            raise RuntimeError("Failed to mount staff-umbrella")

    annotations_path = raw_conflab_path / "release/annotations/pose/coco/"

    video_segments_path = raw_conflab_path / "processed/annotation/videoSegments/"

    categories = [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON,
        }
    ]

    categories[0]["skeleton"] = (np.array(categories[0]["skeleton"]) + 1).tolist()

    train_processed_annotations: dict[str, list[dict, Any]] = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    test_processed_annotations: dict[str, list[dict, Any]] = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }

    train_image_path = processed_conflab_path / "images_train"
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_output_annot_path = processed_conflab_path / "keypoints_and_bboxes_train.json"

    test_image_path = processed_conflab_path / "images_test"
    test_image_path.mkdir(parents=True, exist_ok=True)
    test_output_annot_path = processed_conflab_path / "keypoints_and_bboxes_test.json"

    for folder in [train_image_path, test_image_path, train_output_annot_path, test_output_annot_path]:
        exit_if_file_exists_or_folder_not_empty(folder)

    total_train_images = 0
    total_test_images = 0
    j = 0
    for annotation_file in tqdm(
        sorted(annotations_path.glob("*.json")), desc="Video segment"
    ):
        # # Break early
        # print(annotation_file.stem)
        # if annotation_file.stem not in ["cam2_vid2_seg8_coco", "cam2_vid2_seg9_coco", "cam8_vid2_seg9_coco"]:
        #     continue

        cam, vid, seg, _ = annotation_file.name.split("_")

        if cam in TRAIN_CAMERAS:
            image_path = train_image_path
            total_images = total_train_images
            processed_annotations = train_processed_annotations
        elif cam in TEST_CAMERAS:
            image_path = test_image_path            
            total_images = total_test_images
            processed_annotations = test_processed_annotations
        else:
            print("Skipping", cam)
            continue

        segment_path = video_segments_path / cam / f"{vid}-{seg}-scaled-denoised.mp4"

        if not segment_path.exists():
            raise FileNotFoundError(f"Could not find {segment_path}")

        cmd = ["ffmpeg", "-y", "-i", str(segment_path)]

        # Reduce the size of the test set by only storing one frame per second
        if cam in TEST_CAMERAS:            
            cmd += ["-vf", "select=not(mod(n\,60))", "-vsync", "vfr"]

        # To only process one second of video, put before the start_number: "-t", str(1)
        cmd += ["-start_number", str(total_images), image_path / f"%09d.jpg"]
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode != 0:
            raise RuntimeError(f"Failed to split segment in frames {segment_path}")

        new_images_start = total_images
        total_images = len(list(image_path.glob("*.jpg")))

        with annotation_file.open() as f:
            conflab_annot = json.load(f)

        for i in range(new_images_start, total_images):
            processed_annotations["images"].append(
                {
                    "id": i,
                    "file_name": f"{i:09d}.jpg",
                    "width": IMAGE_WIDTH,
                    "height": IMAGE_HEIGHT,
                }
            )

        multi_people_annot: dict
        for multi_people_annot in tqdm(
            conflab_annot["annotations"]["skeletons"],
            desc="Annotations",
            leave=False,
        ):
            for single_person_annot in multi_people_annot.values():
                # # Break early
                # if annot["image_id"] not in [0, 30, 50]:
                #     continue
                if cam in TEST_CAMERAS and single_person_annot["image_id"] % 60 != 0:
                    continue

                if cam in TEST_CAMERAS:
                    single_person_annot["image_id"] = single_person_annot["image_id"] // 60 + new_images_start
                else:
                    single_person_annot["image_id"] += new_images_start

                single_person_annot["id"] = j
                single_person_annot["keypoints"], keypoints, valid_keypoints_mask = parse_keypoints(
                    single_person_annot["keypoints"],
                    single_person_annot["occluded"],
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                )
                single_person_annot["bbox"] = get_bbox(
                    keypoints,
                    valid_keypoints_mask,
                    image_width=IMAGE_WIDTH,
                    image_height=IMAGE_HEIGHT,
                )
                processed_annotations["annotations"].append(single_person_annot)
                j += 1

        if cam in TRAIN_CAMERAS:
            total_train_images = total_images
        elif cam in TEST_CAMERAS:
            total_test_images = total_images

    for annot_path, processed_annotations in zip([train_output_annot_path, test_output_annot_path], [train_processed_annotations, test_processed_annotations]):
        annot_path.parent.mkdir(parents=True, exist_ok=True)
        with annot_path.open("w") as f:
            json.dump(processed_annotations, f, indent=4)


if __name__ == "__main__":
    main()
