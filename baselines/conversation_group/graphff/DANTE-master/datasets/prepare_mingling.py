#!/usr/bin/env python3
"""Prepare Mingling ViTPose camera datasets for the original DANTE pipeline.

The PyTorch/GraphFF pipeline reads isolated per-batch CSV files from
``data/mingling1`` and ``data/mingling2``. DANTE expects a dataset directory with
``DS_utils/features.txt`` and ``DS_utils/group_names.txt``. This script builds
one DANTE dataset per camera:

    mingling1/cam06
    mingling1/cam08
    mingling1/cam10
    mingling2/cam01
    mingling2/cam03

Each camera dataset concatenates its available batches in chronological order.
DANTE expands every retained frame into pairwise samples, so the default output
keeps one frame out of every 300 aligned ViTPose frames. Frame times are remapped
to DANTE-safe tokens (``t000000``...) because DANTE uses colon-separated sample
ids internally. The original frame times are stored in ``time_map.csv``.
"""

import argparse
import csv
import math
import re
import shutil
from pathlib import Path


NUM_PARTICIPANTS = 32
SESSION_CONFIGS = {
    "mingling1": {
        "cameras": ("cam06", "cam08", "cam10"),
        "source_root": Path("data/mingling1"),
        "output_root": Path("DANTE-master/datasets/mingling1"),
    },
    "mingling2": {
        "cameras": ("cam01", "cam03"),
        "source_root": Path("data/mingling2"),
        "output_root": Path("DANTE-master/datasets/mingling2"),
    },
}
ID_RE = re.compile(r"ID_(\d+)")
GROUP_RE = re.compile(r"<([^>]*)>")


def dante_id(participant_id):
    # Match the original DANTE convention: 1 -> ID_001, 10 -> ID_0010.
    return "ID_00" + str(int(participant_id))


def is_missing(value):
    if value is None:
        return True
    text = str(value).strip()
    if text == "":
        return True
    try:
        return math.isnan(float(text))
    except ValueError:
        return text.lower() in ("nan", "none", "fake")


def parse_group_line(line):
    line = line.strip()
    if not line:
        return "", []
    parts = line.split(maxsplit=1)
    original_time = parts[0]
    groups = []
    for group_text in GROUP_RE.findall(line):
        members = []
        for match in ID_RE.findall(group_text):
            members.append(int(match))
        if members:
            groups.append(members)
    return original_time, groups


def format_group_line(dante_time, groups, valid_participants):
    formatted_groups = []
    for group in groups:
        filtered = [pid for pid in group if pid in valid_participants]
        if not filtered:
            continue
        formatted = " ".join(dante_id(pid) for pid in sorted(filtered))
        formatted_groups.append("< " + formatted + " >")
    if not formatted_groups:
        return dante_time
    return dante_time + " " + " ".join(formatted_groups)


def row_to_feature_tokens(row):
    tokens = []
    valid_participants = set()
    for participant_id in range(1, NUM_PARTICIPANTS + 1):
        values = [
            row.get("ID" + str(participant_id)),
            row.get("X" + str(participant_id)),
            row.get("Y" + str(participant_id)),
            row.get("theta_H" + str(participant_id)),
            row.get("theta_B" + str(participant_id)),
        ]
        if any(is_missing(value) for value in values):
            tokens.extend(["fake"] * 5)
            continue
        valid_participants.add(participant_id)
        tokens.extend(
            [
                dante_id(participant_id),
                str(values[1]),
                str(values[2]),
                str(values[3]),
                str(values[4]),
            ]
        )
    return tokens, valid_participants


def camera_batches(vitpose_root, camera):
    return sorted(vitpose_root.glob(camera + "_batch*/features.csv"))


def prepare_camera(camera, vitpose_root, output_root, overwrite=False, frame_stride=300):
    if frame_stride < 1:
        raise ValueError("--frame-stride must be >= 1")

    dataset_dir = output_root / camera
    ds_utils_dir = dataset_dir / "DS_utils"

    if overwrite and dataset_dir.exists():
        shutil.rmtree(str(dataset_dir))

    ds_utils_dir.mkdir(parents=True, exist_ok=True)

    features_out = ds_utils_dir / "features.txt"
    groups_out = ds_utils_dir / "group_names.txt"
    time_map_out = dataset_dir / "time_map.csv"

    if not overwrite:
        for path in (features_out, groups_out, time_map_out):
            if path.exists():
                raise FileExistsError(
                    str(path) + " exists. Re-run with --overwrite to replace generated files."
                )

    seen = 0
    written = 0
    batches = camera_batches(vitpose_root, camera)
    if not batches:
        raise FileNotFoundError("No batches found for " + camera + " under " + str(vitpose_root))

    with features_out.open("w", newline="") as features_handle, \
            groups_out.open("w", newline="") as groups_handle, \
            time_map_out.open("w", newline="") as time_map_handle:
        time_map_writer = csv.writer(time_map_handle)
        time_map_writer.writerow(["dante_time", "original_time", "source_batch"])

        for features_path in batches:
            batch_dir = features_path.parent
            group_names_path = batch_dir / "group_names.txt"
            if not group_names_path.exists():
                raise FileNotFoundError(str(group_names_path))

            with features_path.open(newline="") as batch_features_handle, \
                    group_names_path.open() as batch_groups_handle:
                reader = csv.DictReader(batch_features_handle)
                for row, group_line in zip(reader, batch_groups_handle):
                    original_time = row["time"]
                    group_time, groups = parse_group_line(group_line)
                    if group_time != original_time:
                        raise ValueError(
                            "Time mismatch in "
                            + str(batch_dir)
                            + ": features="
                            + original_time
                            + ", group_names="
                            + group_time
                        )

                    if seen % frame_stride != 0:
                        seen += 1
                        continue

                    dante_time = "t" + str(written).zfill(6)
                    feature_tokens, valid_participants = row_to_feature_tokens(row)
                    features_handle.write(dante_time + " " + " ".join(feature_tokens) + "\n")
                    groups_handle.write(
                        format_group_line(dante_time, groups, valid_participants) + "\n"
                    )
                    time_map_writer.writerow([dante_time, original_time, batch_dir.name])
                    written += 1
                    seen += 1

    return {
        "camera": camera,
        "frames": written,
        "source_frames": seen,
        "frame_stride": frame_stride,
        "batches": [path.parent.name for path in batches],
        "dataset_dir": str(dataset_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create DANTE DS_utils files for Mingling ViTPose cameras."
    )
    parser.add_argument(
        "--session",
        choices=sorted(SESSION_CONFIGS),
        default="mingling1",
        help="Which Mingling session to prepare.",
    )
    parser.add_argument(
        "--vitpose-root",
        type=Path,
        default=None,
        help="Override root containing camXX_batchYY GraphFF/LSTM CSV files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Where to write camera datasets.",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated camera prefixes, e.g. cam06,cam08,cam10.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace generated files if they already exist.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=300,
        help=(
            "Keep one out of every N aligned ViTPose frames per camera. "
            "Use 1 to keep every frame."
        ),
    )
    args = parser.parse_args()

    config = SESSION_CONFIGS[args.session]
    vitpose_root = args.vitpose_root or config["source_root"]
    output_root = args.output_root or config["output_root"]
    default_cameras = ",".join(config["cameras"])
    cameras_arg = args.cameras or default_cameras
    cameras = [camera.strip() for camera in cameras_arg.split(",") if camera.strip()]
    summaries = [
        prepare_camera(
            camera,
            vitpose_root,
            output_root,
            overwrite=args.overwrite,
            frame_stride=args.frame_stride,
        )
        for camera in cameras
    ]

    for summary in summaries:
        print(
            "{camera}: {frames}/{source_frames} frames kept with stride {frame_stride} -> {dataset_dir}".format(
                camera=summary["camera"],
                frames=summary["frames"],
                source_frames=summary["source_frames"],
                frame_stride=summary["frame_stride"],
                dataset_dir=summary["dataset_dir"],
            )
        )
        print("  batches: " + ", ".join(summary["batches"]))


if __name__ == "__main__":
    main()
