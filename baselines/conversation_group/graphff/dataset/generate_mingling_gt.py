#!/usr/bin/env python3
"""Generate GraphFF/LSTM CSV files for Mingling ViTPose sessions.

The ViTPose pickle files remain the raw source of keypoints under
``data/vitpose_dataframe``.  Generated LSTM-ready files are written into
session-specific dataset directories:

- ``data/mingling1/camXX_batchYY``
- ``data/mingling2/camXX_batchYY``

Each generated batch directory contains:

- features.csv
- GT.csv
- group_names.txt
- scene_continuity.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


NUM_PARTICIPANTS = 32
FRAME_RATE = 60
TIME_RE = re.compile(r"^(\d{2}):(\d{2}):(\d{2});(\d{2})$")

SESSION_CONFIGS = {
    "mingling1": {
        "session_label": "Mingling 1",
        "annotation": Path("data/group_annotations_v3/mingling_1_expanded.csv"),
        "output_root": Path("data/mingling1"),
        "summary": Path("data/mingling1/gt_generation_summary.csv"),
        "cameras": ("cam06", "cam08", "cam10"),
        "raw_segment_start": "13:45:00",
        "raw_segment_end": "14:20:00",
        "annotation_interval": "REF to Balloon2",
        "notes": "Session 1 has frame-level REF timing.",
    },
    "mingling2": {
        "session_label": "Mingling 2",
        "annotation": Path("data/group_annotations_v3/mingling_2_expanded.csv"),
        "output_root": Path("data/mingling2"),
        "summary": Path("data/mingling2/gt_generation_summary.csv"),
        "cameras": ("cam01", "cam03", "cam05"),
        "raw_segment_start": "14:52:00",
        "raw_segment_end": "15:27:00",
        "annotation_interval": "REF to Balloon2",
        "notes": "Session 2 event rows are aligned to ;00 because raw frame annotations are not reliable.",
    },
}


def parse_time_to_frame(time_value: str) -> int:
    match = TIME_RE.match(str(time_value))
    if not match:
        raise ValueError(f"Invalid time format: {time_value!r}")
    hours, minutes, seconds, frame = (int(part) for part in match.groups())
    return (((hours * 60) + minutes) * 60 + seconds) * FRAME_RATE + frame


def parse_groups(group_text: str) -> list[set[int]]:
    groups: list[set[int]] = []
    for group in re.findall(r"\{([^}]*)\}", str(group_text)):
        members = {
            int(value.strip())
            for value in group.split(",")
            if value.strip()
        }
        if members:
            groups.append(members)
    return groups


def load_annotation_map(annotation_csv: Path) -> dict[str, dict[int, int]]:
    annotations = pd.read_csv(annotation_csv)
    required = {"time_association", "conversational_groups"}
    missing = required - set(annotations.columns)
    if missing:
        raise ValueError(f"{annotation_csv} missing columns: {sorted(missing)}")

    annotation_map: dict[str, dict[int, int]] = {}
    for row in annotations.itertuples(index=False):
        time_key = str(row.time_association)
        pid_to_group: dict[int, int] = {}
        for group_index, group in enumerate(parse_groups(row.conversational_groups)):
            for participant_id in group:
                if participant_id in pid_to_group:
                    raise ValueError(
                        f"Duplicate participant {participant_id} at annotation time {time_key}"
                    )
                if not 1 <= participant_id <= NUM_PARTICIPANTS:
                    raise ValueError(
                        f"Invalid participant {participant_id} at annotation time {time_key}"
                    )
                pid_to_group[participant_id] = group_index
        annotation_map[time_key] = pid_to_group
    return annotation_map


def load_annotation_groups(annotation_csv: Path) -> dict[str, list[set[int]]]:
    annotations = pd.read_csv(annotation_csv)
    return {
        str(row.time_association): parse_groups(row.conversational_groups)
        for row in annotations.itertuples(index=False)
    }


def detected_participants(space_feat: object) -> set[int]:
    if not isinstance(space_feat, dict) or "head" not in space_feat:
        return set()
    head = np.asarray(space_feat["head"], dtype=object)
    if head.ndim != 2 or head.shape[1] < 1:
        return set()

    participants: set[int] = set()
    for raw_id in head[:, 0]:
        if pd.isna(raw_id):
            continue
        try:
            participant_id = int(str(raw_id).strip())
        except ValueError:
            continue
        if 1 <= participant_id <= NUM_PARTICIPANTS:
            participants.add(participant_id)
    return participants


def body_part_by_pid(space_feat: object, part_name: str) -> dict[int, tuple[float, float, float]]:
    if not isinstance(space_feat, dict) or part_name not in space_feat:
        return {}
    values = np.asarray(space_feat[part_name], dtype=object)
    if values.ndim != 2 or values.shape[1] < 4:
        return {}

    mapped: dict[int, tuple[float, float, float]] = {}
    for raw_id, raw_x, raw_y, raw_orientation in values[:, :4]:
        if pd.isna(raw_id):
            continue
        try:
            participant_id = int(str(raw_id).strip())
        except ValueError:
            continue
        if not 1 <= participant_id <= NUM_PARTICIPANTS:
            continue
        mapped[participant_id] = (
            float(raw_x) if not pd.isna(raw_x) else np.nan,
            float(raw_y) if not pd.isna(raw_y) else np.nan,
            float(raw_orientation) if not pd.isna(raw_orientation) else np.nan,
        )
    return mapped


def first_valid_position(
    participant_id: int,
    parts: dict[str, dict[int, tuple[float, float, float]]],
    order: tuple[str, ...] = ("hip", "foot", "head", "shoulder"),
) -> tuple[float, float]:
    for part_name in order:
        value = parts[part_name].get(participant_id)
        if value is None:
            continue
        x_value, y_value, _ = value
        if not pd.isna(x_value) and not pd.isna(y_value):
            return x_value, y_value
    return np.nan, np.nan


def first_valid_orientation(
    participant_id: int,
    parts: dict[str, dict[int, tuple[float, float, float]]],
    order: tuple[str, ...],
) -> float:
    for part_name in order:
        value = parts[part_name].get(participant_id)
        if value is None:
            continue
        orientation = value[2]
        if not pd.isna(orientation):
            return orientation
    return np.nan


def build_feature_row(frame_time: str, space_feat: object, valid_pids: set[int]) -> dict[str, object]:
    parts = {
        "head": body_part_by_pid(space_feat, "head"),
        "shoulder": body_part_by_pid(space_feat, "shoulder"),
        "hip": body_part_by_pid(space_feat, "hip"),
        "foot": body_part_by_pid(space_feat, "foot"),
    }

    row: dict[str, object] = {"time": frame_time}
    for participant_id in range(1, NUM_PARTICIPANTS + 1):
        if participant_id not in valid_pids:
            row[f"ID{participant_id}"] = np.nan
            row[f"X{participant_id}"] = np.nan
            row[f"Y{participant_id}"] = np.nan
            row[f"theta_H{participant_id}"] = np.nan
            row[f"theta_B{participant_id}"] = np.nan
            continue

        x_value, y_value = first_valid_position(participant_id, parts)
        row[f"ID{participant_id}"] = participant_id
        row[f"X{participant_id}"] = x_value
        row[f"Y{participant_id}"] = y_value
        row[f"theta_H{participant_id}"] = first_valid_orientation(
            participant_id, parts, ("head", "shoulder", "hip", "foot")
        )
        row[f"theta_B{participant_id}"] = first_valid_orientation(
            participant_id, parts, ("shoulder", "hip", "head", "foot")
        )
    return row


def build_matrix(detected: set[int], pid_to_group: dict[int, int]) -> np.ndarray:
    matrix = np.full((NUM_PARTICIPANTS, NUM_PARTICIPANTS), -1, dtype=int)
    valid = sorted(detected & set(pid_to_group))

    for pid_i in valid:
        group_i = pid_to_group[pid_i]
        for pid_j in valid:
            matrix[pid_i - 1, pid_j - 1] = int(group_i == pid_to_group[pid_j])
    return matrix


def write_gt_csv(path: Path, frame_times: list[str], matrices: list[np.ndarray]) -> None:
    header = ["time", "matrix"] + [""] * NUM_PARTICIPANTS
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame_time, matrix in zip(frame_times, matrices):
            for row_index, row in enumerate(matrix):
                time_cell = frame_time if row_index == 0 else ""
                writer.writerow([time_cell, *row.tolist(), ""])


def feature_columns() -> list[str]:
    columns = ["time"]
    for participant_id in range(1, NUM_PARTICIPANTS + 1):
        columns.extend(
            [
                f"ID{participant_id}",
                f"X{participant_id}",
                f"Y{participant_id}",
                f"theta_H{participant_id}",
                f"theta_B{participant_id}",
            ]
        )
    return columns


def group_names_line(frame_time: str, groups: list[set[int]]) -> str:
    formatted_groups = []
    for group in groups:
        ids = " ".join(f"ID_{participant_id:03d}" for participant_id in sorted(group))
        formatted_groups.append(f"< {ids} >")
    if formatted_groups:
        return f"{frame_time} {' '.join(formatted_groups)}"
    return frame_time


def filter_groups_for_valid_participants(groups: list[set[int]], valid_pids: set[int]) -> list[set[int]]:
    filtered_groups: list[set[int]] = []
    for group in groups:
        filtered_group = group & valid_pids
        if filtered_group:
            filtered_groups.append(filtered_group)
    return filtered_groups


def write_group_names(path: Path, frame_times: list[str], groups: list[list[set[int]]]) -> None:
    with path.open("w", newline="") as handle:
        for frame_time, frame_groups in zip(frame_times, groups):
            handle.write(group_names_line(frame_time, frame_groups) + "\n")


def write_scene_continuity_csv(path: Path, continuity: list[int]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for value in continuity:
            writer.writerow([value])


def process_pickle(
    pickle_path: Path,
    output_dir: Path,
    annotation_map: dict[str, dict[int, int]],
    annotation_groups: dict[str, list[set[int]]],
) -> dict[str, object]:
    dataframe = pd.read_pickle(pickle_path)
    required = {"time", "spaceFeat"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{pickle_path} missing columns: {sorted(missing)}")

    frame_times: list[str] = []
    matrices: list[np.ndarray] = []
    feature_rows: list[dict[str, object]] = []
    group_rows: list[list[set[int]]] = []
    continuity: list[int] = []
    previous_valid_pids: set[int] | None = None
    previous_frame_number: int | None = None

    skipped_no_annotation = 0
    skipped_empty_detection = 0
    skipped_no_valid_participants = 0

    for row in dataframe.itertuples(index=False):
        frame_time = str(row.time)
        pid_to_group = annotation_map.get(frame_time)
        if pid_to_group is None:
            skipped_no_annotation += 1
            continue

        detected = detected_participants(row.spaceFeat)
        if not detected:
            skipped_empty_detection += 1
            continue

        valid_pids = detected & set(pid_to_group)
        if not valid_pids:
            skipped_no_valid_participants += 1
            continue

        matrix = build_matrix(detected, pid_to_group)
        feature_row = build_feature_row(frame_time, row.spaceFeat, valid_pids)
        frame_number = parse_time_to_frame(frame_time)
        is_discontinuous = (
            previous_valid_pids is None
            or valid_pids != previous_valid_pids
            or previous_frame_number is None
            or frame_number != previous_frame_number + 1
        )

        frame_times.append(frame_time)
        matrices.append(matrix)
        feature_rows.append(feature_row)
        group_rows.append(filter_groups_for_valid_participants(annotation_groups[frame_time], valid_pids))
        continuity.append(int(is_discontinuous))
        previous_valid_pids = set(valid_pids)
        previous_frame_number = frame_number

    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "features.csv"
    gt_path = output_dir / "GT.csv"
    group_names_path = output_dir / "group_names.txt"
    continuity_path = output_dir / "scene_continuity.csv"
    pd.DataFrame(feature_rows, columns=feature_columns()).to_csv(features_path, index=False)
    write_gt_csv(gt_path, frame_times, matrices)
    write_group_names(group_names_path, frame_times, group_rows)
    write_scene_continuity_csv(continuity_path, continuity)

    return {
        "batch": output_dir.name,
        "source_pickle": str(pickle_path),
        "input_rows": len(dataframe),
        "written_frames": len(frame_times),
        "first_written_time": frame_times[0] if frame_times else "",
        "last_written_time": frame_times[-1] if frame_times else "",
        "skipped_no_annotation": skipped_no_annotation,
        "skipped_empty_detection": skipped_empty_detection,
        "skipped_no_valid_participants": skipped_no_valid_participants,
        "features_path": str(features_path),
        "gt_path": str(gt_path),
        "group_names_path": str(group_names_path),
        "scene_continuity_path": str(continuity_path),
    }


def default_expected_batches(cameras: tuple[str, ...]) -> list[str]:
    return [f"{camera}_batch{batch_index:02d}" for camera in cameras for batch_index in range(1, 8)]


def write_dataset_info(
    path: Path,
    *,
    session_name: str,
    config: dict[str, object],
    annotation_csv: Path,
    vitpose_root: Path,
    output_root: Path,
    summaries: list[dict[str, object]],
) -> None:
    cameras = tuple(config["cameras"])
    source_batches = [str(summary["batch"]) for summary in summaries]
    nonempty_summaries = [summary for summary in summaries if int(summary["written_frames"]) > 0]
    empty_batches = [str(summary["batch"]) for summary in summaries if int(summary["written_frames"]) == 0]
    available_batches = [str(summary["batch"]) for summary in nonempty_summaries]
    source_set = set(source_batches)
    expected_batches = default_expected_batches(cameras)
    missing_source_batches = [batch for batch in expected_batches if batch not in source_set]

    dataset_info = {
        "dataset": session_name,
        "session": config["session_label"],
        "source_root": str(vitpose_root),
        "generated_root": str(output_root),
        "annotation_source": str(annotation_csv),
        "raw_segment_start": config["raw_segment_start"],
        "raw_segment_end": config["raw_segment_end"],
        "annotation_interval": config["annotation_interval"],
        "notes": config["notes"],
        "participants": {
            "count": NUM_PARTICIPANTS,
            "slot_mapping": "slot k is participant id k",
            "evaluation_id_format": "ID_001 through ID_032",
        },
        "scene_continuity": {
            "1": "first retained frame, participant set changed, or frame number not consecutive",
            "0": "same detected participant set and consecutive frame",
        },
        "cameras": list(cameras),
        "source_batches": source_batches,
        "available_batches": available_batches,
        "empty_annotation_batches": empty_batches,
        "missing_source_batches": missing_source_batches,
        "frame_ranges": [
            {
                "batch": summary["batch"],
                "input_rows": int(summary["input_rows"]),
                "written_frames": int(summary["written_frames"]),
                "first_written_time": "" if pd.isna(summary["first_written_time"]) else str(summary["first_written_time"]),
                "last_written_time": "" if pd.isna(summary["last_written_time"]) else str(summary["last_written_time"]),
            }
            for summary in summaries
        ],
        "example_dataset_path": f"{session_name}/{available_batches[0]}" if available_batches else "",
    }
    path.write_text(json.dumps(dataset_info, indent=2) + "\n")


def resolve_config(
    session_name: str,
    annotation: Path | None,
    output_root: Path | None,
    summary: Path | None,
    cameras: tuple[str, ...] | None,
) -> dict[str, object]:
    base = dict(SESSION_CONFIGS[session_name])
    if annotation is not None:
        base["annotation"] = annotation
    if output_root is not None:
        base["output_root"] = output_root
    if summary is not None:
        base["summary"] = summary
    if cameras is not None:
        base["cameras"] = cameras
    return base


def generate_session(
    session_name: str,
    *,
    vitpose_root: Path,
    annotation: Path | None = None,
    output_root: Path | None = None,
    summary: Path | None = None,
    cameras: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    config = resolve_config(session_name, annotation, output_root, summary, cameras)
    annotation_csv = Path(config["annotation"])
    output_root_path = Path(config["output_root"])
    summary_path = Path(config["summary"])
    camera_names = tuple(config["cameras"])

    annotation_map = load_annotation_map(annotation_csv)
    annotation_groups = load_annotation_groups(annotation_csv)
    pickle_paths: list[Path] = []
    for camera in camera_names:
        pickle_paths.extend(sorted(vitpose_root.glob(f"{camera}_batch*/vitpose_dataframe.pkl")))
    if not pickle_paths:
        raise FileNotFoundError(f"No vitpose_dataframe.pkl files found for cameras {camera_names}")

    summaries = []
    for pickle_path in pickle_paths:
        output_dir = output_root_path / pickle_path.parent.name
        print(f"{session_name}: generating {pickle_path.parent.name} -> {output_dir}", flush=True)
        summaries.append(process_pickle(pickle_path, output_dir, annotation_map, annotation_groups))

    summary_df = pd.DataFrame(summaries)
    output_root_path.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    write_dataset_info(
        output_root_path / "dataset_info.json",
        session_name=session_name,
        config=config,
        annotation_csv=annotation_csv,
        vitpose_root=vitpose_root,
        output_root=output_root_path,
        summaries=summaries,
    )
    return summary_df


def write_metadata_from_summary(
    session_name: str,
    *,
    vitpose_root: Path,
    annotation: Path | None = None,
    output_root: Path | None = None,
    summary: Path | None = None,
    cameras: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    config = resolve_config(session_name, annotation, output_root, summary, cameras)
    annotation_csv = Path(config["annotation"])
    output_root_path = Path(config["output_root"])
    summary_path = Path(config["summary"])
    summary_df = pd.read_csv(summary_path)
    summaries = summary_df.to_dict("records")
    write_dataset_info(
        output_root_path / "dataset_info.json",
        session_name=session_name,
        config=config,
        annotation_csv=annotation_csv,
        vitpose_root=vitpose_root,
        output_root=output_root_path,
        summaries=summaries,
    )
    return summary_df


def parse_camera_list(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    cameras = tuple(item.strip() for item in value.split(",") if item.strip())
    if not cameras:
        raise ValueError("--cameras must contain at least one camera prefix")
    return cameras


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate isolated GraphFF/LSTM CSV files for Mingling ViTPose sessions."
    )
    parser.add_argument(
        "--session",
        choices=("mingling1", "mingling2", "all"),
        default="mingling1",
        help="Session to generate. Use all to generate both Mingling sessions.",
    )
    parser.add_argument(
        "--vitpose-root",
        type=Path,
        default=Path("data/vitpose_dataframe"),
        help="Directory containing camXX_batchYY/vitpose_dataframe.pkl files.",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Override expanded annotation CSV. Only valid for a single session.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Override generated dataset root. Only valid for a single session.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Override generation summary path. Only valid for a single session.",
    )
    parser.add_argument(
        "--cameras",
        default=None,
        help="Comma-separated camera prefixes, for example cam01,cam03,cam05. Only valid for a single session.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Rewrite dataset_info.json from an existing gt_generation_summary.csv without regenerating CSV files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cameras = parse_camera_list(args.cameras)
    has_single_session_overrides = any(
        value is not None for value in (args.annotation, args.output_root, args.summary, cameras)
    )
    if args.session == "all" and has_single_session_overrides:
        raise ValueError("--annotation, --output-root, --summary, and --cameras require a single --session")

    session_names = ("mingling1", "mingling2") if args.session == "all" else (args.session,)
    for session_name in session_names:
        if args.metadata_only:
            summary_df = write_metadata_from_summary(
                session_name,
                vitpose_root=args.vitpose_root,
                annotation=args.annotation,
                output_root=args.output_root,
                summary=args.summary,
                cameras=cameras,
            )
        else:
            summary_df = generate_session(
                session_name,
                vitpose_root=args.vitpose_root,
                annotation=args.annotation,
                output_root=args.output_root,
                summary=args.summary,
                cameras=cameras,
            )
        print(f"\n{session_name}:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
