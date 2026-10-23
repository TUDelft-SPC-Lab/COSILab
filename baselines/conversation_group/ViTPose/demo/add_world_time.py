"""
add_world_time.py

Give every frame of the ViTPose dataframes its world-clock time.

The dataframes' ``time`` column is a session video label: it counts 60 frames
per second from the session video start (mingling_session_videos/camXX.mp4),
while the GoPros record at 59.94 fps, so it falls behind the world clock by
about 1 ms per second. Each session video is a gap-free run of consecutive raw
GoPro frames, continuing across file boundaries, so each session video frame is
one raw frame:

    raw frame = raw_start_frame + session frame

where ``raw_start_frame`` (counted across the camera's raw files in playback
order) comes from session_video_raw_start.csv, measured by
video_postprocess/match_session_video_to_raw.py. The world time of that raw
frame is read from the drift-corrected per-frame timestamp CSVs of the raw
files (videos/timestamps/), which are the ground truth and are not changed.

For each camera the script writes ``cam<XX>_world_time.csv`` with one row per
session video frame:

    session_frame, video_time, raw_file, raw_frame, world_timestamp

and adds a ``world_time`` column to every camXX_batchYY/vitpose_dataframe.pkl.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from vitpose_to_dataframe import (
    _frame_to_time_str,
    add_world_time_column,
    batch_number_from_name,
    cam_number_from_name,
)

TIMESTAMPS_ROOT_DEFAULT = Path(
    "/tudelft.net/staff-umbrella/neon/cosilab_project/4TU_upload/processed/"
    "videos/timestamps"
)
RAW_START_CSV_DEFAULT = Path(__file__).with_name("session_video_raw_start.csv")


def load_raw_starts(csv_path: str | Path) -> dict[str, dict[str, str]]:
    """Read session_video_raw_start.csv into {cam_number: row}, e.g. {"01": {...}}."""
    with open(csv_path, newline="") as f:
        return {row["camera"].removeprefix("camera_"): row for row in csv.DictReader(f)}


def build_world_time_table(
    cam_number: str,
    raw_start: dict[str, str],
    timestamps_root: str | Path,
) -> pd.DataFrame:
    """One row per session video frame of camera ``cam_number``, see the module docstring."""
    camera, session = raw_start["camera"], raw_start["session"]
    csv_paths = sorted((Path(timestamps_root) / session / f"{camera}-{session}").glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No timestamp CSVs for {camera} under {timestamps_root}/{session}")

    # Concatenate the per-file timestamps so they can be indexed by global raw frame
    world_timestamps, raw_files, raw_frames = [], [], []
    for csv_path in csv_paths:
        world = pd.read_csv(csv_path, usecols=["world_timestamp"], dtype=str)["world_timestamp"].to_numpy()
        world_timestamps.append(world)
        raw_files.append(np.full(len(world), csv_path.stem + ".MP4", dtype=object))
        raw_frames.append(np.arange(len(world)))
    world_timestamps = np.concatenate(world_timestamps)
    raw_files = np.concatenate(raw_files)
    raw_frames = np.concatenate(raw_frames)

    start = int(raw_start["raw_start_frame"])
    session_frames = int(raw_start["session_frames"])
    if raw_files[start] != raw_start["raw_start_file"] or raw_frames[start] != int(raw_start["raw_start_local_frame"]):
        raise ValueError(f"{camera}: timestamp CSVs do not line up with the raw files used to measure the start")
    if start + session_frames > len(world_timestamps):
        raise ValueError(f"{camera}: session video runs past the end of the timestamp CSVs")

    selected = slice(start, start + session_frames)
    return pd.DataFrame(
        {
            "session_frame": np.arange(session_frames),
            "video_time": [_frame_to_time_str(cam_number, 1, s) for s in range(session_frames)],
            "raw_file": raw_files[selected],
            "raw_frame": raw_frames[selected],
            "world_timestamp": world_timestamps[selected],
        }
    )


def process_dataframe_root(
    dataframe_root: str | Path,
    output_dir: str | Path | None = None,
    timestamps_root: str | Path = TIMESTAMPS_ROOT_DEFAULT,
    raw_start_csv: str | Path = RAW_START_CSV_DEFAULT,
) -> None:
    """Write cam<XX>_world_time.csv tables and add world_time to every dataframe under dataframe_root.

    Outputs go to ``output_dir`` (same layout), or replace the inputs when it is not set.
    """
    dataframe_root = Path(dataframe_root)
    output_dir = Path(output_dir) if output_dir is not None else dataframe_root
    raw_starts = load_raw_starts(raw_start_csv)

    tables: dict[str, pd.DataFrame] = {}
    for pkl_path in sorted(dataframe_root.glob("cam*_batch*/vitpose_dataframe.pkl")):
        cam_name = pkl_path.parent.name
        cam_number = str(cam_number_from_name(cam_name)).zfill(2)
        batch_number = batch_number_from_name(cam_name)

        if cam_number not in tables:
            if cam_number not in raw_starts:
                raise KeyError(f"cam{cam_number} is missing from {raw_start_csv}")
            tables[cam_number] = build_world_time_table(cam_number, raw_starts[cam_number], timestamps_root)
            table_path = output_dir / f"cam{cam_number}_world_time.csv"
            table_path.parent.mkdir(parents=True, exist_ok=True)
            tables[cam_number].to_csv(table_path, index=False)
            print(f"  -> {table_path}")

        df = add_world_time_column(pd.read_pickle(pkl_path), tables[cam_number], batch_number)
        out_path = output_dir / cam_name / pkl_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(out_path)
        print(f"  -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add world-clock times to the ViTPose dataframe pickles."
    )
    parser.add_argument(
        "dataframe_root",
        help="Directory containing camXX_batchYY/vitpose_dataframe.pkl",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional directory for the outputs. Defaults to updating dataframe_root in place.",
    )
    parser.add_argument(
        "--timestamps_root",
        default=str(TIMESTAMPS_ROOT_DEFAULT),
        help="Root of the per-frame world timestamp CSVs of the raw GoPro files.",
    )
    parser.add_argument(
        "--raw_start_csv",
        default=str(RAW_START_CSV_DEFAULT),
        help="CSV from match_session_video_to_raw.py with each camera's raw start frame.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_dataframe_root(
        args.dataframe_root,
        output_dir=args.output_dir,
        timestamps_root=args.timestamps_root,
        raw_start_csv=args.raw_start_csv,
    )
