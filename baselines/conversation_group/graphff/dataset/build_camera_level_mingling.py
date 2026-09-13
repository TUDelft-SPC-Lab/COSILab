#!/usr/bin/env python3
"""Build camera-level GraphFF/LSTM datasets from Mingling batch folders.

The batch-level generator writes one dataset per camXX_batchYY folder. For
camera-wise benchmarking and cross-transfer evaluation, we also need one
dataset per camera, where the available batches are concatenated in
chronological order and batch boundaries are marked in scene_continuity.csv so
LSTM windows cannot cross them.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SESSION_CAMERAS = {
    "mingling1": ("cam06", "cam08", "cam10"),
    "mingling2": ("cam01", "cam03"),
}


def count_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return handle.readlines()


def append_csv_without_repeated_header(output_handle, input_path: Path, write_header: bool) -> int:
    lines = read_lines(input_path)
    if not lines:
        return 0
    start = 0 if write_header else 1
    output_handle.writelines(lines[start:])
    return max(0, len(lines) - 1)


def write_text_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.writelines(lines)


def build_camera(session_root: Path, camera: str, overwrite: bool = False) -> dict:
    output_dir = session_root / camera
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(str(output_dir) + " exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_dirs = sorted(
        path.parent for path in session_root.glob(camera + "_batch*/features.csv")
    )
    if not batch_dirs:
        raise FileNotFoundError("No batch features found for " + camera + " under " + str(session_root))

    used_batches = []
    skipped_batches = []
    total_frames = 0
    feature_header_written = False
    gt_header_written = False
    group_lines: list[str] = []
    continuity_lines: list[str] = []

    with (output_dir / "features.csv").open("w", encoding="utf-8", newline="") as features_out, (
        output_dir / "GT.csv"
    ).open("w", encoding="utf-8", newline="") as gt_out:
        for batch_dir in batch_dirs:
            features_path = batch_dir / "features.csv"
            gt_path = batch_dir / "GT.csv"
            groups_path = batch_dir / "group_names.txt"
            continuity_path = batch_dir / "scene_continuity.csv"

            num_frames = count_data_rows(features_path)
            num_groups = sum(1 for _ in groups_path.open("r", encoding="utf-8")) if groups_path.exists() else 0
            num_continuity = (
                sum(1 for _ in continuity_path.open("r", encoding="utf-8"))
                if continuity_path.exists()
                else 0
            )
            if num_frames == 0 or num_groups == 0 or num_continuity == 0:
                skipped_batches.append({
                    "batch": batch_dir.name,
                    "frames": num_frames,
                    "group_rows": num_groups,
                    "continuity_rows": num_continuity,
                })
                continue

            feature_rows = append_csv_without_repeated_header(
                features_out, features_path, write_header=not feature_header_written
            )
            feature_header_written = True
            append_csv_without_repeated_header(gt_out, gt_path, write_header=not gt_header_written)
            gt_header_written = True

            batch_group_lines = read_lines(groups_path)
            batch_continuity_lines = read_lines(continuity_path)
            if len(batch_group_lines) != feature_rows or len(batch_continuity_lines) != feature_rows:
                raise ValueError(
                    "Row mismatch in {}: features={}, groups={}, continuity={}".format(
                        batch_dir, feature_rows, len(batch_group_lines), len(batch_continuity_lines)
                    )
                )

            # Force every concatenated batch start to be a discontinuity so
            # sequence windows cannot bridge two acquisition batches.
            batch_continuity_lines[0] = "1\n"
            group_lines.extend(batch_group_lines)
            continuity_lines.extend(batch_continuity_lines)
            total_frames += feature_rows
            used_batches.append({"batch": batch_dir.name, "frames": feature_rows})

    if not used_batches:
        raise ValueError("No non-empty batches for " + camera)

    write_text_lines(output_dir / "group_names.txt", group_lines)
    write_text_lines(output_dir / "scene_continuity.csv", continuity_lines)

    info = {
        "session_root": str(session_root),
        "camera": camera,
        "output_dir": str(output_dir),
        "frames": total_frames,
        "used_batches": used_batches,
        "skipped_batches": skipped_batches,
        "notes": [
            "Batch-level files are concatenated in lexical batch order.",
            "scene_continuity.csv marks every batch start as 1 to prevent LSTM windows crossing batches.",
        ],
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=sorted(SESSION_CAMERAS), required=True)
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--cameras", help="Comma-separated camera prefixes. Defaults to session cameras.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_root = args.root / args.session
    cameras = SESSION_CAMERAS[args.session]
    if args.cameras:
        cameras = tuple(item.strip() for item in args.cameras.split(",") if item.strip())

    for camera in cameras:
        info = build_camera(session_root, camera, overwrite=args.overwrite)
        print("{session}/{camera}: {frames} frames, {batches} batches -> {output}".format(
            session=args.session,
            camera=camera,
            frames=info["frames"],
            batches=len(info["used_batches"]),
            output=info["output_dir"],
        ))


if __name__ == "__main__":
    main()
