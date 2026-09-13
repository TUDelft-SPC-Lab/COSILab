#!/usr/bin/env python3
"""Reconstruct the per-camera distance scalers the trained LSTM weights used.

``data.get_mingling_data_fast`` normalises the distance feature by the min/max
over the whole camera, computed after striding and over the valid windows only.
This reproduces that exact scope without allocating the full four-dimensional
tensor, so the manifest can be rebuilt from features.csv alone -- no cached
tensors and no GPU.

The result feeds scripts/evaluate_matrix_lstm.py, which uses it to hand a frozen
model distances on the scale it was trained with. See scripts/distance_scalers.py
for why that is necessary.

Ported from the original cross-transfer code's
compute_mingling_distance_scalers.py so the numbers reproduce; --verify-against
checks that they do, against the manifest that code shipped.

Example usage:
    python scripts/compute_distance_scalers.py
    python scripts/compute_distance_scalers.py --data-root /path/to/LSTM
    python scripts/compute_distance_scalers.py --verify-against \\
        config/mingling_existing_weight_distance_scalers.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
for _path in reversed((PROJECT_ROOT, SCRIPT_DIR)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import camera_registry  # noqa: E402
import distance_scalers  # noqa: E402
import graphff_paths  # noqa: E402

DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "config", "mingling_distance_scalers.json")
DESCRIPTION = (
    "Full-camera min/max distance scalers matching the normalisation in "
    "data.get_mingling_data_fast. A frozen model must receive target-camera "
    "distances converted to its own source scale."
)
NORMALIZATION_SCOPE = "full_camera_pre_split"


def downsample_continuity(scene_continuity, source_indices):
    """Mirrors data.downsample_scene_continuity."""
    downsampled = np.zeros(len(source_indices), dtype=scene_continuity.dtype)
    if len(source_indices) == 0:
        return downsampled
    downsampled[0] = scene_continuity[source_indices[0]]
    for idx in range(1, len(source_indices)):
        start = source_indices[idx - 1] + 1
        end = source_indices[idx] + 1
        downsampled[idx] = 1 if np.any(scene_continuity[start:end] == 1) else 0
    return downsampled


def compute_scaler(dataset_dir, frame_stride, seq_len, num_nodes):
    """min/max over exactly the pair distances that reach the normalisation."""
    dataset_dir = Path(dataset_dir)
    id_cols = ["ID{}".format(person_id) for person_id in range(1, num_nodes + 1)]
    x_cols = ["X{}".format(person_id) for person_id in range(1, num_nodes + 1)]
    y_cols = ["Y{}".format(person_id) for person_id in range(1, num_nodes + 1)]
    features = pd.read_csv(dataset_dir / "features.csv", usecols=id_cols + x_cols + y_cols)
    continuity = pd.read_csv(
        dataset_dir / "scene_continuity.csv", header=None).to_numpy().squeeze()
    if len(features) != len(continuity):
        raise ValueError("Feature/continuity row mismatch in {}".format(dataset_dir))

    source_indices = np.arange(0, len(features), frame_stride, dtype=np.int64)
    features = features.iloc[source_indices].reset_index(drop=True)
    continuity = downsample_continuity(continuity, source_indices)

    visible = features[id_cols].notna().to_numpy()
    x = features[x_cols].to_numpy(dtype=np.float64)
    y = features[y_cols].to_numpy(dtype=np.float64)
    num_frames = len(features)
    num_segments = num_frames - seq_len + 1
    if num_segments <= 0:
        raise ValueError("Not enough frames in {}".format(dataset_dir))

    # a window is valid when no scene change falls inside it and the self person
    # is visible throughout -- the same two tests get_mingling_data_fast applies
    changes = (continuity == 1).astype(np.int64)
    change_prefix = np.concatenate(([0], np.cumsum(changes)))
    continuity_ok = (
        change_prefix[seq_len:] - change_prefix[1:(num_segments + 1)]) == 0

    visible_prefix = np.concatenate(
        (np.zeros((1, visible.shape[1]), dtype=np.int64),
         np.cumsum(visible.astype(np.int64), axis=0)),
        axis=0,
    )
    visible_all = (visible_prefix[seq_len:] - visible_prefix[:num_segments]) == seq_len
    valid_windows = continuity_ok[:, None] & visible_all
    valid_starts, valid_people = np.nonzero(valid_windows)

    # Mark frames at which a person is the centre of at least one valid window.
    # That is exactly the set of self-person rows contributing to the min/max.
    eligible_delta = np.zeros((num_frames + 1, num_nodes), dtype=np.int32)
    np.add.at(eligible_delta, (valid_starts, valid_people), 1)
    np.add.at(eligible_delta, (valid_starts + seq_len, valid_people), -1)
    eligible_self = np.cumsum(eligible_delta[:-1], axis=0) > 0

    min_distance = np.inf
    max_distance = -np.inf
    num_values = 0
    for frame_idx in range(num_frames):
        finite_position = np.isfinite(x[frame_idx]) & np.isfinite(y[frame_idx])
        self_mask = eligible_self[frame_idx] & finite_position
        neighbor_mask = visible[frame_idx] & finite_position
        if not np.any(self_mask) or not np.any(neighbor_mask):
            continue

        dx = x[frame_idx][None, :] - x[frame_idx][:, None]
        dy = y[frame_idx][None, :] - y[frame_idx][:, None]
        distance = np.sqrt((dx ** 2) + (dy ** 2))
        pair_mask = self_mask[:, None] & neighbor_mask[None, :]
        np.fill_diagonal(pair_mask, False)
        values = distance[pair_mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        min_distance = min(min_distance, float(values.min()))
        max_distance = max(max_distance, float(values.max()))
        num_values += int(values.size)

    if not np.isfinite(min_distance) or not np.isfinite(max_distance):
        raise ValueError("No finite pair distances in {}".format(dataset_dir))

    return {
        "min_distance": min_distance,
        "max_distance": max_distance,
        "frame_stride": int(frame_stride),
        "seq_len": int(seq_len),
        "num_frames_after_stride": int(num_frames),
        "num_valid_windows": int(len(valid_starts)),
        "num_distance_values": int(num_values),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild the per-camera distance scaler manifest.")
    parser.add_argument(
        "--cameras", default="all",
        help="cameras to compute: all, or a comma-separated list (default: all).")
    parser.add_argument(
        "--data-root", default=None,
        help="overrides GRAPHFF_DATA_ROOT for this run.")
    parser.add_argument(
        "--frame-stride", type=int, default=20,
        help="stride the models were trained with (default: %(default)s).")
    parser.add_argument(
        "--seq-len", type=int, default=10,
        help="window length the models were trained with (default: %(default)s).")
    parser.add_argument(
        "--num-nodes", type=int, default=32,
        help="people per camera (default: %(default)s).")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="where to write the manifest (default: %(default)s).")
    parser.add_argument(
        "--verify-against", default=None,
        help="compare the computed scalers with an existing manifest and report "
             "the differences; useful for checking that this data matches the "
             "data the released weights were trained on.")
    return parser.parse_args()


def verify(computed, reference_path):
    """Print a per-camera comparison; returns True when everything matches."""
    reference = distance_scalers.load_manifest(reference_path)["datasets"]
    fields = ("min_distance", "max_distance", "num_frames_after_stride",
              "num_valid_windows", "num_distance_values")
    all_match = True

    print("\nverifying against " + str(reference_path))
    for dataset in sorted(computed):
        if dataset not in reference:
            print("  %-16s not in the reference manifest" % dataset)
            all_match = False
            continue
        mismatches = []
        for field in fields:
            if field not in reference[dataset]:
                continue
            mine, theirs = computed[dataset][field], reference[dataset][field]
            if isinstance(mine, float) or isinstance(theirs, float):
                # the reference is JSON round-tripped, so compare at that precision
                same = abs(float(mine) - float(theirs)) <= 1e-9 * max(1.0, abs(float(theirs)))
            else:
                same = mine == theirs
            if not same:
                mismatches.append("%s %s != %s" % (field, mine, theirs))
        if mismatches:
            all_match = False
            print("  %-16s DIFFERS: %s" % (dataset, "; ".join(mismatches)))
        else:
            print("  %-16s matches" % dataset)

    if all_match:
        print("\nall scalers match: this data is the data those weights were trained on.")
    else:
        print("\n[WARN] scalers differ from the reference. The features.csv here is not")
        print("       the data behind that manifest, so use the manifest computed now")
        print("       and expect different numbers from the released results.")
    return all_match


def main():
    args = parse_args()
    if args.data_root:
        os.environ["GRAPHFF_DATA_ROOT"] = args.data_root

    cameras = camera_registry.parse_camera_selection(args.cameras)
    print("data root    : " + str(graphff_paths.get_data_root()))
    print("cameras      : " + ", ".join(cameras))
    print("frame_stride : " + str(args.frame_stride))
    print("seq_len      : " + str(args.seq_len))
    print("output       : " + args.output)
    print("")

    computed = {}
    for camera in cameras:
        dataset = camera_registry.dataset_of(camera)
        dataset_dir = graphff_paths.dataset_dir(dataset)
        scaler = compute_scaler(dataset_dir, args.frame_stride, args.seq_len,
                                args.num_nodes)
        computed[dataset] = scaler
        print("%-16s min=%.12g max=%.12g  windows=%d  values=%d"
              % (dataset, scaler["min_distance"], scaler["max_distance"],
                 scaler["num_valid_windows"], scaler["num_distance_values"]))

    manifest = {
        "version": distance_scalers.MANIFEST_VERSION,
        "description": DESCRIPTION,
        "normalization_scope": NORMALIZATION_SCOPE,
        "datasets": computed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("\nWrote " + str(output))

    if args.verify_against:
        verify(computed, args.verify_against)


if __name__ == "__main__":
    main()
