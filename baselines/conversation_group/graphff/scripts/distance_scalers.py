#!/usr/bin/env python3
"""Per-camera distance scalers for cross-camera LSTM/GraphFF evaluation.

``data.get_mingling_data_fast`` normalises the distance feature with that
camera's own global min/max, taken over the whole camera before the fold split::

    data[:, :, 5, :] = (distance - min_distance) / (max_distance - min_distance)

The five Mingling cameras disagree wildly on that scale -- max distance runs from
about 5.3 (cam10) to 48.0 (cam06), since a single far-away detection sets the
maximum. A model trained on cam06 therefore reads a cam10 tensor as though every
pair were ~9x further apart than it is, predicts that nobody is grouped, and
scores near zero. That is a property of the normalisation, not of the model's
ability to transfer.

So a frozen model must be handed distances on the scale it was trained with:
undo the target camera's normalisation, re-apply the source camera's. Padded
entries (indicator != 1, value -999) are left untouched.

Ported from the original cross-transfer code's ``cross_transfer_utils`` so the
numbers reproduce; kept stdlib-only apart from the tensor it is handed.
"""

import json
import math

MANIFEST_VERSION = 1
# feature layout of the LSTM input tensor [sample, time, feature, neighbour]
DISTANCE_INDEX = 5
INDICATOR_INDEX = 6

MODE_DISABLED = "disabled"
MODE_SAME_CAMERA = "unchanged_same_camera"
MODE_RESCALED = "target_cache_to_frozen_source_scale"


def load_manifest(path):
    path = str(path)
    try:
        with open(path, "r") as handle:
            manifest = json.load(handle)
    except IOError:
        raise IOError(
            "Distance scaler manifest not found: " + path + "\n"
            "Generate it with scripts/compute_distance_scalers.py, or pass "
            "--no-distance-rescale to evaluate without rescaling.")

    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(
            "Unsupported distance scaler manifest version: %r"
            % (manifest.get("version"),))
    if "datasets" not in manifest:
        raise ValueError("Distance scaler manifest has no datasets mapping")
    return manifest


def get_scaler(manifest, dataset, frame_stride, seq_len):
    """The scaler recorded for one dataset, checked against this run's settings.

    A scaler is only valid for the stride and window length it was computed
    with, because those decide which frames and which windows the min/max ran
    over. Mismatches are refused rather than silently applied.
    """
    if dataset not in manifest["datasets"]:
        raise KeyError(
            "No distance scaler for dataset %r in the manifest (have: %s)"
            % (dataset, ", ".join(sorted(manifest["datasets"]))))
    scaler = manifest["datasets"][dataset]

    if int(scaler["frame_stride"]) != int(frame_stride):
        raise ValueError(
            "Distance scaler frame_stride mismatch for %s: manifest %s != run %s"
            % (dataset, scaler["frame_stride"], frame_stride))
    if int(scaler["seq_len"]) != int(seq_len):
        raise ValueError(
            "Distance scaler seq_len mismatch for %s: manifest %s != run %s"
            % (dataset, scaler["seq_len"], seq_len))

    min_distance = float(scaler["min_distance"])
    max_distance = float(scaler["max_distance"])
    if not math.isfinite(min_distance) or not math.isfinite(max_distance):
        raise ValueError("Non-finite distance scaler for " + dataset)
    if max_distance <= min_distance:
        raise ValueError("Invalid distance scaler range for " + dataset)
    return {
        "min_distance": min_distance,
        "max_distance": max_distance,
        "frame_stride": int(scaler["frame_stride"]),
        "seq_len": int(scaler["seq_len"]),
    }


def rescale_to_source(data, source_scaler, target_scaler,
                      distance_index=DISTANCE_INDEX,
                      indicator_index=INDICATOR_INDEX):
    """Convert a target-normalised distance channel to the source model's scale.

    Modifies ``data`` in place and returns what it did, for the record. Only
    visible entries are touched: a padded neighbour carries -999 by convention,
    and rescaling that would turn padding into a plausible-looking distance.
    """
    if data.ndim != 4:
        raise ValueError("Expected data with shape [sample, time, feature, neighbour]")
    if data.shape[2] <= max(distance_index, indicator_index):
        raise ValueError("Input tensor does not contain distance/indicator features")

    source_min = float(source_scaler["min_distance"])
    source_max = float(source_scaler["max_distance"])
    target_min = float(target_scaler["min_distance"])
    target_max = float(target_scaler["max_distance"])
    source_range = source_max - source_min
    target_range = target_max - target_min
    if source_range <= 0 or target_range <= 0:
        raise ValueError("Distance scaler range must be positive")

    visible = data[:, :, indicator_index, :] == 1
    distance = data[:, :, distance_index, :]
    visible_distance = distance[visible]
    if visible_distance.numel() == 0:
        raise ValueError("No visible pair distances found in the target tensor")

    raw_distance = visible_distance * target_range + target_min
    source_normalized = (raw_distance - source_min) / source_range
    distance[visible] = source_normalized

    return {
        "mode": MODE_RESCALED,
        "num_rescaled_values": int(visible_distance.numel()),
        "raw_distance_min": float(raw_distance.min().item()),
        "raw_distance_max": float(raw_distance.max().item()),
        "source_normalized_min": float(source_normalized.min().item()),
        "source_normalized_max": float(source_normalized.max().item()),
    }
