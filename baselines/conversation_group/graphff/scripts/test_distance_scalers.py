"""Tests for the cross-camera distance rescale.

Adapted from the original cross-transfer code's test_cross_transfer_utils.py,
which covered the same arithmetic before it was ported into distance_scalers.py.
The padding case is the one that matters: -999 means "this neighbour is not
there", and rescaling it would turn the sentinel into a plausible distance.

    python scripts/test_distance_scalers.py
    python -m unittest discover -s scripts -p "test_*.py" -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import distance_scalers  # noqa: E402


class DistanceScalerTest(unittest.TestCase):
    def test_rescale_preserves_padding(self):
        data = torch.zeros((1, 1, 7, 3), dtype=torch.float32)
        data[:, :, distance_scalers.DISTANCE_INDEX, :] = torch.tensor([0.0, 0.5, -999.0])
        data[:, :, distance_scalers.INDICATOR_INDEX, :] = torch.tensor([1.0, 1.0, 0.0])
        source = {"min_distance": 0.0, "max_distance": 40.0}
        target = {"min_distance": 10.0, "max_distance": 20.0}

        summary = distance_scalers.rescale_to_source(data, source, target)

        # 0.0 -> 10 m -> 0.25 of the source range, 0.5 -> 15 m -> 0.375,
        # and the padded entry is left alone
        self.assertTrue(
            torch.allclose(
                data[:, :, distance_scalers.DISTANCE_INDEX, :],
                torch.tensor([[[0.25, 0.375, -999.0]]]),
            )
        )
        self.assertEqual(summary["num_rescaled_values"], 2)
        self.assertEqual(summary["mode"], distance_scalers.MODE_RESCALED)

    def test_rescale_to_same_scale_is_identity(self):
        """What the diagonal relies on: source == target changes nothing."""
        data = torch.zeros((1, 1, 7, 2), dtype=torch.float32)
        data[:, :, distance_scalers.DISTANCE_INDEX, :] = torch.tensor([0.125, 0.875])
        data[:, :, distance_scalers.INDICATOR_INDEX, :] = torch.tensor([1.0, 1.0])
        scaler = {"min_distance": 0.5, "max_distance": 7.5}

        distance_scalers.rescale_to_source(data, scaler, scaler)

        self.assertTrue(
            torch.allclose(
                data[:, :, distance_scalers.DISTANCE_INDEX, :],
                torch.tensor([[[0.125, 0.875]]]),
            )
        )

    def test_manifest_validation(self):
        payload = {
            "version": distance_scalers.MANIFEST_VERSION,
            "datasets": {
                "mingling1/cam06": {
                    "min_distance": 0.1,
                    "max_distance": 4.0,
                    "frame_stride": 20,
                    "seq_len": 10,
                }
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "scalers.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            manifest = distance_scalers.load_manifest(path)

        scaler = distance_scalers.get_scaler(
            manifest, "mingling1/cam06", frame_stride=20, seq_len=10)
        self.assertEqual(scaler["max_distance"], 4.0)

        # a scaler is only valid for the stride and window it was computed with
        with self.assertRaises(ValueError):
            distance_scalers.get_scaler(
                manifest, "mingling1/cam06", frame_stride=1, seq_len=10)
        with self.assertRaises(ValueError):
            distance_scalers.get_scaler(
                manifest, "mingling1/cam06", frame_stride=20, seq_len=5)
        with self.assertRaises(KeyError):
            distance_scalers.get_scaler(
                manifest, "mingling1/cam08", frame_stride=20, seq_len=10)


if __name__ == "__main__":
    unittest.main()
