"""Single source of truth for DANTE training input/output locations.

Everything the training stage reads or writes is derived from two roots:

``DANTE_DATA_ROOT``
    Read-only benchmark artifacts, laid out as ``<session>/<camera>/DS_utils``
    and ``<session>/<camera>/fold_<k>``.

``DANTE_EXPERIMENT_ROOT``
    Where runs are written, laid out as
    ``<session>/<camera>/pair_predictions_<run_id>/fold_<k>``.

Both default to the shared cluster locations and can be overridden with the
corresponding environment variables, which is all that is needed to run the same
code against a local copy of the data.

Kept deliberately dependency-free and Python 3.7 compatible: the DANTE container
environment is Python 3.7.1.
"""

import os
from pathlib import Path


DEFAULT_DATA_ROOT = (
    "/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed"
    "/benchmark_tasks/benchmark_2/baselines/DANTE"
)
DEFAULT_EXPERIMENT_ROOT = (
    "/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp"
    "/B2_pipeline/DANTE/experiments"
)

DEFAULT_RUN_ID = "1"


def get_data_root():
    return Path(os.environ.get("DANTE_DATA_ROOT", DEFAULT_DATA_ROOT))


def get_experiment_root():
    return Path(os.environ.get("DANTE_EXPERIMENT_ROOT", DEFAULT_EXPERIMENT_ROOT))


def get_run_id():
    return os.environ.get("RUN_ID", DEFAULT_RUN_ID)


# ----------------------------- data (read) -----------------------------

def dataset_dir(dataset):
    """e.g. <data_root>/mingling1/cam06"""
    return get_data_root() / dataset


def ds_utils_dir(dataset):
    """Holds features.txt and group_names.txt, needed by the F1 callback."""
    return dataset_dir(dataset) / "DS_utils"


def fold_data_dir(dataset, fold):
    """Holds train.p / val.p / test.p for one fold."""
    return dataset_dir(dataset) / ("fold_" + str(fold))


# -------------------------- experiments (write) --------------------------

def experiment_dir(dataset):
    """e.g. <experiment_root>/mingling1/cam06"""
    return get_experiment_root() / dataset


def run_dir(dataset, run_id, no_pointnet=False):
    """e.g. <experiment_root>/mingling1/cam06/pair_predictions_1"""
    base = experiment_dir(dataset)
    if no_pointnet:
        base = base / "no_pointnet"
    return base / ("pair_predictions_" + str(run_id))


def fold_output_dir(dataset, run_id, fold, no_pointnet=False):
    """Everything one fold produces lives here, including its TensorBoard events."""
    return run_dir(dataset, run_id, no_pointnet) / ("fold_" + str(fold))


def tensorboard_dir(dataset, run_id, fold, no_pointnet=False):
    return fold_output_dir(dataset, run_id, fold, no_pointnet) / "tb"


def logs_dir(dataset):
    """Per-fold console logs, one directory per camera."""
    return experiment_dir(dataset) / "logs"
