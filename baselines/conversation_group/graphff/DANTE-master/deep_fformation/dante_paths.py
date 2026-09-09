"""Single source of truth for DANTE training input/output locations.

Everything the training stage reads or writes is derived from two roots:

``DANTE_DATA_ROOT``
    Read-only benchmark artifacts, laid out as ``<session>/<camera>/DS_utils``
    and ``<session>/<camera>/fold_<k>``.

``DANTE_EXPERIMENT_ROOT``
    Where runs are written, laid out as ``exp_<run_id>/<camera>/fold_<k>``. The
    run id is the top level, so one experiment holds every camera and fold, and
    re-running with the same id rewrites it in place. Outputs are keyed by camera
    alone: camera numbers are unique across sessions, and results are compared
    camera against camera rather than session against session.

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
    "/B2_pipeline/DANTE"
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


def camera_of(dataset):
    """'mingling1/cam06' -> 'cam06'.

    The data root keeps the session directory because that is how the benchmark
    artifacts ship; the experiment root does not, so a camera's results sit at
    one predictable place regardless of which session recorded it.
    """
    return str(dataset).rstrip("/").rsplit("/", 1)[-1]


def ds_utils_dir(dataset):
    """Holds features.txt and group_names.txt, needed by the F1 callback."""
    return dataset_dir(dataset) / "DS_utils"


def fold_data_dir(dataset, fold):
    """Holds train.p / val.p / test.p for one fold."""
    return dataset_dir(dataset) / ("fold_" + str(fold))


# -------------------------- experiments (write) --------------------------

def experiment_dir(run_id):
    """e.g. <experiment_root>/exp_1 -- holds every camera and fold of one run."""
    return get_experiment_root() / ("exp_" + str(run_id))


def dataset_output_dir(dataset, run_id, no_pointnet=False):
    """e.g. <experiment_root>/exp_1/cam06"""
    base = experiment_dir(run_id) / camera_of(dataset)
    if no_pointnet:
        base = base / "no_pointnet"
    return base


def fold_output_dir(dataset, run_id, fold, no_pointnet=False):
    """Everything one fold produces lives here, including its TensorBoard events."""
    return dataset_output_dir(dataset, run_id, no_pointnet) / ("fold_" + str(fold))


def tensorboard_dir(dataset, run_id, fold, no_pointnet=False):
    return fold_output_dir(dataset, run_id, fold, no_pointnet) / "tb"


def logs_dir(dataset, run_id, no_pointnet=False):
    """Per-fold console logs, sitting alongside that camera's fold directories."""
    return dataset_output_dir(dataset, run_id, no_pointnet) / "logs"
