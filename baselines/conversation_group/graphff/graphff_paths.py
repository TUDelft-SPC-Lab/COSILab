"""Single source of truth for LSTM/GraphFF input/output locations.

Mirrors DANTE-master/deep_fformation/dante_paths.py. Everything is derived from
two roots:

``GRAPHFF_DATA_ROOT``
    Read-only benchmark artifacts, laid out as ``<session>/<camera>/`` holding
    features.csv, GT.csv, group_names.txt and scene_continuity.csv.

``GRAPHFF_EXPERIMENT_ROOT``
    Where runs are written, laid out as ``exp_<run_id>/<camera>/fold_<k>``. The
    run id is the top level, so one experiment holds every camera and fold, and
    re-running with the same id rewrites it in place. Outputs are keyed by camera
    alone: camera numbers are unique across sessions, and results are compared
    camera against camera rather than session against session.

Split tensors cached for ``GRAPHFF_DATASET_MAKE=0`` live outside the experiment
tree, under ``<experiment_root>/_cache/<camera>``: they depend only on the
dataset, frame stride and sequence length, so they are shared across runs and
survive an overwrite.

The trained model does not: it belongs to one ``(run_id, camera, fold)`` and is
written into that fold's output directory, so a re-run under a different RUN_ID
leaves earlier checkpoints intact for a later cross-camera evaluation. Models
written before that move are still read from ``_cache`` as a fallback.

Kept dependency-free and Python 3.7 compatible: the container environment is
Python 3.7.1.
"""

import os
from pathlib import Path


DEFAULT_DATA_ROOT = (
    "/tudelft.net/staff-umbrella/neon/cosilab_project/data_clean/processed"
    "/benchmark_tasks/benchmark_2/baselines/LSTM"
)
DEFAULT_EXPERIMENT_ROOT = (
    "/tudelft.net/staff-umbrella/neon/cosilab_project/data_temp"
    "/B2_pipeline/LSTM"
)

DEFAULT_RUN_ID = "1"


def get_data_root():
    return Path(os.environ.get("GRAPHFF_DATA_ROOT", DEFAULT_DATA_ROOT))


def get_experiment_root():
    return Path(os.environ.get("GRAPHFF_EXPERIMENT_ROOT", DEFAULT_EXPERIMENT_ROOT))


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


# -------------------------- experiments (write) --------------------------

def experiment_dir(run_id):
    """e.g. <experiment_root>/exp_1 -- holds every camera and fold of one run."""
    return get_experiment_root() / ("exp_" + str(run_id))


def dataset_output_dir(dataset, run_id):
    """e.g. <experiment_root>/exp_1/cam06"""
    return experiment_dir(run_id) / camera_of(dataset)


def fold_output_dir(dataset, run_id, fold):
    """Everything one fold produces lives here."""
    return dataset_output_dir(dataset, run_id) / ("fold_" + str(fold))


def logs_dir(dataset, run_id):
    """Per-fold console logs, alongside that camera's fold directories."""
    return dataset_output_dir(dataset, run_id) / "logs"


# ------------------------------- cache --------------------------------

def cache_dir(dataset):
    """Cached split tensors, shared across runs and not wiped by --overwrite.

    The trained model is NOT here: it goes to fold_output_dir, keyed by run id.
    """
    return get_experiment_root() / "_cache" / camera_of(dataset)
