#!/usr/bin/env python3
"""The per-cell record written by the cross-camera evaluators.

One row is one (training camera, evaluation camera, fold) evaluation: the model
trained on ``camera`` fold ``fold``, scored on ``test_camera``'s fold-``fold``
held-out test block. ``scripts/build_matrix_report.py`` reduces these rows to the
5x5 matrix, and the column names are the ones ``camera_matrix.build_matrix``
already expects (``pipeline``, ``camera``, ``test_camera``, ``fold``, ``split``
plus the metric columns), so the rows feed it directly.

A row always exists, even when nothing could be computed: ``status`` says why.
That is what makes a missing checkpoint visible in the results instead of simply
absent from them.

Stdlib-only and Python 3.7 compatible: written from inside both container
environments, and DANTE's has no pandas.
"""

import csv

import camera_registry

STATUS_OK = "ok"
STATUS_MISSING_CHECKPOINT = "missing_checkpoint"
STATUS_ERROR = "error"

METRIC_FIELDS = (
    "auc",
    "f1_1",
    "precision_1",
    "recall_1",
    "f1_2_3",
    "precision_2_3",
    "recall_2_3",
)

CELL_FIELDS = (
    "pipeline",
    "run_id",
    "camera",          # the training camera; named to match the aggregators
    "train_session",
    "test_camera",     # where it was scored
    "test_session",
    "fold",
    "split",           # always 'test': the fold's held-out block on test_camera
    "status",
) + METRIC_FIELDS + (
    "n_eval_samples",
    "n_scenes",
    "checkpoint",
    "seconds",
    "error",
)


def new_row(pipeline, run_id, train_camera, test_camera, fold, status,
            checkpoint="", metrics=None, n_eval_samples="", n_scenes="",
            seconds="", error=""):
    """One cell record, with every field present so the CSV stays rectangular."""
    row = dict.fromkeys(CELL_FIELDS, "")
    row.update({
        "pipeline": pipeline,
        "run_id": run_id,
        "camera": train_camera,
        "train_session": camera_registry.session_of(train_camera),
        "test_camera": test_camera,
        "test_session": camera_registry.session_of(test_camera),
        "fold": fold,
        "split": "test",
        "status": status,
        "checkpoint": checkpoint,
        "n_eval_samples": n_eval_samples,
        "n_scenes": n_scenes,
        "seconds": "" if seconds == "" else "%.1f" % float(seconds),
        # newlines would break the row; a traceback's last line is the useful one
        "error": " ".join(str(error).split()),
    })
    for name in METRIC_FIELDS:
        if metrics is not None and name in metrics:
            row[name] = metrics[name]
    return row


class CellWriter(object):
    """Appends cell rows to one CSV, flushing each one.

    Flushing per row matters: these jobs run for hours and a task killed at the
    wall clock limit should still leave behind every cell it finished.
    """

    def __init__(self, path):
        self.path = str(path)
        self._handle = open(self.path, "w")
        self._writer = csv.DictWriter(
            self._handle, fieldnames=list(CELL_FIELDS), lineterminator="\n")
        self._writer.writeheader()
        self._handle.flush()
        self.count = 0

    def write(self, row):
        self._writer.writerow(row)
        self._handle.flush()
        self.count += 1

    def close(self):
        self._handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def write_metrics_summary(path, metrics, test_camera):
    """One cell, in the per-fold layout the two aggregators already read.

    Written as ``<fold_dir>/eval_<camera>/metrics_summary.csv``; the directory
    name alone identifies the evaluation camera, and the explicit column makes
    the file self-describing if it is ever moved.
    """
    with open(str(path), "w") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("split", "test_camera") + METRIC_FIELDS)
        writer.writerow(
            ("test", test_camera) + tuple(metrics[name] for name in METRIC_FIELDS))
