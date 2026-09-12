#!/usr/bin/env python3
"""Camera x camera generalisation matrix.

One model is trained per (camera, fold), so the natural report is a 5x5 matrix
whose rows are the training camera and whose columns are the evaluation camera.
Every cell is mean +- std over that row camera's 5 folds, so the std is always
fold-level and the spread across cells is camera-level.

One rule covers the whole matrix: the model trained on camera r fold k is scored
on **camera c's fold-k test block**, never on all of camera c. Cameras inside a
session film the same event, so scoring a whole camera would feed the model the
very moments it trained on, seen from another angle. Restricting to fold k's
held-out block keeps the evaluated timestamps out of that fold's training data
whichever column is read, and makes every cell the same size (one block, ~20% of
a recording) instead of comparing a block on the diagonal against a full
recording off it. The diagonal is then just the c == r case of the same rule.

That alignment assumes the two cameras' fold-k blocks cover the same moments,
which holds when both cameras' feature files span the same frames; folds are cut
as equal fractions of each camera's own timeline, so cameras of unequal length
leave a residual overlap this script cannot see.

The five cameras are listed flat, session order first (mingling1: cam06, cam08,
cam10; mingling2: cam01, cam03), so a session is a contiguous block of the
matrix rather than a separate aggregation level. Within-session cells measure
viewpoint robustness on held-out time; the two cross-session blocks (mingling1
rows x mingling2 columns and vice versa) are unseen people as well.

The input is the tidy cell records written by the two cross-camera evaluators in
this directory, which state the training camera, the evaluation camera and the
fold outright. scripts/build_matrix_report.py is the only caller.
"""

from __future__ import annotations

import pandas as pd

# flat camera order (mingling1 first, then mingling2) and the session mapping
# live in camera_registry: the cross-camera evaluators need them inside the
# container, where DANTE's environment has no pandas and so cannot import this
# module.
from camera_registry import CAMERA_ORDER, CAMERA_SESSION  # noqa: F401

DEFAULT_MATRIX_METRICS = ("f1_1", "f1_2_3", "auc")

# every cell is the fold's held-out block on the evaluation camera, which both
# pipelines label "test"
CELL_SPLIT = "test"


def _camera_categorical(values: pd.Series) -> pd.Categorical:
    return pd.Categorical(values, categories=CAMERA_ORDER, ordered=True)


def build_matrix(long_df: pd.DataFrame, metrics=DEFAULT_MATRIX_METRICS) -> pd.DataFrame:
    """Tidy one row per (pipeline, metric, train camera, test camera) cell.

    std is the sample std (ddof=1) over that cell's folds, so it is always
    fold-level; a cell with a single fold gets NaN rather than 0.
    """
    records = []
    group_cols = ["pipeline", "camera", "test_camera"]
    for (pipeline, train_camera, test_camera), rows in long_df.groupby(group_cols, dropna=False):
        subset = rows[rows["split"] == CELL_SPLIT]
        if subset.empty:
            continue

        folds = sorted(subset["fold"].unique().tolist())
        for metric in metrics:
            if metric not in subset.columns:
                continue
            values = pd.to_numeric(subset[metric], errors="coerce").dropna()
            records.append({
                "pipeline": pipeline,
                "metric": metric,
                "train_session": CAMERA_SESSION.get(train_camera),
                "train_camera": train_camera,
                "test_session": CAMERA_SESSION.get(test_camera),
                "test_camera": test_camera,
                "cell": "diagonal" if train_camera == test_camera else (
                    "within_session" if CAMERA_SESSION.get(train_camera) == CAMERA_SESSION.get(test_camera)
                    else "cross_session"),
                "n_folds": len(values),
                "folds": ";".join(str(fold) for fold in folds),
                "mean": values.mean() if len(values) > 0 else float("nan"),
                # ddof=1: sample std over folds, NaN for a single fold
                "std": values.std(ddof=1) if len(values) > 1 else float("nan"),
            })

    matrix = pd.DataFrame.from_records(records)
    if matrix.empty:
        return matrix
    matrix["_train_order"] = _camera_categorical(matrix["train_camera"])
    matrix["_test_order"] = _camera_categorical(matrix["test_camera"])
    matrix = (
        matrix.sort_values(["pipeline", "metric", "_train_order", "_test_order"])
        .drop(columns=["_train_order", "_test_order"])
        .reset_index(drop=True)
    )
    return matrix


def format_cell(mean, std, decimals=3) -> str:
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return "%.*f" % (decimals, mean)
    return "%.*f ± %.*f" % (decimals, mean, decimals, std)


def to_wide(matrix: pd.DataFrame, metric: str, decimals=3) -> pd.DataFrame:
    """5x5 'mean ± std' table, rows trained on, columns tested on."""
    subset = matrix[matrix["metric"] == metric]
    cells = {}
    for _, row in subset.iterrows():
        cells[(row["train_camera"], row["test_camera"])] = format_cell(
            row["mean"], row["std"], decimals)

    wide = pd.DataFrame(
        [[cells.get((train, test), "") for test in CAMERA_ORDER] for train in CAMERA_ORDER],
        index=list(CAMERA_ORDER),
        columns=list(CAMERA_ORDER),
    )
    wide.index.name = "trained_on"
    return wide


def report_gaps(matrix: pd.DataFrame, metrics, expected_folds=5) -> list[str]:
    """Cells the matrix is missing or short on, as warning lines."""
    messages = []
    if matrix.empty:
        return ["No matrix cells could be built from the parsed metrics."]

    for pipeline in sorted(matrix["pipeline"].unique()):
        for metric in metrics:
            subset = matrix[(matrix["pipeline"] == pipeline) & (matrix["metric"] == metric)]
            if subset.empty:
                messages.append("%s: metric %r is absent from every result file" % (pipeline, metric))
                continue
            present = set(zip(subset["train_camera"], subset["test_camera"]))
            missing = [
                (train, test)
                for train in CAMERA_ORDER for test in CAMERA_ORDER
                if (train, test) not in present
            ]
            if missing:
                messages.append(
                    "%s/%s: %d of 25 cells have no results (%s)"
                    % (pipeline, metric, len(missing),
                       ", ".join("%s->%s" % pair for pair in missing)))
            short = subset[subset["n_folds"] != expected_folds]
            if not short.empty:
                messages.append(
                    "%s/%s: cells not averaged over %d folds (%s)"
                    % (pipeline, metric, expected_folds,
                       ", ".join(
                           "%s->%s n=%d" % (row.train_camera, row.test_camera, row.n_folds)
                           for row in short.itertuples()
                       )))
    return messages
