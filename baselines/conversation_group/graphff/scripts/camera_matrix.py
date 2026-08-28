#!/usr/bin/env python3
"""Camera x camera generalisation matrix, shared by both aggregators.

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

The evaluation camera is read from the result CSV or its path; see
resolve_test_cameras for the accepted conventions. Files carrying no marker are
treated as self-evaluation, which is what the in-repo training run writes, so a
run with no cross-camera results yields a diagonal-only matrix rather than an
error.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# flat camera order: mingling1 first, then mingling2
CAMERA_ORDER = ("cam06", "cam08", "cam10", "cam01", "cam03")
CAMERA_SESSION = {
    "cam06": "mingling1",
    "cam08": "mingling1",
    "cam10": "mingling1",
    "cam01": "mingling2",
    "cam03": "mingling2",
}
DEFAULT_MATRIX_METRICS = ("f1_1", "f1_2_3", "auc")

# Every cell wants the fold's held-out block on the evaluation camera, which both
# pipelines label "test". "all"/"full" are accepted as a fallback for older
# whole-camera evaluations, but those overlap the row camera's training time
# within a session, so cells falling back are flagged rather than blended in
# silently. The label actually used is recorded per cell in split_used.
SPLIT_PREFERENCE = ("test", "all", "full")
FOLD_MATCHED_SPLIT = "test"

# an explicit column wins over anything encoded in the path
TEST_CAMERA_COLUMNS = ("test_camera", "eval_camera", "test_dataset", "eval_dataset")
# .../fold_0/eval_cam08/metrics_summary.csv, .../fold_0/eval_mingling1_cam08/...
_PATH_DIR_PATTERN = re.compile(r"/(?:eval|test)_(?:mingling[12]_)?cam(\d{1,2})/")
# ..._eval=cam08.csv, ..._test=mingling1_cam08.csv, ..._eval_on_cam08.csv
_PATH_NAME_PATTERNS = (
    re.compile(r"[_.](?:eval|test)=(?:mingling[12][_/])?cam(\d{1,2})"),
    re.compile(r"[_.](?:eval|test)_on_(?:mingling[12]_)?cam(\d{1,2})"),
)
_CAMERA_PATTERN = re.compile(r"^cam(\d{1,2})$")


def normalise_camera(value):
    """'mingling1/cam8' | 'mingling1_cam08' | 'cam8' -> 'cam08'; None if unparsable."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    text = text.rsplit("/", 1)[-1]
    if not text.startswith("cam"):
        text = text.rsplit("_", 1)[-1]
    match = _CAMERA_PATTERN.match(text)
    if not match:
        return None
    return "cam%02d" % int(match.group(1))


def resolve_test_cameras(df: pd.DataFrame, path: Path, default_camera: str):
    """Evaluation camera for every row of one result CSV.

    Returns (Series aligned to df, source label). Accepted conventions, in
    priority order:

    1. a column named test_camera / eval_camera / test_dataset / eval_dataset,
       holding cam08, mingling1/cam08 or mingling1_cam08
    2. a directory under fold_<k>/ named eval_cam08 or eval_mingling1_cam08
       (test_ is accepted in place of eval_)
    3. a filename token _eval=cam08, _test=mingling1_cam08 or _eval_on_cam08
    4. nothing, meaning the file scores the camera it was trained on
    """
    for column in TEST_CAMERA_COLUMNS:
        if column not in df.columns:
            continue
        resolved = df[column].map(normalise_camera)
        unparsable = df.loc[resolved.isna(), column].unique()
        if len(unparsable) > 0:
            raise ValueError(
                "Column %r in %s holds values that are not cameras: %s"
                % (column, path, ", ".join(map(str, unparsable)))
            )
        return resolved, column

    text = Path(path).as_posix()
    for pattern in (_PATH_DIR_PATTERN,) + _PATH_NAME_PATTERNS:
        match = pattern.search(text)
        if match:
            camera = "cam%02d" % int(match.group(1))
            return pd.Series([camera] * len(df), index=df.index), "path"

    return pd.Series([default_camera] * len(df), index=df.index), "self"


def add_test_camera_columns(df: pd.DataFrame, path: Path, train_camera: str) -> pd.DataFrame:
    """Insert test_camera / test_session / self_eval next to the training camera."""
    resolved, source = resolve_test_cameras(df, path, train_camera)
    unknown = sorted(set(resolved) - set(CAMERA_ORDER))
    if unknown:
        raise ValueError(
            "Unknown evaluation camera %s in %s (expected one of %s)"
            % (", ".join(unknown), path, ", ".join(CAMERA_ORDER))
        )
    df["test_camera"] = resolved
    df["test_session"] = resolved.map(CAMERA_SESSION)
    df["self_eval"] = resolved == train_camera
    df["test_camera_source"] = source
    return df


def _camera_categorical(values: pd.Series) -> pd.Categorical:
    return pd.Categorical(values, categories=CAMERA_ORDER, ordered=True)


def _pick_split(rows: pd.DataFrame, preference):
    for split in preference:
        subset = rows[rows["split"] == split]
        if not subset.empty:
            return subset, split
    return None, None


def build_matrix(long_df: pd.DataFrame, metrics=DEFAULT_MATRIX_METRICS,
                 split_override: str | None = None) -> pd.DataFrame:
    """Tidy one row per (pipeline, metric, train camera, test camera) cell.

    std is the sample std (ddof=1) over that cell's folds, so it is always
    fold-level; a cell with a single fold gets NaN rather than 0.
    """
    records = []
    group_cols = ["pipeline", "camera", "test_camera"]
    preference = (split_override,) if split_override is not None else SPLIT_PREFERENCE
    for (pipeline, train_camera, test_camera), rows in long_df.groupby(group_cols, dropna=False):
        subset, split_used = _pick_split(rows, preference)
        if subset is None:
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
                "split_used": split_used,
                # False means the cell scores more than the fold's held-out
                # block, so within a session it overlaps this fold's training time
                "fold_matched": split_used == FOLD_MATCHED_SPLIT,
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
                       ", ".join("%s->%s" % pair for pair in missing))
                )
            short = subset[subset["n_folds"] != expected_folds]
            if not short.empty:
                messages.append(
                    "%s/%s: cells not averaged over %d folds (%s)"
                    % (pipeline, metric, expected_folds,
                       ", ".join(
                           "%s->%s n=%d" % (row.train_camera, row.test_camera, row.n_folds)
                           for row in short.itertuples()
                       ))
                )
            leaky = subset[~subset["fold_matched"]]
            if not leaky.empty:
                messages.append(
                    "%s/%s: %d cells score more than the fold's held-out block, so "
                    "within a session they include time this fold trained on and are "
                    "not comparable to the rest of the table (%s)"
                    % (pipeline, metric, len(leaky),
                       ", ".join(
                           "%s->%s split=%s" % (row.train_camera, row.test_camera, row.split_used)
                           for row in leaky.itertuples()
                       ))
                )
    return messages


def write_matrix_outputs(long_df: pd.DataFrame, output_root: Path, out_prefix: str,
                         metrics=DEFAULT_MATRIX_METRICS, split_override=None,
                         decimals=3, expected_folds=5) -> list[Path]:
    """Write the tidy matrix plus one wide table per metric, and print them."""
    matrix = build_matrix(long_df, metrics=metrics, split_override=split_override)
    written = []

    tidy_path = output_root / ("%s_matrix_long.csv" % out_prefix)
    matrix.to_csv(tidy_path, index=False)
    written.append(tidy_path)

    for metric in metrics:
        if matrix.empty or metric not in set(matrix["metric"]):
            continue
        wide = to_wide(matrix, metric, decimals=decimals)
        wide_path = output_root / ("%s_matrix_%s.csv" % (out_prefix, metric))
        wide.to_csv(wide_path)
        written.append(wide_path)

        splits_used = sorted(set(matrix.loc[matrix["metric"] == metric, "split_used"]))
        print("\n%s: rows trained on, columns tested on, mean ± std over folds "
              "(evaluated on the column camera's fold-k held-out block; split: %s)"
              % (metric, ", ".join(splits_used)))
        print(wide.to_string())

    for message in report_gaps(matrix, metrics, expected_folds=expected_folds):
        print("WARNING:", message)

    return written
