#!/usr/bin/env python3
"""Aggregate LSTM Mingling benchmark metrics."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# reuse the root defaults rather than duplicating them
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graphff_paths import get_experiment_root, get_run_id  # noqa: E402


EXPECTED_CAMERAS = {
    "mingling1": ("cam06", "cam08", "cam10"),
    "mingling2": ("cam01", "cam03"),
}
METRIC_COLUMNS = [
    "auc",
    "f1_1",
    "precision_1",
    "recall_1",
    "f1_2_3",
    "precision_2_3",
    "recall_2_3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate LSTM Mingling metrics_summary CSV files."
    )
    parser.add_argument(
        "--models-root",
        default=get_experiment_root(),
        type=Path,
        help="LSTM experiment root containing exp_*/mingling*/cam*/fold_* outputs "
             "(default from GRAPHFF_EXPERIMENT_ROOT).",
    )
    parser.add_argument(
        "--run-id",
        default=get_run_id(),
        help="which experiment to aggregate, i.e. exp_<run-id> (default from RUN_ID, else 1). "
             "Use --all-runs to aggregate every experiment under the root.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="aggregate every exp_* under the root. Folds repeated across experiments "
             "are reported as duplicates.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("output"),
        type=Path,
        help="Directory where aggregate CSV files are written.",
    )
    parser.add_argument(
        "--out-prefix",
        default="lstm_mingling",
        help="Prefix for aggregate CSV files written under --output-root.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write aggregates even if the expected 25 fold summaries are incomplete.",
    )
    parser.add_argument(
        "--exclude-camera",
        action="append",
        default=[],
        metavar="SESSION/CAMERA",
        help="Exclude a camera, for example mingling2/cam03. Can be repeated.",
    )
    return parser.parse_args()


def parse_file(path: Path) -> pd.DataFrame:
    # run id, session, camera and fold come from the directory layout; the frame
    # stride is only recorded in the filename
    path_match = re.search(
        r"/exp_([^/]+)/(mingling[12])/(cam\d+)/fold_(\d+)/[^/]+\.csv$",
        path.as_posix(),
    )
    if not path_match:
        raise ValueError(f"Unexpected metrics path: {path}")
    run_id, session, camera, fold = path_match.groups()

    name_match = re.search(r"_stride=(\d+)\.csv$", path.name)
    stride = int(name_match.group(1)) if name_match else 1

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df.insert(0, "pipeline", "LSTM")
    df.insert(1, "session", session)
    df.insert(2, "camera", camera)
    df.insert(3, "fold", int(fold))
    df.insert(4, "run_id", run_id)
    df.insert(5, "frame_stride", stride)
    df["source_file"] = str(path)
    return df


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flattened = []
    for column in df.columns:
        if isinstance(column, tuple):
            left, right = column
            flattened.append(left if right == "" else f"{left}_{right}")
        else:
            flattened.append(column)
    df.columns = flattened
    return df


def aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)
    out = grouped[METRIC_COLUMNS].agg(["mean", "std"]).reset_index()
    out = flatten_columns(out)
    out.insert(len(group_cols), "n_rows", grouped.size().to_numpy())
    return out


def validate_inputs(df: pd.DataFrame, allow_incomplete: bool, excluded: set[tuple[str, str]]) -> None:
    expected = {
        (session, camera, fold)
        for session, cameras in EXPECTED_CAMERAS.items()
        for camera in cameras
        for fold in range(5)
        if (session, camera) not in excluded
    }
    observed = set(df[["session", "camera", "fold"]].drop_duplicates().itertuples(index=False, name=None))
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = (
        df[["session", "camera", "fold", "split"]]
        .value_counts()
        .loc[lambda counts: counts > 1]
    )

    messages = []
    if missing:
        messages.append("Missing expected summaries: " + ", ".join(map(str, missing)))
    if extra:
        messages.append("Unexpected summaries: " + ", ".join(map(str, extra)))
    if not duplicates.empty:
        messages.append("Duplicate rows:\n" + duplicates.to_string())

    if messages and not allow_incomplete:
        raise SystemExit("\n\n".join(messages))
    for message in messages:
        print("WARNING:", message)


def main() -> None:
    args = parse_args()
    experiment_glob = "exp_*" if args.all_runs else "exp_" + str(args.run_id)
    search_root = args.models_root / experiment_glob
    files = sorted(args.models_root.glob(
        experiment_glob + "/mingling*/cam*/fold_*/*metrics_summary*.csv"))
    if not files:
        raise SystemExit(
            f"No metrics_summary CSV files found under {search_root}\n"
            "Check --models-root / GRAPHFF_EXPERIMENT_ROOT, and --run-id "
            "(or pass --all-runs to search every experiment)."
        )
    print(f"aggregating {len(files)} fold summaries from {search_root}")

    excluded = set()
    for value in args.exclude_camera:
        try:
            session, camera = value.split("/", 1)
        except ValueError as exc:
            raise SystemExit(f"--exclude-camera must look like mingling2/cam03: {value}") from exc
        excluded.add((session, camera))

    long_df = pd.concat([parse_file(path) for path in files], ignore_index=True)
    if excluded:
        long_df = long_df[
            ~long_df[["session", "camera"]]
            .apply(tuple, axis=1)
            .isin(excluded)
        ].reset_index(drop=True)

    validate_inputs(long_df, args.allow_incomplete, excluded)

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)
    long_path = out_root / f"{args.out_prefix}_metrics_long.csv"
    camera_path = out_root / f"{args.out_prefix}_metrics_by_camera.csv"
    session_path = out_root / f"{args.out_prefix}_metrics_by_session.csv"
    overall_path = out_root / f"{args.out_prefix}_metrics_overall.csv"

    long_df.to_csv(long_path, index=False)
    aggregate(long_df, ["pipeline", "session", "camera", "split"]).to_csv(camera_path, index=False)
    aggregate(long_df, ["pipeline", "session", "split"]).to_csv(session_path, index=False)
    aggregate(long_df, ["pipeline", "split"]).to_csv(overall_path, index=False)

    print(f"Wrote {long_path}")
    print(f"Wrote {camera_path}")
    print(f"Wrote {session_path}")
    print(f"Wrote {overall_path}")


if __name__ == "__main__":
    main()
