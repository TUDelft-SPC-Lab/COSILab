#!/usr/bin/env python3
"""Aggregate DANTE Mingling benchmark metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


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
        description="Aggregate DANTE Mingling metrics_summary CSV files."
    )
    parser.add_argument(
        "--models-root",
        default=Path("DANTE-master/deep_fformation/models"),
        type=Path,
        help="DANTE models directory containing mingling*/cam*/pair_predictions_* outputs.",
    )
    parser.add_argument(
        "--output-root",
        default=Path("output"),
        type=Path,
        help="Directory where aggregate CSV files are written.",
    )
    parser.add_argument(
        "--out-prefix",
        default="dante_mingling",
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
    path_text = path.as_posix()
    match = re.search(
        r"/(mingling[12])/(cam\d+)/pair_predictions_(\d+)/val_fold_(\d+)/metrics_summary\.csv$",
        path_text,
    )
    if not match:
        raise ValueError(f"Unexpected metrics path: {path}")

    session, camera, run_id, fold = match.groups()
    df = pd.read_csv(path)
    df.insert(0, "pipeline", "DANTE")
    df.insert(1, "session", session)
    df.insert(2, "camera", camera)
    df.insert(3, "fold", int(fold))
    df.insert(4, "run_id", int(run_id))
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
    files = sorted(args.models_root.glob("mingling*/cam*/pair_predictions_*/val_fold_*/metrics_summary.csv"))
    if not files:
        raise SystemExit(f"No metrics_summary CSV files found under {args.models_root}")

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

    args.output_root.mkdir(parents=True, exist_ok=True)
    long_path = args.output_root / f"{args.out_prefix}_metrics_long.csv"
    camera_path = args.output_root / f"{args.out_prefix}_metrics_by_camera.csv"
    session_path = args.output_root / f"{args.out_prefix}_metrics_by_session.csv"
    overall_path = args.output_root / f"{args.out_prefix}_metrics_overall.csv"

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
