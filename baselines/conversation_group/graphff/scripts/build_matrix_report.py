#!/usr/bin/env python3
"""Reduce cross-camera evaluation cells to the 5x5 matrix and a results file.

Reads the per-camera cell CSVs written by scripts/evaluate_matrix_lstm.py or
scripts/evaluate_matrix_dante.py and writes, into
``<experiment_root>/exp_<id>/evaluations/results``:

    <prefix>_matrix_cells.csv    every cell of the run, merged, with its status
    <prefix>_matrix_long.csv     tidy mean/std per (metric, train cam, test cam)
    <prefix>_matrix_<metric>.csv the rendered 5x5 table, one file per metric
    <prefix>_matrix_report.txt   the same tables plus what is missing and why

The report names every checkpoint that could not be found, and says which rows of
the table are therefore averaged over fewer than five folds. A matrix with holes
in it should say so on its face rather than only in a job log.

Needs pandas, so for the DANTE pipeline this runs in the container's py371
environment rather than in dante_tf1, which has none.

Example usage:
    python scripts/build_matrix_report.py --model lstm  --exp-id 1
    python scripts/build_matrix_report.py --model dante --exp-id 2
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import camera_matrix  # noqa: E402
import camera_registry  # noqa: E402
import matrix_cells  # noqa: E402

MODELS = {
    "lstm": {
        "pipeline": "LSTM",
        "prefix": "lstm_mingling",
        "cells_glob": "lstm_cells_*.csv",
        "paths_module": "graphff_paths",
        "paths_dir": PROJECT_ROOT,
        "evaluator": "scripts/evaluate_matrix_lstm.py",
    },
    "dante": {
        "pipeline": "DANTE",
        "prefix": "dante_mingling",
        "cells_glob": "dante_cells_*.csv",
        "paths_module": "dante_paths",
        "paths_dir": os.path.join(PROJECT_ROOT, "DANTE-master", "deep_fformation"),
        "evaluator": "scripts/evaluate_matrix_dante.py",
    },
}

EXPECTED_FOLDS = camera_registry.NUM_FOLDS


def load_paths_module(model):
    """graphff_paths / dante_paths, imported only for the model being reported.

    Both are dependency-free; their directories are added late so nothing else
    on sys.path is shadowed (the repository root and deep_fformation both hold a
    utils.py, for instance).
    """
    spec = MODELS[model]
    if spec["paths_dir"] not in sys.path:
        sys.path.insert(0, spec["paths_dir"])
    return __import__(spec["paths_module"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build the camera x camera matrix and results file from "
                    "evaluation cells.")
    parser.add_argument(
        "--model", required=True, choices=sorted(MODELS),
        help="which pipeline's cells to reduce.")
    parser.add_argument(
        "--exp-id", default=os.environ.get("RUN_ID", "1"),
        help="experiment to report, i.e. exp_<id> (default from RUN_ID, else 1).")
    parser.add_argument(
        "--experiment-root", default=None,
        help="overrides the pipeline's experiment root for this run.")
    parser.add_argument(
        "--results-root", default=None,
        help="where the results live "
             "(default <experiment_root>/exp_<id>/evaluations/results).")
    parser.add_argument(
        "--cells-dir", default=None,
        help="where the per-camera cell CSVs live (default <results-root>/cells).")
    parser.add_argument(
        "--matrix-metrics", default=",".join(camera_matrix.DEFAULT_MATRIX_METRICS),
        help="comma-separated metrics to render (default: %(default)s).")
    parser.add_argument(
        "--matrix-decimals", type=int, default=3,
        help="decimals in the rendered 'mean +- std' cells (default: %(default)s).")
    return parser.parse_args()


def read_cells(cells_dir, pattern):
    paths = sorted(glob.glob(os.path.join(cells_dir, pattern)))
    if not paths:
        raise SystemExit(
            "No evaluation cells found under %s (looking for %s).\n"
            "Run the evaluator first, or point --cells-dir at the right place."
            % (cells_dir, pattern))

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source_file"] = path
        frames.append(frame)
    cells = pd.concat(frames, ignore_index=True)

    missing_columns = [name for name in ("camera", "test_camera", "fold", "split", "status")
                       if name not in cells.columns]
    if missing_columns:
        raise SystemExit(
            "Cell files under %s are missing the columns %s; they were not "
            "written by the evaluators in scripts/."
            % (cells_dir, ", ".join(missing_columns)))

    cells["fold"] = pd.to_numeric(cells["fold"], errors="coerce")
    unusable = cells["fold"].isna()
    if unusable.any():
        print("WARNING: dropping %d cell row(s) with no fold number" % int(unusable.sum()))
        cells = cells[~unusable].reset_index(drop=True)
    cells["fold"] = cells["fold"].astype(int)

    duplicated = cells.duplicated(subset=["camera", "test_camera", "fold"], keep="last")
    if duplicated.any():
        # a re-run of one camera overwrites its own file, but an old file left
        # behind under a different name would double-count a cell
        print("WARNING: %d duplicate (train camera, test camera, fold) rows; "
              "keeping the last of each. Check for stale files in the cells "
              "directory." % int(duplicated.sum()))
        cells = cells[~duplicated].reset_index(drop=True)
    return cells, paths


def checkpoint_status(cells):
    """(trained, missing, unevaluated) as sorted (camera, fold) lists.

    A checkpoint is one (training camera, fold). It shows up once per evaluation
    camera in the cells, so the statuses are collapsed here: missing anywhere
    means missing everywhere, it is the same file.
    """
    trained, missing = set(), set()
    for row in cells.itertuples():
        key = (row.camera, int(row.fold))
        if row.status == matrix_cells.STATUS_MISSING_CHECKPOINT:
            missing.add(key)
        else:
            trained.add(key)
    missing -= trained

    expected = set((camera, fold)
                   for camera in camera_registry.CAMERA_ORDER
                   for fold in range(EXPECTED_FOLDS))
    unevaluated = expected - trained - missing
    return sorted(trained), sorted(missing), sorted(unevaluated)


def checkpoint_paths(cells, keys):
    """Where each missing checkpoint was looked for, as recorded by the evaluator."""
    lookup = {}
    for row in cells.itertuples():
        key = (row.camera, int(row.fold))
        if key in lookup:
            continue
        if row.status == matrix_cells.STATUS_MISSING_CHECKPOINT:
            lookup[key] = str(row.checkpoint)
    return [(key, lookup.get(key, "")) for key in keys]


def format_camera_folds(keys):
    """[(cam06, 0), (cam06, 3)] -> 'cam06 folds 0,3'."""
    by_camera = {}
    for camera, fold in keys:
        by_camera.setdefault(camera, []).append(fold)
    ordered = [camera for camera in camera_registry.CAMERA_ORDER if camera in by_camera]
    return "; ".join(
        "%s folds %s" % (camera, ",".join(str(fold) for fold in sorted(by_camera[camera])))
        for camera in ordered)


def build_report(model, args, paths_module, results_root, cells, cell_files,
                 matrix, metrics, decimals):
    spec = MODELS[model]
    lines = []

    def add(text=""):
        lines.append(text)

    trained, missing, unevaluated = checkpoint_status(cells)
    status_counts = cells["status"].value_counts().to_dict()
    failed = cells[cells["status"] == matrix_cells.STATUS_ERROR]

    add("%s camera x camera matrix -- exp_%s" % (spec["pipeline"], args.exp_id))
    add("=" * 72)
    add("generated       : " + datetime.now().isoformat(timespec="seconds"))
    add("experiment root : " + str(paths_module.get_experiment_root()))
    add("experiment      : " + str(paths_module.experiment_dir(args.exp_id)))
    add("evaluations     : " + str(paths_module.evaluations_dir(args.exp_id)))
    add("results root    : " + results_root)
    add("cell files      : " + str(len(cell_files)))
    for path in cell_files:
        add("                  " + path)
    add("cells           : %d (%s)" % (
        len(cells),
        ", ".join("%s=%d" % item for item in sorted(status_counts.items()))))
    add()
    add("Every cell is the model trained on the row camera's fold k, scored on the")
    add("column camera's fold-k held-out test block. Cells are mean +- std over that")
    add("row's folds, so the std is fold-level and camera-level spread is read off by")
    add("comparing cells. The diagonal is the same rule with column == row.")
    add()

    add("CHECKPOINTS")
    add("-" * 72)
    add("expected : %d (%d cameras x %d folds)"
        % (len(camera_registry.CAMERA_ORDER) * EXPECTED_FOLDS,
           len(camera_registry.CAMERA_ORDER), EXPECTED_FOLDS))
    add("found    : %d" % len(trained))
    add("missing  : %d" % len(missing))
    if missing:
        add()
        add("*** %d trained model(s) could not be found. The rows below for %s"
            % (len(missing), ", ".join(sorted(set(camera for camera, _ in missing)))))
        add("*** are averaged over FEWER THAN %d folds, and are not directly"
            % EXPECTED_FOLDS)
        add("*** comparable with the complete rows. Missing:")
        for (camera, fold), path in checkpoint_paths(cells, missing):
            add("      %s fold %d  ->  %s" % (camera, fold, path or "(path not recorded)"))
        add()
        add("    Train those folds, or re-run with --exp-id pointing at the run that")
        add("    trained them, then rebuild this report.")
    else:
        add("all expected checkpoints were found.")
    if unevaluated:
        add()
        add("NOT EVALUATED in this run (no cells at all, e.g. a narrowed")
        add("--train-cam/--fold, or an array task that never finished):")
        add("      " + format_camera_folds(unevaluated))
    add()

    add("FAILED CELLS")
    add("-" * 72)
    if failed.empty:
        add("none.")
    else:
        add("%d cell(s) had a checkpoint but could not be scored:" % len(failed))
        for row in failed.itertuples():
            add("      %s -> %s fold %s: %s"
                % (row.camera, row.test_camera, row.fold, row.error))
    add()

    for metric in metrics:
        add("%s" % metric)
        add("-" * 72)
        if matrix.empty or metric not in set(matrix["metric"]):
            add("no results for this metric.")
            add()
            continue
        wide = camera_matrix.to_wide(matrix, metric, decimals=decimals)
        add("rows trained on, columns tested on, mean +- std over folds")
        add(wide.to_string())
        add()

    add("WARNINGS")
    add("-" * 72)
    warnings = camera_matrix.report_gaps(matrix, metrics, expected_folds=EXPECTED_FOLDS)
    if warnings:
        for message in warnings:
            add("      " + message)
    else:
        add("none.")
    add()
    add("Regenerate with: python %s --model %s --exp-id %s"
        % (os.path.relpath(__file__, PROJECT_ROOT).replace(os.sep, "/"),
           model, args.exp_id))
    add("Cells came from: " + spec["evaluator"])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    spec = MODELS[args.model]
    paths_module = load_paths_module(args.model)

    if args.experiment_root:
        # both paths modules read their root from the environment on every call
        os.environ["GRAPHFF_EXPERIMENT_ROOT" if args.model == "lstm"
                   else "DANTE_EXPERIMENT_ROOT"] = args.experiment_root

    results_root = (args.results_root if args.results_root
                    else str(paths_module.evaluation_results_dir(args.exp_id)))
    cells_dir = args.cells_dir if args.cells_dir else os.path.join(results_root, "cells")
    os.makedirs(results_root, exist_ok=True)

    cells, cell_files = read_cells(cells_dir, spec["cells_glob"])
    metrics = [name.strip() for name in args.matrix_metrics.split(",") if name.strip()]

    # build_matrix wants pipeline/camera/test_camera/fold/split plus the metric
    # columns, which is exactly the cell schema
    matrix = camera_matrix.build_matrix(cells, metrics=metrics)

    written = []
    cells_path = os.path.join(results_root, spec["prefix"] + "_matrix_cells.csv")
    cells.to_csv(cells_path, index=False)
    written.append(cells_path)

    long_path = os.path.join(results_root, spec["prefix"] + "_matrix_long.csv")
    matrix.to_csv(long_path, index=False)
    written.append(long_path)

    for metric in metrics:
        if matrix.empty or metric not in set(matrix["metric"]):
            continue
        wide = camera_matrix.to_wide(matrix, metric, decimals=args.matrix_decimals)
        wide_path = os.path.join(
            results_root, "%s_matrix_%s.csv" % (spec["prefix"], metric))
        wide.to_csv(wide_path)
        written.append(wide_path)

    report = build_report(args.model, args, paths_module, results_root, cells,
                          cell_files, matrix, metrics, args.matrix_decimals)
    report_path = os.path.join(results_root, spec["prefix"] + "_matrix_report.txt")
    with open(report_path, "w") as handle:
        handle.write(report)
    written.append(report_path)

    print(report)
    for path in written:
        print("Wrote " + path)


if __name__ == "__main__":
    main()
