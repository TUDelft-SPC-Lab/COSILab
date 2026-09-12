#!/usr/bin/env python3
"""Cross-camera evaluation of trained DANTE models.

Reads the checkpoints of one experiment and scores each of them on every camera,
producing one cell per (training camera, evaluation camera, fold). Nothing is
trained here.

One rule covers the whole matrix, the same one scripts/camera_matrix.py assumes:
the model trained on camera r fold k is scored on **camera c's fold-k held-out
test block** (that camera's own ``fold_k/test.p``), never on all of camera c.
Cameras inside a session film the same event, so scoring a whole camera would
feed the model the very moments it trained on, seen from another angle. The
diagonal is the c == r case of that rule, so it recomputes what the training run
already reported and is a useful consistency check on this script.

Ground-truth groups and positions come from the **evaluation** camera's DS_utils,
which is what makes an off-diagonal cell a cross-camera score rather than a
mismatched one.

Outputs, all under <experiment_root>/exp_<id>/evaluations:

    cam01@cam03.csv              trained on cam01, scored on cam03: one row per
                                 fold, so one file is one cell of the matrix
    results/cells/dante_cells_<eval camera>.csv
                                 one row per cell of this invocation, statuses
                                 included; build_matrix_report.py merges them
    results/                     the matrix tables and the report

Nothing is written into the training output: a fold directory holds what the
training run put there, and the evaluation of one camera's models against another
belongs to the experiment as a whole. The diagonal (cam01@cam01.csv) is written
too -- it is the same rule with column == row, and reproduces the training run's
own test numbers.

A missing checkpoint is a warning, not an error: the cell is recorded with
status=missing_checkpoint and the run continues, so one absent fold costs a gap
in the table rather than the whole matrix.

Runs in the container's DANTE environment (TensorFlow 1.14 / Keras 2.2.2), which
has no pandas -- hence no import of camera_matrix here.

Example usage:
    python scripts/evaluate_matrix_dante.py --exp-id 1
    python scripts/evaluate_matrix_dante.py --exp-id 1 --eval-cam 06 --train-cam all
"""

import argparse
import os
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DANTE_ROOT = os.path.join(PROJECT_ROOT, "DANTE-master")
# deep_fformation must come first and datasets/ must stay off the path entirely:
# both hold a reformat_data.py, and the datasets/ one reads from a hard-coded
# "../datasets/" prefix, while the deep_fformation one resolves DANTE_DATA_ROOT.
# PROJECT_ROOT is absent for the same reason -- it has its own utils.py, which
# would shadow DANTE's. Inserted in reverse so the listed order is the resolution
# order.
for _path in reversed((os.path.join(DANTE_ROOT, "deep_fformation"), SCRIPT_DIR)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import camera_registry  # noqa: E402
import matrix_cells  # noqa: E402

import dante_paths  # noqa: E402
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import backend as keras_backend  # noqa: E402

from utils import compute_group_metrics, load_matrix  # noqa: E402
from reformat_data import add_time, import_data  # noqa: E402

PIPELINE = "DANTE"
CHECKPOINT_NAME = "best_val_model.h5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score trained DANTE models across cameras.")
    parser.add_argument(
        "--exp-id", default=dante_paths.get_run_id(),
        help="experiment to read, i.e. exp_<id> (default from RUN_ID, else 1)")
    parser.add_argument(
        "--eval-cam", default="all",
        help="evaluation cameras: all, or a comma-separated list (06,cam08). "
             "One Slurm array task normally takes one of them.")
    parser.add_argument(
        "--train-cam", default="all",
        help="which trained models to score: all, or a comma-separated list.")
    parser.add_argument(
        "--fold", default="all",
        help="folds to score: all, or a comma-separated list (0,2).")
    parser.add_argument(
        "--experiment-root", default=None,
        help="overrides DANTE_EXPERIMENT_ROOT for this run.")
    parser.add_argument(
        "--data-root", default=None,
        help="overrides DANTE_DATA_ROOT for this run.")
    parser.add_argument(
        "--results-root", default=None,
        help="where results are written (default <experiment_root>/exp_<id>/results).")
    parser.add_argument(
        "--no-pointnet", action="store_true", default=False,
        help="read the models trained without the Context Transform, i.e. from "
             "<camera>/no_pointnet/fold_<k>.")
    parser.add_argument(
        "--no-pair-files", dest="pair_files", action="store_false", default=True,
        help="do not write the per-pair <train>@<test>.csv files; the cells CSV "
             "is then the only per-cell output.")
    return parser.parse_args()


def find_checkpoint(exp_id, camera, fold, no_pointnet):
    """(path or None, the path looked at)."""
    path = str(dante_paths.fold_output_dir(camera, exp_id, fold,
                                           no_pointnet=no_pointnet)
               / CHECKPOINT_NAME)
    return (path if os.path.exists(path) else None), path


def load_test_block(dataset, fold):
    """The evaluation camera's fold-k held-out block.

    Only test.p is read: train.p is the largest of the three pickles and nothing
    here needs it, so load_data's read-all-three is avoided on purpose.
    """
    path = str(dante_paths.fold_data_dir(dataset, fold) / "test.p")
    if not os.path.exists(path):
        raise IOError("missing evaluation data: " + path)
    return load_matrix(path)


def count_frames(timestamps):
    """Distinct frames behind one block; timestamps look like time:i:j:orientation."""
    return len(set(str(stamp).split(":")[0] for stamp in timestamps))


def load_model(checkpoint, max_people):
    """Fresh graph per model: 25 loads in one process otherwise keep the old ones."""
    keras_backend.clear_session()
    return keras.models.load_model(
        checkpoint, custom_objects={"tf": tf, "max_people": max_people})


def check_input_shape(model, max_people, d):
    """Explain a camera whose blocks do not fit the model, instead of a Keras trace."""
    try:
        shape = model.inputs[0].shape
        model_people, model_d = int(shape[2]), int(shape[3])
    except Exception:  # shape introspection differs across Keras versions
        return
    if (model_people, model_d) != (max_people, d):
        raise ValueError(
            "model expects (max_people={}, d={}) but this block is "
            "(max_people={}, d={})".format(model_people, model_d, max_people, d))


def write_pair_file(exp_id, train_camera, test_camera, rows):
    """One cell's folds as <experiment>/evaluations/<train>@<test>.csv."""
    path = str(dante_paths.evaluation_pair_file(train_camera, test_camera, exp_id))
    matrix_cells.write_pair_file(path, train_camera, test_camera, rows)
    return path


def main():
    args = parse_args()

    # set before anything resolves a path: dante_paths reads the environment on
    # every call
    if args.experiment_root:
        os.environ["DANTE_EXPERIMENT_ROOT"] = args.experiment_root
    if args.data_root:
        os.environ["DANTE_DATA_ROOT"] = args.data_root

    eval_cameras = camera_registry.parse_camera_selection(args.eval_cam)
    train_cameras = camera_registry.parse_camera_selection(args.train_cam)
    folds = camera_registry.parse_fold_selection(args.fold)

    evaluations_dir = str(dante_paths.evaluations_dir(args.exp_id))
    results_root = (args.results_root if args.results_root
                    else str(dante_paths.evaluation_results_dir(args.exp_id)))
    cells_dir = os.path.join(results_root, "cells")
    # exist_ok: the array tasks start together and would otherwise race here,
    # one creating the directory between another's isdir check and its mkdir
    os.makedirs(cells_dir, exist_ok=True)
    os.makedirs(evaluations_dir, exist_ok=True)

    print("pipeline        : " + PIPELINE)
    print("data root       : " + str(dante_paths.get_data_root()))
    print("experiment root : " + str(dante_paths.get_experiment_root()))
    print("experiment      : " + str(dante_paths.experiment_dir(args.exp_id)))
    print("evaluations     : " + evaluations_dir)
    print("results root    : " + results_root)
    print("eval cameras    : " + ", ".join(eval_cameras))
    print("train cameras   : " + ", ".join(train_cameras))
    print("folds           : " + ", ".join(str(fold) for fold in folds))
    print("no_pointnet     : " + str(args.no_pointnet))
    print("pair files      : " + str(args.pair_files))
    print("")

    missing_checkpoints = []
    failed_cells = []
    written = []
    pair_files = []

    for eval_camera in eval_cameras:
        eval_dataset = camera_registry.dataset_of(eval_camera)
        print("\n=========== evaluation camera " + eval_camera
              + " (" + eval_dataset + ") ===========\n")

        # the evaluation camera's own ground truth and positions: this is what
        # makes an off-diagonal cell a cross-camera score
        positions, groups = import_data(eval_dataset)
        groups_at_time = add_time(groups)

        cells_path = os.path.join(
            cells_dir, "dante_cells_" + eval_camera + ".csv")
        # one pair file per training camera, rewritten as each fold lands; this
        # task owns them all, since no other task scores this evaluation camera
        pair_rows = {}
        with matrix_cells.CellWriter(cells_path) as cells:
            for fold in folds:
                # release the previous fold's block before reading the next
                test = X = timestamps = None
                try:
                    test = load_test_block(eval_dataset, fold)
                    X, _, timestamps = test
                    n_pairs, _, max_people, d = X[0].shape
                except Exception as exc:  # a bad fold costs a row, not the run
                    traceback.print_exc()
                    print("[WARN] could not load {} fold {}: {}".format(
                        eval_camera, fold, exc))
                    for train_camera in train_cameras:
                        failed_cells.append((train_camera, eval_camera, fold, str(exc)))
                        cells.write(matrix_cells.new_row(
                            PIPELINE, args.exp_id, train_camera, eval_camera, fold,
                            matrix_cells.STATUS_ERROR,
                            error="evaluation data unavailable: {}: {}".format(
                                type(exc).__name__, exc)))
                    continue

                n_frames = count_frames(timestamps)
                print("fold {}: {} pair samples over {} frames".format(
                    fold, n_pairs, n_frames))

                for train_camera in train_cameras:
                    checkpoint, expected = find_checkpoint(
                        args.exp_id, train_camera, fold, args.no_pointnet)
                    if checkpoint is None:
                        # a warning, not a failure: the rest of the table is
                        # still worth computing
                        print("[WARN] missing checkpoint for {} fold {}; expected: {}"
                              .format(train_camera, fold, expected))
                        missing_checkpoints.append((train_camera, fold, expected))
                        cells.write(matrix_cells.new_row(
                            PIPELINE, args.exp_id, train_camera, eval_camera, fold,
                            matrix_cells.STATUS_MISSING_CHECKPOINT,
                            checkpoint=expected,
                            error="no checkpoint at " + expected))
                        continue

                    started = time.time()
                    try:
                        model = load_model(checkpoint, max_people)
                        check_input_shape(model, max_people, d)
                        metrics = compute_group_metrics(
                            test, model, groups_at_time, eval_dataset, positions)
                    except Exception as exc:  # one bad cell must not end the run
                        traceback.print_exc()
                        print("[WARN] {} -> {} fold {} failed: {}".format(
                            train_camera, eval_camera, fold, exc))
                        failed_cells.append((train_camera, eval_camera, fold, str(exc)))
                        cells.write(matrix_cells.new_row(
                            PIPELINE, args.exp_id, train_camera, eval_camera, fold,
                            matrix_cells.STATUS_ERROR, checkpoint=checkpoint,
                            seconds=time.time() - started,
                            error="{}: {}".format(type(exc).__name__, exc)))
                        continue

                    print("{} -> {} fold {}: f1_1={:.4f} f1_2/3={:.4f} auc={:.4f}".format(
                        train_camera, eval_camera, fold,
                        metrics["f1_1"], metrics["f1_2_3"], metrics["auc"]))
                    cells.write(matrix_cells.new_row(
                        PIPELINE, args.exp_id, train_camera, eval_camera, fold,
                        matrix_cells.STATUS_OK, checkpoint=checkpoint,
                        metrics=metrics, n_eval_samples=n_pairs,
                        n_scenes=n_frames, seconds=time.time() - started))

                    pair_rows.setdefault(train_camera, []).append((fold, metrics))
                    if args.pair_files:
                        write_pair_file(args.exp_id, train_camera, eval_camera,
                                        pair_rows[train_camera])

        print("wrote " + cells_path + " (" + str(cells.count) + " cells)")
        written.append(cells_path)
        if args.pair_files:
            for train_camera in sorted(pair_rows):
                pair_files.append("%s@%s.csv" % (train_camera, eval_camera))

    print("\n----------- SUMMARY -----------\n")
    for path in written:
        print("cells: " + path)
    if pair_files:
        print("pairs: {} file(s) in {}: {}".format(
            len(pair_files), evaluations_dir, ", ".join(pair_files)))
    if missing_checkpoints:
        seen = sorted(set((camera, fold) for camera, fold, _ in missing_checkpoints))
        print("[WARN] {} of the expected checkpoints are missing: {}".format(
            len(seen), ", ".join("%s fold %d" % pair for pair in seen)))
        print("[WARN] the rows for those cameras are averaged over fewer folds; "
              "build_matrix_report.py records this in the results file.")
    if failed_cells:
        print("[WARN] {} cells failed to evaluate; see status=error in the cells "
              "CSV.".format(len(failed_cells)))
    print("\nNext: python scripts/build_matrix_report.py --model dante --exp-id "
          + str(args.exp_id))


if __name__ == "__main__":
    main()
