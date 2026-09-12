#!/usr/bin/env python3
"""Cross-camera evaluation of trained LSTM/GraphFF models.

Reads the checkpoints of one experiment and scores each of them on every camera,
producing one cell per (training camera, evaluation camera, fold). Nothing is
trained here.

One rule covers the whole matrix, the same one scripts/camera_matrix.py assumes:
the model trained on camera r fold k is scored on **camera c's fold-k held-out
test block**, never on all of camera c. Cameras inside a session film the same
event, so scoring a whole camera would feed the model the very moments it trained
on, seen from another angle. The diagonal is the c == r case of that rule, so it
recomputes what the training run already reported and is a useful consistency
check on this script.

Distances are normalised per camera inside get_data, so a cross-camera cell uses
the evaluation camera's own normalisation. That is the only sensible choice here
(the training camera's min/max is not a property of the model), but it is part of
what an off-diagonal number means.

Outputs, all under <experiment_root>/exp_<id>/results:

    cells/lstm_cells_<eval camera>.csv   one row per cell, this invocation
    ...                                  build_matrix_report.py merges them

and, for every off-diagonal cell, a copy at

    exp_<id>/<train camera>/fold_<k>/eval_<eval camera>/metrics_summary.csv

which is one of the layouts the two aggregators already recognise, so they pick
the cross-camera results up as well. The diagonal is deliberately not mirrored
back: the training run already wrote metrics_summary there, and a second copy
would trip the aggregators' duplicate check.

A missing checkpoint is a warning, not an error: the cell is recorded with
status=missing_checkpoint and the run continues, so one absent fold costs a gap
in the table rather than the whole matrix.

Example usage:
    python scripts/evaluate_matrix_lstm.py --exp-id 1
    python scripts/evaluate_matrix_lstm.py --exp-id 1 --eval-cam 06 --train-cam all
"""

import argparse
import os
import sys
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# the pipeline modules live in the repository root, this script's helpers in
# scripts/. Inserted in reverse so the listed order is the resolution order.
for _path in reversed((PROJECT_ROOT, SCRIPT_DIR)):
    if _path in sys.path:
        sys.path.remove(_path)
    sys.path.insert(0, _path)

import torch  # noqa: E402
import sklearn.metrics  # noqa: E402

import camera_registry  # noqa: E402
import matrix_cells  # noqa: E402

import graphff_paths  # noqa: E402
import parameters  # noqa: E402
import evaluate_scene  # noqa: E402
import model as model_module  # noqa: E402,F401  torch.load needs Skynet importable
from data import (  # noqa: E402
    condense_to_group_mat,
    convert_personwise_to_scene,
    get_data,
    split_data,
)
from utils import get_train_val_test_scenes  # noqa: E402
from analysis import get_model_predictions  # noqa: E402
from evaluate_scene import get_scenes_correctness, process_scene_gt  # noqa: E402

PIPELINE = "LSTM"


class _CachedCsvReader(object):
    """pandas stand-in for evaluate_scene, caching GT.csv by path.

    get_scenes_correctness re-reads and re-parses the evaluation camera's GT.csv
    on every call, and one array task calls it 50 times (5 training cameras x 5
    folds x 2 thresholds). The frames are identical every time, so this keeps it
    to one read per camera. Only the no-argument form is cached; anything else
    falls through to pandas untouched.
    """

    def __init__(self, pandas_module):
        self._pd = pandas_module
        self._cache = {}

    def read_csv(self, path, *args, **kwargs):
        if args or kwargs:
            return self._pd.read_csv(path, *args, **kwargs)
        key = str(path)
        if key not in self._cache:
            self._cache[key] = self._pd.read_csv(key)
        return self._cache[key]

    def __getattr__(self, name):
        return getattr(self._pd, name)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Score trained LSTM/GraphFF models across cameras.")
    parser.add_argument(
        "--exp-id", default=graphff_paths.get_run_id(),
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
        help="overrides GRAPHFF_EXPERIMENT_ROOT for this run.")
    parser.add_argument(
        "--data-root", default=None,
        help="overrides GRAPHFF_DATA_ROOT for this run.")
    parser.add_argument(
        "--results-root", default=None,
        help="where results are written (default <experiment_root>/exp_<id>/results).")
    parser.add_argument(
        "--seq-len", type=int, default=parameters.seq_len,
        help="sequence length the models were trained with (default %(default)s).")
    parser.add_argument(
        "--frame-stride", type=int, default=parameters.frame_stride,
        help="frame stride the models were trained with (default %(default)s).")
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"),
        help="where to run inference (default %(default)s).")
    parser.add_argument(
        "--rebuild-data", action="store_true", default=False,
        help="rebuild the split tensors from features.csv instead of reading the "
             "cached ones under <experiment_root>/_cache.")
    parser.add_argument(
        "--no-mirror-cells", dest="mirror_cells", action="store_false", default=True,
        help="do not write eval_<camera>/metrics_summary.csv into the fold "
             "directories; the cells CSV is then the only output.")
    return parser.parse_args()


def summarize_group_correctness(scenes_correctness):
    """Mirrors main_parallel.summarize_group_correctness exactly.

    Precision and recall are averaged over scenes first and F1 is formed from
    those averages, so the diagonal of this matrix reproduces the number the
    training run wrote rather than a differently-pooled one.
    """
    if len(scenes_correctness) == 0:
        return float("nan"), float("nan"), float("nan")
    precision = float(sum(item[-2] for item in scenes_correctness) / len(scenes_correctness))
    recall = float(sum(item[-1] for item in scenes_correctness) / len(scenes_correctness))
    if precision * recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall


def auc_from_predictions(labels, predictions):
    """Mirrors analysis.evaluate_AUC_score, reusing predictions already computed.

    NaN rather than an exception when the block holds a single class: that is a
    property of the evaluated block, and it should cost one number, not the cell.
    """
    last = torch.flatten(labels[:, -1, :]).detach().cpu().numpy()
    last_pred = torch.flatten(predictions[:, -1, :]).detach().cpu().numpy()
    try:
        return float(sklearn.metrics.roc_auc_score(last, last_pred))
    except ValueError:
        return float("nan")


def artifact_prefix(camera, seq_len, frame_stride):
    """Mirrors main_parallel's dataset_artifact_prefix for one camera."""
    prefix = "dataset=" + camera_registry.dataset_label_of(camera) + "_seq=" + str(seq_len)
    if frame_stride != 1:
        prefix += "_stride=" + str(frame_stride)
    return prefix


def find_checkpoint(exp_id, camera, fold, seq_len, frame_stride):
    """(path, [paths looked at]). path is None when the fold was never trained.

    Checkpoints belong to one (run_id, camera, fold) and live in that fold's own
    directory; the shared _cache is the pre-move location and is still read, the
    same fallback main_parallel applies when reloading.
    """
    name = (artifact_prefix(camera, seq_len, frame_stride)
            + "_model_fold" + str(fold) + ".pt")
    candidates = [
        str(graphff_paths.fold_output_dir(camera, exp_id, fold) / name),
        str(graphff_paths.cache_dir(camera) / name),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate, candidates
    return None, candidates


def load_fold_test_data(dataset, fold, seq_len, frame_stride, scenes_per_fold,
                        built, rebuild):
    """(test_list, test_set, trackers) for one camera and fold.

    Prefers the cached split tensors -- they are keyed by dataset, stride and
    seq_len, so they are exactly this fold's split and cost one torch.load
    instead of a full rebuild. ``built`` memoises the rebuilt arrays so a camera
    is read from features.csv at most once per invocation.
    """
    camera = graphff_paths.camera_of(dataset)
    cache_dir = str(graphff_paths.cache_dir(dataset))
    prefix = artifact_prefix(camera, seq_len, frame_stride) + "_fold" + str(fold)
    cached = dict(
        (name, os.path.join(cache_dir, prefix + "_" + name + ".pt"))
        for name in ("test_list", "test_set", "trackers"))

    if not rebuild and all(os.path.exists(path) for path in cached.values()):
        print("using cached split tensors: " + cached["test_set"])
        return (torch.load(cached["test_list"]),
                torch.load(cached["test_set"]),
                torch.load(cached["trackers"]))

    if dataset not in built:
        print("building split tensors from " + str(graphff_paths.dataset_dir(dataset)))
        built[dataset] = get_data(
            dataset, seq_len, parameters.feature_size,
            parameters.num_nodes, parameters.num_neighbors)
    data, labels, trackers = built[dataset]

    _, _, test_list, _, _, test_set = split_data(
        data, labels, trackers,
        scenes_per_fold[fold]["val_scenes"], scenes_per_fold[fold]["test_scenes"])
    return test_list, test_set, trackers


def score_one_cell(checkpoint, device, test_set, scene_group_idx_dict,
                   gt_groups_at_time, seq_len, num_nodes):
    """Metrics for one (model, evaluation block) pair, plus the scene count."""
    skynet = torch.load(checkpoint, map_location=device)
    skynet = skynet.to(device)
    skynet.eval()

    predictions = get_model_predictions(skynet, test_set)
    metrics = {"auc": auc_from_predictions(test_set.labels, predictions)}

    scene_seq_mat_dict = condense_to_group_mat(
        predictions, scene_group_idx_dict, seq_len, num_nodes)
    for label, threshold in (("1", 1.0), ("2_3", 2.0 / 3.0)):
        _, scenes_correctness, _, _, _, _ = get_scenes_correctness(
            num_nodes, scene_seq_mat_dict, gt_groups_at_time, threshold)
        f1, precision, recall = summarize_group_correctness(scenes_correctness)
        metrics["f1_" + label] = f1
        metrics["precision_" + label] = precision
        metrics["recall_" + label] = recall

    return metrics, len(scene_seq_mat_dict)


def mirror_cell(exp_id, train_camera, fold, test_camera, metrics):
    """Write the cell where the existing aggregators look for it."""
    directory = (graphff_paths.fold_output_dir(train_camera, exp_id, fold)
                 / ("eval_" + test_camera))
    os.makedirs(str(directory), exist_ok=True)
    path = str(directory / "metrics_summary.csv")
    matrix_cells.write_metrics_summary(path, metrics, test_camera)
    return path


def main():
    args = parse_args()

    # set before anything resolves a path: graphff_paths reads the environment on
    # every call, so this reaches the pipeline modules too
    if args.experiment_root:
        os.environ["GRAPHFF_EXPERIMENT_ROOT"] = args.experiment_root
    if args.data_root:
        os.environ["GRAPHFF_DATA_ROOT"] = args.data_root
    # get_data reads the stride from the environment rather than from an
    # argument, so --frame-stride has to reach it the same way, or the rebuilt
    # tensors and the artifact names it is keyed by would disagree
    os.environ["GRAPHFF_FRAME_STRIDE"] = str(args.frame_stride)

    eval_cameras = camera_registry.parse_camera_selection(args.eval_cam)
    train_cameras = camera_registry.parse_camera_selection(args.train_cam)
    folds = camera_registry.parse_fold_selection(args.fold)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    experiment_root = graphff_paths.get_experiment_root()
    results_root = (args.results_root if args.results_root
                    else str(graphff_paths.experiment_dir(args.exp_id) / "results"))
    cells_dir = os.path.join(results_root, "cells")
    # exist_ok: the array tasks start together and would otherwise race here,
    # one creating the directory between another's isdir check and its mkdir
    os.makedirs(cells_dir, exist_ok=True)

    print("pipeline        : " + PIPELINE)
    print("data root       : " + str(graphff_paths.get_data_root()))
    print("experiment root : " + str(experiment_root))
    print("experiment      : " + str(graphff_paths.experiment_dir(args.exp_id)))
    print("results root    : " + results_root)
    print("eval cameras    : " + ", ".join(eval_cameras))
    print("train cameras   : " + ", ".join(train_cameras))
    print("folds           : " + ", ".join(str(fold) for fold in folds))
    print("seq_len         : " + str(args.seq_len))
    print("frame_stride    : " + str(args.frame_stride))
    print("device          : " + str(device))
    print("rebuild data    : " + str(args.rebuild_data))
    print("mirror cells    : " + str(args.mirror_cells))
    print("")

    # one parse of GT.csv per camera instead of one per call
    evaluate_scene.pd = _CachedCsvReader(evaluate_scene.pd)

    num_nodes = parameters.num_nodes
    built = {}
    missing_checkpoints = []
    failed_cells = []
    written = []

    for eval_camera in eval_cameras:
        eval_dataset = camera_registry.dataset_of(eval_camera)
        print("\n=========== evaluation camera " + eval_camera
              + " (" + eval_dataset + ") ===========\n")

        # get_scenes_correctness reads GT.csv through this module-level name,
        # which `from parameters import *` bound to the training dataset. Every
        # other input is passed explicitly, so this one rebind is what makes the
        # scoring follow the evaluation camera.
        evaluate_scene.dataset_path = eval_dataset
        gt_groups_at_time = process_scene_gt(eval_dataset)
        scenes_per_fold, _ = get_train_val_test_scenes(eval_dataset)

        cells_path = os.path.join(
            cells_dir, "lstm_cells_" + eval_camera + ".csv")
        with matrix_cells.CellWriter(cells_path) as cells:
            for fold in folds:
                # release the previous fold's tensors before building the next
                test_list = test_set = trackers = None
                try:
                    test_list, test_set, trackers = load_fold_test_data(
                        eval_dataset, fold, args.seq_len, args.frame_stride,
                        scenes_per_fold, built, args.rebuild_data)
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

                scene_group_idx_dict = convert_personwise_to_scene(trackers, test_list)
                print("fold {}: {} test samples, {} scenes".format(
                    fold, test_set.size, len(scene_group_idx_dict)))

                for train_camera in train_cameras:
                    checkpoint, candidates = find_checkpoint(
                        args.exp_id, train_camera, fold,
                        args.seq_len, args.frame_stride)
                    if checkpoint is None:
                        # a warning, not a failure: the rest of the table is
                        # still worth computing
                        print("[WARN] missing checkpoint for {} fold {}; looked at: {}"
                              .format(train_camera, fold, ", ".join(candidates)))
                        missing_checkpoints.append((train_camera, fold, candidates[0]))
                        cells.write(matrix_cells.new_row(
                            PIPELINE, args.exp_id, train_camera, eval_camera, fold,
                            matrix_cells.STATUS_MISSING_CHECKPOINT,
                            checkpoint=candidates[0],
                            error="no checkpoint at " + " or ".join(candidates)))
                        continue

                    started = time.time()
                    try:
                        metrics, n_scenes = score_one_cell(
                            checkpoint, device, test_set, scene_group_idx_dict,
                            gt_groups_at_time, args.seq_len, num_nodes)
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
                        metrics=metrics, n_eval_samples=test_set.size,
                        n_scenes=n_scenes, seconds=time.time() - started))

                    if args.mirror_cells and train_camera != eval_camera:
                        mirror_cell(args.exp_id, train_camera, fold,
                                    eval_camera, metrics)

        print("wrote " + cells_path + " (" + str(cells.count) + " cells)")
        written.append(cells_path)
        # the built arrays for this camera are no longer needed
        built.pop(eval_dataset, None)

    print("\n----------- SUMMARY -----------\n")
    for path in written:
        print("cells: " + path)
    if missing_checkpoints:
        seen = sorted(set((camera, fold) for camera, fold, _ in missing_checkpoints))
        print("[WARN] {} of the expected checkpoints are missing: {}".format(
            len(seen), ", ".join("%s fold %d" % pair for pair in seen)))
        print("[WARN] the rows for those cameras are averaged over fewer folds; "
              "build_matrix_report.py records this in the results file.")
    if failed_cells:
        print("[WARN] {} cells failed to evaluate; see status=error in the cells "
              "CSV.".format(len(failed_cells)))
    print("\nNext: python scripts/build_matrix_report.py --model lstm --exp-id "
          + str(args.exp_id))


if __name__ == "__main__":
    main()
