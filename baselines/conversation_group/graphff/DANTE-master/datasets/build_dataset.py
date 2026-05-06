import argparse
import os
import pickle

import numpy as np


"""
Build DANTE fold pickle files from Mingling DANTE artifacts.

Example usage from DANTE-master/datasets:
python build_dataset.py -p mingling1/cam06
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="dataset path relative to DANTE-master/datasets, e.g. mingling1/cam06",
    )
    return parser.parse_args()


def build_X(people_lines, max_people, d):
    num_examples = len(people_lines)
    X_group = np.zeros((num_examples, 1, max_people, d), dtype=np.float32)
    X_pairs = np.zeros((num_examples, 1, 2, d), dtype=np.float32)

    for i in range(num_examples):
        line = people_lines[i]
        split = line.split() if type(line) == str else line

        j = 1
        while j < len(split):
            representation = split[j:j + d]
            vect = np.array([float(x) for x in representation], dtype=np.float32)
            person_idx = int((j - 1) / d)

            if j < len(split) - 2 * d:
                X_group[i, 0, person_idx, :] = vect
            else:
                X_pairs[i, 0, person_idx - max_people, :] = vect

            j += d

    return X_group, X_pairs


def build_Y(group_lines):
    output_size = len(group_lines[0].split()) - 1
    num_examples = len(group_lines)

    Y = np.zeros((num_examples, output_size), dtype=np.float32)
    timestamps = []

    for i in range(num_examples):
        line = group_lines[i]
        split = line.split()
        timestamps.append(split[0])
        Y[i] = np.asarray(split[1:], dtype=np.float32)
    return Y, timestamps


def split_test_train_val(start_test, end_test, val_start_ends, data):
    val = []
    for start, end in val_start_ends:
        val.append(data[start:end])
    val = np.concatenate(val)

    test = data[start_test:end_test]
    start = min([start for start, end in val_start_ends] + [start_test])
    end = max([end for start, end in val_start_ends] + [end_test])
    train = np.concatenate((data[:start], data[end:]))

    return test, train, val


def get_start_end_timechange(fold, num_folds, X, path):
    with open(path + "/timechanges.txt", "r") as handle:
        timechanges = [int(val) for val in handle.readlines()]

    timechanges.append(X.shape[0])
    num_times = len(timechanges) - 1

    start_test_idx = int(num_times / num_folds * fold)
    end_test_idx = int(num_times / num_folds * (fold + 1))
    val_fold_idx_diff = int(num_times / num_folds / 2)

    start_test = timechanges[start_test_idx]
    end_test = timechanges[end_test_idx]

    if start_test == 0:
        val_start = end_test
        val_end_idx = end_test_idx + 2 * val_fold_idx_diff
        val_end = timechanges[val_end_idx]
        val_start_ends = [(val_start, val_end)]
    elif fold == num_folds - 1:
        val_start_idx = start_test_idx - 2 * val_fold_idx_diff
        val_start = timechanges[val_start_idx]
        val_end = start_test
        val_start_ends = [(val_start, val_end)]
    else:
        val_start_idx = start_test_idx - val_fold_idx_diff
        val_start = timechanges[val_start_idx]
        val_end = start_test
        val_start_ends = [(val_start, val_end)]

        val_start = end_test
        val_end_idx = end_test_idx + val_fold_idx_diff
        val_end = timechanges[val_end_idx]
        val_start_ends.append((val_start, val_end))

    return start_test, end_test, val_start_ends


def dump(path, data):
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


if __name__ == "__main__":
    args = get_args()

    num_folds = 5
    path = args.path
    d = 6

    with open(os.path.join(path, "coordinates.txt"), "r") as people_file:
        people_lines = people_file.readlines()

    max_people = int((len(people_lines[0].split()) - 1) / d) - 2
    X_group, X_pairs = build_X(people_lines, max_people, d)

    with open(os.path.join(path, "affinities.txt"), "r") as group_file:
        group_lines = group_file.readlines()

    Y, timestamps = build_Y(group_lines)

    for k in range(num_folds):
        print("fold number: ", k)
        start_test, end_test, val_start_ends = get_start_end_timechange(k, num_folds, X_group, path)

        X_group_test, X_group_train, X_group_val = split_test_train_val(start_test, end_test, val_start_ends, X_group)
        X_pairs_test, X_pairs_train, X_pairs_val = split_test_train_val(start_test, end_test, val_start_ends, X_pairs)
        Y_test, Y_train, Y_val = split_test_train_val(start_test, end_test, val_start_ends, Y)
        timestamps_test, timestamps_train, timestamps_val = split_test_train_val(start_test, end_test, val_start_ends, timestamps)

        temp_path = os.path.join(path, "fold_" + str(k))
        if not os.path.isdir(temp_path):
            os.makedirs(temp_path)

        print("shape:", X_group_test.shape)
        dump(temp_path + "/test.p", ([X_group_test, X_pairs_test], Y_test, timestamps_test))
        dump(temp_path + "/train.p", ([X_group_train, X_pairs_train], Y_train, timestamps_train))
        dump(temp_path + "/val.p", ([X_group_val, X_pairs_val], Y_val, timestamps_val))
