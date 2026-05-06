import pickle
import argparse
import matplotlib.pyplot as plt

from utils import load_data

from reformat_data import add_time, import_data
from F1_calc import F1_calc


from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

import numpy as np
import tensorflow as tf
import keras
import os


"""
Makes predictions on the training and test sets using a model and then
saves the results to a .txt file. Can also be used to instead calculate the F1
score on the test set with the --F1 flag.

Example usage:
python evaluate_model.py -k 0 -m models/mingling1/cam06/pair_predictions_1 -d mingling1/cam06 -f
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k_fold', type=str, default='0', 
        help="the fold being considered")
    parser.add_argument('-m', '--model_path', type=str, 
        help="path to the desired model directory (e.g. models/mingling1/cam06/pair_predictions_1/)", required=True)
    parser.add_argument('-d', '--dataset', type=str, required=True,
        help="which dataset to use (e.g. mingling1/cam06)")
    parser.add_argument('-f', '--F1', action='store_true', default=False, 
        help="calculates the F1 score on the test set, otherwise saves predictions to an output file") 
    parser.add_argument('--non_reusable', action='store_true', default=False, 
        help="doesn't reuse the same sets in GDSR calc")  

    return parser.parse_args()

# returns a prediction matrix for the training, test, and concatenated data
def build_predictions(model, fold, X_train, X_test):
    fold = int(fold)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    fold_len = test_preds.shape[0]
    test_idx = fold_len * fold

    preds = np.concatenate((train_preds[:test_idx], test_preds, train_preds[test_idx:]))

    return train_preds, test_preds, preds

# saves the predictions to the output file. Preds should be
# of length equal to the entire dataset
def save_predictions(preds, output_file_name, group_lines):
    print(preds.shape, len(group_lines)) # currently not including val data or something
    if preds.shape[0] != len(group_lines):
        throw("ERROR: prediction not for full data")

    print("saving predictions to " + output_file_name)
    output = open(output_file_name, 'w+')
    num_examples, output_len = preds.shape

    for i in range(num_examples):
        line = group_lines[i]
        split = line.split()
        timestamp = split[0]

        output.write(timestamp)
        for j in range(output_len):
            output.write(" ")
            output.write(str(preds[i][j]))

        output.write("\n")
    output.close()


def dump(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def is_mingling_dataset(dataset):
    return dataset.startswith("mingling1/") or dataset.startswith("mingling2/")

if __name__ == "__main__":
    args = get_args()

    test, train, val = load_data("../datasets/" + args.dataset + "/fold_" + str(args.k_fold))
    X, y, timestamps = test
    num_test, _, max_people, d = X[0].shape

    model = keras.models.load_model(args.model_path + "/val_fold_" + str(args.k_fold) 
        + "/best_val_model.h5", custom_objects={'tf':tf , 'max_people':max_people})

    preds = model.predict(X)

    if args.F1: # calculate F1
        if is_mingling_dataset(args.dataset):
            positions, groups = import_data(args.dataset)
            groups_at_time = add_time(groups)
            n_people = 32
            n_features = 5
        else:
            raise ValueError("unrecognized dataset: " + args.dataset)

        # f_2_3, _, _, original_affinities, frames= F1_calc(2/3, preds, timestamps, groups_at_time, positions,
        # n_people, 1e-5, n_features)

        f_1, _, _ , original_affinities, frames= F1_calc(1, preds, timestamps, groups_at_time, positions,
        n_people, 1e-5, n_features)

        print("FRAMES:",frames)

        path = args.model_path + "/affinities"
        if not os.path.isdir(path):
            os.makedirs(path)
        dump(path + '/affinities', original_affinities)
        dump(path + '/frames', frames)

        # print("F_2/3: ", f_2_3)
        print("F_1: ", f_1)



    else: # save predictions
        path = args.model_path + "/preds"
        if not os.path.isdir(path):
            os.makedirs(path)

        dump(path + '/preds', preds)
        dump(path + '/timestamps', timestamps)
