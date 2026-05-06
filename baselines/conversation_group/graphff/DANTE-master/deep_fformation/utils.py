import numpy as np

import os
import argparse
import pickle
import importlib

import tensorflow as tf
import keras as keras

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Reshape, MaxPooling2D, Concatenate, Lambda, Dot, BatchNormalization, Flatten

from F1_calc import F1_calc
import sys
sys.path.append("../datasets")
from reformat_data import add_time, import_data

"""
Holds code to build, train, and save models, as well as loading data.
Primarily used through run_models.py, but a specific model architecture can
be created by calling this script directly and modifing the model architecture
in __main__ at the bottom of the file.
"""


def install_numpy_pickle_compat():
    """Allow NumPy 2 pickles to load in the older NumPy used by TF1."""
    try:
        numpy_core = importlib.import_module("numpy.core")
        sys.modules.setdefault("numpy._core", numpy_core)
        for module_name in (
            "multiarray",
            "numeric",
            "_multiarray_umath",
            "fromnumeric",
            "umath",
            "shape_base",
        ):
            old_name = "numpy.core." + module_name
            new_name = "numpy._core." + module_name
            try:
                sys.modules.setdefault(new_name, importlib.import_module(old_name))
            except ImportError:
                pass
    except ImportError:
        pass


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-k', '--fold',type=str, default='0')
    parser.add_argument('-r', '--reg', type=float, default=0.0000001)
    parser.add_argument('-d', '--dropout', type=float, default=0.35)
    parser.add_argument('-e', '--epochs', type=int, default=600)
    parser.add_argument('--dataset', type=str, default="mingling1/cam06")
    parser.add_argument('-p', '--no_pointnet', action="store_true", default=False)
    parser.add_argument('-s', '--symmetric', action="store_true", default=False)
    parser.add_argument('-b', '--batch_size', type=int, default=1024)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--f1_eval_every', type=int, default=10)

    return parser.parse_args()

def load_matrix(file):
    install_numpy_pickle_compat()
    with open(file, 'rb') as f:
        return pickle.load(f)

# must have run build_dataset.py first
def load_data(path):
    train = load_matrix(path + '/train.p')
    test = load_matrix(path + '/test.p')
    val = load_matrix(path + '/val.p')
    return test, train, val

def is_mingling_dataset(dataset):
    return dataset.startswith("mingling1/") or dataset.startswith("mingling2/")

# creates a new directory to save the model to
def get_path(dataset, no_pointnet=False):
    path = 'models/' + dataset
    if not os.path.isdir(path):
        os.makedirs(path)

    if no_pointnet:
        path += '/no_pointnet'
        if not os.path.isdir(path):
            os.makedirs(path)

    path = path + '/pair_predictions_'
    i = 1
    while True:
        if not os.path.isdir(path + str(i)):
            path = path + str(i)
            os.makedirs(path)
            print('saving model to ' + path)
            break
        else:
            i += 1

        if i == 10000:
            raise ValueError("ERROR: could not find models directory")
    return path


# gives T=1 and T=2/3 F1 scores
def predict(data, model, groups_at_time, dataset="mingling1/cam06", positions=None):
    X, y, timestamps = data
    preds = model.predict(X)
    if is_mingling_dataset(dataset):
        n_people = 32
        n_features = 5
    else:
        raise ValueError("unknown dataset: " + dataset)

    return F1_calc(2/3, preds, timestamps, groups_at_time, positions,
        n_people, 1e-5, n_features), F1_calc(1, preds, timestamps, groups_at_time, positions,
        n_people, 1e-5, n_features)

def binary_auc_score(y_true, y_score):
    y_true = np.ravel(y_true).astype(float)
    y_score = np.ravel(y_score).astype(float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    rank_sum_pos = np.sum(ranks[y_true == 1])
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

def compute_auc(data, model):
    X, y, _ = data
    preds = model.predict(X)
    return binary_auc_score(y, preds)

def compute_group_metrics(data, model, groups_at_time, dataset, positions):
    auc = compute_auc(data, model)
    (f1_two_thirds, p_two_thirds, r_two_thirds, _, _), (f1_one, p_one, r_one, _, _) = predict(
        data, model, groups_at_time, dataset=dataset, positions=positions)
    return {
        'auc': auc,
        'f1_1': f1_one,
        'precision_1': p_one,
        'recall_1': r_one,
        'f1_2_3': f1_two_thirds,
        'precision_2_3': p_two_thirds,
        'recall_2_3': r_two_thirds,
    }

def write_metrics_summary(path, rows):
    with open(path, 'w') as handle:
        handle.write('split,auc,f1_1,precision_1,recall_1,f1_2_3,precision_2_3,recall_2_3\n')
        for split_name, metrics in rows:
            handle.write(
                split_name + ',' +
                str(metrics['auc']) + ',' +
                str(metrics['f1_1']) + ',' +
                str(metrics['precision_1']) + ',' +
                str(metrics['recall_1']) + ',' +
                str(metrics['f1_2_3']) + ',' +
                str(metrics['precision_2_3']) + ',' +
                str(metrics['recall_2_3']) + '\n'
            )

class ValLoss(keras.callbacks.Callback):
    # record train and val losses and mse
    def __init__(self, val_data, dataset, f1_eval_every=10):
        super(ValLoss, self).__init__()
        self.val_data = val_data
        self.dataset = dataset
        self.f1_eval_every = max(0, int(f1_eval_every))

        if is_mingling_dataset(dataset):
            self.positions, groups = import_data(dataset)
            self.groups_at_time = add_time(groups)
        else:
            raise ValueError("unrecognized dataset: " + dataset)

        self.best_val_weights = None
        self.best_val_mse = float("inf")
        self.best_epoch = -1

        self.val_f1_one_obj = {"f1s": [], "epochs": [], "best_f1": float('-inf'), "epoch": None}
        self.val_f1_two_thirds_obj = {"f1s": [], "epochs": [], "best_f1": float('-inf'), "epoch": None}

        self.val_losses = []
        self.train_losses = []

        self.val_mses = []
        self.train_mses = []

    def on_epoch_end(self, epoch, logs={}):
        val_mse = logs.get('val_mean_squared_error', logs.get('val_mse'))
        if val_mse is not None and val_mse < self.best_val_mse:
            self.best_val_weights = self.model.get_weights()
            self.best_val_mse = val_mse
            self.best_epoch = epoch

        self.val_losses.append(logs['val_loss'])
        self.train_losses.append(logs['loss'])
        self.val_mses.append(val_mse)
        self.train_mses.append(logs.get('mean_squared_error', logs.get('mse')))

        if self.f1_eval_every <= 0:
            return
        if (epoch + 1) % self.f1_eval_every != 0:
            return

        print("Running validation F1 evaluation at epoch {}".format(epoch + 1))
        (f1_two_thirds, _, _,original_affinities,frames), (f1_one, _, _,orignal_affinities, frames) = predict(self.val_data, self.model, self.groups_at_time,
            dataset=self.dataset, positions=self.positions)

        for f1, obj in [(f1_one, self.val_f1_one_obj), (f1_two_thirds, self.val_f1_two_thirds_obj)]:
            if f1 > obj['best_f1']:
                obj['best_f1'] = f1
                obj['epoch'] = epoch
            obj['f1s'].append(f1)
            obj['epochs'].append(epoch)

# saves the information in the model.history object to a .txt file
def write_history(file_name, history, test, model):
    file = open(file_name, 'w+')

    file.write("best_val: " + str(history.best_val_mse))
    file.write("\nepoch: " + str(history.best_epoch))

    file.write("\nbest_val_f1_1: " + str(history.val_f1_one_obj['best_f1']))
    file.write("\nepoch: " + str(history.val_f1_one_obj['epoch']))
    file.write("\nbest_val_f1_2/3: " + str(history.val_f1_two_thirds_obj['best_f1']))
    file.write("\nepoch: " + str(history.val_f1_two_thirds_obj['epoch']))

    val_metrics = compute_group_metrics(
        history.val_data, model, history.groups_at_time, history.dataset, history.positions)
    test_metrics = compute_group_metrics(
        test, model, history.groups_at_time, history.dataset, history.positions)

    metrics_summary_path = os.path.join(os.path.dirname(file_name), 'metrics_summary.csv')
    write_metrics_summary(metrics_summary_path, [('val', val_metrics), ('test', test_metrics)])

    file.write("\nval_metrics_best_val_mse_model:")
    file.write("\nval_auc: " + str(val_metrics['auc']))
    file.write("\nval_f1_1: " + str(val_metrics['f1_1']))
    file.write("\nval_precision_1: " + str(val_metrics['precision_1']))
    file.write("\nval_recall_1: " + str(val_metrics['recall_1']))
    file.write("\nval_f1_2/3: " + str(val_metrics['f1_2_3']))
    file.write("\nval_precision_2/3: " + str(val_metrics['precision_2_3']))
    file.write("\nval_recall_2/3: " + str(val_metrics['recall_2_3']))

    file.write("\ntest_metrics_best_val_mse_model:")
    file.write("\ntest_auc: " + str(test_metrics['auc']))
    file.write("\ntest_f1_1: " + str(test_metrics['f1_1']))
    file.write("\ntest_precision_1: " + str(test_metrics['precision_1']))
    file.write("\ntest_recall_1: " + str(test_metrics['recall_1']))
    file.write("\ntest_f1_2/3: " + str(test_metrics['f1_2_3']))
    file.write("\ntest_precision_2/3: " + str(test_metrics['precision_2_3']))
    file.write("\ntest_recall_2/3: " + str(test_metrics['recall_2_3']))

    # Backward-compatible compact lines used by older parsing scripts.
    file.write("\ntest_f1s_best_val_mse_model:")
    file.write("\ntest_f1s: " + str(test_metrics['f1_2_3']) + " " + str(test_metrics['f1_1']))
    file.write('\nprecisions: ' + str(test_metrics['precision_2_3']) + " " + str(test_metrics['precision_1']))
    file.write('\nrecalls: ' + str(test_metrics['recall_2_3']) + " " + str(test_metrics['recall_1']))

    file.write("\ntrain loss:")
    for loss in history.train_losses:
        file.write('\n' + str(loss))
    file.write("\nval loss:")
    for loss in history.val_losses:
        file.write('\n' + str(loss))
    file.write("\ntrain mse:")
    for loss in history.train_mses:
        file.write('\n' + str(loss))
    file.write("\nval mse:")
    for loss in history.val_mses:
        file.write('\n' + str(loss))
    file.write("\nval 1 f1:")
    for epoch, f1 in zip(history.val_f1_one_obj['epochs'], history.val_f1_one_obj['f1s']):
        file.write('\n' + str(epoch) + '\t' + str(f1))
    file.write("\nval 2/3 f1:")
    for epoch, f1 in zip(history.val_f1_two_thirds_obj['epochs'], history.val_f1_two_thirds_obj['f1s']):
        file.write('\n' + str(epoch) + '\t' + str(f1))
    file.close()

def conv(filters, reg, name=None):
    return Conv2D(filters=filters, kernel_size=1, padding='valid', kernel_initializer="he_normal",
        use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu, name=name)

def build_model(reg_amt, drop_amt, max_people, d, global_filters,
    individual_filters, combined_filters, no_pointnet=False, symmetric=False):

    group_inputs = keras.layers.Input(shape=(1, max_people, d))
    pair_inputs = keras.layers.Input(shape=(1, 2, d))

    reg = keras.regularizers.l2(reg_amt)

    y = pair_inputs

    # Dyad Transform
    for filters in individual_filters:
        y = conv(filters, reg)(y)
        y = Dropout(drop_amt)(y)
        y = BatchNormalization()(y)

    y_0 = Lambda(lambda input: tf.slice(input, [0, 0, 0, 0], [-1, -1, 1, -1]))(y)
    y_1 = Lambda(lambda input: tf.slice(input, [0, 0, 1, 0], [-1, -1, 1, -1]))(y)

    if no_pointnet:
        concat = Concatenate(name='concat')([Flatten()(y_0), Flatten()(y_1)])
    else:
        x = group_inputs

        # Context Transform
        for filters in global_filters:
            x = conv(filters, reg)(x)
            x = Dropout(drop_amt)(x)
            x = BatchNormalization()(x)


        x = MaxPooling2D(name="global_pool", pool_size=[1, max_people], strides=1, padding='valid')(x)
        x = Dropout(drop_amt)(x)
        x = BatchNormalization()(x)
        x_flat = Flatten()(x)

        # enforce symmetric affinity predictions by doing pointnet on 2 people
        if symmetric:
            y = MaxPooling2D(name="symmetric_pool", pool_size=[1, 2], strides=1, padding='valid')(y)
            concat = Concatenate(name='concat')([x_flat, Flatten()(y)])
        else:
            concat = Concatenate(name='concat')([x_flat, Flatten()(y_0), Flatten()(y_1)])

    # Final MLP from paper
    for filters in combined_filters:
        concat = Dense(units=filters, use_bias='True', kernel_regularizer=reg, activation=tf.nn.relu,
            kernel_initializer="he_normal")(concat)
        concat = Dropout(drop_amt)(concat)
        concat = BatchNormalization()(concat)

    # final pred
    affinity = Dense(units=1, use_bias="True", kernel_regularizer=reg, activation=tf.nn.sigmoid,
        name='affinity', kernel_initializer="glorot_normal")(concat)
    # affinity = Dense(units=1, use_bias="True", kernel_regularizer=reg, activation=tf.nn.sigmoid,
    #     name='affinity', kernel_initializer="glorot_normal")(concat)

    model = Model(inputs=[group_inputs, pair_inputs], outputs=affinity)

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False, clipvalue=0.5)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=['mse'])

    return model

# constructs a model, trains it with early stopping based on validation MSE, and then
# saves the output to a .txt file.
def train_and_save_model(global_filters, individual_filters, combined_filters,
    train, val, test, epochs, dataset, reg=0.0000001, dropout=.35, fold_num=0,
    no_pointnet=False, symmetric=False, batch_size=1024, patience=50,
    min_delta=0.0, f1_eval_every=10):

    # ensures repeatability
    tf.set_random_seed(0)
    np.random.seed(0)

    num_train, _, max_people, d = train[0][0].shape
    # save achitecture
    path = get_path(dataset, no_pointnet)
    file = open(path + '/architecture.txt', 'w+')
    file.write("global: " + str(global_filters) + "\nindividual: " +
        str(individual_filters) + "\ncombined: " + str(combined_filters) +
        "\nreg= " + str(reg) + "\ndropout= " + str(dropout) +
        "\nepochs= " + str(epochs) + "\nbatch_size= " + str(batch_size) +
        "\npatience= " + str(patience) + "\nmin_delta= " + str(min_delta) +
        "\nearly_stop_monitor= val_mean_squared_error" +
        "\nf1_eval_every= " + str(f1_eval_every))

    best_val_mses = []
    best_val_f1s_one = []
    best_val_f1s_two_thirds = []
    X_train, Y_train, timestamps_train = train
    X_val, Y_val, timestamps_val = val

    # build model
    model = build_model(reg, dropout, max_people, d,
        global_filters, individual_filters, combined_filters,
        no_pointnet=no_pointnet, symmetric=symmetric)

    # train model
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_mean_squared_error',
        patience=patience,
        min_delta=min_delta,
        mode='min')
    history = ValLoss(val, dataset, f1_eval_every=f1_eval_every)
    print("MODEL IS IN {}".format(path))
    print("training config: epochs={}, batch_size={}, patience={}, min_delta={}, early_stop_monitor=val_mean_squared_error, f1_eval_every={}".format(
        epochs, batch_size, patience, min_delta, f1_eval_every))
    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs')

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
        validation_data=(X_val, Y_val), callbacks=[tensorboard, history, early_stop])

    if history.best_val_weights is not None:
        print("Restoring best validation-MSE weights from epoch {}".format(history.best_epoch))
        model.set_weights(history.best_val_weights)

    best_val_mses.append(history.best_val_mse)
    best_val_f1s_one.append(history.val_f1_one_obj['best_f1'])
    best_val_f1s_two_thirds.append(history.val_f1_two_thirds_obj['best_f1'])

    # save model
    name = path + '/val_fold_' + str(fold_num)
    if not os.path.isdir(name):
        os.makedirs(name)

    write_history(name + '/results.txt', history, test, model)

    model.save(name + '/best_val_model.h5')
    print("saved best val model as " + '/best_val_model.h5')

    file.write("\n\nbest overall val loss: " + str(min(best_val_mses)))
    file.write("\nbest val losses per fold: " + str(best_val_mses))

    file.write("\n\nbest overall f1 1: " + str(max(best_val_f1s_one)))
    file.write("\nbest f1 1s per fold: " + str(best_val_f1s_one))

    file.write("\n\nbest overall f1 2/3: " + str(max(best_val_f1s_two_thirds)))
    file.write("\nbest f1 2/3s per fold: " + str(best_val_f1s_two_thirds))

    file.close()

if __name__ == "__main__":
    args = get_args()

    # get data
    test, train, val = load_data("../datasets/" + args.dataset + "/fold_" + args.fold)

    # set model architecture
    global_filters = [64, 128, 512]
    individual_filters = [16, 64, 128]
    combined_filters = [256, 64]

    train_and_save_model(global_filters, individual_filters, combined_filters,
        train, val, test, args.epochs, args.dataset,
        reg=args.reg, dropout=args.dropout, fold_num=args.fold, no_pointnet=args.no_pointnet,
        symmetric=args.symmetric, batch_size=args.batch_size,
        patience=args.patience, min_delta=args.min_delta,
        f1_eval_every=args.f1_eval_every)
