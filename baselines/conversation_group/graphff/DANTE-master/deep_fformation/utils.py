import numpy as np

import os
import argparse
import json
import pickle
import importlib
import shutil

import tensorflow as tf
import keras as keras

from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Dropout, Conv2D, Reshape, MaxPooling2D, Concatenate, Lambda, Dot, BatchNormalization, Flatten

from F1_calc import F1_calc
import sys
sys.path.append("../datasets")
from reformat_data import add_time, import_data

import dante_paths

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
    parser.add_argument('--run_id', type=str, default=dante_paths.get_run_id())
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=True)
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false')

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


# What a fold writes while it trains. best_val_model.h5 is the file evaluation
# reads; the other two exist so a fold killed at the wall clock limit can be
# resumed where it stopped instead of restarting from epoch 0. last_model.h5 is
# removed once the fold finishes, training_state.json is kept.
BEST_MODEL_NAME = 'best_val_model.h5'
LAST_MODEL_NAME = 'last_model.h5'
TRAINING_STATE_NAME = 'training_state.json'
STATE_VERSION = 1

# changing any of these invalidates a checkpoint: the weights in it no longer
# belong to the model the run would build
ARCH_CONFIG_KEYS = ('dataset', 'fold', 'no_pointnet', 'symmetric', 'arch_seed',
    'global_filters', 'individual_filters', 'combined_filters')


def tmp_name_for(path):
    # keeps the .h5 suffix, which some Keras versions use to pick the format
    return path[:-3] + '.tmp.h5' if path.endswith('.h5') else path + '.tmp'


# Saving through a temporary file in the same directory and renaming it into
# place means a kill during the write leaves the previous checkpoint intact
# rather than a truncated file. Returns False rather than raising: a checkpoint
# that cannot be written must not end a run that is otherwise fine.
def atomic_save_model(model, path):
    tmp_path = tmp_name_for(path)
    try:
        model.save(tmp_path)
        os.replace(tmp_path, path)
        return True
    except Exception as err:
        print("[WARN] could not write checkpoint {}: {}".format(path, err))
        remove_quietly(tmp_path)
        return False


def atomic_copy(src, dst):
    tmp_path = tmp_name_for(dst)
    try:
        shutil.copyfile(src, tmp_path)
        os.replace(tmp_path, dst)
        return True
    except Exception as err:
        print("[WARN] could not copy {} to {}: {}".format(src, dst, err))
        remove_quietly(tmp_path)
        return False


def atomic_write_json(path, payload):
    tmp_path = path + '.tmp'
    try:
        with open(tmp_path, 'w') as handle:
            # default=float: the numbers come from Keras logs as np.float32,
            # which json cannot serialise on its own
            json.dump(payload, handle, indent=2, default=float)
        os.replace(tmp_path, path)
        return True
    except Exception as err:
        print("[WARN] could not write {}: {}".format(path, err))
        remove_quietly(tmp_path)
        return False


def remove_quietly(path):
    try:
        os.remove(path)
    except OSError:
        pass


def load_training_state(path):
    state_path = os.path.join(str(path), TRAINING_STATE_NAME)
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path) as handle:
            state = json.load(handle)
    except Exception as err:
        print("[WARN] ignoring unreadable {}: {}".format(state_path, err))
        return None
    if state.get('version') != STATE_VERSION:
        print("[WARN] ignoring {}: state version {}, expected {}".format(
            state_path, state.get('version'), STATE_VERSION))
        return None
    return state


# A checkpoint is only resumable into a run that builds the same model. Rather
# than silently training a mismatched architecture for another day, stop and say
# which field differs.
def check_resume_compatible(saved_config, config, state_path):
    mismatched = []
    for key in ARCH_CONFIG_KEYS:
        if key in saved_config and saved_config[key] != config.get(key):
            mismatched.append((key, saved_config[key], config.get(key)))
    if mismatched:
        lines = ["[ERROR] cannot resume {}: the run differs from the checkpoint".format(state_path)]
        for key, was, now in mismatched:
            lines.append("        {}: checkpoint has {!r}, this run has {!r}".format(key, was, now))
        lines.append("        Use a different --run_id, or drop --resume to retrain the fold.")
        raise SystemExit("\n".join(lines))

    for key in ('batch_size', 'reg', 'dropout'):
        if key in saved_config and saved_config[key] != config.get(key):
            print("[WARN] resuming with a different {}: checkpoint has {}, this run has {}".format(
                key, saved_config[key], config.get(key)))

# creates the output directory for one fold of one run
# the directory is fully determined by (dataset, run_id, fold), so the folds of a
# 5-fold array job land side by side under a single exp_<run_id> instead of
# racing for an auto-incremented one
def get_path(dataset, run_id, fold, no_pointnet=False, overwrite=True, resume=False):
    path = dante_paths.fold_output_dir(dataset, run_id, fold, no_pointnet=no_pointnet)

    if path.exists():
        # resume wins over overwrite: the point of the flag is to keep what is
        # already there. Without a state file there is nothing to continue from,
        # so fall through to the usual overwrite decision.
        if resume:
            if (path / TRAINING_STATE_NAME).exists():
                print('resuming existing fold output at ' + str(path))
                return str(path)
            print('[WARN] resume requested but no ' + TRAINING_STATE_NAME +
                ' in ' + str(path) + '; training this fold from scratch')
        if not overwrite:
            raise SystemExit(
                "[ERROR] fold output already exists: " + str(path) + "\n"
                "Re-run with --overwrite to replace it, or pass a different "
                "--run_id / RUN_ID to write a separate run."
            )
        print('replacing existing fold output at ' + str(path))
        shutil.rmtree(str(path))

    os.makedirs(str(path))
    print('saving model to ' + str(path))
    return str(path)


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

    # everything a resumed run needs to carry on as if it had never stopped:
    # the loss curves results.txt reports, and the best-so-far that decides
    # whether a later epoch may overwrite best_val_model.h5
    def state_dict(self):
        return {
            'best_val_mse': self.best_val_mse,
            'best_epoch': self.best_epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mses': self.train_mses,
            'val_mses': self.val_mses,
            'val_f1_one': self.val_f1_one_obj,
            'val_f1_two_thirds': self.val_f1_two_thirds_obj,
        }

    def load_state(self, state):
        self.best_val_mse = float(state.get('best_val_mse', float("inf")))
        self.best_epoch = int(state.get('best_epoch', -1))
        self.train_losses = list(state.get('train_losses', []))
        self.val_losses = list(state.get('val_losses', []))
        self.train_mses = list(state.get('train_mses', []))
        self.val_mses = list(state.get('val_mses', []))
        self.val_f1_one_obj = state.get('val_f1_one', self.val_f1_one_obj)
        self.val_f1_two_thirds_obj = state.get('val_f1_two_thirds', self.val_f1_two_thirds_obj)
        # best_val_weights stays None; the weights themselves come back from
        # best_val_model.h5 rather than from the state file

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

class ResumableEarlyStopping(keras.callbacks.EarlyStopping):
    """EarlyStopping whose patience counter survives a restart.

    on_train_begin resets wait and best, which would hand a resumed fold a fresh
    patience budget and let it train well past the point the original run would
    have stopped. The recorded counters are put back right afterwards.
    """
    def __init__(self, resume_state=None, **kwargs):
        super(ResumableEarlyStopping, self).__init__(**kwargs)
        self.resume_state = resume_state or {}

    def on_train_begin(self, logs=None):
        super(ResumableEarlyStopping, self).on_train_begin(logs)
        if 'wait' in self.resume_state:
            self.wait = int(self.resume_state['wait'])
        if self.resume_state.get('best') is not None:
            self.best = float(self.resume_state['best'])
            print("restored early stopping: best={}, {} epochs without improvement".format(
                self.best, self.wait))

    def state_dict(self):
        # `best` only exists once training has begun; this is also read when a
        # resumed fold skips straight to evaluation
        return {'wait': getattr(self, 'wait', 0), 'best': getattr(self, 'best', None)}


class CheckpointWriter(keras.callbacks.Callback):
    """Writes the fold's checkpoints and training state at the end of each epoch.

    Must be the LAST callback in the list: it reads the state that ValLoss and
    the early stopper have just updated for this epoch.

    last_model.h5 carries the optimizer state as well as the weights, so a
    resumed run continues with the same Adam moments rather than restarting the
    optimizer mid-training. training_state.json is written after both .h5 files,
    so a kill in between makes the recorded epoch a lower bound on what the
    checkpoints hold -- a resumed run then repeats at most one epoch, which is
    harmless, whereas the opposite order would resume from weights older than
    the state claims.
    """
    def __init__(self, path, history, early_stop, config):
        super(CheckpointWriter, self).__init__()
        self.path = str(path)
        self.history = history
        self.early_stop = early_stop
        self.config = config
        self.best_path = os.path.join(self.path, BEST_MODEL_NAME)
        self.last_path = os.path.join(self.path, LAST_MODEL_NAME)
        self.state_path = os.path.join(self.path, TRAINING_STATE_NAME)

    def on_epoch_end(self, epoch, logs=None):
        is_best = self.history.best_epoch == epoch
        saved_last = atomic_save_model(self.model, self.last_path)
        if is_best:
            # identical bytes, so copy the file just written instead of
            # serialising the same model a second time
            if not (saved_last and atomic_copy(self.last_path, self.best_path)):
                atomic_save_model(self.model, self.best_path)
            self.write_best_marker()
        self.write_state(epoch)

    # says which epoch best_val_model.h5 holds, which is the only record of it
    # when the fold is killed before results.txt is written
    def write_best_marker(self):
        try:
            with open(os.path.join(self.path, 'best_checkpoint.txt'), 'w') as handle:
                handle.write("epoch: " + str(self.history.best_epoch + 1) + "\n")
                handle.write("val_mean_squared_error: " + str(self.history.best_val_mse) + "\n")
                handle.write("checkpoint: " + BEST_MODEL_NAME + "\n")
        except IOError as err:
            print("[WARN] could not write best_checkpoint.txt: {}".format(err))
        print("checkpointed best val MSE {} from epoch {}".format(
            self.history.best_val_mse, self.history.best_epoch + 1))

    def write_state(self, epoch, completed=False):
        state = {
            'version': STATE_VERSION,
            # epochs finished, so it is exactly the initial_epoch a resumed run
            # passes to fit()
            'last_epoch': epoch + 1,
            # self.model is unset when the final write happens after a fold that
            # skipped fit(), so read it defensively
            'early_stopped': bool(getattr(getattr(self, 'model', None), 'stop_training', False)),
            'completed': completed,
            'config': self.config,
            'early_stopping': self.early_stop.state_dict(),
        }
        state.update(self.history.state_dict())
        atomic_write_json(self.state_path, state)


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


# Takes the model back off disk for a resumed fold. A full-model .h5 carries the
# optimizer state, so training continues with the same Adam moments instead of
# restarting the optimizer mid-run. Deserialising the Lambda layers build_model
# uses is the fragile part across Keras versions, so a failure there falls back
# to rebuilding the architecture and loading the weights into it.
def load_checkpoint_model(path, reg_amt, drop_amt, max_people, d, global_filters,
    individual_filters, combined_filters, no_pointnet=False, symmetric=False):

    last_path = os.path.join(str(path), LAST_MODEL_NAME)
    best_path = os.path.join(str(path), BEST_MODEL_NAME)
    # last_model.h5 is removed once a fold finishes, so a rerun of a finished
    # fold picks up the best checkpoint and only re-evaluates it
    checkpoint_path = last_path if os.path.exists(last_path) else best_path
    if not os.path.exists(checkpoint_path):
        raise SystemExit(
            "[ERROR] cannot resume: " + str(path) + " has " + TRAINING_STATE_NAME +
            " but neither " + LAST_MODEL_NAME + " nor " + BEST_MODEL_NAME + ".\n"
            "        Drop the resume flag to retrain the fold from scratch.")

    try:
        model = keras.models.load_model(checkpoint_path,
            custom_objects={'tf': tf, 'max_people': max_people})
        print("resuming from " + checkpoint_path + " (optimizer state included)")
        return model
    except Exception as err:
        print("[WARN] could not load {} as a full model ({}); rebuilding the "
            "architecture and loading weights only, which resets the optimizer "
            "state".format(checkpoint_path, err))

    model = build_model(reg_amt, drop_amt, max_people, d, global_filters,
        individual_filters, combined_filters, no_pointnet=no_pointnet,
        symmetric=symmetric)
    model.load_weights(checkpoint_path)
    return model

# constructs a model, trains it with early stopping based on validation MSE, and then
# saves the output to a .txt file.
def train_and_save_model(global_filters, individual_filters, combined_filters,
    train, val, test, epochs, dataset, reg=0.0000001, dropout=.35, fold_num=0,
    no_pointnet=False, symmetric=False, batch_size=1024, patience=50,
    min_delta=0.0, f1_eval_every=10, run_id=dante_paths.DEFAULT_RUN_ID,
    overwrite=True, arch_seed=None, resume=False):

    # ensures repeatability
    tf.set_random_seed(0)
    np.random.seed(0)

    num_train, _, max_people, d = train[0][0].shape
    # everything this fold produces goes in one directory. architecture.txt lives
    # here rather than one level up because the architecture is resampled per
    # fold, so a run-level file would be overwritten by each fold in turn.
    path = get_path(dataset, run_id, fold_num, no_pointnet=no_pointnet,
        overwrite=overwrite, resume=resume)

    config = {
        'dataset': dataset,
        'fold': int(fold_num),
        'no_pointnet': bool(no_pointnet),
        'symmetric': bool(symmetric),
        'arch_seed': None if arch_seed is None else int(arch_seed),
        'global_filters': [int(f) for f in global_filters],
        'individual_filters': [int(f) for f in individual_filters],
        'combined_filters': [int(f) for f in combined_filters],
        'reg': float(reg),
        'dropout': float(dropout),
        'epochs': int(epochs),
        'batch_size': int(batch_size),
        'patience': int(patience),
        'min_delta': float(min_delta),
    }

    state = load_training_state(path) if resume else None
    if state is not None:
        check_resume_compatible(state.get('config', {}), config,
            os.path.join(path, TRAINING_STATE_NAME))

    # a resumed fold appends to architecture.txt; truncating it would throw away
    # the record of the run that produced the checkpoint
    file = open(path + '/architecture.txt', 'a' if state is not None else 'w+')
    if state is not None:
        file.write("\n\nresumed at epoch " + str(state['last_epoch'] + 1) +
            " of " + str(epochs) + "\n")
    file.write("global: " + str(global_filters) + "\nindividual: " +
        str(individual_filters) + "\ncombined: " + str(combined_filters) +
        "\nreg= " + str(reg) + "\ndropout= " + str(dropout) +
        "\nepochs= " + str(epochs) + "\nbatch_size= " + str(batch_size) +
        "\npatience= " + str(patience) + "\nmin_delta= " + str(min_delta) +
        "\nearly_stop_monitor= val_mean_squared_error" +
        "\nf1_eval_every= " + str(f1_eval_every) +
        "\narch_seed= " + ("none" if arch_seed is None else str(arch_seed)))

    best_val_mses = []
    best_val_f1s_one = []
    best_val_f1s_two_thirds = []
    X_train, Y_train, timestamps_train = train
    X_val, Y_val, timestamps_val = val

    best_path = os.path.join(path, BEST_MODEL_NAME)

    # build model, or take it back from the checkpoint a previous attempt left
    if state is not None:
        model = load_checkpoint_model(path, reg, dropout, max_people, d,
            global_filters, individual_filters, combined_filters,
            no_pointnet=no_pointnet, symmetric=symmetric)
    else:
        model = build_model(reg, dropout, max_people, d,
            global_filters, individual_filters, combined_filters,
            no_pointnet=no_pointnet, symmetric=symmetric)

    # train model
    early_stop = ResumableEarlyStopping(
        resume_state=(state or {}).get('early_stopping'),
        monitor='val_mean_squared_error',
        patience=patience,
        min_delta=min_delta,
        mode='min')
    history = ValLoss(val, dataset, f1_eval_every=f1_eval_every)
    if state is not None:
        history.load_state(state)
    checkpointer = CheckpointWriter(path, history, early_stop, config)
    print("MODEL IS IN {}".format(path))
    print("training config: epochs={}, batch_size={}, patience={}, min_delta={}, early_stop_monitor=val_mean_squared_error, f1_eval_every={}".format(
        epochs, batch_size, patience, min_delta, f1_eval_every))
    tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(path, 'tb'))

    initial_epoch = int(state['last_epoch']) if state is not None else 0
    already_finished = state is not None and (
        state.get('completed') or state.get('early_stopped') or initial_epoch >= epochs)

    if already_finished:
        # the fold trained to the end already; go straight to scoring the best
        # checkpoint, which is what a rerun after a kill during write_history
        # needs, and costs nothing when results.txt is simply being regenerated
        print("training already finished at epoch {} (completed={}, early_stopped={}); "
            "skipping to evaluation".format(initial_epoch, state.get('completed'),
                state.get('early_stopped')))
    else:
        if initial_epoch > 0:
            print("resuming training at epoch {} of {} (best val MSE {} at epoch {})".format(
                initial_epoch + 1, epochs, history.best_val_mse, history.best_epoch + 1))
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
            initial_epoch=initial_epoch, validation_data=(X_val, Y_val),
            callbacks=[tensorboard, history, early_stop, checkpointer])

    if history.best_val_weights is not None:
        print("Restoring best validation-MSE weights from epoch {}".format(history.best_epoch))
        model.set_weights(history.best_val_weights)
    elif os.path.exists(best_path):
        # a resumed run that never beat the earlier best has no best weights in
        # memory, and `model` currently holds the last epoch rather than the
        # best one, so the weights have to come back off disk
        print("Restoring best validation-MSE weights from {} (epoch {})".format(
            BEST_MODEL_NAME, history.best_epoch + 1))
        model.load_weights(best_path)

    best_val_mses.append(history.best_val_mse)
    best_val_f1s_one.append(history.val_f1_one_obj['best_f1'])
    best_val_f1s_two_thirds.append(history.val_f1_two_thirds_obj['best_f1'])

    # save model
    write_history(path + '/results.txt', history, test, model)

    # the per-epoch checkpoint already holds these weights; rewriting it here
    # keeps the final file identical to the restored model
    model.save(best_path)
    print("saved best val model as " + best_path)

    # the fold is done: record that, so a resubmission regenerates results
    # instead of training again, and drop the resume-only copy of the weights
    checkpointer.write_state(
        max(initial_epoch, len(history.val_losses)) - 1, completed=True)
    remove_quietly(os.path.join(path, LAST_MODEL_NAME))

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
    test, train, val = load_data(str(dante_paths.fold_data_dir(args.dataset, args.fold)))

    # set model architecture
    global_filters = [64, 128, 512]
    individual_filters = [16, 64, 128]
    combined_filters = [256, 64]

    train_and_save_model(global_filters, individual_filters, combined_filters,
        train, val, test, args.epochs, args.dataset,
        reg=args.reg, dropout=args.dropout, fold_num=args.fold, no_pointnet=args.no_pointnet,
        symmetric=args.symmetric, batch_size=args.batch_size,
        patience=args.patience, min_delta=args.min_delta,
        f1_eval_every=args.f1_eval_every, run_id=args.run_id,
        overwrite=args.overwrite)
