import numpy as np

from utils import load_data, train_and_save_model
import dante_paths

import argparse

"""
Trains models with random architectures on each fold for a particular dataset.
Use the --no_pointnet flag to remove the Context Transform.
Use the --symmetric flag to cause the Dyad Tranform to use the same type of
symmetric architecture as the Context Transform.

Example usage:
python run_models.py -d mingling1/cam06 -f 0
"""

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dataset', help="which dataset to use", required=True)
    parser.add_argument('-p', '--no_pointnet', help="dont use global features", action='store_true')
    parser.add_argument('-s', '--symmetric', action="store_true", default=False, help="use this to run pointnet Dyad")
    parser.add_argument('-e', '--epochs', type=int, default=600, help="max number of epochs to run for")
    parser.add_argument('-f', '--fold',type=int, help="fold number")
    parser.add_argument('-b', '--batch_size', type=int, default=1024, help="training batch size")
    parser.add_argument('--patience', type=int, default=50, help="early stopping patience on validation MSE")
    parser.add_argument('--min_delta', type=float, default=0.0, help="minimum validation-MSE improvement for early stopping")
    parser.add_argument('--f1_eval_every', type=int, default=10,
        help="run validation F1/dominant-set evaluation every N epochs; 0 disables epoch-end F1")
    parser.add_argument('--run_id', type=str, default=dante_paths.get_run_id(),
        help="run identifier; output goes to <experiment_root>/exp_<run_id>/<dataset>/fold_<fold> "
             "(default from RUN_ID, else 1)")
    parser.add_argument('--overwrite', dest='overwrite', action='store_true', default=True,
        help="replace an existing fold output directory (default)")
    parser.add_argument('--no-overwrite', dest='overwrite', action='store_false',
        help="refuse to run if the fold output directory already exists")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    # folds =  # which folds to train on
    # while True: #??????
    
    fold = args.fold

    # get data
    fold_dir = dante_paths.fold_data_dir(args.dataset, fold)
    print("data root:       ", dante_paths.get_data_root())
    print("experiment root: ", dante_paths.get_experiment_root())
    print("fold data dir:   ", fold_dir)
    test, train, val = load_data(str(fold_dir))
    for j in range(1):
        # set model architecture
        # Context tranform in the paper
        global_filters = [np.random.choice([16, 32, 64])]
        global_filters += [np.random.choice([128, 256])]
        if np.random.choice([True, False]):
            global_filters += [np.random.choice([512, 1024])]
        if np.random.choice([True, False]): 
            global_filters += [np.random.choice([1024, 2048])]
        if np.random.choice([True, False]): 
            global_filters += [np.random.choice([1024, 2048])]
        global_filters.sort()

        # Dyad transform in the paper
        individual_filters = [np.random.choice([16, 32])]
        if np.random.choice([True, False]):
            individual_filters += [np.random.choice([32, 64])]
        if np.random.choice([True, False]):
            individual_filters += [np.random.choice([64, 128])]
        individual_filters.sort()

        # Final MLP in paper
        combined_filters = [np.random.choice([512, 1024])]
        if np.random.choice([True, False]):
            combined_filters += [np.random.choice([128, 256, 512])]
        if np.random.choice([True, False]):
            combined_filters += [np.random.choice([64, 128, 256, 512])]
        if np.random.choice([True, False]): 
            combined_filters += [np.random.choice([64, 128, 256])]
        combined_filters.sort(reverse=True)

        reg = 10 ** (float(np.random.randint(-90, -40))/10)
        dropout = float(np.random.randint(0, 30))/100

        print("global filters:", global_filters)
        print("combined filters:", combined_filters)
        train_and_save_model(global_filters, individual_filters, combined_filters,
        train, val, test, args.epochs, args.dataset,
        reg=reg, dropout=dropout, fold_num=fold, no_pointnet=args.no_pointnet,
        symmetric=args.symmetric, batch_size=args.batch_size,
        patience=args.patience, min_delta=args.min_delta,
        f1_eval_every=args.f1_eval_every, run_id=args.run_id,
        overwrite=args.overwrite)
