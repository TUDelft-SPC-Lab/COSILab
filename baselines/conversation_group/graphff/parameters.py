import os
import torch
from utils import get_train_val_test_scenes
from dataset_registry import get_dataset_config, get_dataset_frame_stride, get_dataset_label


def _get_env_bool(name, default):
	value = os.environ.get(name)
	if value is None:
		return default
	return value.strip().lower() in ("1", "true", "yes", "y", "on")


def _get_env_int(name, default):
	value = os.environ.get(name)
	if value is None:
		return default
	return int(value)


def _get_env_float(name, default):
	value = os.environ.get(name)
	if value is None:
		return default
	return float(value)

# ----------- data parameters -----------
dataset_path = os.environ.get('GRAPHFF_DATASET', 'mingling1/cam06')
dataset_label = get_dataset_label(dataset_path)
dataset_config = get_dataset_config(dataset_path)
frame_stride = get_dataset_frame_stride(dataset_path)

num_nodes = dataset_config['num_nodes']
num_of_actual_people = dataset_config.get('num_of_actual_people', num_nodes)
num_neighbors = num_nodes - 1
feature_size = dataset_config['feature_size'] # including indicator feature


dataset_make_flag = _get_env_bool('GRAPHFF_DATASET_MAKE', False)

# ----------- training parameters -----------
train_flag = _get_env_bool('GRAPHFF_TRAIN', False)
batch_size = _get_env_int('GRAPHFF_BATCH_SIZE', 128)
num_epochs = _get_env_int('GRAPHFF_NUM_EPOCHS', 600)
lr = _get_env_float('GRAPHFF_LR', 0.001)
early_stop_patience = _get_env_int('GRAPHFF_PATIENCE', 50)
early_stop_min_delta = _get_env_float('GRAPHFF_MIN_DELTA', 0.0)
f1_eval_every = _get_env_int('GRAPHFF_F1_EVAL_EVERY', 10)
detect_anomaly = _get_env_bool('GRAPHFF_DETECT_ANOMALY', False)
loss_fn = torch.nn.MSELoss()

# ----------- model parameters ------------
hidden_dim = _get_env_int('GRAPHFF_HIDDEN_DIM', 8)  # parameter sweep candidate: [5,10,15,20]
seq_len = _get_env_int('GRAPHFF_SEQ_LEN', 10) # parameter sweep candidate:  [5,10,15,20]
threshold = _get_env_float('GRAPHFF_THRESHOLD', 1.0)

# ----------- fold parameters -----------

# ----------- result filename -----------
output_dir = os.environ.get('GRAPHFF_OUTPUT_DIR', 'output')
loss_path = 'loss_curve'
AUC_path = 'AUC'
f1_path = 'f1'

path_suffix = ''
