import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset_registry import get_dataset_data_dir, get_dataset_frame_stride, is_mingling_dataset
from parameters import *
from utils import *

torch.set_default_dtype(torch.float32)


class Tracker:
	def __init__(self, scene_start, scene_end, self_person, num_nodes, scene_eval_idx=None):
		self.scene_start = scene_start
		self.scene_end = scene_end
		self.self_person = self_person
		if scene_eval_idx is None:
			scene_eval_idx = scene_end - 1
		self.scene_eval_idx = scene_eval_idx
		self.neighbors = list(range(num_nodes))
		self.neighbors.remove(self_person)


class DataStruct:
	def __init__(self, data, labels, trackers):
		self.data = data
		self.labels = labels
		self.trackers = trackers
		self.size = data.shape[0]
		if data.shape[0] != labels.shape[0]:
			raise ValueError('Incorrect size in DataStruct')
		if data.shape[0] != len(trackers):
			raise ValueError('Incorrect size in DataStruct')


def process_gt_matrices(gt, num_nodes):
	gt_matrix = gt.iloc[:, 1:-1]
	gt_matrix = gt_matrix.dropna()
	gt_matrix = gt_matrix.values
	gt_matrices = []
	i = 0
	while i < len(gt_matrix):
		chunk = gt_matrix[i:i + num_nodes, :]
		gt_matrices.append(chunk)
		i = i + num_nodes
	return np.array(gt_matrices, dtype=int)


def load_scene_continuity(dataset_dir):
	return pd.read_csv(os.path.join(dataset_dir, 'scene_continuity.csv'), header=None).to_numpy().squeeze()


def _wrap_angle_np(alpha):
	return np.arctan2(np.sin(alpha), np.cos(alpha))


def _nan_mean_angle_np(alpha, axis):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		c = np.nanmean(np.cos(alpha), axis=axis)
		s = np.nanmean(np.sin(alpha), axis=axis)
	return _wrap_angle_np(np.arctan2(s, c))


def _feature_matrix(features, prefix, num_nodes):
	cols = [prefix + str(person_id) for person_id in range(1, num_nodes + 1)]
	return features[cols].to_numpy(dtype=np.float64)


def downsample_scene_continuity(scene_continuity, source_indices):
	source_indices = np.asarray(source_indices, dtype=np.int64)
	if len(source_indices) == len(scene_continuity) and np.all(source_indices == np.arange(len(scene_continuity))):
		return scene_continuity

	downsampled = np.zeros(len(source_indices), dtype=scene_continuity.dtype)
	if len(source_indices) == 0:
		return downsampled
	downsampled[0] = scene_continuity[source_indices[0]]
	for idx in range(1, len(source_indices)):
		start = source_indices[idx - 1] + 1
		end = source_indices[idx] + 1
		downsampled[idx] = 1 if np.any(scene_continuity[start:end] == 1) else 0
	return downsampled


def downsample_mingling_frames(features, gt_matrices, scene_continuity, frame_stride):
	source_indices = np.arange(len(features))
	if frame_stride <= 1:
		return features, gt_matrices, scene_continuity, source_indices

	source_indices = source_indices[::frame_stride]
	features = features.iloc[source_indices].reset_index(drop=True)
	gt_matrices = gt_matrices[source_indices]
	scene_continuity = downsample_scene_continuity(scene_continuity, source_indices)
	print('frame stride : ', frame_stride)
	print('retained frames: ', len(source_indices))
	return features, gt_matrices, scene_continuity, source_indices


def get_mingling_data_fast(features, gt_matrices, scene_continuity,
				 seq_len, feature_size, num_nodes, num_neighbors,
				 source_indices=None):
	if feature_size != 7:
		raise ValueError('Mingling vectorized path expects feature_size=7')

	num_scenes = features.shape[0]
	if source_indices is None:
		source_indices = np.arange(num_scenes)
	else:
		source_indices = np.asarray(source_indices, dtype=np.int64)
	num_segments = num_scenes - seq_len + 1
	if num_segments <= 0:
		raise ValueError('Not enough scenes for seq_len')

	print('using vectorized Mingling preprocessing')
	print('num scenes   : ', num_scenes)
	print('num segments : ', num_segments)

	x = _feature_matrix(features, 'X', num_nodes)
	y = _feature_matrix(features, 'Y', num_nodes)
	theta_h = _wrap_angle_np(_feature_matrix(features, 'theta_H', num_nodes))
	theta_b = _wrap_angle_np(_feature_matrix(features, 'theta_B', num_nodes))

	changes = (scene_continuity == 1).astype(np.int64)
	change_prefix = np.concatenate(([0], np.cumsum(changes)))
	continuity_ok = (change_prefix[seq_len:] - change_prefix[1:(num_segments + 1)]) == 0

	visible = np.diagonal(gt_matrices, axis1=1, axis2=2) == 1
	visible_prefix = np.concatenate(
		([np.zeros(num_nodes, dtype=np.int64)], np.cumsum(visible.astype(np.int64), axis=0)),
		axis=0,
	)
	visible_all = (visible_prefix[seq_len:] - visible_prefix[:num_segments]) == seq_len
	valid_mask = continuity_ok[:, None] & visible_all
	valid_pairs = np.argwhere(valid_mask)
	num_valid = valid_pairs.shape[0]

	print('valid samples: ', num_valid)
	if num_valid == 0:
		raise ValueError('No valid Mingling samples after continuity and visibility filtering')

	data = np.full(
		(num_valid, seq_len, feature_size, num_neighbors),
		np.nan,
		dtype=np.float64,
	)
	labels = np.full(
		(num_valid, seq_len, num_neighbors),
		np.nan,
		dtype=np.float64,
	)

	trackers = [
		Tracker(
			int(scene_start),
			int(scene_start) + seq_len,
			int(self_person),
			num_nodes,
			scene_eval_idx=int(source_indices[int(scene_start) + seq_len - 1]),
		)
		for scene_start, self_person in valid_pairs
	]

	time_offsets = np.arange(seq_len)
	chunk_size = 4096

	for self_person in range(num_nodes):
		out_indices = np.flatnonzero(valid_pairs[:, 1] == self_person)
		if len(out_indices) == 0:
			continue

		neighbors = np.array(
			[person for person in range(num_nodes) if person != self_person],
			dtype=np.int64,
		)

		for chunk_start in range(0, len(out_indices), chunk_size):
			chunk_out = out_indices[chunk_start:(chunk_start + chunk_size)]
			starts = valid_pairs[chunk_out, 0]
			scene_idx = starts[:, None] + time_offsets[None, :]

			x_self = x[scene_idx, self_person]
			y_self = y[scene_idx, self_person]
			theta_h_self = theta_h[scene_idx, self_person]
			theta_b_self = theta_b[scene_idx, self_person]

			x_nei = x[scene_idx][:, :, neighbors]
			y_nei = y[scene_idx][:, :, neighbors]
			theta_h_nei = theta_h[scene_idx][:, :, neighbors]
			theta_b_nei = theta_b[scene_idx][:, :, neighbors]

			dx = x_nei - x_self[:, :, None]
			dy = y_nei - y_self[:, :, None]
			theta_d = _wrap_angle_np(np.arctan2(dy, dx))
			dist = np.sqrt((dx ** 2) + (dy ** 2))

			pair_gt = gt_matrices[scene_idx][:, :, self_person, :][:, :, neighbors]
			indicator = ((pair_gt == 0) | (pair_gt == 1)).astype(np.float64)

			chunk_data = np.empty(
				(len(chunk_out), seq_len, feature_size, num_neighbors),
				dtype=np.float64,
			)
			chunk_data[:, :, 0, :] = theta_h_self[:, :, None]
			chunk_data[:, :, 1, :] = -theta_h_nei
			chunk_data[:, :, 2, :] = theta_b_self[:, :, None]
			chunk_data[:, :, 3, :] = -theta_b_nei
			chunk_data[:, :, 4, :] = theta_d
			chunk_data[:, :, 5, :] = dist
			chunk_data[:, :, 6, :] = indicator
			chunk_data[:, :, 0:6, :] = np.where(
				indicator[:, :, None, :] == 1,
				chunk_data[:, :, 0:6, :],
				np.nan,
			)

			data[chunk_out] = chunk_data
			labels[chunk_out] = pair_gt

	reference_b = _nan_mean_angle_np(data[:, :, 2, :], axis=(1, 2))
	for angle_index in [0, 1, 2, 3, 4]:
		data[:, :, angle_index, :] = _wrap_angle_np(
			data[:, :, angle_index, :] - reference_b[:, None, None]
		)

	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		max_distance = np.nanmax(data[:, :, 5, :])
		min_distance = np.nanmin(data[:, :, 5, :])
	data[:, :, 5, :] = (data[:, :, 5, :] - min_distance) / (max_distance - min_distance)

	data[np.isnan(data)] = -999.
	labels[np.abs(labels + 1) < 1e-8] = 0.

	data = torch.from_numpy(data).float()
	labels = torch.from_numpy(labels).float()

	if torch.isnan(data).any():
		raise ValueError('Nan found in data')
	if torch.isnan(labels).any():
		raise ValueError('Nan found in labels')

	print('data shape   : ', list(data.shape))
	print('labels shape : ', list(labels.shape))
	print('trackers len : ', len(trackers), '\n')

	return data, labels, trackers


def get_data(dataset_path, seq_len, feature_size, num_nodes, num_neighbors):
	if not is_mingling_dataset(dataset_path):
		raise ValueError('Only Mingling datasets are supported: ' + dataset_path)

	print("loading data . . .\n")
	dataset_dir = get_dataset_data_dir(dataset_path)
	features = pd.read_csv(os.path.join(dataset_dir, 'features.csv'))
	gt = pd.read_csv(os.path.join(dataset_dir, 'GT.csv'))
	gt_matrices = process_gt_matrices(gt, num_nodes)
	scene_continuity = load_scene_continuity(dataset_dir)

	active_frame_stride = get_dataset_frame_stride(dataset_path)
	features, gt_matrices, scene_continuity, source_indices = downsample_mingling_frames(
		features, gt_matrices, scene_continuity, active_frame_stride
	)
	return get_mingling_data_fast(
		features,
		gt_matrices,
		scene_continuity,
		seq_len,
		feature_size,
		num_nodes,
		num_neighbors,
		source_indices=source_indices,
	)


def split_data(data, labels, trackers, val_scenes, test_scenes):
	print('Splitting train-val-test data . . .\n')

	train_list = []
	val_list = []
	test_list = []

	for idx in range(len(trackers)):
		curr_tracker = trackers[idx]
		val_assigned = False
		test_assigned = False

		for seg in range(0, len(val_scenes)):
			if (curr_tracker.scene_start >= val_scenes[seg][0]) and (curr_tracker.scene_end < val_scenes[seg][1]):
				val_list.append(idx)
				val_assigned = True
				break

		for seg in range(0, len(test_scenes)):
			if (curr_tracker.scene_start >= test_scenes[seg][0]) and (curr_tracker.scene_end < test_scenes[seg][1]):
				test_list.append(idx)
				test_assigned = True
				break

		if val_assigned == False and test_assigned == False:
			train_list.append(idx)

	train_set = DataStruct(data[train_list, :, :, :], labels[train_list, :, :], [trackers[i] for i in train_list])
	val_set = DataStruct(data[val_list, :, :, :], labels[val_list, :, :], [trackers[i] for i in val_list])
	test_set = DataStruct(data[test_list, :, :, :], labels[test_list, :, :], [trackers[i] for i in test_list])

	print('# train samples : ', train_set.size)
	print('# val-- samples : ', val_set.size)
	print('# test- samples : ', test_set.size, '\n')

	return train_list, val_list, test_list, train_set, val_set, test_set


def convert_personwise_to_scene(trackers, infer_list):
	print("grouping person wise to scenewise group predictions")

	scene_group_idx_dict = dict()

	for idx in range(0, len(infer_list)):
		curr_tracker = trackers[infer_list[idx]]
		self_person_id = curr_tracker.self_person
		neighbors = curr_tracker.neighbors
		scene_times = (curr_tracker.scene_start, curr_tracker.scene_end, curr_tracker.scene_eval_idx)

		if scene_times in scene_group_idx_dict.keys():
			scene_group_idx_dict[scene_times].append((infer_list[idx], idx, self_person_id, neighbors))
		else:
			scene_group_idx_dict[scene_times] = [(infer_list[idx], idx, self_person_id, neighbors)]

	return scene_group_idx_dict


def condense_to_group_mat(set_predictions, scene_group_idx_dict, seq_len, num_nodes):
	print("condensing person wise to scenewise group predictions")

	scene_seq_mat_dict = dict()
	for key in scene_group_idx_dict.keys():
		scene_seq_mat = torch.ones(seq_len, num_nodes, num_nodes)

		for info in scene_group_idx_dict[key]:
			idx = info[1]
			self_person_id = info[2]
			neighbors = info[3]

			predictions = set_predictions[idx, :, :]
			for t in range(0, predictions.size()[0]):
				row_wise_data = torch.ones(num_nodes)
				row_wise_data[neighbors] = predictions[t, :]
				scene_seq_mat[t, self_person_id, :] = row_wise_data
		scene_seq_mat_dict[key] = scene_seq_mat
	return scene_seq_mat_dict
