import torch
import re
import numpy as np
import pandas as pd
import os
from parameters import *
from dominant_sets import dominant_set_extraction
from data import process_gt_matrices
from dataset_registry import get_dataset_data_dir


def load_gt_groups(dataset_path):
	gt_groups =  np.genfromtxt(os.path.join(get_dataset_data_dir(dataset_path), 'group_names.txt'), dtype = 'str', delimiter = ',')
	return gt_groups
# run this to generate Groups_at_time, Groups is from import_gc_data()
# Groups is of the form: time < ID001 ID002 > < ID003 > etc.
# returns dictionary from time to array of group arrays
# eg. scene number -> [[ID001, ID002], [ID003], ...]
def add_time(Groups):
	Groups_at_time = {}
	for idx in range(0,len(Groups)):
		groups = Groups[idx]
		groups_arr = re.split(" < | > < ", groups)
		Groups_at_time[idx] = []
		last_index = -1

		for group in groups_arr[1:]:
			last_index += 1
			Groups_at_time[idx].append(re.split(" ",group))

		# remove last > character
		if len(groups_arr[1:])==0:
			continue
		Groups_at_time[idx][last_index] = Groups_at_time[idx][last_index][:-1]

	return Groups_at_time

def process_scene_gt(dataset_path):
	gt_groups = load_gt_groups(dataset_path)
	gt_groups_at_time = add_time(gt_groups)
	return gt_groups_at_time

# for a set of vectors of the form [0,1,0,...,1], return a set of vectors of group names
# for more efficiency later, we should represent groups the first way, but for now we do this
def format_person_id(person_id):
	return "ID_" + str(int(person_id)).zfill(3)


def group_names(bool_groups, num_nodes, participant_ids=None):
	if participant_ids is None:
		participant_ids = list(range(1, num_nodes + 1))
	groups = []
	for bool_group in bool_groups:
		group = []
		for i in range(num_nodes):
			if (bool_group[i]):
				group.append(format_person_id(participant_ids[i]))
		groups.append(group)
	return groups


def filter_truth_groups(truth, participant_ids):
	valid_people = set(format_person_id(person_id) for person_id in participant_ids)
	filtered_truth = []
	for group in truth:
		filtered_group = [person for person in group if person in valid_people]
		if len(filtered_group) > 0:
			filtered_truth.append(filtered_group)
	return filtered_truth


## calculates true positives, false negatives, and false positives
## given the guesses, the true groups, and the threshold T
def group_correctness(guesses, truth, T, non_reusable = False):
	TP = 0
	FN = 0
	FP = 0

	n_true_groups = len(truth)
	n_guess_groups = len(guesses)

	for true_group in truth:
		if len(true_group) <= 1:
			n_true_groups -= 1

	for guess in guesses:
		if len(guess) <= 1:
			n_guess_groups -= 1
			continue

	for true_group in truth:
		if len(true_group) <= 1:
			continue

		for guess in guesses:
			if len(guess) <= 1:
				continue

			n_found = 0
			for person in guess:
				if person in true_group:
					n_found += 1

			if float(n_found) / max(len(true_group), len(guess)) >= T:
				if non_reusable == True:
					guesses.remove(guess)
				TP += 1

	if n_true_groups == 0 and n_guess_groups == 0:
		return 0,0,0,1,1

	elif n_true_groups == 0:
		return 0,n_guess_groups,0,0,1

	elif n_guess_groups == 0:
		return 0,0,n_true_groups, 1, 0

	else:
		FP = n_guess_groups - TP
		FN = n_true_groups - TP
		precision = float(TP) / (TP + FP)
		recall = float(TP) / (TP + FN)
		return TP, FN, FP, precision, recall


def get_scenes_correctness(num_nodes,scene_seq_mat_dict, gt_groups_at_time, threshold):
	scenes_f1 = []
	scenes_correctness = []
	scenes_no = []
	guesses = []
	truths = []
	aff_values = []

	gt = pd.read_csv(os.path.join(get_dataset_data_dir(dataset_path), 'GT.csv'))
	gt_matrices = process_gt_matrices(gt, num_nodes)

	for scenes in scene_seq_mat_dict.keys():
		
		if len(scenes) >= 3:
			scene_idx = scenes[2]
		else:
			scene_idx = scenes[1]-1 # last time point, -1 to exclude scene-end (retract one time step to match with the overall time idx from the group_at_time dictionary)
		scenes_no.append(scene_idx)
		# scene-evaluation
		A = scene_seq_mat_dict[scenes][-1].detach().cpu().numpy() # -1: idx correponds to seq_len
		gt_matrix = gt_matrices[scene_idx]
		
		valid_idx = np.where(np.diag(gt_matrix) == 1)[0]
		participant_ids = [idx + 1 for idx in valid_idx]
		A = A[np.ix_(valid_idx, valid_idx)]


		aff_values.append(A)
		# threshold affinity values
		# idx = (A<0.5)
		# A[idx] = 0.
		# A[~idx] = 1.

		# print("scenes, aff:", (scenes,A))
		# set diagnoal to zero - avoid self loop
		np.fill_diagonal(A, 0)
		
		if A.shape[0] == 0:
			predicted_groups = []
		else:
			predicted_groups = dominant_set_extraction(A,A.shape[0],op='avg')
		
		guess = group_names(predicted_groups, A.shape[0], participant_ids)
		truth = filter_truth_groups(gt_groups_at_time[scene_idx], participant_ids)

		guesses.append(guess)
		truths.append(truth)

		correctness = group_correctness(guess,truth, threshold, False)
		scenes_correctness.append(correctness) # correctness follows: TP, FN, FP, precision, recall
		
		precision = correctness[-2]
		recall = correctness[-1]

		if precision*recall == 0:
			f1= 0
		else:
			f1 = 2*(precision*recall)/(precision+recall)
		scenes_f1.append(f1)

	return scenes_no, scenes_correctness, scenes_f1, guesses, truths, aff_values
