import torch
import numpy as np
import pandas as pd
import os
import sys
import pickle

from parameters import *
from data import *
from model import *
from train import *
from analysis import *
from evaluate_scene import *

# reproducibility
torch.manual_seed(0)
if detect_anomaly:
	torch.autograd.set_detect_anomaly(True)
	print('torch anomaly detection: enabled')
else:
	print('torch anomaly detection: disabled')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


print("\n----------- PARAMETER OVERRIDES -----------")
print("*    (don't forget to remove it later)    *\n")

fold_index = int(sys.argv[1])

fold = fold_index
print('fold:', fold)
print('frame_stride:', frame_stride)
print('early_stop_patience:', early_stop_patience)
print('early_stop_min_delta:', early_stop_min_delta)
print('f1_eval_every:', f1_eval_every)

scenes_per_fold, num_folds= get_train_val_test_scenes(dataset_path)


# override path suffix for output file names
path_suffix += '_seq='+str(seq_len) + '_hidden='+str(hidden_dim)+'_threshold='+str(threshold)
if frame_stride != 1:
	path_suffix += '_stride='+str(frame_stride)
print('path_suffix :', path_suffix, '\n')
dataset_artifact_prefix = 'dataset=' + dataset_label
if frame_stride != 1:
	dataset_artifact_prefix += '_stride=' + str(frame_stride)


print("\n----------- DATA PREPROCESS -----------\n")

model_dir = os.path.join('models', dataset_path)
os.makedirs(model_dir, exist_ok=True)

if (dataset_make_flag == True):
	# load data
	print("DATASET",dataset_path)
	data, labels, trackers = get_data(dataset_path, 
									seq_len, feature_size, 
									num_nodes, num_neighbors)

	scenes_per_fold, num_folds = get_train_val_test_scenes(dataset_path)

	folds = list(range(0,num_folds)) # cross-validation

	fold = folds[fold_index]

	# split data
	train_list,val_list,test_list,train_set, val_set, test_set= split_data(data, labels, trackers,
											scenes_per_fold[fold]['val_scenes'], scenes_per_fold[fold]['test_scenes'])

	# batch-based loader for training
	train_tensor_dataset = TensorDataset(train_set.data, train_set.labels)
	train_loader = DataLoader(dataset=train_tensor_dataset, batch_size=batch_size, shuffle=True)

	# save data
	torch.save(train_list, model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_train_list'+'.pt')
	torch.save(val_list, model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_val_list'+'.pt')
	torch.save(test_list, model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_test_list'+'.pt')
	torch.save(train_set, model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_train_set'+'.pt')
	torch.save(val_set, model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_val_set'+'.pt')
	torch.save(test_set, model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_test_set'+'.pt')

	torch.save(trackers, model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_trackers'+'.pt')

elif (dataset_make_flag == False):
	scenes_per_fold, num_folds = get_train_val_test_scenes(dataset_path)

	folds = list(range(0,num_folds)) # cross-validation

	fold = folds[fold_index]

	train_list = torch.load(model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_train_list'+'.pt')
	val_list = torch.load(model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_val_list'+'.pt')
	test_list = torch.load(model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_test_list'+'.pt')
	train_set = torch.load(model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_train_set'+'.pt')
	val_set = torch.load(model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_val_set'+'.pt')
	test_set = torch.load(model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_test_set'+'.pt')

	trackers = torch.load(model_dir+'/'+dataset_artifact_prefix+'_fold'+str(fold)+'_trackers'+'.pt')

	# batch-based loader for training
	train_tensor_dataset = TensorDataset(train_set.data, train_set.labels)
	train_loader = DataLoader(dataset=train_tensor_dataset, batch_size=batch_size, shuffle=True)


print("\n----------- TRAINING -----------\n")

# create model
skynet = Skynet(seq_len, feature_size, num_nodes, hidden_dim)
skynet = skynet.to(device)

def summarize_group_correctness(scenes_correctness):
	if len(scenes_correctness) == 0:
		return float('nan'), float('nan'), float('nan')
	precision = float(np.mean([item[-2] for item in scenes_correctness]))
	recall = float(np.mean([item[-1] for item in scenes_correctness]))
	if precision * recall == 0:
		f1 = 0.0
	else:
		f1 = 2 * (precision * recall) / (precision + recall)
	return f1, precision, recall

def evaluate_group_metrics(scene_seq_mat_dict, gt_groups_at_time, auc_value):
	metrics = {'auc': auc_value}
	details = {}
	for label, threshold_value in [('1', 1.0), ('2_3', 2.0 / 3.0)]:
		scenes_no, scenes_correctness, scenes_f1, guesses, truths, aff_values = get_scenes_correctness(
			num_nodes, scene_seq_mat_dict, gt_groups_at_time, threshold_value)
		f1, precision, recall = summarize_group_correctness(scenes_correctness)
		metrics['f1_' + label] = f1
		metrics['precision_' + label] = precision
		metrics['recall_' + label] = recall
		metrics['mean_scene_f1_' + label] = (
			float(np.mean(scenes_f1)) if len(scenes_f1) > 0 else float('nan')
		)
		metrics['n_scenes_' + label] = len(scenes_no)
		details[label] = {
			'scenes_no': scenes_no,
			'scenes_correctness': scenes_correctness,
			'scenes_f1': scenes_f1,
			'guesses': guesses,
			'truths': truths,
			'aff_values': aff_values,
		}
	return metrics, details

def build_validation_f1_callback():
	if f1_eval_every <= 0:
		return None

	gt_groups_for_f1 = process_scene_gt(dataset_path)
	val_scene_group_idx_dict_for_f1 = convert_personwise_to_scene(trackers, val_list)

	def callback(epoch_iter, model):
		if (epoch_iter + 1) % f1_eval_every != 0:
			return None

		print("Running validation F1 evaluation at epoch {}".format(epoch_iter + 1))
		was_training = model.training
		model.eval()
		val_predictions_for_f1 = get_model_predictions(model, val_set)
		val_scene_seq_mat_dict_for_f1 = condense_to_group_mat(
			val_predictions_for_f1,
			val_scene_group_idx_dict_for_f1,
			seq_len,
			num_nodes,
		)
		_, correctness_one, scene_f1_one, _, _, _ = get_scenes_correctness(
			num_nodes, val_scene_seq_mat_dict_for_f1, gt_groups_for_f1, 1.0)
		_, correctness_two_thirds, scene_f1_two_thirds, _, _, _ = get_scenes_correctness(
			num_nodes, val_scene_seq_mat_dict_for_f1, gt_groups_for_f1, 2.0 / 3.0)
		if was_training:
			model.train()

		f1_one, precision_one, recall_one = summarize_group_correctness(correctness_one)
		f1_two_thirds, precision_two_thirds, recall_two_thirds = summarize_group_correctness(
			correctness_two_thirds)
		mean_scene_f1_one = float(np.mean(scene_f1_one)) if len(scene_f1_one) > 0 else float('nan')
		mean_scene_f1_two_thirds = (
			float(np.mean(scene_f1_two_thirds)) if len(scene_f1_two_thirds) > 0 else float('nan')
		)

		print("val f1 @1: {:.6f}, val f1 @2/3: {:.6f}".format(
			f1_one, f1_two_thirds))
		return {
			'epoch': epoch_iter,
			'f1_1': f1_one,
			'precision_1': precision_one,
			'recall_1': recall_one,
			'mean_scene_f1_1': mean_scene_f1_one,
			'f1_2_3': f1_two_thirds,
			'precision_2_3': precision_two_thirds,
			'recall_2_3': recall_two_thirds,
			'mean_scene_f1_2_3': mean_scene_f1_two_thirds,
		}

	return callback

def add_best_validation_f1(training_info):
	history = training_info.get('epoch_metrics', [])
	training_info['f1_eval_every'] = f1_eval_every
	if len(history) == 0:
		training_info['best_val_f1_1'] = float('nan')
		training_info['best_val_f1_1_epoch'] = -1
		training_info['best_val_f1_2_3'] = float('nan')
		training_info['best_val_f1_2_3_epoch'] = -1
		return
	best_one = max(history, key=lambda item: item['f1_1'])
	best_two_thirds = max(history, key=lambda item: item['f1_2_3'])
	training_info['best_val_f1_1'] = best_one['f1_1']
	training_info['best_val_f1_1_epoch'] = best_one['epoch']
	training_info['best_val_f1_2_3'] = best_two_thirds['f1_2_3']
	training_info['best_val_f1_2_3_epoch'] = best_two_thirds['epoch']

# train model
if(train_flag == True):
	validation_f1_callback = build_validation_f1_callback()
	skynet, train_loss, val_loss, training_info = train_model(skynet, train_loader, val_set,
												loss_fn, lr, num_epochs, device=device,
												patience=early_stop_patience,
												min_delta=early_stop_min_delta,
												epoch_end_callback=validation_f1_callback)
	add_best_validation_f1(training_info)
	# save best validation-loss model
	torch.save(skynet.cpu(), model_dir+'/'+dataset_artifact_prefix+'_model_fold'+str(fold)+'.pt')
	torch.save(training_info, model_dir+'/'+dataset_artifact_prefix+'_training_info_fold'+str(fold)+'.pt')
	skynet = skynet.to(device)
else:
	skynet = torch.load(model_dir+'/'+dataset_artifact_prefix+'_model_fold'+str(fold)+'.pt', map_location=device)
	skynet = skynet.to(device)
	skynet.eval()
	train_loss = None
	val_loss = None
	training_info_path = model_dir+'/'+dataset_artifact_prefix+'_training_info_fold'+str(fold)+'.pt'
	training_info = torch.load(training_info_path) if os.path.exists(training_info_path) else None
	if training_info is not None and 'f1_eval_every' not in training_info:
		add_best_validation_f1(training_info)

print("\n----------- EVALUATION -----------\n")

# build time and group names GT map
gt_groups_at_time = process_scene_gt(dataset_path)

# create output dir
os.makedirs(output_dir,exist_ok=True)

# compute AUC score
train_AUC = evaluate_AUC_score(skynet, train_set, 'train')
val_AUC   = evaluate_AUC_score(skynet, val_set,   'val--')
test_AUC  = evaluate_AUC_score(skynet, test_set,  'test-')

# condense predictions to scenes
train_predictions = get_model_predictions(skynet,train_set)
# val
val_predictions = get_model_predictions(skynet, val_set)
# # test
test_predictions = get_model_predictions(skynet, test_set)

print("condense to group")
# train_scene_group_idx_map: dictionary keyed by (scene_start, scene_end), values : (train_list[idx],idx,self_person_id,neighbors), first idx is global based on data, second idx for the train/val/test list (length) only, 
train_scene_group_idx_dict= convert_personwise_to_scene(trackers, train_list)
train_scene_seq_mat_dict = condense_to_group_mat(train_predictions, train_scene_group_idx_dict,seq_len,num_nodes)

# val_scene_group_idx_map: dictionary keyed by (scene_start, scene_end), values : (val_list[idx],idx,self_person_id,neighbors), first idx is global based on data, second idx for the val/test list (length) only, 
val_scene_group_idx_dict= convert_personwise_to_scene(trackers, val_list)
val_scene_seq_mat_dict = condense_to_group_mat(val_predictions, val_scene_group_idx_dict,seq_len,num_nodes) # val prections

# test_scene_group_idx_map
test_scene_group_idx_dict= convert_personwise_to_scene(trackers, test_list)
test_scene_seq_mat_dict = condense_to_group_mat(test_predictions, test_scene_group_idx_dict,seq_len,num_nodes) # test predictions

# save intermediate scene_seq_mat_dict
dict_path = os.path.join(output_dir, dataset_label+"_test_scene_seq_mat_dict_fold_"+str(fold)+".pk")
filehandler = open(dict_path,"wb")
pickle.dump(test_scene_seq_mat_dict,filehandler)
filehandler.close()

# get scenes_correctness
print("get scenes correctness")
print("val split")
val_metrics, val_details = evaluate_group_metrics(
	val_scene_seq_mat_dict, gt_groups_at_time, val_AUC)
print("test split")
test_metrics, test_details = evaluate_group_metrics(
	test_scene_seq_mat_dict, gt_groups_at_time, test_AUC)

# export loss curves
print("exporting results")
if train_loss is not None:
	export_loss_curves(train_loss, val_loss, output_dir + '/' +'dataset_'+dataset_label+"_"+ 'fold'+str(fold)+"_"+loss_path + path_suffix)
if training_info is not None:
	export_table(
		[[training_info['best_epoch'], training_info['best_val_loss'],
		  training_info['epochs_ran'], training_info['patience'],
		  training_info['min_delta'], training_info.get('f1_eval_every', f1_eval_every),
		  training_info.get('best_val_f1_1', float('nan')),
		  training_info.get('best_val_f1_1_epoch', -1),
		  training_info.get('best_val_f1_2_3', float('nan')),
		  training_info.get('best_val_f1_2_3_epoch', -1)]],
		output_dir + '/' +'dataset_'+dataset_label+"_"+ 'fold='+str(fold)+"_"+'training_info' + path_suffix,
		['best_epoch','best_val_loss','epochs_ran','patience','min_delta',
		 'f1_eval_every','best_val_f1_1','best_val_f1_1_epoch',
		 'best_val_f1_2_3','best_val_f1_2_3_epoch'])
	if len(training_info.get('epoch_metrics', [])) > 0:
		f1_history_rows = [
			[
				item['epoch'], item['f1_1'], item['precision_1'], item['recall_1'],
				item['mean_scene_f1_1'], item['f1_2_3'], item['precision_2_3'],
				item['recall_2_3'], item['mean_scene_f1_2_3'],
			]
			for item in training_info['epoch_metrics']
		]
		export_table(
			f1_history_rows,
			output_dir + '/' +'dataset_'+dataset_label+"_"+ 'fold='+str(fold)+"_"+'val_f1_history' + path_suffix,
			['epoch','f1_1','precision_1','recall_1','mean_scene_f1_1',
			 'f1_2_3','precision_2_3','recall_2_3','mean_scene_f1_2_3'])

# export results
export_list([[seq_len,hidden_dim,train_AUC, val_AUC, test_AUC]], 
			output_dir + '/' +'dataset_'+dataset_label+"_"+ 'fold='+str(fold)+"_"+AUC_path + path_suffix)

# export_table([train_scenes_no,train_scenes_f1,train_guesses,train_truths,train_scenes_correctness, train_aff_values],
# 			 output_dir + '/' +'dataset_'+dataset_label+"_" + 'fold='+str(fold)+"_"+f1_path+'_train' + path_suffix,
# 			 ['scenes_no','f1','guesses','truths','correctness','aff'])
# export_table([val_scenes_no,val_scenes_f1,val_guesses,val_truths,val_scenes_correctness, val_aff_values],
# 			 output_dir + '/' +'dataset_'+dataset_label+"_" + 'fold='+str(fold)+"_"+f1_path+'_val' + path_suffix,
# 			 ['scenes_no','f1','guesses','truths','correctness','aff'])
metrics_rows = [
	[
		'val', val_metrics['auc'],
		val_metrics['f1_1'], val_metrics['precision_1'], val_metrics['recall_1'],
		val_metrics['f1_2_3'], val_metrics['precision_2_3'], val_metrics['recall_2_3'],
	],
	[
		'test', test_metrics['auc'],
		test_metrics['f1_1'], test_metrics['precision_1'], test_metrics['recall_1'],
		test_metrics['f1_2_3'], test_metrics['precision_2_3'], test_metrics['recall_2_3'],
	],
]
export_table(
	metrics_rows,
	output_dir + '/' +'dataset_'+dataset_label+"_"+ 'fold='+str(fold)+"_"+'metrics_summary' + path_suffix,
	['split','auc','f1_1','precision_1','recall_1',
	 'f1_2_3','precision_2_3','recall_2_3'])
for split_name, details_by_threshold in [('val', val_details), ('test', test_details)]:
	for threshold_label, details in details_by_threshold.items():
		export_table(
			[
				details['scenes_no'],
				details['scenes_f1'],
				details['guesses'],
				details['truths'],
				details['scenes_correctness'],
				details['aff_values'],
			],
			output_dir + '/' +'dataset_'+dataset_label+"_" + 'fold='+str(fold)+"_"+f1_path+'_'+split_name+'_T'+threshold_label + path_suffix,
			['scenes_no','f1','guesses','truths','correctness','aff'])


print("\n---------- ENDING FOLD",str(fold),"----------\n")





