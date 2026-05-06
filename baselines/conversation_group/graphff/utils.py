import numpy as np
import torch
import pandas as pd
from dataset_registry import get_dataset_data_dir, get_dataset_frame_stride, is_mingling_dataset

###############################################################
# Helper functions for entire code:

# compute angle from x and y coordinates
def compute_orientation(x,y):
	return torch.atan2(y,x)

# wrap angle
def wrap_angle(alpha):
	return compute_orientation(torch.cos(alpha),torch.sin(alpha))

# circular mean
def mean_angle(alpha):
	c = torch.mean(torch.cos(alpha))
	s = torch.mean(torch.sin(alpha))
	angle = torch.atan2(s,c)
	return wrap_angle(angle)

# circular mean while ignoring nan
def nan_mean_angle(alpha):
	c = torch.nanmean(torch.cos(alpha))
	s = torch.nanmean(torch.sin(alpha))
	angle = torch.atan2(s,c)
	return wrap_angle(angle)

# max while ignoring nan
def nan_numpy_max(torch_mat):
	val = np.nanmax(torch_mat.detach().numpy())
	return val

# min while ignoring nan
def nan_numpy_min(torch_mat):
	val = np.nanmin(torch_mat.detach().numpy())
	return val

def ranges(nums):
	nums = sorted(set(nums))
	gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
	edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
	return np.array([(s, e+1) for s, e in zip(edges, edges)])

def generate_index_from_length(l, fold):
	fold_size = l/5
	indices_list = []
	if fold==0:
		indices_list.append([[int(2*fold_size),int(5*fold_size)]])
		indices_list.append([[int(1*fold_size),int(2*fold_size)]])
		indices_list.append([[int(0*fold_size),int(1*fold_size)]])
	elif fold==1:
		indices_list.append([[int(0*fold_size),int(0.5*fold_size)],[int(2.5*fold_size),int(5*fold_size)]])
		indices_list.append([[int(0.5*fold_size),int(1*fold_size)],[int(2*fold_size),int(2.5*fold_size)]])
		indices_list.append([[int(1*fold_size),int(2*fold_size)]])
	elif fold==2:
		indices_list.append([[int(0*fold_size),int(1.5*fold_size)],[int(3.5*fold_size),int(5*fold_size)]])
		indices_list.append([[int(1.5*fold_size),int(2*fold_size)],[int(3*fold_size),int(3.5*fold_size)]])
		indices_list.append([[int(2*fold_size),int(3*fold_size)]])
	elif fold==3:
		indices_list.append([[int(0*fold_size),int(2.5*fold_size)],[int(4.5*fold_size),int(5*fold_size)]])
		indices_list.append([[int(2.5*fold_size),int(3*fold_size)],[int(4*fold_size),int(4.5*fold_size)]])
		indices_list.append([[int(3*fold_size),int(4*fold_size)]])
	elif fold==4:
		indices_list.append([[int(0*fold_size),int(3*fold_size)]])
		indices_list.append([[int(3*fold_size),int(4*fold_size)]])
		indices_list.append([[int(4*fold_size),int(5*fold_size)]])
	return indices_list

def generate_index(dataset,fold):
	l = len(pd.read_csv(get_dataset_data_dir(dataset) + '/features.csv'))
	if is_mingling_dataset(dataset):
		stride = get_dataset_frame_stride(dataset)
		l = len(range(0, l, stride))
	return generate_index_from_length(l, fold)

def get_train_val_test_scenes(dataset):
	if is_mingling_dataset(dataset):

		fold_no = 5
		folds = list(range(fold_no))

		splits = ['train', 'val', 'test']
		num_scenes = len(pd.read_csv(get_dataset_data_dir(dataset) + '/features.csv'))
		stride = get_dataset_frame_stride(dataset)
		num_scenes = len(range(0, num_scenes, stride))

		scenes_per_fold = dict()

		for fold in range(0,len(folds)):
			indices_list = generate_index_from_length(num_scenes, fold)
			scenes_per_fold[fold] = dict()

			for split in range(0,len(splits)):
				if splits[split]=='train':
					scenes_per_fold[fold][splits[split]+"_scenes"] = np.array(indices_list[0])
				elif splits[split]=='val':
					scenes_per_fold[fold][splits[split]+"_scenes"] = np.array(indices_list[1])
				elif splits[split]=='test':
					scenes_per_fold[fold][splits[split]+"_scenes"] = np.array(indices_list[2])
					
				
	else:
		raise ValueError('get_train_val_test_scenes not implemented for', dataset)
	
	return scenes_per_fold, fold_no














































