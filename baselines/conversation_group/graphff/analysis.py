import torch
import sklearn.metrics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _get_model_device(model):
	return next(model.parameters()).device

def export_loss_curves(train_loss, val_loss, path):
	if (train_loss is not None) and (val_loss is not None):
		# plot loss curves
		matplotlib.use('Agg')
		plt.plot(train_loss, label='train')
		plt.plot(val_loss, label='val')
		plt.legend(loc = 'upper right')
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.savefig(path+'.png')
		# plt.close()

def get_model_predictions(model,test_set):
	device = _get_model_device(model)
	x = test_set.data.to(device)

	with torch.no_grad():
		y_pred = model(x)
	return y_pred.detach().cpu()


def evaluate_AUC_score(model, test_set, prefix=''):

	y = test_set.labels
	y_pred = get_model_predictions(model,test_set)

	# get prediction for the last time step
	last = torch.flatten(y[:,-1,:]).detach().cpu().numpy()
	last_pred = torch.flatten(y_pred[:,-1,:]).detach().cpu().numpy()

	auc_test = sklearn.metrics.roc_auc_score(last,last_pred)

	print(prefix,'AUC :',auc_test)

	return auc_test

def export_list(data, path):
	np.savetxt(path+'.csv', np.array(data), delimiter=",", fmt='%1.3f')

def export_table(data, path, headers=None):

	if type(data)==type([]):
		try:
			data = np.array(data)
		except ValueError:
			if headers is not None and len(data) == len(headers):
				df = pd.DataFrame({headers[idx]: data[idx] for idx in range(len(headers))})
				print(path)
				df.to_csv(path+".csv",header=headers)
				return
			data = np.array(data, dtype=object)
	if type(data)==type(torch.tensor([])):
		data = data.detach().cpu().numpy()
	if headers is not None:
		if len(data.shape) > 1 and data.shape[1] == len(headers):
			pass
		elif len(data.shape) > 0 and data.shape[0] == len(headers):
			data = data.transpose()
		else:
			raise ValueError('Wrong headers list provided in export_table()')

	df=pd.DataFrame.from_records(data)
	if(headers):
		df.columns = headers
	print(path)
	df.to_csv(path+".csv",header=headers)
