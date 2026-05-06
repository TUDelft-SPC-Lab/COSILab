import torch
import numpy as np
import copy

def train_model(model, train_loader, val_set,
				loss_fn, lr, num_epochs, device=None,
				patience=0, min_delta=0.0,
				epoch_end_callback=None):

	if device is None:
		device = next(model.parameters()).device
	
	optimizer = torch.optim.Adam(model.parameters(),lr = lr)
	
	model.train()
	
	train_loss = []
	val_loss = []
	best_val_loss = float('inf')
	best_epoch = -1
	best_state_dict = copy.deepcopy(model.state_dict())
	epochs_without_improvement = 0
	use_early_stopping = patience > 0
	epoch_metrics = []

	x_val = val_set.data.to(device)
	y_val = val_set.labels.to(device)
	
	for epoch_iter in range(num_epochs):
		
		epoch_loss = 0.0

		for iteration, batch in enumerate(train_loader,0):

			# extract data
			x_train = batch[0].to(device)
			y_train = batch[1].to(device)

			# optimize
			optimizer.zero_grad()

			# predict
			y_train_pred = model(x_train)
			
			# compute loss
			loss = loss_fn(y_train_pred,y_train) 

			# backprop
			loss.backward()

			# optimize
			optimizer.step()

			# update running epoch_loss
			epoch_loss += loss.item()
		
		train_loss.append(epoch_loss/len(train_loader))
		
		with torch.no_grad():
			y_val_pred = model(x_val)
			val_epoch_loss = loss_fn(y_val_pred,y_val)
			val_loss.append(val_epoch_loss.item())

		print("Epoch", str(epoch_iter), " : ", 
			np.around(train_loss[-1],4), " | ", np.around(val_loss[-1],4))

		improved = val_loss[-1] < (best_val_loss - min_delta)
		if improved:
			best_val_loss = val_loss[-1]
			best_epoch = epoch_iter
			best_state_dict = copy.deepcopy(model.state_dict())
			epochs_without_improvement = 0
		else:
			epochs_without_improvement += 1

		if epoch_end_callback is not None:
			callback_metrics = epoch_end_callback(epoch_iter, model)
			if callback_metrics is not None:
				epoch_metrics.append(callback_metrics)

		if use_early_stopping and epochs_without_improvement >= patience:
			print("Early stopping at epoch {}. Best validation loss {:.6f} at epoch {}.".format(
				epoch_iter, best_val_loss, best_epoch))
			break
	
	model.load_state_dict(best_state_dict)
	training_info = {
		'best_val_loss': best_val_loss,
		'best_epoch': best_epoch,
		'epochs_ran': len(train_loss),
		'patience': patience,
		'min_delta': min_delta,
		'epoch_metrics': epoch_metrics,
	}
	print("Restored best validation-loss model from epoch {} with val_loss {:.6f}.".format(
		best_epoch, best_val_loss))
	return model, train_loss, val_loss, training_info
