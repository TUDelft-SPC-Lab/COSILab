import torch

class Skynet(torch.nn.Module):

	def __init__(self, seq_len, feature_size, num_nodes, hidden_dim):
		super(Skynet, self).__init__()
		
		# args
		self.seq_len = seq_len
		self.feature_size = feature_size
		self.num_nodes = num_nodes
		self.num_neighbors = num_nodes - 1

		self.hidden_dim = hidden_dim
		self.output_dim = 1

		# trainable models
		self.lstm_cell = torch.nn.LSTMCell(self.feature_size-1, self.hidden_dim)
		self.context_lambda = torch.nn.Parameter(torch.tensor([0.])) #will go into a sigmoid later
		self.affinity_predictor = torch.nn.Linear(self.hidden_dim, self.output_dim)


	def forward(self, x):
		
		# get batch_size
		batch_size = x.shape[0]

		# asserts
		assert self.seq_len==x.shape[1], "seq_len doesn't match"
		assert self.feature_size==x.shape[2], "feature_size doesn't match"
		assert self.num_neighbors==x.shape[3], "num_nodes doesn't match"

		# separate features and indicator variables
		z = x[:,:,0:(self.feature_size-1),:]
		# indicator variables
		idx = x[:,:,(self.feature_size-1):self.feature_size,:]

		# init hidden and cell states
		hidden_states = torch.zeros(
			batch_size, self.hidden_dim, self.num_neighbors,
			device=x.device, dtype=x.dtype
		)
		cell_states = torch.zeros(
			batch_size, self.hidden_dim, self.num_neighbors,
			device=x.device, dtype=x.dtype
		)

		# init output data structure
		A = torch.zeros(
			(batch_size, self.seq_len, self.num_neighbors),
			device=x.device, dtype=x.dtype
		)

		# pre-compute context_weight (using sigmoid to ensure weight in range [0,1])
		context_weight = torch.sigmoid(self.context_lambda)

		# begin time loop
		for t in range(self.seq_len):
			
			# begin loop over neighbors
			for n in range(self.num_neighbors):

				# lstm_cell prediction
				hidden_states[:,:,n], cell_states[:,:,n] = self.lstm_cell(
																z[:,t,:,n],
																(hidden_states[:,:,n].clone(),
																cell_states[:,:,n].clone()))

				# masking
				hidden_states[:,:,n] = hidden_states[:,:,n].clone() * idx[:,t,:,n]
				pass

			# pooling
			# Note: idx is summed with dim=2 coz the time dimensions vanishes after [:,t,:,:]
			hidden_pool = torch.sum(hidden_states, dim=2) / torch.sum(idx[:,t,:,:], dim=2)

			# recombining pooled context
			for n in range(self.num_neighbors):
				hidden_states[:,:,n] = (context_weight*hidden_states[:,:,n].clone()
										+ (1.-context_weight)*hidden_pool)


			# predict affinity values
			for n in range(self.num_neighbors):
				embedding = hidden_states[:,:,n].clone()
				A[:,t,n:(n+1)] = torch.sigmoid(self.affinity_predictor(embedding))*idx[:,t,:,n]

		return A
