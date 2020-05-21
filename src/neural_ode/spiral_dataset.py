import torch
from torch.utils.data import Dataset
import numpy as np



class SpiralDataset(Dataset):

	def __init__(self, hparams):

		super(SpiralDataset, self).__init__()

		self.hparams = hparams

		self.generate_data()
		



	def generate_data(self):

		n = self.hparams.num_data
		start = 0
		stop = 10*np.pi

		a = torch.linspace(start, stop, steps=n).reshape(-1, 1)
		b = torch.linspace(start, stop, steps=n).reshape(-1, 1)
		c = torch.linspace(start, stop, steps=n).reshape(-1, 1)
		self.y = torch.cat([a, b, c], dim=1)

		self.t = torch.linspace(start, stop, steps=n).reshape(-1, 1)
		
		self.y_0 = torch.cat([torch.randn(n, 1), torch.randn(n, 2)], dim=1)

		a_0 = torch.linspace(start-1, stop-1, steps=n).reshape(-1, 1)
		b_0 = torch.linspace(start-1, stop-1, steps=n).reshape(-1, 1)
		c_0 = torch.linspace(start, stop, steps=n).reshape(-1, 1)
		self.y_0 = torch.cat([a_0, b_0, c_0], dim=1)


	def __len__(self):
		return self.hparams.num_data

	def __getitem__(self, idx):

		y_0 = self.y_0[idx]
		y = self.y[idx]
		t = self.t[idx]

		return y[0], t, y

	def __iter__(self):
		
		bs = self.hparams.batch_size
		num_batch = self.y.shape[0]//self.hparams.batch_size
		rem_batch = self.y.shape[0]%self.hparams.batch_size
		
		while True:
			for i in range(num_batch):
				i1, i2 = i*bs, (i+1)*self.hparams.batch_size
			
				yield self.y_0[i1:i2], self.t[i1:i2], self.y[i1:i2]
			
			"""
			i1 = -rem_batch
			i2 = 0
			yield self.y_0[i1:i2], self.t[i1:i2], self.y[i1:i2]
			"""

	