import torch
from torchdiffeq import odeint


class LinearBlock(torch.nn.Module):

	def __init__(self, hparams):

		super(LinearBlock, self).__init__()

		self.hparams = hparams

		self.linear = torch.nn.ModuleDict()

		self.define_network()

		self.n = 0

	def define_network(self):

		self.relu = torch.nn.LeakyReLU()
		self.tanh = torch.nn.Tanh()

		self.linear["l{}".format(0)] = torch.nn.Linear(self.hparams.linear_in_features,
			 self.hparams.linear_hidden_features)
		
		for i in range(1, self.hparams.num_linear_layers):
			self.linear["l{}".format(i)] = torch.nn.Linear(self.hparams.linear_hidden_features,
			 self.hparams.linear_hidden_features)

		self.linear["l{}".format(self.hparams.num_linear_layers)] = torch.nn.Linear(self.hparams.linear_hidden_features,
			 self.hparams.linear_out_features)
			
	
	def forward(self, t, x):

		#out = torch.Tensor(x)
		out = x
		
		for i in range(self.hparams.num_linear_layers+1):
			out = self.linear["l{}".format(i)](out)
			out = self.relu(out)

		self.n+=1

		if self.n%10 == 0:
			print(self.n)

		return out

class ConvBlock(torch.nn.Module):

	def __init__(self, hparams):

		super(ConvBlock, self).__init__()

		self.hparams = hparams

	def define_network(self):
		pass

	def forward(self, t, x):
		self.nfe += 1
		out = self.norm1(x)
		out = self.relu(out)
		out = self.conv1(t, out)
		out = self.norm2(out)
		out = self.relu(out)
		out = self.conv2(t, out)
		out = self.norm3(out)
		return out


class ODEBlock(torch.nn.Module):

	def __init__(self, hparams, odefunc):

		super(ODEBlock, self).__init__()

		self.hparams = hparams
		self.odefunc = odefunc
		self.integration_time = torch.tensor([0, 1]).float()

	
	def forward(self, x):
		self.hparams.int_time = self.integration_time.type_as(x)
		out = odeint(self.odefunc, x, self.integration_time, rtol=self.hparams.tol, atol=self.hparams.tol)
		return out[1]

