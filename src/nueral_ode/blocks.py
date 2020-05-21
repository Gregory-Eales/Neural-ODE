import torch
from torchdiffeq import odeint


class LinearBlock(torch.nn.Module):

	def __init__(self, hparams):

		super(LinearBlock, self).__init__()

		self.hparams = hparams

		self.linear = torch.nn.ModuleDict()

	def define_network(self):

		self.relu = torch.nn.LeakyReLU()

		self.linear["l{}".format(0)] = torch.nn.Linear(self.hparam.linear_in_features,
			 self.hparam.linear_hidden_features)
		
		for i in range(1, self.hparam.num_linear_units):
			self.linear["l{}".format(i)] = torch.nn.Linear(self.hparam.linear_in_features,
			 self.hparam.linear_hidden_features)

		self.linear["l{}".format(self.hparam.num_linear_block)] = torch.nn.Linear(self.hparam.linear_in_features,
			 self.hparam.linear_hidden_features)
			
	
	def forward(self, x):

		out = torch.Tensor(x)
		
		for i in range(0, self.hparam.num_linear_units+1):
			out = self.linear["l{}".format(0)](out)
			out = self.relu(out)

		return out

class ConvBlock(torch.nn.Module):

	def __init__(self, hparams):

		super(ConvBlock, self).__init__()

		self.hparams = hparams

	def define_network(self):
		self.norm1 = torch.nn.GroupNorm(min(32, dim), dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = torch.nn.GroupNorm(min(32, dim), dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = torch.nn.GroupNorm(min(32, dim), dim)
        self.nfe = 0

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


class ODEBlock(torch.nn.module):

	def __init__(self, hparams, odefunc):

		super(ODEBlock, self).__init__()

		self.hparams = hparams
		self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

	def define_network(self):
		pass
	
	def forward(self, x):
		self.hparams.int_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=args.tol, atol=args.tol)
        return out[1]

