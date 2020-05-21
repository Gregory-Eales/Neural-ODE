"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import os
from pytorch_lightning import Trainer
from argparse import ArgumentParser
import gym
import torch
import numpy as np

from neural_ode.neural_ode import NeuralODE

def main(args):
	neural_ode = NeuralODE(hparams=args)

if __name__ == '__main__':


	torch.manual_seed(0)
	np.random.seed(0)

	parser = ArgumentParser()

	# general params
	parser.add_argument("--gpu", type=int, default=0, help="number of gpus")
	parser.add_argument("--num_epochs", type=int, default=200, help="number of gpus")
	parser.add_argument("--batch_size", type=int, default=64, help="size of training batch")
	parser.add_argument("--lr", type=int, default=1e-2, help="learning rate")
	parser.add_argument("--accumulate_grad_batches", type=int, default=64, help="grad batches")
	
	# linear block params
	parser.add_argument("--linear_block", type=bool, default=True, help="bool that initiates linear block")
	parser.add_argument("--linear_in_features", type=int, default=3, help="in features for linear block")
	parser.add_argument("--linear_hidden_features", type=int, default=3, help="hidden dim of linear block")
	parser.add_argument("--linear_linear_units", type=int, default=3, help="hidden dim of linear block")
	parser.add_argument("--linear_out_features", type=int, default=3, help="out features for linear block")

	# conv block params
	parser.add_argument("--conv_block", type=bool, default=False, help="bool that initiates conv block")
	parser.add_argument("--conv_in_features", type=int, default=3, help="in features for conv block")
	parser.add_argument("--conv_hidden_features", type=int, default=3, help="hidden dim of conv block")
	parser.add_argument("--conv_linear_units", type=int, default=3, help="hidden dim of conv block")
	parser.add_argument("--conv_out_features", type=int, default=3, help="out features for conv block")

	# run
	args = parser.parse_args()
	main(args)