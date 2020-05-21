"""
This file defines the core research contribution   
"""
import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

import pytorch_lightning as pl
import torchdiffeq
from torchdiffeq import odeint

from .blocks import LinearBlock, ConvBlock, ODEBlock


class NeuralODE(pl.LightningModule):

    def __init__(self, hparams):
        super(NeuralODE, self).__init__()
      
        self.hparams = hparams

        self.define_network()

    def define_network(self):

        if self.hparams.linear_block:

            self.odefunc = LinearBlock(hparams=self.hparams)

        elif self.hparams.conv_block:

            self.odefunc = ConvBlock(hparams=self.hparams)

        self.ode_block = ODEBlock(self.hparams, self.odefunc)
        

    def forward(self, x):
        return self.ode_block.forward(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED

        try:
            batch_y0, batch_t, batch_y = batch

            pred_y = odeint(self.odefunc, batch_y0, batch_t)
            loss = torch.mean(torch.abs(pred_y - batch_y))

            tensorboard_logs = {'train_loss': loss}

            return {'loss': loss, 'log': tensorboard_logs}

        except:
            print("#################################")
            print("#################################")
            batch_y0, batch_t, batch_y = batch
            print(batch_y0.shape, batch_t.shape, batch_y.shape)
            pred_y = odeint(self.odefunc, batch_y0, batch_t)
            print(pred_y.shape)
            print("#################################")
            print("#################################")


    """
    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'avg_val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()

        tensorboard_logs = {'test_val_loss': avg_loss}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}
    """

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        # REQUIRED
        #return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)
        pass
    """
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=self.hparams.batch_size)
    """

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser


