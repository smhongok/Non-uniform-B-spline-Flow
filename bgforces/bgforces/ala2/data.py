
import os
import torch
import numpy as np

import pytorch_lightning as pl

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from simtk.unit import kilojoules_per_mole, MOLAR_GAS_CONSTANT_R, kelvin

from bgmol.datasets import AImplicitUnconstrained


__all__ = ["read_dataset", "Ala2Data"]


def read_dataset():
    return AImplicitUnconstrained(read=True)

        
class Ala2Data(pl.LightningDataModule):
    
    def __init__(self, batch_size=128, slice=1, val_fraction=0.1, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.slice = slice
        self.val_fraction = val_fraction
     
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Ala2Data")
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--slice", type=int, default=1)       
        parser.add_argument("--val-fraction", type=float, default=0.1)
        return parent_parser
    
    @staticmethod
    def read_dataset():
        return AImplicitUnconstrained(read=True)

    def prepare_data(self):
        Ala2Data.read_dataset()
        
    def setup(self, stage=None):
        dataset = Ala2Data.read_dataset()
        
        # unit conversion
        kBT = dataset.temperature * kelvin * MOLAR_GAS_CONSTANT_R
        forces = dataset.forces * kilojoules_per_mole / kBT
        
        #slicing
        all_xyz = dataset.xyz.reshape(-1, dataset.dim)[::self.slice]
        all_f = forces.reshape(-1, dataset.dim)[::self.slice]

        # split into test and validation set
        training_xyz, val_xyz, training_f, val_f = train_test_split(
            all_xyz, all_f, test_size=self.val_fraction
        )
        self.train_xyz = torch.tensor(training_xyz)
        self.val_xyz = torch.tensor(val_xyz)
        self.train_f = torch.tensor(training_f)
        self.val_f = torch.tensor(val_f)
        self.dataset = dataset

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(self.train_xyz, self.train_f),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()
        )
    
    def val_dataloader(self):
        return DataLoader(
            TensorDataset(self.val_xyz, self.val_f),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count()
        )
