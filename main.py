# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:30:45 2022

@author: Yuanhang Zhang
"""

from model import TransformerModel
from Hamiltonian import Ising, IsingY, XYZ
from optimizer import Optimizer

import os
import numpy as np
import torch
import argparse
import neptune
import os
from dotenv import load_dotenv

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["Ising", "IsingY"], default="Ising")
args = parser.parse_args()

# Load environment variables from .env file
load_dotenv()

# Initialize Neptune logger
logger = None
if 'NEPTUNE_API_TOKEN' in os.environ and 'NEPTUNE_PROJECT' in os.environ:
    logger = neptune.init_run(
        api_token=os.environ['NEPTUNE_API_TOKEN'],
        project=os.environ['NEPTUNE_PROJECT']
    )
else:
    print("Warning: Neptune logging disabled. Set NEPTUNE_API_TOKEN and NEPTUNE_PROJECT in .env file to enable logging.")

torch.set_default_tensor_type(
    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
)
# torch.set_default_tensor_type(torch.FloatTensor)
try:
    os.mkdir("results/")
except FileExistsError:
    pass

system_sizes = np.arange(10, 41, 2).reshape(-1, 1)

hamiltonian_class = IsingY if args.model == 'IsingY' else Ising
Hamiltonians = [hamiltonian_class(system_size_i, periodic=False) for system_size_i in system_sizes]

param_dim = Hamiltonians[0].param_dim
embedding_size = 32
n_head = 8
n_hid = embedding_size
n_layers = 8
dropout = 0
minibatch = 10000

model = TransformerModel(
    system_sizes,
    param_dim,
    embedding_size,
    n_head,
    n_hid,
    n_layers,
    dropout=dropout,
    minibatch=minibatch,
)
num_params = sum([param.numel() for param in model.parameters()])
print("Number of parameters: ", num_params)
folder = "results/"
name = type(Hamiltonians[0]).__name__
save_str = f"{name}_{embedding_size}_{n_head}_{n_layers}"
# missing_keys, unexpected_keys = model.load_state_dict(torch.load(f'{folder}ckpt_100000_{save_str}_0.ckpt'),
#                                                       strict=False)
# print(f'Missing keys: {missing_keys}')
# print(f'Unexpected keys: {unexpected_keys}')

param_range = None  # use default param range
# param = torch.tensor([1.0])
# param_range = torch.tensor([[param], [param]])
point_of_interest = None
use_SR = False

optim = Optimizer(model, Hamiltonians, point_of_interest=point_of_interest, logger=logger)
optim.train(
    100000,
    batch=1000000,
    max_unique=100,
    param_range=param_range,
    fine_tuning=False,
    use_SR=use_SR,
    ensemble_id=int(use_SR),
)
