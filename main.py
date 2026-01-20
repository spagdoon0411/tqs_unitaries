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

# Log run configuration to Neptune
if logger:
    logger["run_config/gpu/available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        logger["run_config/gpu/device_count"] = torch.cuda.device_count()
        logger["run_config/gpu/device_name"] = torch.cuda.get_device_name(0)
        logger["run_config/gpu/memory_total"] = torch.cuda.get_device_properties(0).total_memory
    
    logger["run_config/hamiltonian/class"] = args.model
    logger["run_config/system/sizes"] = str(system_sizes.flatten().tolist())
    logger["run_config/system/min_size"] = int(system_sizes.min())
    logger["run_config/system/max_size"] = int(system_sizes.max())
    logger["run_config/system/num_sizes"] = len(system_sizes)

hamiltonian_class = IsingY if args.model == 'IsingY' else Ising
Hamiltonians = [hamiltonian_class(system_size_i, periodic=False) for system_size_i in system_sizes]

# Log Hamiltonian parameters to Neptune
if logger:
    sample_ham = Hamiltonians[0]
    logger["run_config/hamiltonian/param_dim"] = sample_ham.param_dim
    logger["run_config/hamiltonian/param_range"] = str(sample_ham.param_range.tolist())
    logger["run_config/hamiltonian/periodic"] = sample_ham.periodic
    if hasattr(sample_ham, 'J'):
        logger["run_config/hamiltonian/J"] = sample_ham.J
    if hasattr(sample_ham, 'h'):
        logger["run_config/hamiltonian/h"] = sample_ham.h

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

# Log model configuration to Neptune
if logger:
    logger["run_config/model/param_dim"] = param_dim
    logger["run_config/model/embedding_size"] = embedding_size
    logger["run_config/model/n_head"] = n_head
    logger["run_config/model/n_hid"] = n_hid
    logger["run_config/model/n_layers"] = n_layers
    logger["run_config/model/dropout"] = dropout
    logger["run_config/model/minibatch"] = minibatch
    logger["run_config/model/num_params"] = num_params
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
