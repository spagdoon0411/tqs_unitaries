# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:30:45 2022

@author: Yuanhang Zhang
"""

from model import TransformerModel
from Hamiltonian import Ising, IsingY, XYZ
from optimizer import Optimizer
from diagnostic_config import DiagnosticConfig, DEFAULT_CONFIG
from diagnostic_logger import DiagnosticLogger

import os
import numpy as np
import torch
import argparse
import neptune
import os
from dotenv import load_dotenv
import signal
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["Ising", "IsingY"], default="Ising")
parser.add_argument("--diagnostics", action="store_true", default=True,
                    help="Enable diagnostic logging (default: True)")
parser.add_argument("--results-dir", default="results",
                    help="Directory for saving results and checkpoints (default: results)")
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
    
    def signal_handler(sig, frame):
        print("\nInterrupted! Stopping Neptune logging...")
        if logger:
            logger.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
else:
    print("Warning: Neptune logging disabled. Set NEPTUNE_API_TOKEN and NEPTUNE_PROJECT in .env file to enable logging.")

torch.set_default_tensor_type(
    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
)
# torch.set_default_tensor_type(torch.FloatTensor)
os.makedirs(args.results_dir, exist_ok=True)

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
folder = args.results_dir
name = type(Hamiltonians[0]).__name__
save_str = f"{name}_{embedding_size}_{n_head}_{n_layers}"
# missing_keys, unexpected_keys = model.load_state_dict(torch.load(f'{folder}/ckpt_100000_{save_str}_0.ckpt'),
#                                                       strict=False)
# print(f'Missing keys: {missing_keys}')
# print(f'Unexpected keys: {unexpected_keys}')

param_range = None  # use default param range
# param = torch.tensor([1.0])
# param_range = torch.tensor([[param], [param]])
point_of_interest = None
use_SR = False

optim = Optimizer(model, Hamiltonians, point_of_interest=point_of_interest, logger=logger, results_dir=args.results_dir)

# Initialize diagnostic logger if enabled
diagnostic_logger = None
if args.diagnostics and logger:
    diagnostic_logger = DiagnosticLogger(logger, DEFAULT_CONFIG)
    diagnostic_logger.register_hooks(model)
    optim.diagnostic_logger = diagnostic_logger
    
    # Log diagnostic config to Neptune
    logger["run_config/diagnostics/enabled"] = True
    logger["run_config/diagnostics/global_frequency"] = DEFAULT_CONFIG.global_frequency
    logger["run_config/diagnostics/mean_activations"] = DEFAULT_CONFIG.mean_activations.enabled
    logger["run_config/diagnostics/std_activations"] = DEFAULT_CONFIG.std_activations.enabled
    logger["run_config/diagnostics/activations_below_threshold"] = DEFAULT_CONFIG.activations_below_threshold.enabled
    logger["run_config/diagnostics/threshold_value"] = DEFAULT_CONFIG.activations_below_threshold.threshold
    logger["run_config/diagnostics/gradient_l2_norm"] = DEFAULT_CONFIG.gradient_l2_norm.enabled
    logger["run_config/diagnostics/weight_l2_norm"] = DEFAULT_CONFIG.weight_l2_norm.enabled
    logger["run_config/diagnostics/update_to_weight_ratio"] = DEFAULT_CONFIG.update_to_weight_ratio.enabled
    logger["run_config/diagnostics/activation_percentiles"] = DEFAULT_CONFIG.activation_percentiles.enabled
elif logger:
    logger["run_config/diagnostics/enabled"] = False

optim.train(
    100000,
    batch=1000000,
    max_unique=100,
    param_range=param_range,
    fine_tuning=False,
    use_SR=use_SR,
    ensemble_id=int(use_SR),
)
