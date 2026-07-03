# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:30:45 2022

@author: Yuanhang Zhang
"""

from model import TransformerModel
from Hamiltonian import IsingThreeSpin
from optimizer import Optimizer

import os
import numpy as np
import torch
import wandb

wandb_project = "tqs-unitaries"


def main():
    config = {
        "hamiltonian": "IsingThreeSpin",
        "system_sizes": np.arange(10, 41, 2).reshape(-1, 1).tolist(),
        "periodic": True,
        "embedding_size": 32,
        "n_head": 8,
        "n_hid": 32,
        "n_layers": 8,
        "dropout": 0,
        "minibatch": 10000,
        "n_iter": 100000,
        "batch": 1000000,
        "max_unique": 100,
        "use_SR": False,
        "fine_tuning": False,
        "param_range": None,
        "point_of_interest": None,
    }

    torch.set_default_tensor_type(
        torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    )
    try:
        os.mkdir("results/")
    except FileExistsError:
        pass

    system_sizes = np.array(config["system_sizes"])
    Hamiltonians = [
        IsingThreeSpin(system_size_i, periodic=config["periodic"])
        for system_size_i in system_sizes
    ]
    config["param_dim"] = Hamiltonians[0].param_dim
    config["J2"] = Hamiltonians[0].J2
    config["J3"] = Hamiltonians[0].J3
    config["param_range"] = (
        Hamiltonians[0].param_range.tolist()
        if config["param_range"] is None
        else config["param_range"]
    )

    model = TransformerModel(
        system_sizes,
        config["param_dim"],
        config["embedding_size"],
        config["n_head"],
        config["n_hid"],
        config["n_layers"],
        dropout=config["dropout"],
        minibatch=config["minibatch"],
    )
    config["num_params"] = sum(param.numel() for param in model.parameters())
    print("Number of parameters: ", config["num_params"])

    name = type(Hamiltonians[0]).__name__
    save_str = f"{name}_{config['embedding_size']}_{config['n_head']}_{config['n_layers']}"
    # missing_keys, unexpected_keys = model.load_state_dict(
    #     torch.load(f"results/ckpt_100000_{save_str}_0.ckpt"), strict=False
    # )
    # print(f'Missing keys: {missing_keys}')
    # print(f'Unexpected keys: {unexpected_keys}')

    wandb.init(project=wandb_project, config=config)

    optim = Optimizer(model, Hamiltonians, point_of_interest=config["point_of_interest"])
    optim.train(
        config["n_iter"],
        batch=config["batch"],
        max_unique=config["max_unique"],
        param_range=torch.tensor(config["param_range"]),
        fine_tuning=config["fine_tuning"],
        use_SR=config["use_SR"],
        ensemble_id=int(config["use_SR"]),
    )
    wandb.finish()


if __name__ == "__main__":
    main()
