# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:30:45 2022

@author: Yuanhang Zhang
"""

from model import TransformerModel
from Hamiltonian import IsingThreeSpin
from optimizer import Optimizer

import argparse
import json
import os
import datetime
from pathlib import Path

import numpy as np
import torch
import wandb

wandb_project = "tqs-unitaries"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to an existing run's checkpoint folder (containing run_summary.json). "
        "Resumes training from that run's last checkpoint, using its recorded config.",
    )
    return parser.parse_args()


def load_resume_state(resume_dir):
    summary_path = resume_dir / "run_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No run_summary.json found in '{resume_dir}'.")
    with open(summary_path) as f:
        run_summary = json.load(f)
    if not run_summary.get("checkpoints"):
        raise SystemExit(f"'{summary_path}' has no checkpoint index; nothing to resume from.")

    config = run_summary["config"]
    last_iter = run_summary["last_iter"]
    ckpt_path = Path(run_summary["checkpoints"][str(last_iter)])
    if not ckpt_path.exists():
        ckpt_path = resume_dir / ckpt_path.name
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint for iteration {last_iter} not found (looked for '{ckpt_path}').")

    return config, last_iter, ckpt_path, run_summary["wandb_run_id"], run_summary["wandb_run_name"]


def main():
    args = parse_args()
    resuming = args.resume is not None

    if resuming:
        config, last_iter, ckpt_path, wandb_run_id, wandb_run_name = load_resume_state(args.resume)
        checkpoint_dir = str(args.resume)
        start_iter = last_iter + 1
    else:
        config = {
            "hamiltonian": "IsingThreeSpin",
            "system_sizes": np.arange(10, 41, 2).reshape(-1, 1).tolist(),
            "periodic": False,
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
            "checkpoint_freq": 10,
        }
        start_iter = None

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

    if resuming:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Resuming from '{ckpt_path}' at iteration {last_iter}; continuing from iteration {start_iter}.")
        wandb.init(project=wandb_project, config=config, id=wandb_run_id, resume="must")
    else:
        wandb.init(project=wandb_project, config=config)
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_dir = os.path.join("checkpoints", f"{timestamp}_{wandb.run.name}")
        os.makedirs(checkpoint_dir, exist_ok=True)

    optim = Optimizer(
        model, Hamiltonians, point_of_interest=config["point_of_interest"]
    )
    try:
        optim.train(
            config["n_iter"],
            batch=config["batch"],
            max_unique=config["max_unique"],
            param_range=torch.tensor(config["param_range"]),
            fine_tuning=config["fine_tuning"],
            use_SR=config["use_SR"],
            ensemble_id=int(config["use_SR"]),
            start_iter=start_iter,
            checkpoint_dir=checkpoint_dir,
            checkpoint_freq=config["checkpoint_freq"],
            run_config=config,
            wandb_run_id=wandb.run.id,
            wandb_run_name=wandb.run.name,
        )
    except KeyboardInterrupt:
        print("Training interrupted; finishing wandb run before exiting.")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
