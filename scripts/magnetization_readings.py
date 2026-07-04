"""
Draws quantum-state readings of |magnetization| across a sweep of h from one checkpoint
or a range of checkpoints, saving the raw (non-averaged) per-reading tensor to disk.
sigma^z is diagonal in the sampled basis, so each reading is read directly off the
sampled spin bits:

    m(x) = | mean_i (2 x_i - 1) |

Run from the repo root:

    uv run python scripts/magnetization_readings.py checkpoints/20260701_120000_flowery-frog-3 --n-h 51
    uv run python scripts/magnetization_readings.py checkpoints/20260701_120000_flowery-frog-3 --n-h 51 \\
        --iteration-min 500 --iteration-max 2000
"""

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Hamiltonian import Ising, IsingThreeSpin
from model import TransformerModel
from model_utils import sample

HAMILTONIAN_CLASSES = {
    "Ising": Ising,
    "IsingThreeSpin": IsingThreeSpin,
}

BATCH = 131_072
MAX_UNIQUE = 2048


def _reading(model: TransformerModel, symmetry) -> float:
    samples, sample_weight = sample(model, batch=BATCH, max_unique=MAX_UNIQUE, symmetry=symmetry)
    spins_pm = 2 * samples.to(torch.get_default_dtype()) - 1
    per_config = spins_pm.mean(dim=0).abs()
    return (per_config * sample_weight).sum().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_dir", type=Path, help="Run directory with run_summary.json and model_iter_*.pt weights.")
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Single checkpoint iteration to load (default: latest available). "
        "Mutually exclusive with --iteration-min/--iteration-max.",
    )
    parser.add_argument(
        "--iteration-min", type=int, default=None, help="Lower bound (inclusive) of a range of iterations to sweep."
    )
    parser.add_argument(
        "--iteration-max", type=int, default=None, help="Upper bound (inclusive) of a range of iterations to sweep."
    )
    parser.add_argument(
        "--iteration-skip",
        type=int,
        default=1,
        help="Only keep iterations that are multiples of this value (default: 1, i.e. no skipping).",
    )
    parser.add_argument("--n-h", type=int, required=True, help="Number of h values to sweep (granularity).")
    parser.add_argument(
        "--n-readings", type=int, default=10, help="Independent quantum-state readings per h (default: 10)."
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=1,
        help="Number of model instances to load and sample from concurrently (default: 1, sequential).",
    )
    parser.add_argument(
        "--system-size-idx",
        type=int,
        default=-1,
        help="Index into the trained system_sizes list to evaluate at (default: -1, the largest size).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output tensor path (default: scripts/data/<run>_iter<N>_readings.pt).",
    )
    args = parser.parse_args()

    if args.iteration is not None and (args.iteration_min is not None or args.iteration_max is not None):
        raise SystemExit("--iteration is mutually exclusive with --iteration-min/--iteration-max.")
    if (args.iteration_min is None) != (args.iteration_max is None):
        raise SystemExit("--iteration-min and --iteration-max must be given together.")
    if args.parallelism <= 0:
        raise SystemExit(f"--parallelism must be positive, got {args.parallelism}.")

    summary_path = args.ckpt_dir / "run_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No run_summary.json found in '{args.ckpt_dir}'.")
    with open(summary_path) as f:
        metadata = json.load(f)
    if not metadata.get("checkpoints"):
        raise SystemExit(f"'{summary_path}' has no checkpoint index; re-run training with the updated main.py.")
    config = metadata.get("config")
    if config is None:
        raise SystemExit(f"'{summary_path}' has no 'config' entry; re-run training with the updated main.py.")

    print(
        tabulate(
            [[k, v] for k, v in config.items()],
            headers=["Parameter", "Value"],
            tablefmt="rounded_outline",
        )
    )
    iterations = sorted(int(k) for k in metadata["checkpoints"])
    print(tabulate([[i] for i in iterations], headers=["Iteration available"], tablefmt="rounded_outline"))

    if args.iteration is not None:
        if args.iteration not in iterations:
            raise SystemExit(f"No checkpoint at iteration {args.iteration}. Available: {iterations}")
        selected_iterations = [args.iteration]
    elif args.iteration_min is not None:
        selected_iterations = [i for i in iterations if args.iteration_min <= i <= args.iteration_max]
        if not selected_iterations:
            raise SystemExit(
                f"No checkpoints in range [{args.iteration_min}, {args.iteration_max}]. Available: {iterations}"
            )
    else:
        selected_iterations = [iterations[-1]]

    if args.iteration_skip > 1:
        selected_iterations = [i for i in selected_iterations if i % args.iteration_skip == 0]
        if not selected_iterations:
            raise SystemExit(f"No iterations are multiples of --iteration-skip={args.iteration_skip}.")
    print(f"\nUsing iterations {selected_iterations}")

    torch.set_default_tensor_type(
        torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config["hamiltonian"] not in HAMILTONIAN_CLASSES:
        raise SystemExit(
            f"Unknown Hamiltonian '{config['hamiltonian']}'; expected one of {sorted(HAMILTONIAN_CLASSES)}."
        )
    ham_cls = HAMILTONIAN_CLASSES[config["hamiltonian"]]
    system_sizes = np.array(config["system_sizes"])
    system_size = system_sizes[args.system_size_idx]
    H = ham_cls(system_size, periodic=config["periodic"])
    h_min, h_max = config["param_range"][0][0], config["param_range"][1][0]

    # Each concurrent reading gets its own model instance, since set_param mutates
    # model state (system_size, param, prefix) and sharing one across threads would race.
    # The Hamiltonian is only read from (for its symmetry object), so it is safe to share.
    models = []
    for _ in range(args.parallelism):
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
        model.param_range = torch.tensor(config["param_range"])
        models.append(model)

    system_size_tensor = torch.tensor(system_size, dtype=torch.int64)
    L = int(system_size_tensor.prod().item())
    h_values = np.linspace(h_min, h_max, args.n_h)
    readings = torch.zeros(len(selected_iterations), args.n_readings, args.n_h)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        for iter_idx, iteration in enumerate(selected_iterations):
            ckpt_path = Path(metadata["checkpoints"][str(iteration)])
            if not ckpt_path.exists():
                ckpt_path = args.ckpt_dir / ckpt_path.name
            state_dict = torch.load(ckpt_path, map_location=device)
            for model in models:
                model.load_state_dict(state_dict)
                model.eval()

            for h_idx, h in enumerate(tqdm(h_values, desc=f"iter {iteration}: h sweep")):
                param = torch.tensor([float(h)])
                for model in models:
                    model.set_param(system_size=system_size_tensor, param=param)

                futures = [
                    executor.submit(_reading, models[reading_idx % args.parallelism], H.symmetry)
                    for reading_idx in range(args.n_readings)
                ]
                for reading_idx, future in enumerate(futures):
                    readings[iter_idx, reading_idx, h_idx] = future.result()

    iter_suffix = (
        str(selected_iterations[0])
        if len(selected_iterations) == 1
        else f"{selected_iterations[0]}-{selected_iterations[-1]}"
    )
    out = (
        args.out
        if args.out is not None
        else Path(__file__).parent / "data" / f"{args.ckpt_dir.name}_iter{iter_suffix}_readings.pt"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "readings": readings,
            "h_values": torch.tensor(h_values),
            "h_name": "h",
            "iterations": torch.tensor(selected_iterations),
            "L": L,
            "wandb_run_name": metadata.get("wandb_run_name"),
        },
        out,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
