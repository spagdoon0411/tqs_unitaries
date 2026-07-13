"""
Generates the readings tensor for a 3D magnetization sweep -- magnetization vs. training
iteration number and h -- from a single NQS checkpoint directory, by driving
scripts/magnetization_readings.py with a fixed, self-documenting parameter set.

This produces ONLY the per-reading readings tensor (shape (iteration, reading, h)); it
does not plot anything. Feed the written *.pt into scripts/plot_magnetization_readings.py
--mode 3d separately to render the surface.

Every knob of the sweep is lifted to the SWEEP PARAMETERS block below so denser sweeps
(more h-points, more readings, finer iteration steps) can be produced by editing only the
constants, never the logic. The output filename and an embedded `sweep_params` provenance
record are both built from those constants, so the tensor stays traceable to exactly the
inputs that produced it.

Run from the repo root:

    uv run python scripts/magnetization_sweep_3d.py
"""

import json
import subprocess
import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------------------------------
# SWEEP PARAMETERS -- edit only these to run a different / denser sweep.
# ----------------------------------------------------------------------------------------
CKPT_DIR = Path("checkpoints/20260703-224128_devout-sun-10")  # run dir with run_summary.json + weights

CHAIN_LENGTH = 10  # evaluate the model at this trained system size (must be in its system_sizes)

H_MIN = -2.0  # lower end of the h sweep
H_MAX = 3.0  # upper end of the h sweep
N_H = 10  # number of h-points across [H_MIN, H_MAX]

ITER_MIN = 0  # first training-iteration checkpoint to include (inclusive)
ITER_MAX = 1000  # last training-iteration checkpoint to include (inclusive)
ITER_SKIP = 100  # keep only checkpoints whose iteration is a multiple of this (step size)

N_READINGS = 10  # independent quantum-state readings per (iteration, h) point
# ----------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
READINGS_SCRIPT = Path(__file__).resolve().parent / "magnetization_readings.py"
DATA_DIR = Path(__file__).resolve().parent / "data"


def _resolve_system_size_idx(ckpt_dir: Path, chain_length: int) -> int:
    """Map CHAIN_LENGTH to the index magnetization_readings.py expects, from the run config."""
    summary_path = ckpt_dir / "run_summary.json"
    if not summary_path.exists():
        raise SystemExit(f"No run_summary.json found in '{ckpt_dir}'.")
    with open(summary_path) as f:
        config = json.load(f).get("config")
    if config is None:
        raise SystemExit(f"'{summary_path}' has no 'config' entry.")
    system_sizes = [tuple(s) for s in config["system_sizes"]]
    if (chain_length,) not in system_sizes:
        raise SystemExit(
            f"Chain length {chain_length} is not among the trained system_sizes "
            f"{[s[0] for s in system_sizes]} of '{ckpt_dir.name}'."
        )
    return system_sizes.index((chain_length,))


def main() -> None:
    system_size_idx = _resolve_system_size_idx(CKPT_DIR, CHAIN_LENGTH)

    out_name = (
        f"{CKPT_DIR.name}_L{CHAIN_LENGTH}"
        f"_h{H_MIN}-{H_MAX}_nh{N_H}"
        f"_iter{ITER_MIN}-{ITER_MAX}-step{ITER_SKIP}"
        f"_nr{N_READINGS}_3dsweep_readings.pt"
    )
    out_path = DATA_DIR / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(READINGS_SCRIPT),
        str(CKPT_DIR),
        "--h-min", str(H_MIN),
        "--h-max", str(H_MAX),
        "--n-h", str(N_H),
        "--iteration-min", str(ITER_MIN),
        "--iteration-max", str(ITER_MAX),
        "--iteration-skip", str(ITER_SKIP),
        "--n-readings", str(N_READINGS),
        "--system-size-idx", str(system_size_idx),
        "--out", str(out_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    # Stamp the tensor with the full set of inputs that produced it, so the file is
    # self-describing regardless of how it is later renamed or moved.
    data = torch.load(out_path, map_location="cpu", weights_only=False)
    data["sweep_params"] = {
        "checkpoint_dir": str(CKPT_DIR),
        "chain_length": CHAIN_LENGTH,
        "system_size_idx": system_size_idx,
        "h_min": H_MIN,
        "h_max": H_MAX,
        "n_h": N_H,
        "iteration_min": ITER_MIN,
        "iteration_max": ITER_MAX,
        "iteration_skip": ITER_SKIP,
        "n_readings": N_READINGS,
    }
    torch.save(data, out_path)
    print(f"Wrote {out_path} (labeled with sweep_params)")


if __name__ == "__main__":
    main()
