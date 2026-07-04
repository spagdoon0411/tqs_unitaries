"""
Exact-diagonalization cross-check of the NQS magnetization-vs-h plot (see
scripts/magnetization_readings.py and scripts/plot_magnetization_readings.py), and of the
DMRG cross-check (scripts/magnetization_sweep_dmrg.py), for the three-spin cluster-Ising
chain:

    H = -(J2/2) sum_i Z_i Z_{i+1} - (J3/2) sum_i Z_{i-1} X_i Z_{i+1} - (h/2) sum_i X_i

Ground states are computed by full diagonalization (scipy eigsh on the exact 2^L x 2^L
Hamiltonian, via Hamiltonian.IsingThreeSpin.full_H), which is only tractable for small L
(L <= ~14 or so) but has no truncation, warm-start, or branch-switching concerns at all --
unlike DMRG, this is the actual exact ground state. As with the DMRG script, the
order-parameter magnitude is estimated from the zz structure factor rather than a raw
single-site expectation value (which vanishes identically by the model's Z_2 symmetry):

    m(h) = sqrt( (1/L^2) sum_{i,j} <Z_i Z_j> )

computed directly from the exact ground-state wavefunction's basis-state probabilities,
since Z is diagonal in the computational basis.

Run from the repo root:

    uv run python scripts/magnetization_sweep_exact.py --L 10 --open --h-min -1 --h-max 0 --n-h 101
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Hamiltonian import IsingThreeSpin


def ground_state_structure_factor(H: IsingThreeSpin, h: float, L: int) -> tuple[float, float, float]:
    Hmat = H.full_H(param=h)
    E, psi = eigsh(Hmat, k=1, which="SA")
    psi = psi[:, 0]
    idxs = np.arange(2**L)
    bits = (idxs[:, None] >> np.arange(L)[None, :]) & 1
    spins = 1 - 2 * bits  # (2**L, L), +-1
    probs = np.abs(psi) ** 2
    weighted_spins = spins * probs[:, None]
    corr = spins.T @ weighted_spins  # (L, L)
    sf = float(corr.mean())
    m = float(np.sqrt(max(sf, 0.0)))
    return sf, m, float(E[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=10, help="Chain length (default: 10; must be small, <=~14).")
    parser.add_argument(
        "--open",
        dest="periodic",
        action="store_false",
        help="Use open boundary conditions instead of the default periodic ring.",
    )
    parser.add_argument("--h-min", type=float, default=-0.5, help="Lower end of the h sweep (default: -0.5).")
    parser.add_argument("--h-max", type=float, default=2.5, help="Upper end of the h sweep (default: 2.5).")
    parser.add_argument("--n-h", type=int, default=41, help="Number of h values to sweep (default: 41).")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output tensor path (default: scripts/data/<run>_readings.pt).",
    )
    parser.set_defaults(periodic=True)
    args = parser.parse_args()

    H = IsingThreeSpin(args.L, periodic=args.periodic)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bc_tag = "periodic" if args.periodic else "open"
    run_name = f"{timestamp}_exact_L{args.L}_{bc_tag}_J2-{H.J2}_J3-{H.J3}"
    data_out = args.out if args.out is not None else Path(__file__).parent / "data" / f"{run_name}_readings.pt"

    h_values = np.linspace(args.h_min, args.h_max, args.n_h)
    magnetizations = np.zeros(args.n_h)
    structure_factors = np.zeros(args.n_h)
    energies = np.zeros(args.n_h)

    for idx, h in enumerate(h_values):
        sf, m, E = ground_state_structure_factor(H, float(h), args.L)
        magnetizations[idx] = m
        structure_factors[idx] = sf
        energies[idx] = E
        print(f"h={h:6.3f}  sf={sf:.6f}  m={m:.6f}  E={E:.6f}")

    readings = torch.tensor(magnetizations, dtype=torch.float32).view(1, 1, args.n_h)
    data_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "readings": readings,
            "structure_factors": torch.tensor(structure_factors, dtype=torch.float32).view(1, 1, args.n_h),
            "energies": torch.tensor(energies, dtype=torch.float32).view(1, 1, args.n_h),
            "h_values": torch.tensor(h_values),
            "h_name": "h",
            "iterations": torch.tensor([0]),
            "L": args.L,
            "wandb_run_name": run_name,
            "method": "Exact diagonalization",
        },
        data_out,
    )
    print(f"Wrote {data_out}")


if __name__ == "__main__":
    main()
