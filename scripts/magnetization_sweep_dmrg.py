"""
DMRG cross-check of the NQS magnetization-vs-h plot (see scripts/magnetization_readings.py
and scripts/plot_magnetization_readings.py) for the three-spin cluster-Ising chain:

    H = -(J2/2) sum_i Z_i Z_{i+1} - (J3/2) sum_i Z_{i-1} X_i Z_{i+1} - (h/2) sum_i X_i

Ground states are computed with finite-system DMRG (TeNPy), sweeping h and warm-starting
each DMRG run from the previous h's converged state for speed. Since the exact ground state
respects the global Z_2 symmetry prod_i X_i (so <Z_i> = 0 identically at any finite size),
the order-parameter magnitude is estimated from the zz structure factor rather than a raw
single-site expectation value:

    m(h) = sqrt( (1/L^2) sum_{i,j} <Z_i Z_j> )

which is the same quantity scripts/magnetization_readings.py estimates from NQS samples as
sqrt(mean(m(x)^2)), since Z is diagonal in the sampled basis (E_samples[m(x)^2] = (1/L^2)
sum_ij <Z_i Z_j> exactly). Match --periodic/--open and --L to whatever a given NQS
checkpoint was actually trained with (check its run_summary.json) for a fair comparison --
boundary condition and system size both change the ground-state physics.

Warm-started, finite-chi DMRG on a periodic ring is prone to a specific artifact: at a
fixed h, the optimizer can settle into a metastable variational branch (biased by the
previous h's state, or by the initial product state at the very first h), then suddenly
snap to a different branch a few h-steps later -- producing a spurious kink that is not
part of the true ground-state curve. This script has a few diagnostics for that:
  - --sweep-direction both runs the sweep forward (from h-min) and backward (from h-max)
    independently and stores both as separate 'readings' entries (aligned to the same h
    grid), so plot_magnetization_readings.py's mean/error-bar machinery directly shows
    where the two directions disagree.
  - --initial-state controls the product state the very first point of each direction
    warm-starts from (up/down/right/left), to check sensitivity to that choice.
  - Each step prints the raw structure factor (pre-sqrt) and the DMRG energy, since the
    sqrt can visually exaggerate small numerical changes and an energy jump between
    consecutive h is a direct sign of a branch switch.
  - --bulk-sites N excludes N sites from each edge of the chain when computing the
    structure factor, to check whether an open-boundary artifact is edge-driven.

Run from the repo root:

    uv run python scripts/magnetization_sweep_dmrg.py
    uv run python scripts/magnetization_sweep_dmrg.py --L 10 --open --h-min -4 --h-max 4
    uv run python scripts/magnetization_sweep_dmrg.py --L 10 --open --h-min -1 --h-max 0 \\
        --sweep-direction both --initial-state up
"""

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tenpy.algorithms import dmrg
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import SpinHalfSite

DPI = 300

# 'right'/'left' are the Sigmax eigenstates (+1/-1), given as explicit local superpositions
# since SpinHalfSite only defines 'up'/'down' as named state labels.
INITIAL_STATES = {
    "up": "up",
    "down": "down",
    "right": np.array([1.0, 1.0]) / np.sqrt(2),
    "left": np.array([1.0, -1.0]) / np.sqrt(2),
}


class IsingThreeSpinModel(CouplingMPOModel):
    """TeNPy model for H = -(J2/2) ZZ - (J3/2) ZXZ - (h/2) X."""

    def init_sites(self, model_params):
        return SpinHalfSite(conserve=None)

    def init_terms(self, model_params):
        J2 = model_params.get("J2", 1.0)
        J3 = model_params.get("J3", 1.0)
        h = model_params.get("h", 0.0)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-h / 2, u, "Sigmax")
        self.add_multi_coupling(-J2 / 2, [("Sigmaz", 0, 0), ("Sigmaz", 1, 0)])
        self.add_multi_coupling(-J3 / 2, [("Sigmaz", -1, 0), ("Sigmax", 0, 0), ("Sigmaz", 1, 0)])


def run_dmrg(
    h: float, psi: MPS | None, L: int, J2: float, J3: float, periodic: bool, chi_max: int, svd_min: float,
    max_e_err: float, max_sweeps: int, use_mixer: bool, initial_state: str,
) -> tuple[MPS, dict]:
    model_params = dict(
        L=L,
        J2=J2,
        J3=J3,
        h=h,
        bc_MPS="finite",
        bc_x="periodic" if periodic else "open",
        order="default",
    )
    model = IsingThreeSpinModel(model_params)
    if psi is None:
        p_state = [INITIAL_STATES[initial_state]] * L
        psi = MPS.from_product_state(model.lat.mps_sites(), p_state, bc=model.lat.bc_MPS)
    dmrg_params = {
        "trunc_params": {"chi_max": chi_max, "svd_min": svd_min},
        "mixer": use_mixer,
        "max_E_err": max_e_err,
        "max_sweeps": max_sweeps,
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="final DMRG state not in canonical form")
        info = dmrg.run(psi, model, dmrg_params)
    psi.canonical_form()
    return psi, info


def zz_structure_factor_magnetization(psi: MPS, L: int, bulk_sites: int) -> tuple[float, float]:
    """
    Returns (raw structure factor, sqrt(max(structure factor, 0))). Uses correlation_function
    explicitly over the full (or bulk-trimmed) set of site pairs, rather than relying on
    its default site range, so this is unambiguously (1/n^2) sum_ij <Z_i Z_j> over the
    sites actually being measured.
    """
    sites = list(range(bulk_sites, L - bulk_sites))
    n = len(sites)
    corr = psi.correlation_function("Sigmaz", "Sigmaz", sites1=sites, sites2=sites)
    sf = float(np.sum(corr).real / (n * n))
    m = float(np.sqrt(max(sf, 0.0)))
    return sf, m


def sweep(
    h_values: np.ndarray, initial_state: str, L: int, J2: float, J3: float, periodic: bool, chi_max: int,
    svd_min: float, max_e_err: float, max_sweeps: int, use_mixer: bool, bulk_sites: int, label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnetizations = np.zeros(len(h_values))
    structure_factors = np.zeros(len(h_values))
    energies = np.zeros(len(h_values))
    psi = None
    for idx, h in enumerate(h_values):
        psi, info = run_dmrg(
            float(h), psi, L, J2, J3, periodic, chi_max, svd_min, max_e_err, max_sweeps, use_mixer, initial_state,
        )
        sf, m = zz_structure_factor_magnetization(psi, L, bulk_sites)
        magnetizations[idx] = m
        structure_factors[idx] = sf
        energies[idx] = info["E"]
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}h={h:6.3f}  sf={sf:.6f}  m={m:.6f}  E={info['E']:.6f}  max_chi={max(psi.chi)}")
    return magnetizations, structure_factors, energies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--L", type=int, default=30, help="Chain length (default: 30).")
    parser.add_argument("--J2", type=float, default=1.0, help="ZZ coupling (default: 1.0).")
    parser.add_argument("--J3", type=float, default=1.0, help="ZXZ (cluster) coupling (default: 1.0).")
    parser.add_argument(
        "--open",
        dest="periodic",
        action="store_false",
        help="Use open boundary conditions instead of the default periodic ring.",
    )
    parser.add_argument("--h-min", type=float, default=-0.5, help="Lower end of the h sweep (default: -0.5).")
    parser.add_argument("--h-max", type=float, default=2.5, help="Upper end of the h sweep (default: 2.5).")
    parser.add_argument("--n-h", type=int, default=41, help="Number of h values to sweep (default: 41).")
    parser.add_argument("--chi-max", type=int, default=150, help="DMRG max bond dimension (default: 150).")
    parser.add_argument("--svd-min", type=float, default=1e-10)
    parser.add_argument("--max-e-err", type=float, default=1e-8)
    parser.add_argument("--max-sweeps", type=int, default=15)
    parser.add_argument("--no-mixer", dest="use_mixer", action="store_false")
    parser.add_argument(
        "--initial-state",
        choices=sorted(INITIAL_STATES),
        default="up",
        help="Product state each sweep direction warm-starts its first h from (default: up).",
    )
    parser.add_argument(
        "--sweep-direction",
        choices=["forward", "backward", "both"],
        default="forward",
        help="forward sweeps h-min->h-max, backward sweeps h-max->h-min, both runs and records "
        "both independently (as separate 'readings' entries) to check for branch-switching "
        "artifacts that depend on sweep direction (default: forward).",
    )
    parser.add_argument(
        "--bulk-sites",
        type=int,
        default=0,
        help="Exclude this many sites from each edge of the chain when computing the structure "
        "factor, to check whether an artifact is edge-driven (default: 0, use the full chain).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output tensor path (default: scripts/data/<run>_readings.pt).",
    )
    parser.set_defaults(periodic=True, use_mixer=True)
    args = parser.parse_args()

    if 2 * args.bulk_sites >= args.L:
        raise SystemExit(f"--bulk-sites {args.bulk_sites} leaves no sites for L={args.L}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bc_tag = "periodic" if args.periodic else "open"
    run_name = f"{timestamp}_dmrg_L{args.L}_{bc_tag}_J2-{args.J2}_J3-{args.J3}_chi{args.chi_max}"
    data_out = args.out if args.out is not None else Path(__file__).parent / "data" / f"{run_name}_readings.pt"

    h_values = np.linspace(args.h_min, args.h_max, args.n_h)

    directions = []
    if args.sweep_direction in ("forward", "both"):
        directions.append(("forward", h_values, False))
    if args.sweep_direction in ("backward", "both"):
        directions.append(("backward", h_values[::-1], True))

    all_m, all_sf, all_e = [], [], []
    for label, h_seq, reverse_output in directions:
        m_seq, sf_seq, e_seq = sweep(
            h_seq, args.initial_state, args.L, args.J2, args.J3, args.periodic, args.chi_max, args.svd_min,
            args.max_e_err, args.max_sweeps, args.use_mixer, args.bulk_sites,
            label if args.sweep_direction == "both" else "",
        )
        if reverse_output:
            m_seq, sf_seq, e_seq = m_seq[::-1], sf_seq[::-1], e_seq[::-1]
        all_m.append(m_seq)
        all_sf.append(sf_seq)
        all_e.append(e_seq)

    # Matches scripts/magnetization_readings.py's on-disk schema (readings shaped
    # (iteration, reading, h)) so plot_magnetization_readings.py can load this unmodified,
    # including overlaying it against an NQS *_readings.pt file. DMRG is deterministic (no
    # Monte Carlo noise), so the "iteration" axis is singleton; chi_max stands in for
    # "iteration" as the DMRG convergence knob. The "reading" axis holds one entry per sweep
    # direction requested; with --sweep-direction both, the mean/error-bar machinery in
    # plot_magnetization_readings.py directly visualizes forward/backward disagreement.
    readings = torch.tensor(np.stack(all_m), dtype=torch.float32).unsqueeze(0)
    structure_factors = torch.tensor(np.stack(all_sf), dtype=torch.float32).unsqueeze(0)
    energies = torch.tensor(np.stack(all_e), dtype=torch.float32).unsqueeze(0)
    data_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "readings": readings,
            "structure_factors": structure_factors,
            "energies": energies,
            "h_values": torch.tensor(h_values),
            "h_name": "h",
            "iterations": torch.tensor([args.chi_max]),
            "L": args.L,
            "wandb_run_name": run_name,
        },
        data_out,
    )
    print(f"Wrote {data_out}")


if __name__ == "__main__":
    main()
