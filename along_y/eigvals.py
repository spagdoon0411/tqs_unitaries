import sys

sys.path.append("..")

from Hamiltonian import Ising, IsingY
import numpy as np
import argparse
import matplotlib.pyplot as plt

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Compute ground states of Ising and IsingY Hamiltonians"
)
parser.add_argument(
    "--show-observables",
    action="store_true",
    help="Print the Pauli string operators and sites they act on",
)
parser.add_argument(
    "--plot-groundstate",
    action="store_true",
    help="Plot real and imaginary components of ground state",
)
parser.add_argument(
    "--h",
    type=float,
    default=1.0,
    help="External field parameter (default: 1.0)",
)
parser.add_argument(
    "--system-size",
    type=int,
    default=10,
    help="Number of spins in 1D chain (default: 10)",
)
args = parser.parse_args()

# System parameters
system_size = [args.system_size]  # 1D chain
param = args.h  # External field parameter


def print_hamiltonian_observables(hamiltonian, name):
    """Print the Pauli string operators and sites they act on."""
    print(f"\n{name} Hamiltonian observables:")
    for i, (pauli_strs, coefs, spin_idx) in enumerate(hamiltonian.H):
        print(f"  Term {i+1}:")
        for j, (pauli_str, coef) in enumerate(zip(pauli_strs, coefs)):
            if hasattr(spin_idx, "__len__") and len(spin_idx) > 0:
                if hasattr(spin_idx[0], "__len__"):
                    # Multiple sites per operator
                    sites_str = ", ".join(
                        [f"{tuple(int(s) for s in sites)}" for sites in spin_idx[:5]]
                    )
                    if len(spin_idx) > 5:
                        sites_str += f", ... ({len(spin_idx)} total)"
                    print(f"    {pauli_str}: coef={coef}, sites=[{sites_str}]")
                else:
                    # Single site
                    print(
                        f"    {pauli_str}: coef={coef}, sites={tuple(int(s) for s in spin_idx)}"
                    )


def plot_ground_state(psi, name, n_spins):
    """Plot real and imaginary components of ground state."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    indices = np.arange(len(psi))

    # Real part
    ax1.stem(indices, np.real(psi), linefmt="b-", markerfmt="bo", basefmt=" ")
    ax1.set_ylabel("Real part")
    ax1.set_title(f"{name} Ground State (n={n_spins} spins)")
    ax1.grid(True, alpha=0.3)

    # Imaginary part
    ax2.stem(indices, np.imag(psi), linefmt="r-", markerfmt="ro", basefmt=" ")
    ax2.set_xlabel("Basis state index")
    ax2.set_ylabel("Imaginary part")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'groundstate_{name.lower().replace(" ", "_")}.png', dpi=150)
    print(f"  Saved plot to groundstate_{name.lower().replace(' ', '_')}.png")


# Create Ising Hamiltonian
print("=" * 60)
print("Ising Hamiltonian (ZZ + X)")
print("=" * 60)
ising = Ising(system_size, periodic=False)
print(f"System size: {system_size}")
print(f"Number of spins: {ising.n}")
print(f"Parameter (h): {param}")

if args.show_observables:
    print_hamiltonian_observables(ising, "Ising")

print("\nGenerating full Hamiltonian...")
H_ising = ising.full_H(param)
print(f"Hamiltonian matrix size: {H_ising.shape}")

print("\nComputing ground state energy...")
E_ground_ising = ising.calc_E_ground(param)
print(f"Ground state energy: {E_ground_ising:.10f}")
print(f"Ground state energy per spin: {E_ground_ising / ising.n:.10f}")

if args.plot_groundstate:
    plot_ground_state(ising.psi_ground, "Ising", ising.n)

# Create IsingY Hamiltonian
print("\n" + "=" * 60)
print("IsingY Hamiltonian (ZZ + Y)")
print("=" * 60)
ising_y = IsingY(system_size, periodic=False)
print(f"System size: {system_size}")
print(f"Number of spins: {ising_y.n}")
print(f"Parameter (h): {param}")

if args.show_observables:
    print_hamiltonian_observables(ising_y, "IsingY")

print("\nGenerating full Hamiltonian...")
H_ising_y = ising_y.full_H(param)
print(f"Hamiltonian matrix size: {H_ising_y.shape}")

print("\nComputing ground state energy...")
E_ground_ising_y = ising_y.calc_E_ground(param)
print(f"Ground state energy: {E_ground_ising_y:.10f}")
print(f"Ground state energy per spin: {E_ground_ising_y / ising_y.n:.10f}")

if args.plot_groundstate:
    plot_ground_state(ising_y.psi_ground, "IsingY", ising_y.n)

# Comparison
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"Ising (ZZ + X) ground state energy:  {E_ground_ising:.10f}")
print(f"IsingY (ZZ + Y) ground state energy: {E_ground_ising_y:.10f}")
print(f"Difference: {abs(E_ground_ising - E_ground_ising_y):.10e}")
