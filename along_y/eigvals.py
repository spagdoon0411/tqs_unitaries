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
parser.add_argument(
    "--periodic",
    action="store_true",
    help="Use periodic boundary conditions (default: False)",
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


def plot_ground_state(psi, name, n_spins, periodic):
    """Plot real and imaginary components of ground state."""
    fig = plt.figure(figsize=(10, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 2, 2, 1.5], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    indices = np.arange(len(psi))
    bc_str = "periodic" if periodic else "open"

    # Real part
    ax1.vlines(indices, 0, np.real(psi), colors="b", linewidth=1.5)
    ax1.set_ylabel("Real part")
    ax1.set_title(f"{name} Ground State (n={n_spins} spins, {bc_str} BC)")
    ax1.grid(True, alpha=0.3)
    ax1.set_xticklabels([])
    ax1.set_xlim(-0.5, len(psi) - 0.5)

    # Imaginary part
    ax2.vlines(indices, 0, np.imag(psi), colors="r", linewidth=1.5)
    ax2.set_ylabel("Imaginary part")
    ax2.grid(True, alpha=0.3)
    ax2.set_xticklabels([])
    ax2.set_xlim(-0.5, len(psi) - 0.5)

    # Probability
    prob = np.abs(psi) ** 2
    ax3.vlines(indices, 0, prob, colors="g", linewidth=1.5)
    ax3.set_ylabel("Probability")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticklabels([])
    ax3.set_xlim(-0.5, len(psi) - 0.5)

    # Create basis state heatmap
    # Each basis state is represented as a binary number
    # where each bit represents a spin (0 = down/↓, 1 = up/↑)
    basis_states = np.zeros((n_spins, len(psi)))
    for i in range(len(psi)):
        # Convert index to binary representation
        binary = format(i, f'0{n_spins}b')
        for j, bit in enumerate(binary):
            basis_states[j, i] = int(bit)

    # Plot heatmap with extent to match the x-axis of the plots above
    im = ax4.imshow(basis_states, aspect='auto', cmap='bwr',
                     interpolation='nearest', vmin=0, vmax=1,
                     extent=[-0.5, len(psi) - 0.5, n_spins - 0.5, -0.5])
    ax4.set_ylabel("Spin index")
    ax4.set_xlabel("Basis state index")
    ax4.set_yticks(range(n_spins))
    ax4.set_yticklabels(range(n_spins))
    ax4.set_xlim(-0.5, len(psi) - 0.5)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.15, aspect=30)
    cbar.set_label('Spin state (0=↓, 1=↑)', rotation=0)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['↓', '↑'])

    plt.savefig(f'groundstate_{name.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
    print(f"  Saved plot to groundstate_{name.lower().replace(' ', '_')}.png")


# Create Ising Hamiltonian
print("=" * 60)
print("Ising Hamiltonian (ZZ + X)")
print("=" * 60)
ising = Ising(system_size, periodic=args.periodic)
print(f"System size: {system_size}")
print(f"Number of spins: {ising.n}")
print(f"Periodic boundary conditions: {args.periodic}")
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
    plot_ground_state(ising.psi_ground, "Ising", ising.n, args.periodic)

# Create IsingY Hamiltonian
print("\n" + "=" * 60)
print("IsingY Hamiltonian (ZZ + Y)")
print("=" * 60)
ising_y = IsingY(system_size, periodic=args.periodic)
print(f"System size: {system_size}")
print(f"Number of spins: {ising_y.n}")
print(f"Periodic boundary conditions: {args.periodic}")
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
    plot_ground_state(ising_y.psi_ground, "IsingY", ising_y.n, args.periodic)

# Comparison
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"Ising (ZZ + X) ground state energy:  {E_ground_ising:.10f}")
print(f"IsingY (ZZ + Y) ground state energy: {E_ground_ising_y:.10f}")
print(f"Difference: {abs(E_ground_ising - E_ground_ising_y):.10e}")

plt.savefig("groundstate_comparison.png", dpi=150, bbox_inches='tight')