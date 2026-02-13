# import sys
# sys.path.insert(0, "..")

import argparse
import torch
import os

from model import TransformerModel
from model_utils import compute_psi
from Hamiltonian import IsingY, Ising
from Hamiltonian_utils import dec2bin
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--system-size", type=int, required=True)
parser.add_argument("--param", type=float, required=True)
parser.add_argument("--ckpt-path", type=str)
parser.add_argument(
    "--periodic",
    action="store_true",
    help="Use periodic boundary conditions (default: False)",
)
parser.add_argument(
    "--out-dir",
    type=str,
    default="default",
    help="Subdirectory within out/ for output files (created if needed)",
)
parser.add_argument(
    "--model",
    type=str,
    choices=["Ising", "IsingY"],
    required=True,
    help="Hamiltonian model to use for exact diagonalization",
)
parser.add_argument(
    "--low-k",
    type=int,
    default=5,
    help="Number of lowest-energy eigenstates to save (default: 5)",
)
args = parser.parse_args()

MODEL_CLASSES = {"Ising": Ising, "IsingY": IsingY}

torch.set_default_tensor_type(
    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
)

system_sizes = torch.arange(10, 41, 2, device="cpu").reshape(-1, 1)
param_dim = 1
embedding_size = 32
n_head = 8
n_hid = embedding_size
n_layers = 8
dropout = 0
minibatch = 10000

eval_system_size = torch.tensor([args.system_size], dtype=torch.int64, device="cpu")
eval_param = torch.tensor([args.param])

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
model.load_state_dict(torch.load(args.ckpt_path))
model.eval()

n = eval_system_size.prod().item()
model.set_param(eval_system_size, eval_param)

all_dec = torch.arange(2**n, dtype=torch.int64)
all_configs = dec2bin(all_dec, n).T  # (n, 2^n)

with torch.no_grad():
    # Create a Hamiltonian of the requested parameters to get 
    # the symmetry object
    ham = MODEL_CLASSES[args.model]([args.system_size], periodic=args.periodic)
    ham.update_param(args.param)

    log_amp, log_phase = compute_psi(model, all_configs, check_duplicate=False, symmetry=ham.symmetry)

print(log_amp.exp().sum())


def reproduce_psi(log_amp: torch.Tensor, log_phase: torch.Tensor) -> torch.Tensor:
    """
    Produces full imaginary wave function pointwise as a function of log_amps
    and log_phases, as output by the TQS model.
    """

    return torch.sqrt(log_amp.exp()) * torch.exp(1j * log_phase)


psi_tqs = reproduce_psi(log_amp, log_phase)
psi_tqs_cpu = psi_tqs.detach().cpu()

# --- Exact diagonalization ---
hamiltonian = MODEL_CLASSES[args.model]([args.system_size], periodic=args.periodic)
E_ground = hamiltonian.calc_E_ground(args.param)
psi_exact = torch.from_numpy(hamiltonian.psi_ground).to(dtype=torch.complex128)
print(f"Exact ground state energy: {E_ground:.10f}")

# --- k lowest-energy eigenstates ---
full_H = hamiltonian.full_H(args.param)
low_energies, low_states = eigsh(full_H, k=args.low_k, which="SA")
order = low_energies.argsort()
low_energies = torch.from_numpy(low_energies[order])
low_states = torch.from_numpy(low_states[:, order]).to(dtype=torch.complex128)
print(f"Lowest {args.low_k} energies: {low_energies}")

# Inner product between TQS and exact ground states
overlap = torch.vdot(psi_exact, psi_tqs_cpu.to(dtype=torch.complex128))
print(f"|<psi_exact|psi_tqs>| = {overlap.abs().item():.10f}")
print(f"<psi_exact|psi_tqs>  = {overlap}")
self_overlap = torch.vdot(psi_tqs_cpu, psi_tqs_cpu)
print(f"<psi_tqs|psi_tqs>  = {self_overlap}")


def save_torch(psi_tqs, log_amp, log_phase, psi_exact, overlap, low_energies, low_states, out_dir, model_tag, system_size, param):
    """Save wave functions, amplitudes, and phases to .pt, preserving PyTorch tensor format."""
    out_path = os.path.join("out", out_dir)
    os.makedirs(out_path, exist_ok=True)
    pt_path = os.path.join(
        out_path,
        f"tqs_vs_exact_{model_tag}_n{system_size}_param{param:.4f}.pt",
    )
    torch.save(
        {
            "psi_tqs": psi_tqs,
            "log_amp": log_amp,
            "log_phase": log_phase,
            "psi_exact": psi_exact,
            "overlap": overlap,
            "low_energies": low_energies,
            "low_states": low_states,
        },
        pt_path,
    )
    print(f"  Saved wave functions, amplitudes, and phases to {pt_path}")



def plot_ground_state(psi, psi_exact, name, n_spins, periodic, out_dir, tqs_configs,
                      overlap, log_phase):
    """Plot real and imaginary components of ground state in a two-column layout.

    tqs_configs: (n_spins, 2^n) torch tensor — the exact spin configurations
        passed into the TQS model, as produced by dec2bin(...).T.
    overlap: complex — inner product <psi_exact|psi_tqs>.
    log_phase: 1-D torch tensor — raw log-phase output from the TQS model.
    """
    out_path = os.path.join("out", out_dir)
    os.makedirs(out_path, exist_ok=True)

    # Save wave functions, amplitudes, and phases to the same directory as the plot
    save_torch(psi, log_amp.detach().cpu(), log_phase, psi_exact, overlap, low_energies, low_states, out_dir, name.lower(), n_spins, args.param)

    fig = plt.figure(figsize=(18, 26))
    gs = fig.add_gridspec(8, 2, height_ratios=[2, 2, 2, 2, 2, 2, 1.5, 0.6], hspace=0.35, wspace=0.25)

    indices = torch.arange(len(psi), device="cpu")
    bc_str = "periodic" if periodic else "open"
    n_basis = len(psi)
    x_margin = max(n_basis * 0.02, 1.0)
    xlim = (-x_margin, n_basis - 1 + x_margin)

    # TQS basis: the exact spin chains passed into the model
    basis_tqs = tqs_configs  # (n_spins, 2^n), from dec2bin

    # Exact diag basis: derived from Kronecker product construction of full_H.
    # In full_H, the inner loop `for j in range(n)` builds
    #   kron(Op_0, Op_1, ..., Op_{n-1})
    # so spin j is the j-th Kronecker factor (from left). The leftmost factor
    # controls the MSB of the matrix index. Therefore basis state index k has
    # spin j in state given by the j-th bit of k (MSB-first).
    basis_exact = torch.zeros((n_spins, n_basis), device="cpu")
    for k in range(n_basis):
        for j in range(n_spins):
            basis_exact[j, k] = (k >> (n_spins - 1 - j)) & 1

    # --- Compute shared y-limits for each paired row ---
    def _symmetric_ylim(*arrays):
        vals = torch.cat([a.flatten() for a in arrays])
        lo, hi = vals.min().item(), vals.max().item()
        margin = max((hi - lo) * 0.05, 1e-8)
        return (lo - margin, hi + margin)

    ylim_real = _symmetric_ylim(psi.real, psi_exact.real)
    ylim_imag = _symmetric_ylim(psi.imag, psi_exact.imag)
    ylim_prob = _symmetric_ylim(psi.abs() ** 2, psi_exact.abs() ** 2)
    ylim_angle = _symmetric_ylim(torch.angle(psi), torch.angle(psi_exact))
    ylim_logphase_raw = _symmetric_ylim(log_phase, torch.angle(psi_exact))
    ylim_logphase_mod = _symmetric_ylim(log_phase % (2 * torch.pi), torch.angle(psi_exact))

    # --- Left column: TQS ---
    ax_real_tqs = fig.add_subplot(gs[0, 0])
    ax_real_tqs.vlines(indices, 0, psi.real, color="b", linewidth=1.0)
    ax_real_tqs.set_ylabel("Real part")
    ax_real_tqs.set_title(f"TQS — {name} Ground State (n={n_spins}, {bc_str} BC)")
    ax_real_tqs.grid(True, alpha=0.3)
    ax_real_tqs.set_xticklabels([])
    ax_real_tqs.set_xlim(xlim)
    ax_real_tqs.set_ylim(ylim_real)

    ax_imag_tqs = fig.add_subplot(gs[1, 0])
    ax_imag_tqs.vlines(indices, 0, psi.imag, color="r", linewidth=1.0)
    ax_imag_tqs.set_ylabel("Imaginary part")
    ax_imag_tqs.grid(True, alpha=0.3)
    ax_imag_tqs.set_xticklabels([])
    ax_imag_tqs.set_xlim(xlim)
    ax_imag_tqs.set_ylim(ylim_imag)

    ax_prob_tqs = fig.add_subplot(gs[2, 0])
    ax_prob_tqs.vlines(indices, 0, psi.abs() ** 2, color="g", linewidth=1.0)
    ax_prob_tqs.set_ylabel(r"$|\psi|^2$")
    ax_prob_tqs.grid(True, alpha=0.3)
    ax_prob_tqs.set_xticklabels([])
    ax_prob_tqs.set_xlim(xlim)
    ax_prob_tqs.set_ylim(ylim_prob)

    # --- Phase reference lines ---
    import numpy as np
    phase_hlines = [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi]

    def _add_phase_hlines(ax):
        for hl in phase_hlines:
            ax.axhline(hl, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # --- Row 3: raw log_phase (left) vs exact arg(psi) (right) ---
    ax_logphase_raw = fig.add_subplot(gs[3, 0])
    ax_logphase_raw.scatter(indices, log_phase, color="teal", s=4)
    _add_phase_hlines(ax_logphase_raw)
    ax_logphase_raw.set_ylabel(r"$\log\,\phi$")
    ax_logphase_raw.set_title(r"TQS raw log-phase")
    ax_logphase_raw.grid(True, alpha=0.3)
    ax_logphase_raw.set_xticklabels([])
    ax_logphase_raw.set_xlim(xlim)
    ax_logphase_raw.set_ylim(ylim_logphase_raw)

    ax_exact_for_raw = fig.add_subplot(gs[3, 1])
    ax_exact_for_raw.scatter(indices, torch.angle(psi_exact), color="darkorange", s=4)
    _add_phase_hlines(ax_exact_for_raw)
    ax_exact_for_raw.set_ylabel(r"$\mathrm{arg}(\psi)$")
    ax_exact_for_raw.set_title(r"Exact $\mathrm{arg}(\psi)$")
    ax_exact_for_raw.grid(True, alpha=0.3)
    ax_exact_for_raw.set_xticklabels([])
    ax_exact_for_raw.set_xlim(xlim)
    ax_exact_for_raw.set_ylim(ylim_logphase_raw)

    # --- Row 4: log_phase mod 2pi (left) vs exact arg(psi) (right) ---
    ax_logphase_mod = fig.add_subplot(gs[4, 0])
    ax_logphase_mod.scatter(indices, log_phase % (2 * torch.pi), color="purple", s=4)
    _add_phase_hlines(ax_logphase_mod)
    ax_logphase_mod.set_ylabel(r"$\log\,\phi\;\mathrm{mod}\;2\pi$")
    ax_logphase_mod.set_title(r"TQS log-phase $\mathrm{mod}\;2\pi$")
    ax_logphase_mod.grid(True, alpha=0.3)
    ax_logphase_mod.set_xticklabels([])
    ax_logphase_mod.set_xlim(xlim)
    ax_logphase_mod.set_ylim(ylim_logphase_mod)

    ax_exact_for_mod = fig.add_subplot(gs[4, 1])
    ax_exact_for_mod.scatter(indices, torch.angle(psi_exact), color="darkorange", s=4)
    _add_phase_hlines(ax_exact_for_mod)
    ax_exact_for_mod.set_ylabel(r"$\mathrm{arg}(\psi)$")
    ax_exact_for_mod.set_title(r"Exact $\mathrm{arg}(\psi)$")
    ax_exact_for_mod.grid(True, alpha=0.3)
    ax_exact_for_mod.set_xticklabels([])
    ax_exact_for_mod.set_xlim(xlim)
    ax_exact_for_mod.set_ylim(ylim_logphase_mod)

    # --- Row 5: arg(psi) for TQS (left) and Exact (right) ---
    ax_angle_tqs = fig.add_subplot(gs[5, 0])
    ax_angle_tqs.scatter(indices, torch.angle(psi), color="darkorange", s=4)
    _add_phase_hlines(ax_angle_tqs)
    ax_angle_tqs.set_ylabel(r"$\mathrm{arg}(\psi)$")
    ax_angle_tqs.set_title(r"TQS $\mathrm{arg}(\psi)$")
    ax_angle_tqs.grid(True, alpha=0.3)
    ax_angle_tqs.set_xticklabels([])
    ax_angle_tqs.set_xlim(xlim)
    ax_angle_tqs.set_ylim(ylim_angle)

    ax_angle_exact = fig.add_subplot(gs[5, 1])
    ax_angle_exact.scatter(indices, torch.angle(psi_exact), color="darkorange", s=4)
    _add_phase_hlines(ax_angle_exact)
    ax_angle_exact.set_ylabel(r"$\mathrm{arg}(\psi)$")
    ax_angle_exact.set_title(r"Exact $\mathrm{arg}(\psi)$")
    ax_angle_exact.grid(True, alpha=0.3)
    ax_angle_exact.set_xticklabels([])
    ax_angle_exact.set_xlim(xlim)
    ax_angle_exact.set_ylim(ylim_angle)

    ax_basis_tqs = fig.add_subplot(gs[6, 0])
    im_tqs = ax_basis_tqs.imshow(basis_tqs, aspect='auto', cmap='bwr',
                                  interpolation='nearest', vmin=0, vmax=1,
                                  extent=[-0.5, n_basis - 0.5, n_spins - 0.5, -0.5])
    ax_basis_tqs.set_ylabel("Spin index")
    ax_basis_tqs.set_xlabel("Basis state index")
    ax_basis_tqs.set_yticks(range(n_spins))
    ax_basis_tqs.set_yticklabels(range(n_spins))
    ax_basis_tqs.set_xlim(xlim)

    # --- Right column: Exact ---
    ax_real_exact = fig.add_subplot(gs[0, 1])
    ax_real_exact.vlines(indices, 0, psi_exact.real, color="b", linewidth=1.0)
    ax_real_exact.set_ylabel("Real part")
    ax_real_exact.set_title(f"Exact — {name} Ground State (n={n_spins}, {bc_str} BC)")
    ax_real_exact.grid(True, alpha=0.3)
    ax_real_exact.set_xticklabels([])
    ax_real_exact.set_xlim(xlim)
    ax_real_exact.set_ylim(ylim_real)

    ax_imag_exact = fig.add_subplot(gs[1, 1])
    ax_imag_exact.vlines(indices, 0, psi_exact.imag, color="r", linewidth=1.0)
    ax_imag_exact.set_ylabel("Imaginary part")
    ax_imag_exact.grid(True, alpha=0.3)
    ax_imag_exact.set_xticklabels([])
    ax_imag_exact.set_xlim(xlim)
    ax_imag_exact.set_ylim(ylim_imag)

    ax_prob_exact = fig.add_subplot(gs[2, 1])
    ax_prob_exact.vlines(indices, 0, psi_exact.abs() ** 2, color="g", linewidth=1.0)
    ax_prob_exact.set_ylabel(r"$|\psi|^2$")
    ax_prob_exact.grid(True, alpha=0.3)
    ax_prob_exact.set_xticklabels([])
    ax_prob_exact.set_xlim(xlim)
    ax_prob_exact.set_ylim(ylim_prob)

    ax_basis_exact = fig.add_subplot(gs[6, 1])
    im_exact = ax_basis_exact.imshow(basis_exact, aspect='auto', cmap='bwr',
                                      interpolation='nearest', vmin=0, vmax=1,
                                      extent=[-0.5, n_basis - 0.5, n_spins - 0.5, -0.5])
    ax_basis_exact.set_ylabel("Spin index")
    ax_basis_exact.set_xlabel("Basis state index")
    ax_basis_exact.set_yticks(range(n_spins))
    ax_basis_exact.set_yticklabels(range(n_spins))
    ax_basis_exact.set_xlim(xlim)

    # Shared colorbar at the bottom
    cbar = fig.colorbar(im_exact, ax=[ax_basis_tqs, ax_basis_exact],
                         orientation='horizontal', pad=0.15, aspect=50)
    cbar.set_label('Spin state (0=↓, 1=↑)', rotation=0)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['↓', '↑'])

    # Overlap annotation spanning both columns
    ax_overlap = fig.add_subplot(gs[7, :])
    ax_overlap.axis('off')
    overlap_abs = overlap.abs().item()
    overlap_re = overlap.real.item()
    overlap_im = overlap.imag.item()
    sign = '+' if overlap_im >= 0 else '-'
    prob_sum_tqs = psi.abs().pow(2).sum().item()
    latex_str = (
        r"$\langle \psi_{\mathrm{exact}} | \psi_{\mathrm{TQS}} \rangle"
        rf" = {overlap_re:.6f} {sign} {abs(overlap_im):.6f}\,i"
        r",\quad"
        r"|\langle \psi_{\mathrm{exact}} | \psi_{\mathrm{TQS}} \rangle|"
        rf" = {overlap_abs:.6f}"
        r",\quad"
        r"\sum |\psi_{\mathrm{TQS}}|^2"
        rf" = {prob_sum_tqs:.6f}$"
    )
    ax_overlap.text(0.5, 0.5, latex_str, transform=ax_overlap.transAxes,
                    ha='center', va='center', fontsize=14)

    fname = f'tqs_vs_exact_{name.lower().replace(" ", "_")}.png'
    save_path = os.path.join(out_path, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {save_path}")


plot_ground_state(
    psi_tqs_cpu, psi_exact, args.model, n, args.periodic, args.out_dir,
    tqs_configs=all_configs.cpu(),
    overlap=overlap,
    log_phase=log_phase.detach().cpu(),
)
