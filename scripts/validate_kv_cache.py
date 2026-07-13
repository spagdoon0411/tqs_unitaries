"""
Validates the KV-cached sampling path (model.TransformerModel.forward_step +
model_utils.sample_without_weight_cached) against the un-cached path and against exact
diagonalization, for chain lengths 10-14 inclusive.

Two checks:

  Part A -- cache correctness (rigorous, training-independent). For each L, the cached,
    teacher-forced log-amplitude of a batch of fixed configurations must equal the un-cached
    forward's log-amplitude to floating-point tolerance. This is the real proof the cache is
    mathematically sound: it compares deterministic wavefunction amplitudes, not noisy
    samples, so it holds at every L regardless of whether the model was trained there.

  Part B -- magnetization vs. exact diagonalization. For each L, sweep h and compare the
    order-parameter magnitude m(h) = sqrt((1/L^2) sum_ij <Z_i Z_j>) computed three ways:
    exact diagonalization (ground truth), un-cached NQS sampling, and cached NQS sampling.
    The devout-sun model was trained on EVEN system sizes only, so NQS is expected to match
    exact diag at L in {10,12,14} but not necessarily at the untrained odd L in {11,13};
    the cached and un-cached NQS curves, however, must agree with each other at every L
    (that agreement is another view of the Part A result, through the sampler).

Run on the GPU remote from the repo root:

    uv run python scripts/validate_kv_cache.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch.nn.functional as F

from Hamiltonian import IsingThreeSpin
from model import TransformerModel
from model_utils import KVCache, sample_without_weight, sample_without_weight_cached

# ----------------------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------------------
CKPT_DIR = Path("checkpoints/20260703-224128_devout-sun-10")
ITERATION = None  # None -> latest checkpoint

CHAIN_LENGTHS = [10, 11, 12, 13, 14]

# Part A (deterministic cache-equivalence check)
A_H_VALUES = [-1.0, 0.5, 2.0]
A_N_CONFIGS = 64  # random configs per (L, h), plus all-up and all-down
A_TOL = 1e-3  # max |cached - uncached| log-amp allowed (fp32, summed over sites)

# Part B (magnetization sweep vs exact diagonalization)
B_H_VALUES = [-1.0, 0.0, 1.0, 2.0]
B_BATCH = 8192  # samples per (L, h) reading
# ----------------------------------------------------------------------------------------


def load_model_and_config():
    summary_path = CKPT_DIR / "run_summary.json"
    with open(summary_path) as f:
        metadata = json.load(f)
    config = metadata["config"]
    iterations = sorted(int(k) for k in metadata["checkpoints"])
    iteration = ITERATION if ITERATION is not None else iterations[-1]
    ckpt_path = Path(metadata["checkpoints"][str(iteration)])
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / ckpt_path.name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerModel(
        np.array(config["system_sizes"]),
        config["param_dim"],
        config["embedding_size"],
        config["n_head"],
        config["n_hid"],
        config["n_layers"],
        dropout=config["dropout"],
        minibatch=config["minibatch"],
    )
    model.param_range = torch.tensor(config["param_range"])
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, config, iteration


def uncached_logamp(model, configs):
    """Raw autoregressive log|psi|^2-style log-amp (no symmetry) via the un-cached forward.
    Mirrors compute_psi's gather with symmetry=None. configs: (n, B) of {0,1}."""
    n, B = configs.shape
    (log_amp,) = model.forward(configs, compute_phase=False)  # (n+1, B, phys)
    log_amp = log_amp[:-1]  # (n, B, phys)
    n_idx = torch.arange(n).reshape(n, 1)
    b_idx = torch.arange(B).reshape(1, B)
    return log_amp[n_idx, b_idx, configs.to(torch.int64)].sum(dim=0)  # (B,)


def cached_logamp(model, configs):
    """Same quantity via the cached forward_step, teacher-forced on the given configs."""
    n, B = configs.shape
    cache = KVCache(model.n_layers)
    prefix = model.prefix
    n_params = prefix.shape[1]
    prefix_exp = prefix.repeat_interleave(B // n_params, dim=1)
    log_amp = model.forward_step(prefix_exp, cache, start_pos=0)  # (prefix_len, B, phys)
    total = log_amp[-1].gather(1, configs[0].to(torch.int64).view(B, 1)).squeeze(1)
    for i in range(n - 1):
        token = torch.zeros(1, B, model.input_dim)
        token[0, :, : model.phys_dim] = F.one_hot(
            configs[i].to(torch.int64), num_classes=model.phys_dim
        ).to(token.dtype)
        log_amp = model.forward_step(token, cache, start_pos=model.seq_prefix_len + i)
        total = total + log_amp[-1].gather(1, configs[i + 1].to(torch.int64).view(B, 1)).squeeze(1)
    return total  # (B,)


def part_a(model):
    print("\n=== Part A: cached vs un-cached log-amplitude (deterministic) ===")
    rows = []
    worst = 0.0
    for L in CHAIN_LENGTHS:
        size = torch.tensor([L], dtype=torch.int64)
        L_worst = 0.0
        for h in A_H_VALUES:
            model.set_param(system_size=size, param=torch.tensor([float(h)]))
            rand = (torch.rand(L, A_N_CONFIGS) < 0.5).to(torch.get_default_dtype())
            edges = torch.stack([torch.zeros(L), torch.ones(L)], dim=1)  # all-down, all-up
            configs = torch.cat([rand, edges], dim=1)  # (L, A_N_CONFIGS+2)
            diff = (cached_logamp(model, configs) - uncached_logamp(model, configs)).abs().max().item()
            L_worst = max(L_worst, diff)
        worst = max(worst, L_worst)
        rows.append([L, f"{L_worst:.2e}", "PASS" if L_worst < A_TOL else "FAIL"])
    print(tabulate(rows, headers=["L", "max|cached-uncached|", f"< {A_TOL:g}?"], tablefmt="rounded_outline"))
    return worst < A_TOL


def structure_factor_m_exact(H, h, L):
    Hmat = H.full_H(param=h)
    E, psi = eigsh(Hmat, k=1, which="SA")
    psi = psi[:, 0]
    idxs = np.arange(2**L)
    bits = (idxs[:, None] >> np.arange(L)[None, :]) & 1
    spins = 1 - 2 * bits
    probs = np.abs(psi) ** 2
    corr = spins.T @ (spins * probs[:, None])
    return float(np.sqrt(max(corr.mean(), 0.0)))


def structure_factor_m_samples(samples):
    """Uniform-weight zz-structure-factor magnitude from i.i.d. samples (n, B) of {0,1}."""
    spins_pm = 2 * samples.to(torch.get_default_dtype()) - 1
    return (spins_pm.mean(dim=0) ** 2).mean().sqrt().item()


def part_b(model, config):
    print("\n=== Part B: magnetization m(h) vs exact diagonalization ===")
    periodic = config["periodic"]
    ok = True
    for L in CHAIN_LENGTHS:
        size = torch.tensor([L], dtype=torch.int64)
        H = IsingThreeSpin([L], periodic=periodic)
        rows = []
        trained = L % 2 == 0
        max_cache_gap = 0.0
        max_exact_gap = 0.0
        for h in B_H_VALUES:
            model.set_param(system_size=size, param=torch.tensor([float(h)]))
            m_uncached = structure_factor_m_samples(sample_without_weight(model, batch=B_BATCH, symmetry=H.symmetry))
            m_cached = structure_factor_m_samples(sample_without_weight_cached(model, batch=B_BATCH, symmetry=H.symmetry))
            m_exact = structure_factor_m_exact(H, float(h), L)
            max_cache_gap = max(max_cache_gap, abs(m_cached - m_uncached))
            max_exact_gap = max(max_exact_gap, abs(m_cached - m_exact))
            rows.append([f"{h:+.2f}", f"{m_exact:.4f}", f"{m_uncached:.4f}", f"{m_cached:.4f}",
                         f"{abs(m_cached - m_uncached):.4f}", f"{abs(m_cached - m_exact):.4f}"])
        tag = "trained size" if trained else "UNTRAINED odd size (NQS-vs-exact gap expected)"
        print(f"\nL = {L}  ({tag}), boundary = {'periodic' if periodic else 'open'}")
        print(tabulate(
            rows,
            headers=["h", "exact", "NQS uncached", "NQS cached", "|cache-unc|", "|cache-exact|"],
            tablefmt="rounded_outline",
        ))
        # cached must track un-cached everywhere (within MC noise); flag if it doesn't.
        if max_cache_gap > 0.05:
            ok = False
            print(f"  WARNING: cached vs un-cached magnetization gap {max_cache_gap:.4f} exceeds MC tolerance.")
        if trained and max_exact_gap > 0.1:
            print(f"  NOTE: NQS vs exact gap {max_exact_gap:.4f} at trained size (model accuracy, not cache).")
    return ok


def main():
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
    model, config, iteration = load_model_and_config()
    print(f"Loaded {CKPT_DIR.name} @ iteration {iteration}; cuda={torch.cuda.is_available()}")

    a_ok = part_a(model)
    b_ok = part_b(model, config)

    print("\n=== Summary ===")
    print(f"Part A (cache exactness): {'PASS' if a_ok else 'FAIL'}")
    print(f"Part B (cached tracks un-cached): {'PASS' if b_ok else 'FAIL'}")
    if not (a_ok and b_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
