"""
Renders a magnetization sweep from a readings tensor written by
scripts/magnetization_sweep_3d.py (or any *_readings.pt with the (iteration, reading, h)
schema) in one of three representations, chosen with --representation:

  surface (default) -- a translucent 3D surface floating above the input plane it is
      defined over. The input plane is the (h, iteration) grid the model was evaluated on:
      the DEPTH axis is h, the RIGHTWARD axis is the training iteration number, and the
      surface height is the reading-averaged |magnetization| m(h). A flat translucent grey
      plane at z=0 marks that input plane.

  heatmap -- a 2D image of the same reading-averaged m(h), with iteration on the x-axis, h
      on the y-axis, and magnetization mapped to color.

  overlay -- every iteration's m(h)-vs-h curve overlaid on one 2D plot. TQS curves are red,
      with per-curve opacity scaling linearly from 0 (earliest iteration) to 1 (latest
      iteration), so training progress reads as the curve "fading in". If --dmrg-readings
      is supplied, the DMRG reference curve is overlaid in solid blue.

--dmrg-readings, --elev, and --azim apply only to their relevant representations (see help).

Run from the repo root:

    uv run python scripts/plot_magnetization_sweep_3d.py \\
        --readings scripts/data/<run>_3dsweep_readings.pt
    uv run python scripts/plot_magnetization_sweep_3d.py \\
        --readings scripts/data/<run>_3dsweep_readings.pt --representation overlay \\
        --dmrg-readings scripts/data/<dmrg run>_readings.pt
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

DPI = 300
MAGNETIZATION_LABEL = r"$m(h)$"

TQS_COLOR = "tab:red"
DMRG_COLOR = "tab:blue"

ALPHA_FLOOR = 0.1  # minimum per-curve opacity in the overlay, so early iterations stay visible

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 19,
        "axes.labelsize": 18,
        "legend.fontsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }
)


def _load(data_path: Path, require_iteration_range: bool) -> dict:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    readings = data["readings"]
    if readings.ndim != 3:
        raise SystemExit(
            f"'{data_path}': expected 'readings' with 3 dims (iteration, reading, h), got shape "
            f"{tuple(readings.shape)}."
        )
    n_iterations = readings.shape[0]
    if require_iteration_range and n_iterations < 2:
        raise SystemExit(
            f"'{data_path}' has only {n_iterations} iteration(s); this representation needs a "
            f"range of iterations (re-run magnetization_sweep_3d.py over an iteration range)."
        )
    return data


def _iteration_alphas(iterations: np.ndarray) -> np.ndarray:
    """Exponential opacity ramp from ALPHA_FLOOR (smallest iteration) to 1 (largest
    iteration), floored so early curves stay visible rather than fading fully to
    transparent. Geometric interpolation ALPHA_FLOOR**(1 - normalized) keeps the result
    within [ALPHA_FLOOR, 1] while ramping up exponentially with iteration."""
    if len(iterations) == 1:
        return np.ones(1)
    lo, hi = iterations.min(), iterations.max()
    if hi == lo:
        return np.ones_like(iterations, dtype=float)
    normalized = (iterations - lo) / (hi - lo)
    return ALPHA_FLOOR ** (1.0 - normalized)


def _plot_surface(data: dict, elev: float, azim: float):
    h_values = data["h_values"].numpy()
    iterations = data["iterations"].numpy()
    h_name = data.get("h_name", "h")

    # mean over readings -> (n_iterations, n_h); transpose so rows index h (depth) and
    # columns index iteration (rightward), matching the requested axis orientation.
    surf_z = data["readings"].mean(dim=1).numpy().T  # (n_h, n_iterations)
    ITER, HH = np.meshgrid(iterations, h_values)  # both (n_h, n_iterations)

    fig = plt.figure(figsize=(10, 7.5))
    ax = fig.add_subplot(projection="3d")

    # Flat translucent input plane at z=0, spanning the full (iteration, h) domain.
    plane_iter, plane_h = np.meshgrid(
        [iterations.min(), iterations.max()], [h_values.min(), h_values.max()]
    )
    ax.plot_surface(
        plane_iter, plane_h, np.zeros_like(plane_iter, dtype=float),
        color="grey", alpha=0.12, zorder=0, shade=False,
    )

    surf = ax.plot_surface(
        ITER, HH, surf_z, cmap="viridis", alpha=0.7, linewidth=0.2, edgecolor="k",
        antialiased=True, zorder=1,
    )
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label=MAGNETIZATION_LABEL)

    ax.set_xlabel("Iteration")  # rightward axis
    ax.set_ylabel(h_name)  # depth axis
    ax.set_zlabel(MAGNETIZATION_LABEL)
    ax.set_zlim(bottom=0.0)
    ax.set_title(f"Magnetization vs. {h_name} and Iteration — L={data['L']}")
    ax.view_init(elev=elev, azim=azim)
    return fig


def _plot_heatmap(data: dict):
    h_values = data["h_values"].numpy()
    iterations = data["iterations"].numpy()
    h_name = data.get("h_name", "h")

    mean = data["readings"].mean(dim=1).numpy()  # (n_iterations, n_h)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    # x = iteration (rightward), y = h; pcolormesh wants Z shaped (len(y), len(x)) = (n_h, n_it).
    mesh = ax.pcolormesh(iterations, h_values, mean.T, cmap="viridis", shading="nearest")
    fig.colorbar(mesh, ax=ax, label=MAGNETIZATION_LABEL)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(h_name)
    ax.set_title(f"Magnetization vs. {h_name} and Iteration — L={data['L']}")
    return fig


def _plot_overlay(data: dict, dmrg_data: dict | None):
    h_values = data["h_values"].numpy()
    iterations = data["iterations"].numpy()
    h_name = data.get("h_name", "h")

    mean = data["readings"].mean(dim=1).numpy()  # (n_iterations, n_h)
    alphas = _iteration_alphas(iterations)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    for iter_idx, alpha in enumerate(alphas):
        ax.plot(h_values, mean[iter_idx], color=TQS_COLOR, alpha=float(alpha), zorder=2)

    handles = [
        Line2D(
            [0], [0], color=TQS_COLOR,
            label=f"TQS reading (iter {int(iterations.min())}→{int(iterations.max())}, "
            f"opacity ∝ iteration)",
        )
    ]

    if dmrg_data is not None:
        dmrg_h = dmrg_data["h_values"].numpy()
        # DMRG's "iteration" axis (chi) and "reading" axis (sweep direction) are typically
        # singleton; average over readings and draw each stored curve in blue.
        dmrg_mean = dmrg_data["readings"].mean(dim=1).numpy()  # (n_dmrg_iter, n_h)
        for row in dmrg_mean:
            ax.plot(dmrg_h, row, color=DMRG_COLOR, linewidth=2, zorder=3)
        handles.append(Line2D([0], [0], color=DMRG_COLOR, linewidth=2, label="DMRG reading"))

    ax.set_xlabel(h_name)
    ax.set_ylabel(MAGNETIZATION_LABEL)
    ax.set_title(f"Magnetization vs. {h_name} — L={data['L']}")
    ax.legend(handles=handles)
    ax.grid(alpha=0.3)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--readings",
        type=Path,
        required=True,
        help="TQS readings *.pt file (from scripts/magnetization_sweep_3d.py).",
    )
    parser.add_argument(
        "--representation",
        choices=["surface", "heatmap", "overlay"],
        default="surface",
        help="Plot representation (default: surface).",
    )
    parser.add_argument(
        "--dmrg-readings",
        type=Path,
        default=None,
        help="Optional DMRG *_readings.pt to overlay in blue. Only used by --representation overlay.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: figures/<wandb run name>_<representation>_<timestamp>.png).",
    )
    parser.add_argument(
        "--elev", type=float, default=25.0,
        help="Camera elevation in degrees (surface representation only; default: 25).",
    )
    parser.add_argument(
        "--azim", type=float, default=-60.0,
        help="Camera azimuth in degrees (surface representation only; default: -60).",
    )
    args = parser.parse_args()

    if args.dmrg_readings is not None and args.representation != "overlay":
        raise SystemExit("--dmrg-readings is only supported with --representation overlay.")

    data = _load(args.readings, require_iteration_range=True)

    if args.representation == "surface":
        fig = _plot_surface(data, args.elev, args.azim)
    elif args.representation == "heatmap":
        fig = _plot_heatmap(data)
    else:
        dmrg_data = (
            _load(args.dmrg_readings, require_iteration_range=False)
            if args.dmrg_readings is not None
            else None
        )
        fig = _plot_overlay(data, dmrg_data)

    if args.out is not None:
        out = args.out
    else:
        run_name = data.get("wandb_run_name") or "run"
        plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(__file__).parent / "figures" / f"{run_name}_{args.representation}_{plot_timestamp}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    print(f"Wrote {out}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
