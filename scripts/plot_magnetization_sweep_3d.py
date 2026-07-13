"""
Renders a 3D magnetization sweep from a readings tensor written by
scripts/magnetization_sweep_3d.py (or any *_readings.pt with the (iteration, reading, h)
schema), as a translucent surface floating above the input plane it is defined over.

The input plane is the (h, iteration) grid the model was evaluated on:
  - the DEPTH axis (into the page) is the h-dimension,
  - the RIGHTWARD axis is the training iteration number,
  - the surface height is the reading-averaged |magnetization| m(h) at each (iteration, h).

A flat translucent grey plane is drawn at z=0 to mark that input plane; the magnetization
surface is drawn semi-transparent above it so the plane, gridlines, and back of the surface
stay visible.

Run from the repo root:

    uv run python scripts/plot_magnetization_sweep_3d.py \\
        --readings scripts/data/<run>_3dsweep_readings.pt
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

DPI = 300
MAGNETIZATION_LABEL = r"$m(h)$"

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


def _load(data_path: Path) -> dict:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    readings = data["readings"]
    if readings.ndim != 3:
        raise SystemExit(
            f"'{data_path}': expected 'readings' with 3 dims (iteration, reading, h), got shape "
            f"{tuple(readings.shape)}."
        )
    n_iterations = readings.shape[0]
    if n_iterations < 2:
        raise SystemExit(
            f"'{data_path}' has only {n_iterations} iteration(s); a 3D sweep needs a range of "
            f"iterations (re-run magnetization_sweep_3d.py over an iteration range)."
        )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--readings",
        type=Path,
        required=True,
        help="Readings *.pt file (from scripts/magnetization_sweep_3d.py).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: figures/<wandb run name>_3dsweep_<plot timestamp>.png).",
    )
    parser.add_argument(
        "--elev", type=float, default=25.0, help="Camera elevation angle in degrees (default: 25)."
    )
    parser.add_argument(
        "--azim", type=float, default=-60.0, help="Camera azimuth angle in degrees (default: -60)."
    )
    args = parser.parse_args()

    data = _load(args.readings)
    h_values = data["h_values"].numpy()
    iterations = data["iterations"].numpy()
    h_name = data.get("h_name", "h")

    # mean over the reading axis -> (n_iterations, n_h); transpose so rows index h (depth)
    # and columns index iteration (rightward), matching the requested axis orientation.
    mean = data["readings"].mean(dim=1).numpy()  # (n_iterations, n_h)
    surf_z = mean.T  # (n_h, n_iterations)
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

    # Translucent magnetization surface above the plane.
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
    ax.view_init(elev=args.elev, azim=args.azim)

    if args.out is not None:
        out = args.out
    else:
        run_name = data.get("wandb_run_name") or "run"
        plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(__file__).parent / "figures" / f"{run_name}_3dsweep_{plot_timestamp}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
