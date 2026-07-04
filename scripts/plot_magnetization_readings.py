"""
Averages the per-reading magnetization tensor(s) written by magnetization_readings.py
(NQS) and/or scripts/dmrg/magnetization_sweep_dmrg.py (DMRG) across readings, and plots
|magnetization| vs. h.

2D mode (default) plots one line per (file, iteration) pair with a shaded +-1 std band.
3D mode plots a surface of mean |magnetization| over h and iteration, for a single file.

At least one of --transformer-readings / --dmrg-readings must be given. If both are given,
a second panel is added showing the transformer readings' relative error against the DMRG
curve (interpolated onto the transformer's h grid).

Run from the repo root:

    uv run python scripts/plot_magnetization_readings.py \\
        --transformer-readings scripts/data/20260701_120000_iter2000_readings.pt
    uv run python scripts/plot_magnetization_readings.py \\
        --transformer-readings scripts/data/20260701_120000_iter500-2000_readings.pt --mode 3d
    uv run python scripts/plot_magnetization_readings.py \\
        --transformer-readings scripts/data/20260701_120000_iter2000_readings.pt \\
        --dmrg-readings scripts/dmrg/data/20260703_010000_dmrg_L30_J2-1.0_J3-1.0_chi150_readings.pt
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

DPI = 300


def _load(data_path: Path) -> dict:
    data = torch.load(data_path, map_location="cpu", weights_only=False)
    readings = data["readings"]
    h_values = data["h_values"].numpy()
    iterations = data["iterations"].numpy()

    if readings.ndim != 3:
        raise SystemExit(
            f"'{data_path}': expected 'readings' with 3 dims (iteration, reading, h), got shape "
            f"{tuple(readings.shape)}."
        )
    n_iterations, _, n_h = readings.shape
    if h_values.ndim != 1 or h_values.shape[0] != n_h:
        raise SystemExit(
            f"'{data_path}': 'h_values' shape {h_values.shape} doesn't match readings' h dimension {n_h}."
        )
    if iterations.ndim != 1 or iterations.shape[0] != n_iterations:
        raise SystemExit(
            f"'{data_path}': 'iterations' shape {iterations.shape} doesn't match readings' "
            f"iteration dimension {n_iterations}."
        )
    return data


def _mean_curve(data: dict, iter_idx: int) -> np.ndarray:
    return data["readings"][iter_idx].mean(dim=0).numpy()


def _plot_2d(ax, data_path: Path, data: dict) -> None:
    h_values = data["h_values"].numpy()
    iterations = data["iterations"].numpy()
    for iter_idx, iteration in enumerate(iterations):
        mean = _mean_curve(data, iter_idx)
        std = data["readings"][iter_idx].std(dim=0).numpy()
        (line,) = ax.plot(h_values, mean, label=f"{data_path.stem} iter {int(iteration)}")
        ax.fill_between(h_values, mean - std, mean + std, alpha=0.2, color=line.get_color())


def _plot_3d(ax, fig, data_path: Path, data: dict) -> None:
    readings = data["readings"]
    h_values = data["h_values"].numpy()
    iterations = data["iterations"].numpy()
    n_iterations = readings.shape[0]
    if n_iterations < 2:
        raise SystemExit(
            f"'{data_path}' has only {n_iterations} iteration(s); --mode 3d needs a range "
            f"(re-run magnetization_readings.py with --iteration-min/--iteration-max)."
        )
    mean = readings.mean(dim=1).numpy()  # (n_iterations, n_h)
    H, ITER = np.meshgrid(h_values, iterations, indexing="xy")
    surf = ax.plot_surface(H, ITER, mean, cmap="viridis", edgecolor="k", linewidth=0.2, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.6, label="|magnetization|")
    ax.view_init(elev=25, azim=-50)
    ax.invert_yaxis()  # later iterations in front, so earlier ones don't obscure them


def _plot_relative_error(
    ax, transformer_path: Path, transformer_data: dict, dmrg_path: Path, dmrg_data: dict
) -> None:
    """
    Plots |transformer - dmrg| / |dmrg| vs h for each transformer iteration. DMRG's mean
    curve (its single iteration/reading) is linearly interpolated onto the transformer's h
    grid so the two can be compared point-by-point even if their h sweeps don't align.
    """
    t_h = transformer_data["h_values"].numpy()
    t_iterations = transformer_data["iterations"].numpy()

    dmrg_h = dmrg_data["h_values"].numpy()
    dmrg_mean = _mean_curve(dmrg_data, 0)
    dmrg_on_t_grid = np.interp(t_h, dmrg_h, dmrg_mean)

    for iter_idx, iteration in enumerate(t_iterations):
        t_mean = _mean_curve(transformer_data, iter_idx)
        rel_error = np.abs(t_mean - dmrg_on_t_grid) / np.abs(dmrg_on_t_grid)
        ax.plot(t_h, rel_error, label=f"{transformer_path.stem} iter {int(iteration)}")

    ax.set_xlabel(transformer_data["h_name"])
    ax.set_ylabel("Relative Error of <|m_z|> with DMRG")
    ax.set_title(f"Relative error vs. DMRG ({dmrg_path.stem})")
    ax.legend()
    ax.grid(alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--transformer-readings",
        type=Path,
        default=None,
        help="NQS *_readings.pt file (from magnetization_readings.py).",
    )
    parser.add_argument(
        "--dmrg-readings",
        type=Path,
        default=None,
        help="DMRG *_readings.pt file (from scripts/dmrg/magnetization_sweep_dmrg.py).",
    )
    parser.add_argument(
        "--mode", choices=["2d", "3d"], default="2d", help="Magnetization panel plot type (default: 2d)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: figures/<wandb run name>_<plot timestamp>.png).",
    )
    args = parser.parse_args()

    if args.transformer_readings is None and args.dmrg_readings is None:
        raise SystemExit("Provide at least one of --transformer-readings / --dmrg-readings.")

    data_paths = [p for p in (args.transformer_readings, args.dmrg_readings) if p is not None]
    show_relative_error = args.transformer_readings is not None and args.dmrg_readings is not None

    if args.mode == "3d" and (show_relative_error or len(data_paths) > 1):
        raise SystemExit("--mode 3d only supports a single input file (depth axis is that file's iterations).")

    loaded = {path: _load(path) for path in data_paths}

    fig = plt.figure(figsize=(14, 6) if show_relative_error else (9, 7) if args.mode == "3d" else (8, 6))
    if show_relative_error:
        ax = fig.add_subplot(1, 2, 1)
        ax_err = fig.add_subplot(1, 2, 2)
    else:
        ax = fig.add_subplot(projection="3d") if args.mode == "3d" else fig.add_subplot()
        ax_err = None

    for data_path in data_paths:
        data = loaded[data_path]
        if args.mode == "2d":
            _plot_2d(ax, data_path, data)
        else:
            _plot_3d(ax, fig, data_path, data)

    last_data = loaded[data_paths[-1]]
    ax.set_xlabel(last_data["h_name"])
    if args.mode == "2d":
        ax.set_ylabel("|magnetization|")
        ax.set_title(f"Magnetization vs. {last_data['h_name']} — L={last_data['L']}")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.set_ylabel("Iteration")
        ax.set_zlabel("|magnetization|")
        ax.set_title(f"Magnetization vs. {last_data['h_name']} and iteration — L={last_data['L']}")

    if show_relative_error:
        _plot_relative_error(
            ax_err,
            args.transformer_readings,
            loaded[args.transformer_readings],
            args.dmrg_readings,
            loaded[args.dmrg_readings],
        )

    fig.tight_layout()

    if args.out is not None:
        out = args.out
    else:
        run_name = last_data.get("wandb_run_name") or "run"
        plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(__file__).parent / "figures" / f"{run_name}_{plot_timestamp}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
