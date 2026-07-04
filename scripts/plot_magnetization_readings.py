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
from scipy.interpolate import PchipInterpolator

DPI = 300

MAGNETIZATION_LABEL = r"$\sqrt{\left\langle\left(\frac{1}{L}\sum_i \sigma^z_i\right)^{2}\right\rangle}$"


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


_CURVE_STYLES = {
    # kind -> (linestyle, color, linewidth, zorder). Higher zorder draws on top.
    "DMRG": ("--", "tab:blue", 1.5, 2),
    "Exact diagonalization": ("-", "tab:orange", 3, 1),
}


def _plot_2d(ax, kind: str, data: dict) -> None:
    h_values = data["h_values"].numpy()
    iterations = data["iterations"].numpy()
    n_readings = data["readings"].shape[1]
    for iter_idx, iteration in enumerate(iterations):
        mean = _mean_curve(data, iter_idx)
        label = f"{kind} reading" if len(iterations) == 1 else f"{kind} reading (iter {int(iteration)})"
        if kind in _CURVE_STYLES:
            linestyle, color, linewidth, zorder = _CURVE_STYLES[kind]
            h_dense = np.linspace(h_values.min(), h_values.max(), 400)
            mean_dense = PchipInterpolator(h_values, mean)(h_dense)
            ax.plot(
                h_dense, mean_dense, linestyle=linestyle, color=color, linewidth=linewidth, zorder=zorder,
                label=label,
            )
        else:
            std = data["readings"][iter_idx].std(dim=0).numpy() if n_readings > 1 else np.zeros_like(mean)
            ax.errorbar(
                h_values, mean, yerr=std, fmt="o", color="tab:red", markersize=3, capsize=2, elinewidth=1,
                zorder=3, label=label,
            )


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
    fig.colorbar(surf, ax=ax, shrink=0.6, label=MAGNETIZATION_LABEL)
    ax.view_init(elev=25, azim=-50)
    ax.invert_yaxis()  # later iterations in front, so earlier ones don't obscure them


def _plot_relative_error(ax, transformer_data: dict, reference_data: dict, reference_label: str) -> None:
    """
    Plots |transformer - reference| / |reference| vs h for each transformer iteration. The
    reference's mean curve (its single iteration/reading) is linearly interpolated onto the
    transformer's h grid so the two can be compared point-by-point even if their h sweeps
    don't align. `reference_data` is exact-diagonalization data if available, else DMRG.
    """
    t_h = transformer_data["h_values"].numpy()
    t_iterations = transformer_data["iterations"].numpy()

    ref_h = reference_data["h_values"].numpy()
    ref_mean = _mean_curve(reference_data, 0)
    ref_on_t_grid = np.interp(t_h, ref_h, ref_mean)

    for iter_idx, iteration in enumerate(t_iterations):
        t_mean = _mean_curve(transformer_data, iter_idx)
        rel_error = np.abs(t_mean - ref_on_t_grid) / np.abs(ref_on_t_grid)
        label = "Relative error" if len(t_iterations) == 1 else f"Relative error (iter {int(iteration)})"
        ax.plot(t_h, rel_error, linestyle=":", marker="o", markersize=3, label=label)

    ax.set_xlabel(transformer_data["h_name"])
    ax.set_ylabel("Relative error")
    ax.set_title(f"Relative Error of Magnetization, vs. {reference_label}")
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
        help="DMRG *_readings.pt file (from scripts/magnetization_sweep_dmrg.py).",
    )
    parser.add_argument(
        "--exact-readings",
        type=Path,
        default=None,
        help="Exact-diagonalization *_readings.pt file (from scripts/magnetization_sweep_exact.py). "
        "Drawn as a solid orange line, under the DMRG curve.",
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
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Zoom the h axis to this range for display only (doesn't affect the underlying data).",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Subfolder under figures/ to write into (e.g. 'v3'), grouping one comparison "
        "setup's plots together. Ignored if --out is given.",
    )
    parser.add_argument(
        "--highlight-region",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Shade this h range on the magnetization panel (low-opacity, DMRG-blue), labeled "
        "'Region of interest' in the legend.",
    )
    parser.add_argument(
        "--title-suffix",
        type=str,
        default=None,
        help="Text appended to the magnetization panel's title (e.g. '(Region of Interest)').",
    )
    args = parser.parse_args()

    if args.transformer_readings is None and args.dmrg_readings is None and args.exact_readings is None:
        raise SystemExit("Provide at least one of --transformer-readings / --dmrg-readings / --exact-readings.")

    data_paths = [
        p for p in (args.exact_readings, args.dmrg_readings, args.transformer_readings) if p is not None
    ]
    reference_path = args.exact_readings if args.exact_readings is not None else args.dmrg_readings
    reference_label = "Exact Diagonalization" if args.exact_readings is not None else "DMRG"
    show_relative_error = args.transformer_readings is not None and reference_path is not None

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
        if data_path == args.exact_readings:
            kind = "Exact diagonalization"
        elif data_path == args.dmrg_readings:
            kind = "DMRG"
        else:
            kind = "TQS"
        if args.mode == "2d":
            _plot_2d(ax, kind, data)
        else:
            _plot_3d(ax, fig, data_path, data)

    last_data = loaded[data_paths[-1]]
    ax.set_xlabel(last_data["h_name"])
    if args.mode == "2d":
        if args.highlight_region is not None:
            ax.axvspan(
                args.highlight_region[0], args.highlight_region[1], color="tab:blue", alpha=0.15,
                label="Region of interest", zorder=0,
            )
        ax.set_ylabel(MAGNETIZATION_LABEL)
        title = f"Magnetization vs. {last_data['h_name']} — L={last_data['L']}"
        if args.title_suffix is not None:
            title = f"{title} {args.title_suffix}"
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.set_ylabel("Iteration")
        ax.set_zlabel(MAGNETIZATION_LABEL)
        ax.set_title(f"Magnetization vs. {last_data['h_name']} and Iteration — L={last_data['L']}")

    if show_relative_error:
        _plot_relative_error(ax_err, loaded[args.transformer_readings], loaded[reference_path], reference_label)

    if args.xlim is not None:
        ax.set_xlim(args.xlim)
        if ax_err is not None:
            ax_err.set_xlim(args.xlim)

    fig.tight_layout()

    if args.out is not None:
        out = args.out
    else:
        run_name = last_data.get("wandb_run_name") or "run"
        plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figures_dir = Path(__file__).parent / "figures"
        if args.version is not None:
            figures_dir = figures_dir / args.version
        out = figures_dir / f"{run_name}_{plot_timestamp}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
