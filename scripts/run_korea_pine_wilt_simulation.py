from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_origin_lab.korea_data import (
    build_density_grid,
    compare_observed_and_simulated,
    load_korea_pine_wilt_points,
    load_manifest,
    simulate_density_rk4,
)


def _format_year_title(year: int, suffix: str = "") -> str:
    return f"{year}{suffix}"


def _save_observed_density_figure(path: Path, grid) -> None:
    n_years = len(grid.years)
    cols = 4
    rows = int(np.ceil(n_years / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    extent = [grid.x_edges[0], grid.x_edges[-1], grid.y_edges[0], grid.y_edges[-1]]
    vmax = max(float(grid.density.max()), 1.0e-12)
    last_image = None
    for idx, year in enumerate(grid.years.tolist()):
        ax = axes_arr[idx]
        last_image = ax.imshow(
            grid.density[idx],
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(_format_year_title(int(year)))
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[n_years:]:
        ax.axis("off")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes_arr[:n_years], shrink=0.82, label="normalized density")
    fig.suptitle("Korea Forest Service pine-wilt observations", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_forecast_figure(path: Path, grid, sim_years: np.ndarray, sim_fields: np.ndarray) -> None:
    target_years = [2016, 2018, 2020, 2023, 2025, 2027, int(sim_years[-1])]
    seen: set[int] = set()
    target_years = [year for year in target_years if year in set(sim_years.tolist()) and not (year in seen or seen.add(year))]
    cols = min(4, len(target_years))
    rows = int(np.ceil(len(target_years) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.1 * rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    extent = [grid.x_edges[0], grid.x_edges[-1], grid.y_edges[0], grid.y_edges[-1]]
    vmax = max(float(sim_fields.max()), float(grid.density.max()), 1.0e-12)
    last_image = None
    year_to_idx = {int(year): idx for idx, year in enumerate(sim_years.tolist())}
    for idx, year in enumerate(target_years):
        ax = axes_arr[idx]
        last_image = ax.imshow(
            sim_fields[year_to_idx[year]],
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        suffix = " observed span" if year <= int(grid.years[-1]) else " forecast"
        ax.set_title(_format_year_title(year, suffix))
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(target_years):]:
        ax.axis("off")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes_arr[: len(target_years)], shrink=0.82, label="normalized density")
    fig.suptitle("2D Fisher-KPP RK4 simulation from 2016 observed density", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_metric_figure(path: Path, rows: list[dict[str, float | int]]) -> None:
    years = np.asarray([row["year"] for row in rows], dtype=float)
    rel_l2 = np.asarray([row["relative_l2"] for row in rows], dtype=float)
    corr = np.asarray([row["correlation"] for row in rows], dtype=float)
    obs_mean = np.asarray([row["observed_mean"] for row in rows], dtype=float)
    sim_mean = np.asarray([row["simulated_mean"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].plot(years, rel_l2, marker="o", label="relative L2")
    axes[0].set_title("Observed-year field error")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("relative L2")
    axes[0].grid(alpha=0.25)
    ax_corr = axes[0].twinx()
    ax_corr.plot(years, corr, marker="s", color="tab:orange", label="correlation")
    ax_corr.set_ylabel("correlation")

    axes[1].plot(years, obs_mean, marker="o", label="observed")
    axes[1].plot(years, sim_mean, marker="o", label="simulated")
    axes[1].set_title("Mean normalized density")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("mean density")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_metric_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fields = ["year", "observed_mean", "simulated_mean", "relative_l2", "correlation"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    points = load_korea_pine_wilt_points()
    grid = build_density_grid(
        points,
        grid_size=args.grid_size,
        pad_m=args.pad_m,
        capacity_percentile=args.capacity_percentile,
        smooth_passes=args.smooth_passes,
    )
    sim_years, sim_fields = simulate_density_rk4(
        grid.density[0],
        start_year=int(grid.years[0]),
        end_year=args.end_year,
        diffusion=args.diffusion,
        reaction=args.reaction,
        steps_per_year=args.steps_per_year,
    )
    rows = compare_observed_and_simulated(grid, sim_years, sim_fields)

    _write_metric_csv(out_dir / "korea_pine_wilt_metrics.csv", rows)
    _save_observed_density_figure(out_dir / "observed_density_by_year.png", grid)
    _save_forecast_figure(out_dir / "rk4_forecast_timeline.png", grid, sim_years, sim_fields)
    _save_metric_figure(out_dir / "observed_vs_simulated_metrics.png", rows)

    summary = {
        "dataset": manifest["dataset"],
        "records": int(len(points.year)),
        "year_min": int(points.year.min()),
        "year_max": int(points.year.max()),
        "grid_size": int(args.grid_size),
        "capacity": float(grid.capacity),
        "diffusion": float(args.diffusion),
        "reaction": float(args.reaction),
        "steps_per_year": int(args.steps_per_year),
        "end_year": int(args.end_year),
        "mean_relative_l2_observed_years": float(np.nanmean([row["relative_l2"] for row in rows])),
        "mean_correlation_observed_years": float(np.nanmean([row["correlation"] for row in rows])),
        "outputs": [
            "korea_pine_wilt_metrics.csv",
            "observed_density_by_year.png",
            "rk4_forecast_timeline.png",
            "observed_vs_simulated_metrics.png",
        ],
    }
    (out_dir / "korea_pine_wilt_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Korea pine-wilt Fisher-KPP RK4 simulation.")
    parser.add_argument("--grid-size", type=int, default=96)
    parser.add_argument("--pad-m", type=float, default=15_000.0)
    parser.add_argument("--capacity-percentile", type=float, default=99.0)
    parser.add_argument("--smooth-passes", type=int, default=1)
    parser.add_argument("--diffusion", type=float, default=0.0015)
    parser.add_argument("--reaction", type=float, default=0.70)
    parser.add_argument("--steps-per-year", type=int, default=80)
    parser.add_argument("--end-year", type=int, default=2030)
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "korea_pine_wilt_csv_simulation")
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
