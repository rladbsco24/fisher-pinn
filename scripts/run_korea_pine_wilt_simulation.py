from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_origin_lab.korea_data import (
    build_density_grid,
    compare_observed_and_simulated,
    fit_korea_pine_wilt_pinn,
    load_korea_pine_wilt_points,
    load_manifest,
    simulate_density_rk4,
)


PROVINCE_GEOJSON = ROOT / "data" / "korea_pine_wilt" / "assets" / "skorea_provinces_2018.geojson"


def _format_year_title(year: int, suffix: str = "") -> str:
    return f"{year}{suffix}"


def _geojson_rings(geometry: dict) -> list[np.ndarray]:
    """Return exterior/interior rings from a Polygon or MultiPolygon GeoJSON geometry."""

    if geometry.get("type") == "Polygon":
        polygons = [geometry.get("coordinates", [])]
    elif geometry.get("type") == "MultiPolygon":
        polygons = geometry.get("coordinates", [])
    else:
        return []
    rings: list[np.ndarray] = []
    for polygon in polygons:
        for ring in polygon:
            arr = np.asarray(ring, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] >= 2:
                rings.append(arr[:, :2])
    return rings


def _load_korea_province_geojson(path: Path = PROVINCE_GEOJSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _simplify_ring(ring: np.ndarray, *, max_points: int = 450) -> np.ndarray:
    if max_points <= 0 or len(ring) <= max_points:
        return ring
    idx = np.unique(np.linspace(0, len(ring) - 1, max_points, dtype=int))
    return ring[idx]


def _province_rings(province_geojson: dict, *, max_points: int = 450) -> list[np.ndarray]:
    rings: list[np.ndarray] = []
    for feature in province_geojson.get("features", []):
        for ring in _geojson_rings(feature.get("geometry", {})):
            rings.append(_simplify_ring(ring, max_points=max_points))
    return rings


def _province_bbox(province_rings: list[np.ndarray]) -> tuple[float, float, float, float]:
    if not province_rings:
        return (124.0, 33.0, 132.0, 39.0)
    coords = np.concatenate(province_rings, axis=0)
    return (float(coords[:, 0].min()), float(coords[:, 1].min()), float(coords[:, 0].max()), float(coords[:, 1].max()))


def _grid_edges_lonlat(grid) -> tuple[np.ndarray, np.ndarray]:
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete environments.
        raise RuntimeError(
            "pyproj is required for Korea map GIF export because the compact pine-wilt "
            "points are EPSG:5179 while the committed province map is longitude/latitude. "
            "Install with `pip install -r requirements.txt`."
        ) from exc

    xx, yy = np.meshgrid(grid.x_edges, grid.y_edges, indexing="xy")
    transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


def _field_for_year(year: int, years: np.ndarray, fields: np.ndarray) -> np.ndarray | None:
    lookup = {int(item): idx for idx, item in enumerate(years.tolist())}
    idx = lookup.get(int(year))
    if idx is None:
        return None
    return fields[idx]


def _draw_korea_base(ax, province_rings: list[np.ndarray], *, redraw: bool = False) -> None:
    ax.set_facecolor("#f8faf7")
    color = "#394239" if redraw else "#8b9389"
    linewidth = 0.58 if redraw else 0.52
    alpha = 0.82 if redraw else 0.72
    zorder = 3 if redraw else 0
    for ring in province_rings:
        ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=linewidth, alpha=alpha, zorder=zorder)


def _draw_map_field_panel(
    ax,
    *,
    title: str,
    field: np.ndarray | None,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    province_rings: list[np.ndarray],
    norm: PowerNorm,
    cmap,
    bbox: tuple[float, float, float, float],
    missing_label: str = "not observed",
):
    _draw_korea_base(ax, province_rings)
    mesh = None
    if field is None:
        ax.text(
            0.5,
            0.52,
            missing_label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            color="#5d655e",
            bbox={"facecolor": "white", "edgecolor": "#c5cbc2", "boxstyle": "round,pad=0.35", "alpha": 0.88},
            zorder=4,
        )
    else:
        visible = np.ma.masked_where(np.asarray(field) <= 0.002, np.asarray(field))
        mesh = ax.pcolormesh(
            lon_edges,
            lat_edges,
            visible,
            shading="auto",
            cmap=cmap,
            norm=norm,
            alpha=0.9,
            zorder=2,
        )
        # Redraw boundaries over the density field so the Korea map remains legible.
        _draw_korea_base(ax, province_rings, redraw=True)
    lon_min, lat_min, lon_max, lat_max = bbox
    ax.set_xlim(lon_min - 0.15, lon_max + 0.15)
    ax.set_ylim(lat_min - 0.12, lat_max + 0.12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.tick_params(labelsize=8, colors="#596276")
    ax.grid(color="#d8ddd5", linewidth=0.45, alpha=0.55)
    return mesh


def _save_korea_map_baseline_gif(
    path: Path,
    grid,
    sim_years: np.ndarray,
    sim_fields: np.ndarray,
    *,
    pinn_years: np.ndarray | None = None,
    pinn_fields: np.ndarray | None = None,
    fps: float = 1.2,
    max_frames: int | None = 15,
) -> dict[str, object]:
    """Save observed/RK4/PINN yearly fields as an animated Korea map GIF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    province_geojson = _load_korea_province_geojson()
    province_rings = _province_rings(province_geojson)
    bbox = _province_bbox(province_rings)
    lon_edges, lat_edges = _grid_edges_lonlat(grid)
    frame_years = np.arange(int(grid.years[0]), int(sim_years[-1]) + 1, dtype=int)
    if max_frames is not None and max_frames > 1 and len(frame_years) > max_frames:
        idx = np.unique(np.linspace(0, len(frame_years) - 1, int(max_frames), dtype=int))
        frame_years = frame_years[idx]
    panels = 3 if pinn_years is not None and pinn_fields is not None else 2
    arrays = [grid.density, sim_fields]
    if pinn_fields is not None:
        arrays.append(pinn_fields)
    stacked = np.concatenate([np.asarray(arr, dtype=float).ravel() for arr in arrays])
    positive = stacked[np.isfinite(stacked) & (stacked > 0.0)]
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    vmax = float(np.clip(vmax, 0.05, 1.0))
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    frames: list[Image.Image] = []
    duration_ms = int(round(1000.0 / max(float(fps), 0.1)))
    last_mesh = None
    for year in frame_years.tolist():
        fig, axes = plt.subplots(1, panels, figsize=(5.0 * panels, 5.6), constrained_layout=True)
        axes_arr = np.atleast_1d(axes)
        observed = _field_for_year(year, grid.years, grid.density)
        rk4 = _field_for_year(year, sim_years, sim_fields)
        last_mesh = _draw_map_field_panel(
            axes_arr[0],
            title=f"{year} observed",
            field=observed,
            lon_edges=lon_edges,
            lat_edges=lat_edges,
            province_rings=province_rings,
            norm=norm,
            cmap=cmap,
            bbox=bbox,
        )
        last_mesh = _draw_map_field_panel(
            axes_arr[1],
            title=f"{year} RK4",
            field=rk4,
            lon_edges=lon_edges,
            lat_edges=lat_edges,
            province_rings=province_rings,
            norm=norm,
            cmap=cmap,
            bbox=bbox,
        ) or last_mesh
        if panels == 3:
            pinn = _field_for_year(year, pinn_years, pinn_fields)  # type: ignore[arg-type]
            last_mesh = _draw_map_field_panel(
                axes_arr[2],
                title=f"{year} PINN",
                field=pinn,
                lon_edges=lon_edges,
                lat_edges=lat_edges,
                province_rings=province_rings,
                norm=norm,
                cmap=cmap,
                bbox=bbox,
            ) or last_mesh
        if last_mesh is not None:
            cbar = fig.colorbar(last_mesh, ax=axes_arr, orientation="horizontal", shrink=0.72, pad=0.04)
            cbar.set_label("normalized pine-wilt density, power-scaled for visibility", fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        fig.suptitle(
            "Korea pine-wilt Fisher-KPP baselines on province map",
            fontsize=14,
            fontweight="bold",
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="white")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE, colors=256))

    if not frames:
        raise ValueError("No frames were available for Korea map GIF export.")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)
    preview_path = path.with_name(f"{path.stem}_preview.png")
    frames[0].convert("RGB").save(preview_path)
    return {
        "path": path.name,
        "preview": preview_path.name,
        "frames": int(len(frames)),
        "fps": float(fps),
        "max_frames": None if max_frames is None else int(max_frames),
        "panels": ["observed", "rk4", "pinn"] if panels == 3 else ["observed", "rk4"],
        "vmax": vmax,
    }


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


def _save_pinn_baseline_figure(path: Path, grid, pinn_years: np.ndarray, pinn_fields: np.ndarray) -> None:
    selected_years = [2016, 2018, 2020, 2023]
    selected_years = [year for year in selected_years if year in set(grid.years.tolist())]
    fig, axes = plt.subplots(len(selected_years), 3, figsize=(9.6, 2.6 * len(selected_years)), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)
    extent = [grid.x_edges[0], grid.x_edges[-1], grid.y_edges[0], grid.y_edges[-1]]
    year_to_obs = {int(year): idx for idx, year in enumerate(grid.years.tolist())}
    year_to_pinn = {int(year): idx for idx, year in enumerate(pinn_years.tolist())}
    vmax = max(float(grid.density.max()), float(pinn_fields.max()), 1.0e-12)
    err_vmax = 1.0e-12
    for year in selected_years:
        obs = grid.density[year_to_obs[year]]
        pred = pinn_fields[year_to_pinn[year]]
        err_vmax = max(err_vmax, float(np.max(np.abs(pred - obs))))
    for row, year in enumerate(selected_years):
        obs = grid.density[year_to_obs[year]]
        pred = pinn_fields[year_to_pinn[year]]
        err = np.abs(pred - obs)
        panels = [
            (f"{year} observed", obs, "magma", 0.0, vmax),
            (f"{year} PINN", pred, "magma", 0.0, vmax),
            (f"{year} |error|", err, "viridis", 0.0, err_vmax),
        ]
        for col, (title, field, cmap, vmin, panel_vmax) in enumerate(panels):
            ax = axes_arr[row, col]
            im = ax.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=panel_vmax,
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 2:
                fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle("Korea pine-wilt PINN baseline: observed years", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_baseline_comparison_figure(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        years = np.asarray([row["year"] for row in method_rows], dtype=float)
        rel_l2 = np.asarray([row["relative_l2"] for row in method_rows], dtype=float)
        corr = np.asarray([row["correlation"] for row in method_rows], dtype=float)
        axes[0].plot(years, rel_l2, marker="o", label=method)
        axes[1].plot(years, corr, marker="o", label=method)
    axes[0].set_title("Observed-year relative L2")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("relative L2")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Observed-year spatial correlation")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("correlation")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
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


def _write_baseline_metric_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = ["method", "year", "observed_mean", "simulated_mean", "relative_l2", "correlation"]
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
        land_mask=grid.land_mask,
    )
    rows = compare_observed_and_simulated(grid, sim_years, sim_fields)
    baseline_rows: list[dict[str, float | int | str]] = [dict(method="rk4", **row) for row in rows]

    pinn_result = None
    if not getattr(args, "skip_pinn", False):
        pinn_result = fit_korea_pine_wilt_pinn(
            grid,
            end_year=args.end_year,
            epochs=getattr(args, "pinn_epochs", 120),
            batch_size=getattr(args, "pinn_batch_size", 4096),
            collocation_points=getattr(args, "pinn_collocation_points", 768),
            boundary_points=getattr(args, "pinn_boundary_points", 128),
            lr=getattr(args, "pinn_lr", 2.0e-3),
            data_weight=getattr(args, "pinn_data_weight", 8.0),
            pde_weight=getattr(args, "pinn_pde_weight", 0.05),
            boundary_weight=getattr(args, "pinn_boundary_weight", 0.01),
            sea_weight=getattr(args, "pinn_sea_weight", 2.0),
            diffusion=args.diffusion,
            reaction=getattr(args, "pinn_initial_reaction", 0.20),
            seed=getattr(args, "seed", 7),
        )
        baseline_rows.extend(dict(method="pinn", **row) for row in pinn_result.metrics)

    _write_metric_csv(out_dir / "korea_pine_wilt_metrics.csv", rows)
    _write_baseline_metric_csv(out_dir / "korea_pine_wilt_baseline_metrics.csv", baseline_rows)
    _save_observed_density_figure(out_dir / "observed_density_by_year.png", grid)
    _save_forecast_figure(out_dir / "rk4_forecast_timeline.png", grid, sim_years, sim_fields)
    _save_metric_figure(out_dir / "observed_vs_simulated_metrics.png", rows)
    _save_baseline_comparison_figure(out_dir / "baseline_metric_comparison.png", baseline_rows)
    map_gif_info = _save_korea_map_baseline_gif(
        out_dir / "korea_map_baselines.gif",
        grid,
        sim_years,
        sim_fields,
        pinn_years=pinn_result.years if pinn_result is not None else None,
        pinn_fields=pinn_result.fields if pinn_result is not None else None,
        fps=getattr(args, "map_gif_fps", 1.2),
        max_frames=getattr(args, "map_gif_max_frames", 15),
    )
    if pinn_result is not None:
        _save_pinn_baseline_figure(out_dir / "pinn_baseline_observed_years.png", grid, pinn_result.years, pinn_result.fields)

    summary = {
        "dataset": manifest["dataset"],
        "records": int(len(points.year)),
        "year_min": int(points.year.min()),
        "year_max": int(points.year.max()),
        "grid_size": int(args.grid_size),
        "capacity": float(grid.capacity),
        "land_mask": {
            "enabled": grid.land_mask is not None,
            "land_cells": int(np.sum(grid.land_mask)) if grid.land_mask is not None else int(grid.density.shape[-1] * grid.density.shape[-2]),
            "sea_cells": int(grid.land_mask.size - np.sum(grid.land_mask)) if grid.land_mask is not None else 0,
        },
        "diffusion": float(args.diffusion),
        "reaction": float(args.reaction),
        "steps_per_year": int(args.steps_per_year),
        "end_year": int(args.end_year),
        "mean_relative_l2_observed_years": float(np.nanmean([row["relative_l2"] for row in rows])),
        "mean_correlation_observed_years": float(np.nanmean([row["correlation"] for row in rows])),
        "baselines": {
            "rk4": {
                "mean_relative_l2_observed_years": float(np.nanmean([row["relative_l2"] for row in rows])),
                "mean_correlation_observed_years": float(np.nanmean([row["correlation"] for row in rows])),
            }
        },
        "outputs": [
            "korea_pine_wilt_metrics.csv",
            "korea_pine_wilt_baseline_metrics.csv",
            "observed_density_by_year.png",
            "rk4_forecast_timeline.png",
            "observed_vs_simulated_metrics.png",
            "baseline_metric_comparison.png",
            "korea_map_baselines.gif",
            "korea_map_baselines_preview.png",
        ],
        "map_gif": map_gif_info,
    }
    if pinn_result is not None:
        summary["baselines"]["pinn"] = {
            "status": pinn_result.status,
            "epochs": int(getattr(args, "pinn_epochs", 120)),
            "physics": pinn_result.physics,
            "sea_weight": float(getattr(args, "pinn_sea_weight", 2.0)),
            "mean_relative_l2_observed_years": float(np.nanmean([row["relative_l2"] for row in pinn_result.metrics])),
            "mean_correlation_observed_years": float(np.nanmean([row["correlation"] for row in pinn_result.metrics])),
            "history": pinn_result.history,
        }
        summary["outputs"].append("pinn_baseline_observed_years.png")
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
    parser.add_argument("--skip-pinn", action="store_true", help="Disable the repository PINN baseline.")
    parser.add_argument("--pinn-epochs", type=int, default=120)
    parser.add_argument("--pinn-batch-size", type=int, default=4096)
    parser.add_argument("--pinn-collocation-points", type=int, default=768)
    parser.add_argument("--pinn-boundary-points", type=int, default=128)
    parser.add_argument("--pinn-lr", type=float, default=2.0e-3)
    parser.add_argument("--pinn-data-weight", type=float, default=8.0)
    parser.add_argument("--pinn-pde-weight", type=float, default=0.05)
    parser.add_argument("--pinn-boundary-weight", type=float, default=0.01)
    parser.add_argument("--pinn-sea-weight", type=float, default=2.0)
    parser.add_argument("--pinn-initial-reaction", type=float, default=0.20)
    parser.add_argument("--map-gif-fps", type=float, default=1.2)
    parser.add_argument("--map-gif-max-frames", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
