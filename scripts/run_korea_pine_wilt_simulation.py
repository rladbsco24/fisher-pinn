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
from matplotlib.ticker import MaxNLocator
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_origin_lab.korea_data import (
    build_density_grid,
    compare_observed_and_simulated,
    fit_korea_pine_wilt_pinn,
    korea_grid_time_label,
    korea_grid_time_values,
    korea_equivalent_dr_pairs,
    korea_physics_prior_from_normalized,
    korea_physics_prior_from_physical,
    load_korea_pine_wilt_action_time_points,
    load_korea_pine_wilt_points,
    load_manifest,
    simulate_density_rk4,
    simulate_density_rk4_at_times,
)
from fisher_origin_lab.plotting import ERROR_CMAP, FIELD_CMAP, TOKENS


PROVINCE_GEOJSON = ROOT / "data" / "korea_pine_wilt" / "assets" / "skorea_provinces_2018.geojson"


def _format_year_title(year: int, suffix: str = "") -> str:
    return f"{year}{suffix}"


def _format_grid_time_title(grid, period: int, suffix: str = "") -> str:
    return korea_grid_time_label(grid, int(period), suffix)


def _save_figure_with_padding(fig: plt.Figure, path: Path, *, dpi: int = 180) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.10)


def _add_vertical_colorbar(fig: plt.Figure, image, ax, *, shrink: float = 0.82):
    cbar = fig.colorbar(image, ax=ax, shrink=shrink, pad=0.035)
    _style_colorbar(cbar, horizontal=False)
    return cbar


def _style_colorbar(cbar, *, horizontal: bool) -> None:
    cbar.ax.tick_params(labelsize=8, pad=3, colors=TOKENS["muted"])
    cbar.outline.set_edgecolor(TOKENS["axis"])
    cbar.outline.set_linewidth(0.8)
    if horizontal:
        cbar.ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    else:
        cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.yaxis.set_ticks_position("right")


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
    cmap = plt.get_cmap(FIELD_CMAP).copy()
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
            title=f"{_format_grid_time_title(grid, year)} observed",
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
            title=f"{_format_grid_time_title(grid, year)} RK4",
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
                title=f"{_format_grid_time_title(grid, year)} PINN",
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
            _style_colorbar(cbar, horizontal=True)
        fig.suptitle(
            "Korea pine-wilt Fisher-KPP baselines on province map",
            fontsize=14,
            fontweight="bold",
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="white", bbox_inches="tight", pad_inches=0.10)
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


def _save_korea_error_gif(
    path: Path,
    grid,
    sim_years: np.ndarray,
    sim_fields: np.ndarray,
    *,
    pinn_years: np.ndarray | None = None,
    pinn_fields: np.ndarray | None = None,
    fps: float = 1.2,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Save observed-period RK4/PINN absolute-error maps as an animated Korea GIF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    province_geojson = _load_korea_province_geojson()
    province_rings = _province_rings(province_geojson)
    bbox = _province_bbox(province_rings)
    lon_edges, lat_edges = _grid_edges_lonlat(grid)
    frame_years = np.asarray(grid.years, dtype=int)
    if max_frames is not None and max_frames > 1 and len(frame_years) > max_frames:
        idx = np.unique(np.linspace(0, len(frame_years) - 1, int(max_frames), dtype=int))
        frame_years = frame_years[idx]

    error_arrays = []
    for year in frame_years.tolist():
        observed = _field_for_year(year, grid.years, grid.density)
        rk4 = _field_for_year(year, sim_years, sim_fields)
        if observed is not None and rk4 is not None:
            error_arrays.append(np.abs(np.asarray(rk4) - np.asarray(observed)))
        if pinn_years is not None and pinn_fields is not None:
            pinn = _field_for_year(year, pinn_years, pinn_fields)
            if observed is not None and pinn is not None:
                error_arrays.append(np.abs(np.asarray(pinn) - np.asarray(observed)))
    if error_arrays:
        stacked = np.concatenate([arr.ravel() for arr in error_arrays])
        finite = stacked[np.isfinite(stacked)]
        err_vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
    else:
        err_vmax = 1.0
    err_vmax = float(np.clip(err_vmax, 1.0e-6, 1.0))
    obs_vmax = max(float(np.max(grid.density)), 1.0e-6)
    norm_obs = PowerNorm(gamma=0.55, vmin=0.0, vmax=obs_vmax)
    norm_err = PowerNorm(gamma=0.70, vmin=0.0, vmax=err_vmax)
    obs_cmap = plt.get_cmap(FIELD_CMAP).copy()
    err_cmap = plt.get_cmap(ERROR_CMAP).copy()
    obs_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    err_cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    panels = 3 if pinn_years is not None and pinn_fields is not None else 2
    frames: list[Image.Image] = []
    duration_ms = int(round(1000.0 / max(float(fps), 0.1)))
    for year in frame_years.tolist():
        fig, axes = plt.subplots(1, panels, figsize=(5.0 * panels, 5.6), constrained_layout=True)
        axes_arr = np.atleast_1d(axes)
        observed = _field_for_year(year, grid.years, grid.density)
        rk4 = _field_for_year(year, sim_years, sim_fields)
        rk4_error = None if observed is None or rk4 is None else np.abs(np.asarray(rk4) - np.asarray(observed))
        last_mesh = _draw_map_field_panel(
            axes_arr[0],
            title=f"{_format_grid_time_title(grid, year)} observed",
            field=observed,
            lon_edges=lon_edges,
            lat_edges=lat_edges,
            province_rings=province_rings,
            norm=norm_obs,
            cmap=obs_cmap,
            bbox=bbox,
        )
        last_mesh = _draw_map_field_panel(
            axes_arr[1],
            title=f"{_format_grid_time_title(grid, year)} RK4 |error|",
            field=rk4_error,
            lon_edges=lon_edges,
            lat_edges=lat_edges,
            province_rings=province_rings,
            norm=norm_err,
            cmap=err_cmap,
            bbox=bbox,
        ) or last_mesh
        if panels == 3:
            pinn = _field_for_year(year, pinn_years, pinn_fields)  # type: ignore[arg-type]
            pinn_error = None if observed is None or pinn is None else np.abs(np.asarray(pinn) - np.asarray(observed))
            last_mesh = _draw_map_field_panel(
                axes_arr[2],
                title=f"{_format_grid_time_title(grid, year)} PINN |error|",
                field=pinn_error,
                lon_edges=lon_edges,
                lat_edges=lat_edges,
                province_rings=province_rings,
                norm=norm_err,
                cmap=err_cmap,
                bbox=bbox,
            ) or last_mesh
        if last_mesh is not None:
            cbar = fig.colorbar(last_mesh, ax=axes_arr, orientation="horizontal", shrink=0.72, pad=0.04)
            cbar.set_label("absolute normalized-density error", fontsize=9)
            _style_colorbar(cbar, horizontal=True)
        fig.suptitle("Korea pine-wilt observed-period absolute error", fontsize=14, fontweight="bold")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor="white", bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
    if not frames:
        raise ValueError("No frames were available for Korea error GIF export.")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0, disposal=2)
    preview_path = path.with_name(f"{path.stem}_preview.png")
    frames[min(len(frames) - 1, max(0, len(frames) // 2))].convert("RGB").save(preview_path)
    return {
        "path": path.name,
        "preview": preview_path.name,
        "frames": int(len(frames)),
        "fps": float(fps),
        "max_frames": None if max_frames is None else int(max_frames),
        "panels": ["observed", "rk4_abs_error", "pinn_abs_error"] if panels == 3 else ["observed", "rk4_abs_error"],
        "error_vmax": err_vmax,
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
            cmap=FIELD_CMAP,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(_format_grid_time_title(grid, int(year)))
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[n_years:]:
        ax.axis("off")
    if last_image is not None:
        _add_vertical_colorbar(fig, last_image, axes_arr[:n_years], shrink=0.82).set_label("normalized density")
    fig.suptitle("Korea Forest Service pine-wilt observations", fontsize=13)
    _save_figure_with_padding(fig, path, dpi=180)
    plt.close(fig)


def _save_forecast_figure(path: Path, grid, sim_years: np.ndarray, sim_fields: np.ndarray) -> None:
    if getattr(grid, "time_unit", "year") == "year":
        target_years = [2016, 2018, 2020, 2023, 2025, 2027, int(sim_years[-1])]
        seen: set[int] = set()
        target_years = [year for year in target_years if year in set(sim_years.tolist()) and not (year in seen or seen.add(year))]
    else:
        idx = np.unique(np.linspace(0, len(sim_years) - 1, min(8, len(sim_years)), dtype=int))
        target_years = [int(sim_years[item]) for item in idx]
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
            cmap=FIELD_CMAP,
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        suffix = " observed span" if year <= int(grid.years[-1]) else " forecast"
        ax.set_title(_format_grid_time_title(grid, year, suffix))
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(target_years):]:
        ax.axis("off")
    if last_image is not None:
        _add_vertical_colorbar(fig, last_image, axes_arr[: len(target_years)], shrink=0.82).set_label("normalized density")
    fig.suptitle("2D Fisher-KPP RK4 simulation from initial observed density", fontsize=13)
    _save_figure_with_padding(fig, path, dpi=180)
    plt.close(fig)


def _save_pinn_baseline_figure(path: Path, grid, pinn_years: np.ndarray, pinn_fields: np.ndarray) -> None:
    if getattr(grid, "time_unit", "year") == "year":
        selected_years = [2016, 2018, 2020, 2023]
        selected_years = [year for year in selected_years if year in set(grid.years.tolist())]
    else:
        idx = np.unique(np.linspace(0, len(grid.years) - 1, min(4, len(grid.years)), dtype=int))
        selected_years = [int(grid.years[item]) for item in idx]
    fig, axes = plt.subplots(len(selected_years), 3, figsize=(10.8, 2.6 * len(selected_years)), constrained_layout=True)
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
            (f"{_format_grid_time_title(grid, year)} observed", obs, FIELD_CMAP, 0.0, vmax),
            (f"{_format_grid_time_title(grid, year)} PINN", pred, FIELD_CMAP, 0.0, vmax),
            (f"{_format_grid_time_title(grid, year)} |error|", err, ERROR_CMAP, 0.0, err_vmax),
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
                _add_vertical_colorbar(fig, im, ax, shrink=0.75)
    fig.suptitle("Korea pine-wilt PINN baseline: observed years", fontsize=13)
    _save_figure_with_padding(fig, path, dpi=180)
    plt.close(fig)


def _save_korea_phase_contour_figure(
    path: Path,
    grid,
    pinn_years: np.ndarray,
    pinn_fields: np.ndarray,
    phase_fields: np.ndarray | None,
    *,
    level: float = 0.10,
) -> None:
    if phase_fields is None or len(phase_fields) == 0:
        return
    year_to_obs = {int(year): idx for idx, year in enumerate(grid.years.tolist())}
    year_to_pinn = {int(year): idx for idx, year in enumerate(pinn_years.tolist())}
    selected = [int(grid.years[0])]
    if int(grid.years[-1]) in year_to_pinn and int(grid.years[-1]) not in selected:
        selected.append(int(grid.years[-1]))
    selected = selected[:2]
    fig, axes = plt.subplots(len(selected), 3, figsize=(11.0, 3.2 * len(selected)), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)
    extent = [grid.x_edges[0], grid.x_edges[-1], grid.y_edges[0], grid.y_edges[-1]]
    phase_lim = float(np.nanpercentile(np.abs(phase_fields), 97.5)) if np.any(np.isfinite(phase_fields)) else 1.0
    phase_lim = max(phase_lim, 0.05)
    for row, year in enumerate(selected):
        obs = grid.density[year_to_obs[year]] if year in year_to_obs else None
        pred = pinn_fields[year_to_pinn[year]]
        phase = phase_fields[year_to_pinn[year]]
        panels = [
            (f"{_format_grid_time_title(grid, year)} observed", obs if obs is not None else pred, FIELD_CMAP, 0.0, 1.0),
            (f"{_format_grid_time_title(grid, year)} PINN + u={level:.2f}", pred, FIELD_CMAP, 0.0, 1.0),
            (f"{_format_grid_time_title(grid, year)} phase psi=0", phase, "coolwarm", -phase_lim, phase_lim),
        ]
        for col, (title, field, cmap, vmin, vmax) in enumerate(panels):
            ax = axes_arr[row, col]
            im = ax.imshow(
                field,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            try:
                if col == 0 and obs is not None:
                    ax.contour(obs, levels=[level], origin="lower", extent=extent, colors=["#f43f5e"], linewidths=1.2)
                if col in {1, 2}:
                    ax.contour(pred, levels=[level], origin="lower", extent=extent, colors=["#f43f5e"], linewidths=1.2)
                    if np.nanmin(phase) <= 0.0 <= np.nanmax(phase):
                        ax.contour(phase, levels=[0.0], origin="lower", extent=extent, colors=["#111827"], linewidths=1.2, linestyles="--")
            except ValueError:
                pass
            _add_vertical_colorbar(fig, im, ax, shrink=0.75)
    axes_arr[0, 1].plot([], [], color="#f43f5e", linewidth=1.2, label=f"u={level:.2f}")
    axes_arr[0, 1].plot([], [], color="#111827", linewidth=1.2, linestyle="--", label="psi=0")
    axes_arr[0, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Korea pine-wilt intrinsic front phase diagnostic", fontsize=13)
    _save_figure_with_padding(fig, path, dpi=180)
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
    axes[0].set_title("Observed-period relative L2")
    axes[0].set_xlabel("Time id")
    axes[0].set_ylabel("relative L2")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].set_title("Observed-period spatial correlation")
    axes[1].set_xlabel("Time id")
    axes[1].set_ylabel("correlation")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    _save_figure_with_padding(fig, path, dpi=180)
    plt.close(fig)


def _save_metric_figure(path: Path, rows: list[dict[str, float | int]]) -> None:
    years = np.asarray([row["year"] for row in rows], dtype=float)
    rel_l2 = np.asarray([row["relative_l2"] for row in rows], dtype=float)
    corr = np.asarray([row["correlation"] for row in rows], dtype=float)
    obs_mean = np.asarray([row["observed_mean"] for row in rows], dtype=float)
    sim_mean = np.asarray([row["simulated_mean"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].plot(years, rel_l2, marker="o", label="relative L2")
    axes[0].set_title("Observed-period field error")
    axes[0].set_xlabel("Time id")
    axes[0].set_ylabel("relative L2")
    axes[0].grid(alpha=0.25)
    ax_corr = axes[0].twinx()
    ax_corr.plot(years, corr, marker="s", color="tab:orange", label="correlation")
    ax_corr.set_ylabel("correlation")

    axes[1].plot(years, obs_mean, marker="o", label="observed")
    axes[1].plot(years, sim_mean, marker="o", label="simulated")
    axes[1].set_title("Mean normalized density")
    axes[1].set_xlabel("Time id")
    axes[1].set_ylabel("mean density")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    _save_figure_with_padding(fig, path, dpi=180)
    plt.close(fig)


def _write_metric_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    preferred = [
        "year",
        "observed_mean",
        "simulated_mean",
        "mean_absolute_error",
        "mass_absolute_error",
        "mass_relative_error",
        "relative_l2",
        "correlation",
    ]
    fields = _ordered_fieldnames(rows, preferred)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_baseline_metric_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    preferred = [
        "method",
        "year",
        "observed_mean",
        "simulated_mean",
        "mean_absolute_error",
        "mass_absolute_error",
        "mass_relative_error",
        "relative_l2",
        "correlation",
    ]
    fields = _ordered_fieldnames(rows, preferred)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_fieldnames(rows: list[dict], preferred: list[str]) -> list[str]:
    keys: set[str] = set()
    for row in rows:
        keys.update(str(key) for key in row.keys())
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(key for key in keys if key not in set(ordered)))
    return ordered


def _mean_metric(rows: list[dict[str, float | int | str]], key: str) -> float | None:
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value_f):
            values.append(value_f)
    return float(np.mean(values)) if values else None


def _method_summary(rows: list[dict[str, float | int | str]]) -> dict[str, float | None]:
    summary = {
        "mean_relative_l2_observed_years": _mean_metric(rows, "relative_l2"),
        "mean_correlation_observed_years": _mean_metric(rows, "correlation"),
        "mean_mass_absolute_error": _mean_metric(rows, "mass_absolute_error"),
        "mean_mass_relative_error": _mean_metric(rows, "mass_relative_error"),
        "mean_support_fnr_005": _mean_metric(rows, "support_fnr_005"),
        "mean_support_fnr_010": _mean_metric(rows, "support_fnr_010"),
        "mean_support_fpr_005": _mean_metric(rows, "support_fpr_005"),
        "mean_support_fpr_010": _mean_metric(rows, "support_fpr_010"),
        "mean_support_dice_005": _mean_metric(rows, "support_dice_005"),
        "mean_support_dice_010": _mean_metric(rows, "support_dice_010"),
        "mean_support_tversky_005": _mean_metric(rows, "support_tversky_005"),
        "mean_support_tversky_010": _mean_metric(rows, "support_tversky_010"),
    }
    summary["mean_relative_l2_observed_periods"] = summary["mean_relative_l2_observed_years"]
    summary["mean_correlation_observed_periods"] = summary["mean_correlation_observed_years"]
    return summary


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    time_axis = str(getattr(args, "time_axis", "year")).lower()
    action_data = None
    if time_axis == "year":
        points = load_korea_pine_wilt_points()
        grid = build_density_grid(
            points,
            grid_size=args.grid_size,
            pad_m=args.pad_m,
            capacity_percentile=args.capacity_percentile,
            smooth_passes=args.smooth_passes,
        )
    elif time_axis == "pre_action_month":
        action_data = load_korea_pine_wilt_action_time_points(
            getattr(args, "raw_csv_dir", None),
            action_cutoff_date=getattr(args, "action_cutoff_date", None),
            cumulative_infected_complete_threshold=int(getattr(args, "action_threshold", 50_000)),
            include_action_month=bool(getattr(args, "include_action_month", False)),
            raw_crs=str(getattr(args, "raw_crs", "EPSG:5181")),
        )
        points = action_data.points
        grid = build_density_grid(
            points,
            years=action_data.period_ids,
            grid_size=args.grid_size,
            pad_m=args.pad_m,
            capacity_percentile=args.capacity_percentile,
            smooth_passes=args.smooth_passes,
            time_values=action_data.time_values,
            time_labels=action_data.time_labels,
            time_unit="pre_action_month",
            action_metadata=action_data.metadata,
        )
    else:
        raise ValueError("time_axis must be either 'year' or 'pre_action_month'.")
    parameterization = str(getattr(args, "parameterization", "physical")).lower()
    length_scale_mode = str(getattr(args, "physics_length_scale", "max_extent"))
    if parameterization == "physical":
        physics_prior = korea_physics_prior_from_physical(
            grid,
            diffusion_km2_per_year=float(getattr(args, "diffusion_km2_per_year", 15.5)),
            reaction_per_year=float(getattr(args, "reaction_per_year", getattr(args, "reaction", 0.70))),
            length_scale_mode=length_scale_mode,
        )
        diffusion = physics_prior.normalized_diffusion
        reaction = physics_prior.normalized_reaction
    elif parameterization == "normalized":
        diffusion = float(args.diffusion)
        reaction = float(args.reaction)
        physics_prior = korea_physics_prior_from_normalized(
            grid,
            normalized_diffusion=diffusion,
            normalized_reaction=reaction,
            length_scale_mode=length_scale_mode,
        )
    else:
        raise ValueError("parameterization must be either 'physical' or 'normalized'.")
    pinn_initial_reaction = getattr(args, "pinn_initial_reaction", None)
    if pinn_initial_reaction is None:
        pinn_initial_reaction = reaction
    if time_axis == "year":
        sim_years, sim_fields = simulate_density_rk4(
            grid.density[0],
            start_year=int(grid.years[0]),
            end_year=args.end_year,
            diffusion=physics_prior.normalized_diffusion_x,
            diffusion_y=physics_prior.normalized_diffusion_y,
            reaction=reaction,
            steps_per_year=args.steps_per_year,
            land_mask=grid.land_mask,
        )
    else:
        sim_years, sim_fields = simulate_density_rk4_at_times(
            grid.density[0],
            output_ids=grid.years,
            output_times=korea_grid_time_values(grid),
            diffusion=physics_prior.normalized_diffusion_x,
            diffusion_y=physics_prior.normalized_diffusion_y,
            reaction=reaction,
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
            end_time_years=None if time_axis == "year" else float(korea_grid_time_values(grid)[-1]),
            epochs=getattr(args, "pinn_epochs", 120),
            batch_size=getattr(args, "pinn_batch_size", 4096),
            collocation_points=getattr(args, "pinn_collocation_points", 768),
            boundary_points=getattr(args, "pinn_boundary_points", 128),
            lr=getattr(args, "pinn_lr", 2.0e-3),
            data_weight=getattr(args, "pinn_data_weight", 8.0),
            pde_weight=getattr(args, "pinn_pde_weight", 0.05),
            boundary_weight=getattr(args, "pinn_boundary_weight", 0.01),
            initial_condition_weight=getattr(args, "pinn_initial_condition_weight", 16.0),
            initial_condition_points=getattr(args, "pinn_initial_condition_points", 2048),
            sea_weight=getattr(args, "pinn_sea_weight", 2.0),
            support_weight=getattr(args, "pinn_support_weight", 1.5),
            support_temperature=getattr(args, "pinn_support_temperature", 0.025),
            support_false_positive_weight=getattr(args, "pinn_support_false_positive_weight", 0.20),
            support_false_negative_weight=getattr(args, "pinn_support_false_negative_weight", 0.80),
            support_area_weight=getattr(args, "pinn_support_area_weight", 0.35),
            mass_trajectory_weight=getattr(args, "pinn_mass_trajectory_weight", 0.75),
            mass_trajectory_points=getattr(args, "pinn_mass_trajectory_points", 2048),
            mass_trajectory_times=getattr(args, "pinn_mass_trajectory_times", 4),
            phase_pde_weight=getattr(args, "pinn_phase_pde_weight", 0.02),
            intrinsic_phase_initial_weight=getattr(args, "pinn_intrinsic_phase_initial_weight", 0.03),
            intrinsic_phase_anchor_points=getattr(args, "pinn_intrinsic_phase_anchor_points", 2048),
            intrinsic_phase_anchor_level=getattr(args, "pinn_intrinsic_phase_anchor_level", 0.10),
            intrinsic_phase_anchor_band=getattr(args, "pinn_intrinsic_phase_anchor_band", 0.025),
            intrinsic_phase_anchor_sign_margin=getattr(args, "pinn_intrinsic_phase_anchor_sign_margin", 0.015),
            intrinsic_phase_gradient_alignment_weight=getattr(
                args,
                "pinn_intrinsic_phase_gradient_alignment_weight",
                0.01,
            ),
            intrinsic_phase_monotonicity_weight=getattr(args, "pinn_intrinsic_phase_monotonicity_weight", 0.01),
            intrinsic_phase_compatibility_points=getattr(args, "pinn_intrinsic_phase_compatibility_points", 512),
            intrinsic_phase_compatibility_low=getattr(args, "pinn_intrinsic_phase_compatibility_low", 0.02),
            intrinsic_phase_compatibility_high=getattr(args, "pinn_intrinsic_phase_compatibility_high", 0.98),
            intrinsic_phase_compatibility_temperature=getattr(
                args,
                "pinn_intrinsic_phase_compatibility_temperature",
                0.03,
            ),
            intrinsic_phase_compatibility_min_grad=getattr(
                args,
                "pinn_intrinsic_phase_compatibility_min_grad",
                1.0e-4,
            ),
            residual_cvar_weight=getattr(args, "pinn_residual_cvar_weight", 0.03),
            residual_cvar_fraction=getattr(args, "pinn_residual_cvar_fraction", 0.10),
            diffusion=diffusion,
            reaction=float(pinn_initial_reaction),
            physics_anchor_weight=getattr(args, "pinn_physics_anchor_weight", 0.08),
            coefficient_field_weight=getattr(args, "pinn_coefficient_field_weight", 0.02),
            physics_length_scale_mode=length_scale_mode,
            checkpoint_path=out_dir / "korea_front_phase_pinn_latest.pt",
            resume_from_checkpoint=not getattr(args, "no_pinn_resume", False),
            checkpoint_every=getattr(args, "pinn_checkpoint_every", 1),
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
    error_gif_info = _save_korea_error_gif(
        out_dir / "korea_error_baselines.gif",
        grid,
        sim_years,
        sim_fields,
        pinn_years=pinn_result.years if pinn_result is not None else None,
        pinn_fields=pinn_result.fields if pinn_result is not None else None,
        fps=getattr(args, "map_gif_fps", 1.2),
        max_frames=getattr(args, "map_gif_max_frames", 15),
    )
    field_payload = {
        "observed_years": grid.years,
        "observed_time_values": korea_grid_time_values(grid),
        "observed_fields": grid.density,
        "rk4_years": sim_years,
        "rk4_time_values": korea_grid_time_values(grid) if time_axis != "year" else sim_years.astype(float) - float(sim_years[0]),
        "rk4_fields": sim_fields,
    }
    if grid.time_labels is not None:
        field_payload["observed_time_labels"] = np.asarray(grid.time_labels)
        field_payload["rk4_time_labels"] = np.asarray(grid.time_labels)
    if pinn_result is not None:
        field_payload["pinn_years"] = pinn_result.years
        field_payload["pinn_time_values"] = korea_grid_time_values(grid) if time_axis != "year" else pinn_result.years.astype(float) - float(pinn_result.years[0])
        field_payload["pinn_fields"] = pinn_result.fields
        if pinn_result.phase_fields is not None:
            field_payload["pinn_phase_fields"] = pinn_result.phase_fields
    np.savez_compressed(out_dir / "korea_pine_wilt_fields.npz", **field_payload)
    if pinn_result is not None:
        _save_pinn_baseline_figure(out_dir / "pinn_baseline_observed_years.png", grid, pinn_result.years, pinn_result.fields)
        _save_korea_phase_contour_figure(
            out_dir / "korea_phase_contours.png",
            grid,
            pinn_result.years,
            pinn_result.fields,
            pinn_result.phase_fields,
            level=getattr(args, "pinn_intrinsic_phase_anchor_level", 0.10),
        )

    rk4_summary = _method_summary([row for row in baseline_rows if row["method"] == "rk4"])
    pinn_rows = [row for row in baseline_rows if row["method"] == "pinn"]
    baseline_protocol = {
        "methods": ["rk4"] + (["pinn"] if pinn_result is not None else []),
        "initial_condition": "first_observed_density",
        "same_initial_density": True,
        "same_observed_period_metrics": True,
        "same_phase_capable_backbone_as_flagship": True,
        "front_phase_head_enabled_for_pinn": pinn_result is not None,
        "grid_size": int(args.grid_size),
        "time_axis": time_axis,
        "time_unit": grid.time_unit,
        "land_mask_enabled": grid.land_mask is not None,
        "sea_cells_forced_zero": grid.land_mask is not None,
        "parameterization": parameterization,
        "length_scale_mode": physics_prior.scale.length_scale_mode,
        "normalized_diffusion_scalar": float(physics_prior.normalized_diffusion),
        "normalized_diffusion_x": float(physics_prior.normalized_diffusion_x),
        "normalized_diffusion_y": float(physics_prior.normalized_diffusion_y),
        "normalized_reaction": float(physics_prior.normalized_reaction),
        "diffusion_km2_per_year": float(physics_prior.diffusion_km2_per_year),
        "reaction_per_year": float(physics_prior.reaction_per_year),
        "steps_per_year": int(args.steps_per_year),
        "end_year": int(args.end_year),
        "end_elapsed_years": float(korea_grid_time_values(grid)[-1]) if time_axis != "year" else float(int(args.end_year) - int(grid.years[0])),
    }
    summary = {
        "dataset": manifest["dataset"],
        "records": int(len(points.year)),
        "time_axis": time_axis,
        "time_unit": grid.time_unit,
        "time_labels": list(grid.time_labels) if grid.time_labels is not None else None,
        "elapsed_years": korea_grid_time_values(grid).tolist(),
        "year_min": int(points.year.min()) if time_axis == "year" else None,
        "year_max": int(points.year.max()) if time_axis == "year" else None,
        "period_min": int(points.year.min()),
        "period_max": int(points.year.max()),
        "action_metadata": grid.action_metadata,
        "grid_size": int(args.grid_size),
        "capacity": float(grid.capacity),
        "land_mask": {
            "enabled": grid.land_mask is not None,
            "land_cells": int(np.sum(grid.land_mask)) if grid.land_mask is not None else int(grid.density.shape[-1] * grid.density.shape[-2]),
            "sea_cells": int(grid.land_mask.size - np.sum(grid.land_mask)) if grid.land_mask is not None else 0,
        },
        "parameterization": parameterization,
        "physics_prior": {
            "diffusion_km2_per_year": float(physics_prior.diffusion_km2_per_year),
            "reaction_per_year": float(physics_prior.reaction_per_year),
            "front_speed_km_per_year": float(physics_prior.front_speed_km_per_year),
            "normalized_diffusion": float(physics_prior.normalized_diffusion),
            "normalized_diffusion_x": float(physics_prior.normalized_diffusion_x),
            "normalized_diffusion_y": float(physics_prior.normalized_diffusion_y),
            "normalized_reaction": float(physics_prior.normalized_reaction),
            "length_scale_mode": physics_prior.scale.length_scale_mode,
            "length_scale_km": float(physics_prior.scale.length_scale_km),
            "width_km": float(physics_prior.scale.width_km),
            "height_km": float(physics_prior.scale.height_km),
        },
        "diffusion": float(diffusion),
        "diffusion_x": float(physics_prior.normalized_diffusion_x),
        "diffusion_y": float(physics_prior.normalized_diffusion_y),
        "reaction": float(reaction),
        "diffusion_km2_per_year": float(physics_prior.diffusion_km2_per_year),
        "reaction_per_year": float(physics_prior.reaction_per_year),
        "front_speed_km_per_year": float(physics_prior.front_speed_km_per_year),
        "steps_per_year": int(args.steps_per_year),
        "end_year": int(args.end_year),
        "end_elapsed_years": float(korea_grid_time_values(grid)[-1]) if time_axis != "year" else float(int(args.end_year) - int(grid.years[0])),
        "mean_relative_l2_observed_years": rk4_summary["mean_relative_l2_observed_years"],
        "mean_correlation_observed_years": rk4_summary["mean_correlation_observed_years"],
        "baselines": {
            "rk4": rk4_summary,
        },
        "baseline_protocol": baseline_protocol,
        "physics_identifiability": {
            "main_identifiability_issue": "The leading Fisher-KPP front speed depends on 2*sqrt(D*r), so field data are needed to separate D and r beyond the product.",
            "equivalent_front_speed_pairs": korea_equivalent_dr_pairs(
                diffusion_km2_per_year=float(physics_prior.diffusion_km2_per_year),
                reaction_per_year=float(physics_prior.reaction_per_year),
            ),
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
            "korea_error_baselines.gif",
            "korea_error_baselines_preview.png",
            "korea_pine_wilt_fields.npz",
        ],
        "map_gif": map_gif_info,
        "error_gif": error_gif_info,
    }
    if pinn_result is not None:
        pinn_summary = _method_summary(pinn_rows)
        summary["baselines"]["pinn"] = {
            "status": pinn_result.status,
            "epochs": int(getattr(args, "pinn_epochs", 120)),
            "physics": pinn_result.physics,
            "sea_weight": float(getattr(args, "pinn_sea_weight", 2.0)),
            "initial_condition_weight": float(getattr(args, "pinn_initial_condition_weight", 16.0)),
            "initial_condition_points": int(getattr(args, "pinn_initial_condition_points", 2048)),
            "support_weight": float(getattr(args, "pinn_support_weight", 1.5)),
            "support_temperature": float(getattr(args, "pinn_support_temperature", 0.025)),
            "support_false_positive_weight": float(getattr(args, "pinn_support_false_positive_weight", 0.20)),
            "support_false_negative_weight": float(getattr(args, "pinn_support_false_negative_weight", 0.80)),
            "support_area_weight": float(getattr(args, "pinn_support_area_weight", 0.35)),
            "mass_trajectory_weight": float(getattr(args, "pinn_mass_trajectory_weight", 0.75)),
            "mass_trajectory_points": int(getattr(args, "pinn_mass_trajectory_points", 2048)),
            "mass_trajectory_times": int(getattr(args, "pinn_mass_trajectory_times", 4)),
            "phase_pde_weight": float(getattr(args, "pinn_phase_pde_weight", 0.02)),
            "intrinsic_phase_initial_weight": float(getattr(args, "pinn_intrinsic_phase_initial_weight", 0.03)),
            "intrinsic_phase_anchor_points": int(getattr(args, "pinn_intrinsic_phase_anchor_points", 2048)),
            "intrinsic_phase_anchor_level": float(getattr(args, "pinn_intrinsic_phase_anchor_level", 0.10)),
            "intrinsic_phase_anchor_band": float(getattr(args, "pinn_intrinsic_phase_anchor_band", 0.025)),
            "intrinsic_phase_anchor_sign_margin": float(getattr(args, "pinn_intrinsic_phase_anchor_sign_margin", 0.015)),
            "intrinsic_phase_gradient_alignment_weight": float(
                getattr(args, "pinn_intrinsic_phase_gradient_alignment_weight", 0.01)
            ),
            "intrinsic_phase_monotonicity_weight": float(getattr(args, "pinn_intrinsic_phase_monotonicity_weight", 0.01)),
            "intrinsic_phase_compatibility_points": int(getattr(args, "pinn_intrinsic_phase_compatibility_points", 512)),
            "physics_anchor_weight": float(getattr(args, "pinn_physics_anchor_weight", 0.08)),
            "coefficient_field_weight": float(getattr(args, "pinn_coefficient_field_weight", 0.02)),
            "resume_from_checkpoint": not bool(getattr(args, "no_pinn_resume", False)),
            "checkpoint_every": int(getattr(args, "pinn_checkpoint_every", 1)),
            "checkpoint_path": "korea_front_phase_pinn_latest.pt",
            **pinn_summary,
            "history": pinn_result.history,
        }
        summary["outputs"].append("pinn_baseline_observed_years.png")
        summary["outputs"].append("korea_phase_contours.png")
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
    parser.add_argument("--time-axis", choices=["year", "pre_action_month"], default="year")
    parser.add_argument("--raw-csv-dir", type=Path, default=None, help="Original KFS annual CSV directory used by --time-axis pre_action_month.")
    parser.add_argument("--raw-crs", default="EPSG:5181", help="CRS of original KFS raw CSV x/y coordinates before conversion to EPSG:5179.")
    parser.add_argument("--action-threshold", type=int, default=50_000, help="Cumulative infected/completed records used to infer large-scale action start.")
    parser.add_argument("--action-cutoff-date", default=None, help="Override large-scale action cutoff date, e.g. 2016-10-02.")
    parser.add_argument("--include-action-month", action="store_true", help="Include the cutoff month in the pre-action monthly comparison.")
    parser.add_argument("--parameterization", choices=["physical", "normalized"], default="physical")
    parser.add_argument("--physics-length-scale", choices=["max_extent", "mean_extent", "geometric_mean_extent", "x_extent", "y_extent"], default="max_extent")
    parser.add_argument("--diffusion-km2-per-year", type=float, default=15.5)
    parser.add_argument("--reaction-per-year", type=float, default=0.70)
    parser.add_argument("--diffusion", type=float, default=0.0015, help="Normalized diffusion used only with --parameterization normalized.")
    parser.add_argument("--reaction", type=float, default=0.70, help="Normalized reaction used only with --parameterization normalized.")
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
    parser.add_argument("--pinn-initial-condition-weight", type=float, default=16.0)
    parser.add_argument("--pinn-initial-condition-points", type=int, default=2048)
    parser.add_argument("--pinn-sea-weight", type=float, default=2.0)
    parser.add_argument("--pinn-support-weight", type=float, default=1.5)
    parser.add_argument("--pinn-support-temperature", type=float, default=0.025)
    parser.add_argument("--pinn-support-false-positive-weight", type=float, default=0.20)
    parser.add_argument("--pinn-support-false-negative-weight", type=float, default=0.80)
    parser.add_argument("--pinn-support-area-weight", type=float, default=0.35)
    parser.add_argument("--pinn-mass-trajectory-weight", type=float, default=0.75)
    parser.add_argument("--pinn-mass-trajectory-points", type=int, default=2048)
    parser.add_argument("--pinn-mass-trajectory-times", type=int, default=4)
    parser.add_argument("--pinn-phase-pde-weight", type=float, default=0.02)
    parser.add_argument("--pinn-intrinsic-phase-initial-weight", type=float, default=0.03)
    parser.add_argument("--pinn-intrinsic-phase-anchor-points", type=int, default=2048)
    parser.add_argument("--pinn-intrinsic-phase-anchor-level", type=float, default=0.10)
    parser.add_argument("--pinn-intrinsic-phase-anchor-band", type=float, default=0.025)
    parser.add_argument("--pinn-intrinsic-phase-anchor-sign-margin", type=float, default=0.015)
    parser.add_argument("--pinn-intrinsic-phase-gradient-alignment-weight", type=float, default=0.01)
    parser.add_argument("--pinn-intrinsic-phase-monotonicity-weight", type=float, default=0.01)
    parser.add_argument("--pinn-intrinsic-phase-compatibility-points", type=int, default=512)
    parser.add_argument("--pinn-intrinsic-phase-compatibility-low", type=float, default=0.02)
    parser.add_argument("--pinn-intrinsic-phase-compatibility-high", type=float, default=0.98)
    parser.add_argument("--pinn-intrinsic-phase-compatibility-temperature", type=float, default=0.03)
    parser.add_argument("--pinn-intrinsic-phase-compatibility-min-grad", type=float, default=1.0e-4)
    parser.add_argument("--pinn-residual-cvar-weight", type=float, default=0.03)
    parser.add_argument("--pinn-residual-cvar-fraction", type=float, default=0.10)
    parser.add_argument("--pinn-initial-reaction", type=float, default=None)
    parser.add_argument("--pinn-physics-anchor-weight", type=float, default=0.08)
    parser.add_argument("--pinn-coefficient-field-weight", type=float, default=0.02)
    parser.add_argument(
        "--no-pinn-resume",
        action="store_true",
        help="Disable resume from output-dir/korea_front_phase_pinn_latest.pt.",
    )
    parser.add_argument("--pinn-checkpoint-every", type=int, default=1, help="Save Korea PINN training state every N epochs.")
    parser.add_argument("--map-gif-fps", type=float, default=1.2)
    parser.add_argument("--map-gif-max-frames", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
