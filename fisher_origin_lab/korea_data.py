from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "korea_pine_wilt"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_POINTS_NPZ = DEFAULT_PROCESSED_DIR / "infected_points_2016_2023.npz"
DEFAULT_POINTS_CSV_GZ = DEFAULT_PROCESSED_DIR / "infected_points_2016_2023.csv.gz"
DEFAULT_MANIFEST = DEFAULT_PROCESSED_DIR / "manifest.json"


@dataclass(frozen=True)
class KoreaPineWiltPoints:
    x: np.ndarray
    y: np.ndarray
    year: np.ndarray
    crs: str = "EPSG:5179"


@dataclass(frozen=True)
class KoreaPineWiltGrid:
    years: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    density: np.ndarray
    raw_counts: np.ndarray
    capacity: float


def load_manifest(path: Path | None = None) -> dict:
    manifest_path = DEFAULT_MANIFEST if path is None else Path(path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_korea_pine_wilt_points(path: Path | None = None) -> KoreaPineWiltPoints:
    """Load compact Korea Forest Service infected-tree point observations.

    The committed compact files contain only infected-tree coordinates and year,
    derived from the larger CP949 annual Korea Forest Service CSV files.
    """

    data_path = Path(path) if path is not None else DEFAULT_POINTS_NPZ
    if data_path.suffix == ".npz":
        data = np.load(data_path)
        return KoreaPineWiltPoints(
            x=data["x"].astype(np.float64),
            y=data["y"].astype(np.float64),
            year=data["year"].astype(np.int16),
        )
    if data_path.suffixes[-2:] == [".csv", ".gz"]:
        xs: list[float] = []
        ys: list[float] = []
        years: list[int] = []
        with gzip.open(data_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                xs.append(float(row["x_5179"]))
                ys.append(float(row["y_5179"]))
                years.append(int(row["year"]))
        return KoreaPineWiltPoints(
            x=np.asarray(xs, dtype=np.float64),
            y=np.asarray(ys, dtype=np.float64),
            year=np.asarray(years, dtype=np.int16),
        )
    raise ValueError(f"Unsupported Korea pine-wilt point file: {data_path}")


def _smooth_2d(field: np.ndarray, passes: int = 1) -> np.ndarray:
    out = np.asarray(field, dtype=np.float64)
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    for _ in range(passes):
        out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 0, out)
        out = np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), 1, out)
    return out


def build_density_grid(
    points: KoreaPineWiltPoints,
    *,
    years: tuple[int, ...] = tuple(range(2016, 2024)),
    grid_size: int = 96,
    pad_m: float = 15_000.0,
    capacity_percentile: float = 99.0,
    smooth_passes: int = 1,
) -> KoreaPineWiltGrid:
    if grid_size < 8:
        raise ValueError("grid_size must be at least 8.")
    years_arr = np.asarray(years, dtype=np.int16)
    x_edges = np.linspace(float(points.x.min() - pad_m), float(points.x.max() + pad_m), grid_size + 1)
    y_edges = np.linspace(float(points.y.min() - pad_m), float(points.y.max() + pad_m), grid_size + 1)
    raw = []
    for year in years_arr:
        mask = points.year == year
        hist, _, _ = np.histogram2d(points.x[mask], points.y[mask], bins=[x_edges, y_edges])
        raw.append(_smooth_2d(hist.T, passes=smooth_passes))
    raw_counts = np.asarray(raw, dtype=np.float64)
    positive = raw_counts[raw_counts > 0.0]
    if positive.size == 0:
        raise ValueError("No positive infected-tree observations found for the requested years.")
    capacity = float(np.percentile(positive, capacity_percentile))
    density = np.clip(raw_counts / max(capacity, 1.0e-12), 0.0, 1.0)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return KoreaPineWiltGrid(
        years=years_arr,
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        density=density,
        raw_counts=raw_counts,
        capacity=capacity,
    )


def _laplacian_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    padded = np.pad(u, 1, mode="edge")
    return (
        padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
        - 4.0 * padded[1:-1, 1:-1]
    ) / dx**2


def _rk4_step_2d(u: np.ndarray, dt: float, dx: float, diffusion: float, reaction: float) -> np.ndarray:
    def rhs(v: np.ndarray) -> np.ndarray:
        return diffusion * _laplacian_neumann(v, dx) + reaction * v * (1.0 - v)

    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    return np.clip(u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)


def simulate_density_rk4(
    initial_density: np.ndarray,
    *,
    start_year: int = 2016,
    end_year: int = 2030,
    diffusion: float = 0.0015,
    reaction: float = 0.70,
    steps_per_year: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a normalized 2D Fisher-KPP RK4 simulation on a square grid."""

    if end_year < start_year:
        raise ValueError("end_year must be >= start_year.")
    u = np.asarray(initial_density, dtype=np.float64).copy()
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("initial_density must be a square 2D field.")
    dx = 1.0 / max(u.shape[0] - 1, 1)
    dt = 1.0 / float(steps_per_year)
    dt_diff_limit = 0.69 * dx**2 / (2.0 * max(diffusion, 1.0e-12))
    if dt > 0.95 * min(dt_diff_limit, 1.0 / max(reaction, 1.0e-12)):
        raise ValueError(
            "RK4 Korea simulation is outside the practical stability estimate; "
            f"dt={dt:.3e}, diffusion_limit={dt_diff_limit:.3e}."
        )

    years = np.arange(start_year, end_year + 1, dtype=np.int16)
    fields = [np.clip(u, 0.0, 1.0)]
    for _year in years[:-1]:
        for _ in range(steps_per_year):
            u = _rk4_step_2d(u, dt, dx, diffusion, reaction)
        fields.append(u.copy())
    return years, np.asarray(fields)


def compare_observed_and_simulated(
    observed_grid: KoreaPineWiltGrid,
    sim_years: np.ndarray,
    sim_fields: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    sim_lookup = {int(year): idx for idx, year in enumerate(sim_years.tolist())}
    for obs_idx, year in enumerate(observed_grid.years.tolist()):
        if int(year) not in sim_lookup:
            continue
        obs = observed_grid.density[obs_idx]
        pred = sim_fields[sim_lookup[int(year)]]
        denom = float(np.linalg.norm(obs)) + 1.0e-12
        rel_l2 = float(np.linalg.norm(pred - obs) / denom)
        if np.std(obs) <= 1.0e-12 or np.std(pred) <= 1.0e-12:
            corr = np.nan
        else:
            corr = float(np.corrcoef(obs.ravel(), pred.ravel())[0, 1])
        rows.append(
            {
                "year": int(year),
                "observed_mean": float(obs.mean()),
                "simulated_mean": float(pred.mean()),
                "relative_l2": rel_l2,
                "correlation": corr,
            }
        )
    return rows

