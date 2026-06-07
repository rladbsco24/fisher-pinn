from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import DomainConfig, ModelConfig, PDEConfig, SeedConfig
from .losses import boundary_neumann_loss, pde_residual
from .models import OriginPINN


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


@dataclass(frozen=True)
class KoreaPineWiltPINNResult:
    years: np.ndarray
    fields: np.ndarray
    history: list[dict[str, float]]
    metrics: list[dict[str, float | int]]
    physics: dict[str, float]
    status: str


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


def _normalized_grid_xy(grid: KoreaPineWiltGrid) -> np.ndarray:
    x_norm = (grid.x_centers - grid.x_edges[0]) / max(grid.x_edges[-1] - grid.x_edges[0], 1.0e-12)
    y_norm = (grid.y_centers - grid.y_edges[0]) / max(grid.y_edges[-1] - grid.y_edges[0], 1.0e-12)
    x, y = np.meshgrid(x_norm, y_norm, indexing="xy")
    return np.stack([x.ravel(), y.ravel()], axis=1).astype(np.float32)


def _korea_training_tensors(
    grid: KoreaPineWiltGrid,
    *,
    start_year: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    xy = _normalized_grid_xy(grid)
    xyt_rows = []
    value_rows = []
    for idx, year in enumerate(grid.years.tolist()):
        t = np.full((xy.shape[0], 1), float(int(year) - start_year), dtype=np.float32)
        xyt_rows.append(np.concatenate([xy, t], axis=1))
        value_rows.append(grid.density[idx].reshape(-1, 1).astype(np.float32))
    xyt = torch.tensor(np.concatenate(xyt_rows, axis=0), dtype=torch.float32, device=device)
    values = torch.tensor(np.concatenate(value_rows, axis=0), dtype=torch.float32, device=device)
    return xyt, values


def predict_korea_pinn_fields(
    model: OriginPINN,
    grid: KoreaPineWiltGrid,
    years: np.ndarray,
    *,
    start_year: int,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    xy = torch.tensor(_normalized_grid_xy(grid), dtype=torch.float32, device=device)
    fields = []
    model.eval()
    with torch.no_grad():
        for year in years.tolist():
            t = torch.full((len(xy), 1), float(int(year) - start_year), dtype=torch.float32, device=device)
            preds = []
            for start in range(0, len(xy), batch_size):
                preds.append(model(xy[start : start + batch_size], t[start : start + batch_size]).detach().cpu())
            field = torch.cat(preds, dim=0).numpy().reshape(grid.density.shape[1:])
            fields.append(np.clip(field, 0.0, 1.0))
    return np.asarray(fields, dtype=np.float64)


def fit_korea_pine_wilt_pinn(
    grid: KoreaPineWiltGrid,
    *,
    end_year: int = 2030,
    epochs: int = 120,
    batch_size: int = 4096,
    collocation_points: int = 768,
    boundary_points: int = 128,
    lr: float = 2.0e-3,
    data_weight: float = 8.0,
    pde_weight: float = 0.05,
    boundary_weight: float = 0.01,
    diffusion: float = 0.0015,
    reaction: float = 0.20,
    seed: int = 7,
    device: str | torch.device | None = None,
) -> KoreaPineWiltPINNResult:
    """Fit the repository's forward PINN architecture to Korea pine-wilt density grids.

    This is a baseline diagnostic, not a calibrated epidemiological model. It uses the
    same compact gridded observations as the RK4 baseline, adds a weak Fisher-KPP PDE
    residual, and reports the same observed-year field metrics.
    """

    start_year = int(grid.years[0])
    model_end_year = max(int(end_year), int(grid.years[-1]))
    t_end = float(model_end_year - start_year)
    if t_end <= 0.0:
        raise ValueError("end_year must be after the first observed year.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    domain = DomainConfig(box=1.0, t_end=t_end, grid=int(grid.density.shape[-1]), truth_steps=0)
    pde = PDEConfig(diffusion=diffusion, reaction=reaction, velocity_x=0.0, velocity_y=0.0, include_advection=False)
    seed_cfg = SeedConfig(center_x=0.5, center_y=0.5, sigma=0.12, amplitude=0.25)
    model_cfg = ModelConfig(
        architecture="pirate",
        fourier_features=16,
        fourier_sigma=1.0,
        hidden=48,
        layers=3,
        use_random_weight_factorization=True,
        learn_diffusion=True,
        learn_reaction=True,
        learn_drift=False,
        use_source_envelope=False,
        use_geo_features=True,
        spatial_fourier_only=True,
        use_seed_front_features=False,
        use_traveling_wave_features=False,
        hard_initial_condition=False,
        use_kpp_front_envelope=False,
    )
    model = OriginPINN(domain, pde, seed_cfg, model_cfg).to(device_obj)
    xyt, values = _korea_training_tensors(grid, start_year=start_year, device=device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_obs = len(xyt)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        idx = torch.randint(0, n_obs, (min(batch_size, n_obs),), device=device_obj)
        pred = model(xyt[idx, :2], xyt[idx, 2:3])
        density_weight = 1.0 + 4.0 * values[idx].detach()
        data_loss = torch.mean(density_weight * (pred - values[idx]) ** 2)

        xy_col = torch.rand(collocation_points, 2, device=device_obj)
        t_col = torch.rand(collocation_points, 1, device=device_obj) * domain.t_end
        residual = pde_residual(model, xy_col, t_col)
        pde_loss = torch.mean(residual**2)
        if boundary_points > 0:
            bc_loss = boundary_neumann_loss(model, boundary_points, device_obj)
        else:
            bc_loss = torch.zeros((), device=device_obj)

        loss = data_weight * data_loss + pde_weight * pde_loss + boundary_weight * bc_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "total": float(loss.detach().cpu()),
                    "data": float(data_loss.detach().cpu()),
                    "pde": float(pde_loss.detach().cpu()),
                    "boundary": float(bc_loss.detach().cpu()),
                    "diffusion": float(model.pde.diffusion().detach().cpu()),
                    "reaction": float(model.pde.reaction().detach().cpu()),
                }
            )

    pred_years = np.arange(start_year, model_end_year + 1, dtype=np.int16)
    fields = predict_korea_pinn_fields(
        model,
        grid,
        pred_years,
        start_year=start_year,
        device=device_obj,
    )
    metrics = compare_observed_and_simulated(grid, pred_years, fields)
    mean_l2 = float(np.nanmean([row["relative_l2"] for row in metrics]))
    status = "diagnostic_baseline"
    if epochs < 50:
        status = "diagnostic_only_low_epoch"
    elif mean_l2 > 1.0:
        status = "diagnostic_high_error"
    return KoreaPineWiltPINNResult(
        years=pred_years,
        fields=fields,
        history=history,
        metrics=metrics,
        physics=model.physics_dict(),
        status=status,
    )
