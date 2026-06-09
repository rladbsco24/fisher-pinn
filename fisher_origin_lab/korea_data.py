from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import torch
from matplotlib.path import Path as MplPath

from .config import DomainConfig, ModelConfig, PDEConfig, SeedConfig
from .losses import boundary_neumann_loss, spatial_coefficient_regularization_loss
from .models import OriginPINN


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "korea_pine_wilt"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_POINTS_NPZ = DEFAULT_PROCESSED_DIR / "infected_points_2016_2023.npz"
DEFAULT_POINTS_CSV_GZ = DEFAULT_PROCESSED_DIR / "infected_points_2016_2023.csv.gz"
DEFAULT_MANIFEST = DEFAULT_PROCESSED_DIR / "manifest.json"
DEFAULT_PROVINCE_GEOJSON = DEFAULT_DATA_DIR / "assets" / "skorea_provinces_2018.geojson"


@dataclass(frozen=True)
class KoreaPineWiltPoints:
    x: np.ndarray
    y: np.ndarray
    year: np.ndarray
    crs: str = "EPSG:5179"
    date_ordinal: np.ndarray | None = None
    status: np.ndarray | None = None


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
    land_mask: np.ndarray | None = None
    time_values: np.ndarray | None = None
    time_labels: tuple[str, ...] | None = None
    time_unit: str = "year"
    action_metadata: dict[str, object] | None = None


@dataclass(frozen=True)
class KoreaPineWiltDomainScale:
    width_km: float
    height_km: float
    length_scale_km: float
    length_scale_mode: str


@dataclass(frozen=True)
class KoreaPineWiltPhysicsPrior:
    diffusion_km2_per_year: float
    reaction_per_year: float
    normalized_diffusion: float
    normalized_diffusion_x: float
    normalized_diffusion_y: float
    normalized_reaction: float
    front_speed_km_per_year: float
    scale: KoreaPineWiltDomainScale


@dataclass(frozen=True)
class KoreaPineWiltPINNResult:
    years: np.ndarray
    fields: np.ndarray
    history: list[dict[str, float]]
    metrics: list[dict[str, float | int]]
    physics: dict[str, float | str]
    status: str


@dataclass(frozen=True)
class KoreaActionTimeData:
    points: KoreaPineWiltPoints
    period_ids: tuple[int, ...]
    time_values: np.ndarray
    time_labels: tuple[str, ...]
    metadata: dict[str, object]


def korea_domain_scale(grid: KoreaPineWiltGrid, mode: str = "max_extent") -> KoreaPineWiltDomainScale:
    """Return the physical length scale used by the normalized Korea model.

    The current PINN/RK4 grid uses x,y in [0, 1]. A physical diffusion D_phys in
    km^2/year therefore maps to D_norm = D_phys / L^2, where L is the chosen
    kilometer length represented by one normalized coordinate unit.
    """

    width_km = float((grid.x_edges[-1] - grid.x_edges[0]) / 1000.0)
    height_km = float((grid.y_edges[-1] - grid.y_edges[0]) / 1000.0)
    if width_km <= 0.0 or height_km <= 0.0:
        raise ValueError("Korea grid must have positive x/y physical extents.")
    mode_key = str(mode).lower()
    if mode_key == "max_extent":
        length_scale_km = max(width_km, height_km)
    elif mode_key == "mean_extent":
        length_scale_km = 0.5 * (width_km + height_km)
    elif mode_key == "geometric_mean_extent":
        length_scale_km = float(np.sqrt(width_km * height_km))
    elif mode_key == "x_extent":
        length_scale_km = width_km
    elif mode_key == "y_extent":
        length_scale_km = height_km
    else:
        raise ValueError(
            "Unsupported Korea length-scale mode. Use one of "
            "'max_extent', 'mean_extent', 'geometric_mean_extent', 'x_extent', or 'y_extent'."
        )
    return KoreaPineWiltDomainScale(
        width_km=width_km,
        height_km=height_km,
        length_scale_km=float(length_scale_km),
        length_scale_mode=mode_key,
    )


def korea_physical_to_normalized_diffusion(
    diffusion_km2_per_year: float,
    grid: KoreaPineWiltGrid,
    *,
    length_scale_mode: str = "max_extent",
) -> float:
    scale = korea_domain_scale(grid, length_scale_mode)
    return float(diffusion_km2_per_year) / max(scale.length_scale_km**2, 1.0e-12)


def korea_physical_to_anisotropic_normalized_diffusion(
    diffusion_km2_per_year: float,
    grid: KoreaPineWiltGrid,
) -> tuple[float, float]:
    """Map physical scalar diffusion to x/y normalized diffusion coefficients."""

    scale = korea_domain_scale(grid, "max_extent")
    diffusion = float(diffusion_km2_per_year)
    dx = diffusion / max(scale.width_km**2, 1.0e-12)
    dy = diffusion / max(scale.height_km**2, 1.0e-12)
    return float(dx), float(dy)


def korea_anisotropic_normalized_diffusion_components(
    normalized_diffusion: float,
    grid: KoreaPineWiltGrid,
    *,
    length_scale_mode: str = "max_extent",
) -> tuple[float, float]:
    """Return x/y coefficients consistent with a scalar physical D prior.

    ``normalized_diffusion`` is defined against the selected length scale L. Since
    the Korea grid stores x and y separately in [0, 1], the physical PDE maps to
    D_x = D_phys / width^2 and D_y = D_phys / height^2.
    """

    scale = korea_domain_scale(grid, length_scale_mode)
    diffusion_phys = float(normalized_diffusion) * scale.length_scale_km**2
    return korea_physical_to_anisotropic_normalized_diffusion(diffusion_phys, grid)


def korea_normalized_to_physical_diffusion(
    normalized_diffusion: float,
    grid: KoreaPineWiltGrid,
    *,
    length_scale_mode: str = "max_extent",
) -> float:
    scale = korea_domain_scale(grid, length_scale_mode)
    return float(normalized_diffusion) * scale.length_scale_km**2


def korea_physics_prior_from_physical(
    grid: KoreaPineWiltGrid,
    *,
    diffusion_km2_per_year: float = 15.5,
    reaction_per_year: float = 0.70,
    length_scale_mode: str = "max_extent",
) -> KoreaPineWiltPhysicsPrior:
    """Build a physical-to-normalized Korea Fisher-KPP prior.

    The default diffusion follows the previous Korea PINN notebook's reported
    D ~= 15.5 km^2/year. Reaction remains configurable because the old PINN
    fitted near-zero growth in a decreasing-control period, while the forward
    Fisher simulation may intentionally use a nonzero spread regime.
    """

    scale = korea_domain_scale(grid, length_scale_mode)
    diffusion_norm = float(diffusion_km2_per_year) / max(scale.length_scale_km**2, 1.0e-12)
    diffusion_x, diffusion_y = korea_physical_to_anisotropic_normalized_diffusion(diffusion_km2_per_year, grid)
    reaction_norm = float(reaction_per_year)
    speed = 2.0 * np.sqrt(max(float(diffusion_km2_per_year) * float(reaction_per_year), 0.0))
    return KoreaPineWiltPhysicsPrior(
        diffusion_km2_per_year=float(diffusion_km2_per_year),
        reaction_per_year=float(reaction_per_year),
        normalized_diffusion=diffusion_norm,
        normalized_diffusion_x=diffusion_x,
        normalized_diffusion_y=diffusion_y,
        normalized_reaction=reaction_norm,
        front_speed_km_per_year=float(speed),
        scale=scale,
    )


def korea_physics_prior_from_normalized(
    grid: KoreaPineWiltGrid,
    *,
    normalized_diffusion: float,
    normalized_reaction: float,
    length_scale_mode: str = "max_extent",
) -> KoreaPineWiltPhysicsPrior:
    scale = korea_domain_scale(grid, length_scale_mode)
    diffusion_phys = float(normalized_diffusion) * scale.length_scale_km**2
    diffusion_x, diffusion_y = korea_physical_to_anisotropic_normalized_diffusion(diffusion_phys, grid)
    reaction_phys = float(normalized_reaction)
    speed = 2.0 * np.sqrt(max(diffusion_phys * reaction_phys, 0.0))
    return KoreaPineWiltPhysicsPrior(
        diffusion_km2_per_year=diffusion_phys,
        reaction_per_year=reaction_phys,
        normalized_diffusion=float(normalized_diffusion),
        normalized_diffusion_x=diffusion_x,
        normalized_diffusion_y=diffusion_y,
        normalized_reaction=float(normalized_reaction),
        front_speed_km_per_year=float(speed),
        scale=scale,
    )


def korea_equivalent_dr_pairs(
    *,
    diffusion_km2_per_year: float,
    reaction_per_year: float,
    multipliers: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> list[dict[str, float]]:
    """Return D/r pairs with the same Fisher front speed 2 sqrt(D r)."""

    product = max(float(diffusion_km2_per_year) * float(reaction_per_year), 0.0)
    rows = []
    for multiplier in multipliers:
        if multiplier <= 0.0:
            continue
        d_value = float(diffusion_km2_per_year) * float(multiplier)
        r_value = product / max(d_value, 1.0e-12)
        rows.append(
            {
                "diffusion_km2_per_year": d_value,
                "reaction_per_year": r_value,
                "front_speed_km_per_year": float(2.0 * np.sqrt(product)),
                "diffusion_multiplier": float(multiplier),
            }
        )
    return rows


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


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(str(value).strip().strip('"'), "%Y-%m-%d").date()


def _month_index(value: date, start: date) -> int:
    return (value.year - start.year) * 12 + (value.month - start.month)


def _month_label_from_index(start: date, index: int) -> str:
    total = start.year * 12 + (start.month - 1) + int(index)
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def _infer_raw_korea_csv_dir(raw_dir: Path | None = None) -> Path:
    if raw_dir is not None:
        raw_path = Path(raw_dir)
        if raw_path.exists():
            return raw_path
        raise FileNotFoundError(f"Korea raw CSV directory does not exist: {raw_path}")
    try:
        hint = Path(str(load_manifest().get("source", {}).get("source_root_hint", "")))
    except Exception:
        hint = Path("")
    candidates = [
        hint,
        REPO_ROOT / "data" / "korea_pine_wilt" / "raw",
        Path.home() / "Downloads" / "korea-pine-wilt-pinn-main" / "korea-pine-wilt-pinn-main" / "Data",
        Path.home() / "Downloads" / "korea-pine-wilt-pinn-main",
    ]
    for candidate in candidates:
        if str(candidate) and candidate.exists() and any(candidate.rglob("*.csv")):
            return candidate
    raise FileNotFoundError(
        "Action-time Korea pine-wilt data needs the original annual CSV files with 조사일자. "
        "Pass --raw-csv-dir pointing to the extracted korea-pine-wilt-pinn Data directory."
    )


def _transform_korea_raw_xy(
    x: np.ndarray,
    y: np.ndarray,
    *,
    raw_crs: str = "EPSG:5181",
    target_crs: str = "EPSG:5179",
) -> tuple[np.ndarray, np.ndarray]:
    if raw_crs == target_crs:
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyproj is required to transform Korea raw CSV coordinates to EPSG:5179.") from exc
    transformer = Transformer.from_crs(raw_crs, target_crs, always_xy=True)
    tx, ty = transformer.transform(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)


def estimate_korea_action_start(
    raw_dir: Path | None = None,
    *,
    cumulative_infected_complete_threshold: int = 50_000,
    infected_label: str = "감염목",
    complete_label: str = "완료",
) -> dict[str, object]:
    """Estimate the first large-scale control-action date from raw KFS rows.

    The raw files do not contain a true cutting/start timestamp. They contain
    조사일자 and 방제완료여부. This estimate therefore reports the first recorded
    completed infected-tree action and the date when cumulative completed infected
    records cross a configurable large-scale threshold.
    """

    raw_path = _infer_raw_korea_csv_dir(raw_dir)
    daily_infected_complete: dict[date, int] = {}
    first_record: date | None = None
    first_infected: date | None = None
    first_infected_complete: date | None = None
    total_rows = 0
    infected_rows = 0
    infected_complete_rows = 0
    csv_files = sorted(raw_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Korea pine-wilt CSV files found below {raw_path}")
    for path in csv_files:
        with path.open("r", encoding="cp949", errors="replace", newline="") as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                continue
            for row in reader:
                if len(row) <= 10:
                    continue
                try:
                    observed_date = _parse_iso_date(row[9])
                except ValueError:
                    continue
                total_rows += 1
                if first_record is None or observed_date < first_record:
                    first_record = observed_date
                infected = row[8].strip() == infected_label
                completed = row[10].strip() == complete_label
                if infected:
                    infected_rows += 1
                    if first_infected is None or observed_date < first_infected:
                        first_infected = observed_date
                if infected and completed:
                    infected_complete_rows += 1
                    daily_infected_complete[observed_date] = daily_infected_complete.get(observed_date, 0) + 1
                    if first_infected_complete is None or observed_date < first_infected_complete:
                        first_infected_complete = observed_date

    cumulative = 0
    threshold_date: date | None = None
    for observed_date in sorted(daily_infected_complete):
        cumulative += daily_infected_complete[observed_date]
        if threshold_date is None and cumulative >= cumulative_infected_complete_threshold:
            threshold_date = observed_date
            break

    if threshold_date is None:
        threshold_date = first_infected_complete
    if first_record is None or first_infected is None or first_infected_complete is None or threshold_date is None:
        raise ValueError("Could not infer action timing from raw Korea CSV files.")
    return {
        "raw_dir": str(raw_path),
        "csv_files": len(csv_files),
        "total_rows": int(total_rows),
        "infected_rows": int(infected_rows),
        "infected_complete_rows": int(infected_complete_rows),
        "first_record_date": first_record.isoformat(),
        "first_infected_date": first_infected.isoformat(),
        "first_infected_complete_date": first_infected_complete.isoformat(),
        "large_scale_threshold": int(cumulative_infected_complete_threshold),
        "large_scale_action_start_date": threshold_date.isoformat(),
        "large_scale_action_start_month": threshold_date.strftime("%Y-%m"),
        "interpretation": (
            "방제완료여부 is not a true cutting-start timestamp. The threshold date is "
            "a data-derived proxy for the start of large-scale completed action records."
        ),
    }


def load_korea_pine_wilt_action_time_points(
    raw_dir: Path | None = None,
    *,
    action_cutoff_date: str | None = None,
    cumulative_infected_complete_threshold: int = 50_000,
    time_bin: str = "month",
    include_action_month: bool = False,
    raw_crs: str = "EPSG:5181",
    infected_label: str = "감염목",
    complete_label: str = "완료",
) -> KoreaActionTimeData:
    """Load infected observations before the estimated large-scale action period.

    Period ids are zero-based bins, not calendar years. ``time_values`` are
    elapsed physical years from the first kept bin, so RK4/PINN dynamics keep
    yearly D/r units even when the observations are monthly.
    """

    if time_bin != "month":
        raise ValueError("Only monthly Korea action-time bins are currently supported.")
    raw_path = _infer_raw_korea_csv_dir(raw_dir)
    action = estimate_korea_action_start(
        raw_path,
        cumulative_infected_complete_threshold=cumulative_infected_complete_threshold,
        infected_label=infected_label,
        complete_label=complete_label,
    )
    cutoff = _parse_iso_date(action_cutoff_date) if action_cutoff_date else _parse_iso_date(str(action["large_scale_action_start_date"]))
    start_date: date | None = None
    rows: list[tuple[float, float, date, str]] = []
    for path in sorted(raw_path.rglob("*.csv")):
        with path.open("r", encoding="cp949", errors="replace", newline="") as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                continue
            for row in reader:
                if len(row) <= 10 or row[8].strip() != infected_label:
                    continue
                try:
                    observed_date = _parse_iso_date(row[9])
                    x = float(row[1])
                    y = float(row[2])
                except ValueError:
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                before_cutoff = observed_date < cutoff
                if include_action_month:
                    before_cutoff = (observed_date.year, observed_date.month) <= (cutoff.year, cutoff.month)
                else:
                    before_cutoff = before_cutoff and (observed_date.year, observed_date.month) < (cutoff.year, cutoff.month)
                if not before_cutoff:
                    continue
                rows.append((x, y, observed_date, row[10].strip()))
                if start_date is None or observed_date < start_date:
                    start_date = observed_date
    if not rows or start_date is None:
        raise ValueError("No pre-action infected observations found for the requested Korea action window.")
    period = np.asarray([_month_index(observed_date, start_date) for _, _, observed_date, _ in rows], dtype=np.int16)
    max_period = int(period.max())
    period_ids = tuple(range(max_period + 1))
    labels = tuple(_month_label_from_index(start_date, idx) for idx in period_ids)
    time_values = np.asarray([idx / 12.0 for idx in period_ids], dtype=np.float64)
    status_lookup = {complete_label: 1}
    metadata = {
        **action,
        "time_axis": "pre_action_month",
        "time_bin": time_bin,
        "include_action_month": bool(include_action_month),
        "action_cutoff_date": cutoff.isoformat(),
        "kept_records": int(len(rows)),
        "kept_periods": int(len(period_ids)),
        "kept_start_month": labels[0],
        "kept_end_month": labels[-1],
        "time_values_are_elapsed_years": True,
    }
    xs = np.asarray([row[0] for row in rows], dtype=np.float64)
    ys = np.asarray([row[1] for row in rows], dtype=np.float64)
    xs, ys = _transform_korea_raw_xy(xs, ys, raw_crs=raw_crs, target_crs="EPSG:5179")
    metadata["raw_crs"] = raw_crs
    metadata["crs"] = "EPSG:5179"
    return KoreaActionTimeData(
        points=KoreaPineWiltPoints(
            x=xs,
            y=ys,
            year=period,
            date_ordinal=np.asarray([row[2].toordinal() for row in rows], dtype=np.int32),
            status=np.asarray([status_lookup.get(row[3], 0) for row in rows], dtype=np.int8),
        ),
        period_ids=period_ids,
        time_values=time_values,
        time_labels=labels,
        metadata=metadata,
    )


def _geojson_polygons(path: Path = DEFAULT_PROVINCE_GEOJSON) -> list[list[np.ndarray]]:
    """Load province GeoJSON polygons as lon/lat rings."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    polygons: list[list[np.ndarray]] = []
    for feature in data.get("features", []):
        geometry = feature.get("geometry", {})
        if geometry.get("type") == "Polygon":
            raw_polygons = [geometry.get("coordinates", [])]
        elif geometry.get("type") == "MultiPolygon":
            raw_polygons = geometry.get("coordinates", [])
        else:
            continue
        for polygon in raw_polygons:
            rings = []
            for ring in polygon:
                arr = np.asarray(ring, dtype=np.float64)
                if arr.ndim == 2 and arr.shape[0] >= 3 and arr.shape[1] >= 2:
                    rings.append(arr[:, :2])
            if rings:
                polygons.append(rings)
    if not polygons:
        raise ValueError(f"No polygon rings found in province GeoJSON: {path}")
    return polygons


def build_land_mask(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    *,
    source_crs: str = "EPSG:5179",
    province_geojson: Path = DEFAULT_PROVINCE_GEOJSON,
) -> np.ndarray:
    """Return True for grid cells whose centers fall inside Korean province polygons."""

    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - dependency guard for incomplete environments.
        raise RuntimeError(
            "pyproj is required to build the Korea land mask because pine-wilt points use EPSG:5179 "
            "and the committed province boundaries are longitude/latitude."
        ) from exc

    xx, yy = np.meshgrid(np.asarray(x_centers, dtype=np.float64), np.asarray(y_centers, dtype=np.float64), indexing="xy")
    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xx, yy)
    lonlat = np.column_stack([np.asarray(lon).ravel(), np.asarray(lat).ravel()])
    mask = np.zeros(lonlat.shape[0], dtype=bool)
    for polygon in _geojson_polygons(province_geojson):
        inside = MplPath(polygon[0]).contains_points(lonlat, radius=1.0e-10)
        for hole in polygon[1:]:
            inside &= ~MplPath(hole).contains_points(lonlat, radius=1.0e-10)
        mask |= inside
    return mask.reshape(xx.shape)


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
    enforce_land_mask: bool = True,
    time_values: np.ndarray | None = None,
    time_labels: tuple[str, ...] | None = None,
    time_unit: str = "year",
    action_metadata: dict[str, object] | None = None,
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
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    land_mask = build_land_mask(x_centers, y_centers, source_crs=points.crs) if enforce_land_mask else None
    if land_mask is not None:
        raw_counts = np.where(land_mask[None, :, :], raw_counts, 0.0)
    positive = raw_counts[raw_counts > 0.0]
    if positive.size == 0:
        raise ValueError("No positive infected-tree observations found for the requested years.")
    capacity = float(np.percentile(positive, capacity_percentile))
    density = np.clip(raw_counts / max(capacity, 1.0e-12), 0.0, 1.0)
    if land_mask is not None:
        density = np.where(land_mask[None, :, :], density, 0.0)
    return KoreaPineWiltGrid(
        years=years_arr,
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=x_centers,
        y_centers=y_centers,
        density=density,
        raw_counts=raw_counts,
        capacity=capacity,
        land_mask=land_mask,
        time_values=None if time_values is None else np.asarray(time_values, dtype=np.float64),
        time_labels=time_labels,
        time_unit=str(time_unit),
        action_metadata=action_metadata,
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


def _laplacian_masked_neumann(u: np.ndarray, dx: float, land_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(land_mask, dtype=bool)
    u_land = np.where(mask, u, 0.0)
    padded_u = np.pad(u_land, 1, mode="edge")
    padded_m = np.pad(mask, 1, mode="edge")
    center = u_land
    up = np.where(padded_m[2:, 1:-1], padded_u[2:, 1:-1], center)
    down = np.where(padded_m[:-2, 1:-1], padded_u[:-2, 1:-1], center)
    right = np.where(padded_m[1:-1, 2:], padded_u[1:-1, 2:], center)
    left = np.where(padded_m[1:-1, :-2], padded_u[1:-1, :-2], center)
    lap = (up + down + right + left - 4.0 * center) / dx**2
    return np.where(mask, lap, 0.0)


def _second_derivatives_neumann(u: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    padded = np.pad(u, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    d2y = (padded[2:, 1:-1] + padded[:-2, 1:-1] - 2.0 * center) / dx**2
    d2x = (padded[1:-1, 2:] + padded[1:-1, :-2] - 2.0 * center) / dx**2
    return d2x, d2y


def _second_derivatives_masked_neumann(
    u: np.ndarray,
    dx: float,
    land_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(land_mask, dtype=bool)
    u_land = np.where(mask, u, 0.0)
    padded_u = np.pad(u_land, 1, mode="edge")
    padded_m = np.pad(mask, 1, mode="edge")
    center = u_land
    up = np.where(padded_m[2:, 1:-1], padded_u[2:, 1:-1], center)
    down = np.where(padded_m[:-2, 1:-1], padded_u[:-2, 1:-1], center)
    right = np.where(padded_m[1:-1, 2:], padded_u[1:-1, 2:], center)
    left = np.where(padded_m[1:-1, :-2], padded_u[1:-1, :-2], center)
    d2y = (up + down - 2.0 * center) / dx**2
    d2x = (right + left - 2.0 * center) / dx**2
    return np.where(mask, d2x, 0.0), np.where(mask, d2y, 0.0)


def _rk4_step_2d(
    u: np.ndarray,
    dt: float,
    dx: float,
    diffusion: float,
    reaction: float,
    land_mask: np.ndarray | None = None,
    diffusion_y: float | None = None,
) -> np.ndarray:
    mask = None if land_mask is None else np.asarray(land_mask, dtype=bool)
    diffusion_x = float(diffusion)
    diffusion_y_value = diffusion_x if diffusion_y is None else float(diffusion_y)

    def rhs(v: np.ndarray) -> np.ndarray:
        if mask is None:
            d2x, d2y = _second_derivatives_neumann(v, dx)
            return diffusion_x * d2x + diffusion_y_value * d2y + reaction * v * (1.0 - v)
        v_land = np.where(mask, v, 0.0)
        d2x, d2y = _second_derivatives_masked_neumann(v_land, dx, mask)
        return np.where(mask, diffusion_x * d2x + diffusion_y_value * d2y + reaction * v_land * (1.0 - v_land), 0.0)

    k1 = rhs(u)
    k2 = rhs(u + 0.5 * dt * k1)
    k3 = rhs(u + 0.5 * dt * k2)
    k4 = rhs(u + dt * k3)
    next_u = np.clip(u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), 0.0, 1.0)
    if mask is not None:
        next_u = np.where(mask, next_u, 0.0)
    return next_u


def korea_grid_time_values(grid: KoreaPineWiltGrid) -> np.ndarray:
    if grid.time_values is not None:
        values = np.asarray(grid.time_values, dtype=np.float64)
        if len(values) != len(grid.years):
            raise ValueError("Korea grid time_values must have the same length as years/period ids.")
        return values
    return np.asarray(grid.years, dtype=np.float64) - float(grid.years[0])


def korea_grid_time_label(grid: KoreaPineWiltGrid, period: int, suffix: str = "") -> str:
    lookup = {int(item): idx for idx, item in enumerate(grid.years.tolist())}
    idx = lookup.get(int(period))
    if idx is not None and grid.time_labels is not None and idx < len(grid.time_labels):
        return f"{grid.time_labels[idx]}{suffix}"
    if grid.time_unit != "year":
        return f"t={float(korea_grid_time_values(grid)[idx]) if idx is not None else float(period):.3f} yr{suffix}"
    return f"{int(period)}{suffix}"


def simulate_density_rk4_at_times(
    initial_density: np.ndarray,
    *,
    output_ids: np.ndarray,
    output_times: np.ndarray,
    diffusion: float = 0.0015,
    diffusion_y: float | None = None,
    reaction: float = 0.70,
    steps_per_year: int = 80,
    land_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run RK4 and return fields at arbitrary elapsed-year output times."""

    ids = np.asarray(output_ids)
    times = np.asarray(output_times, dtype=np.float64)
    if len(ids) != len(times) or len(ids) == 0:
        raise ValueError("output_ids and output_times must be non-empty arrays with the same length.")
    if np.any(np.diff(times) < -1.0e-12):
        raise ValueError("output_times must be sorted in ascending elapsed years.")
    u = np.asarray(initial_density, dtype=np.float64).copy()
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("initial_density must be a square 2D field.")
    mask = None
    if land_mask is not None:
        mask = np.asarray(land_mask, dtype=bool)
        if mask.shape != u.shape:
            raise ValueError(f"land_mask shape {mask.shape} does not match density shape {u.shape}.")
        u = np.where(mask, u, 0.0)
    dx = 1.0 / max(u.shape[0] - 1, 1)
    diffusion_x = float(diffusion)
    diffusion_y_value = diffusion_x if diffusion_y is None else float(diffusion_y)
    fields = [np.where(mask, np.clip(u, 0.0, 1.0), 0.0) if mask is not None else np.clip(u, 0.0, 1.0)]
    current_time = float(times[0])
    for next_time in times[1:]:
        interval = float(next_time - current_time)
        if interval < -1.0e-12:
            raise ValueError("output_times must be sorted in ascending elapsed years.")
        substeps = max(1, int(np.ceil(max(interval, 0.0) * float(steps_per_year))))
        dt = interval / float(substeps) if substeps > 0 else 0.0
        if dt > 0.0:
            dt_diff_limit = 0.69 * dx**2 / (2.0 * max(diffusion_x + diffusion_y_value, 1.0e-12))
            if dt > 0.95 * min(dt_diff_limit, 1.0 / max(float(reaction), 1.0e-12)):
                raise ValueError(
                    "RK4 Korea action-time simulation is outside the practical stability estimate; "
                    f"dt={dt:.3e}, diffusion_limit={dt_diff_limit:.3e}."
                )
            for _ in range(substeps):
                u = _rk4_step_2d(u, dt, dx, diffusion_x, reaction, mask, diffusion_y=diffusion_y_value)
        fields.append(u.copy())
        current_time = float(next_time)
    return ids.astype(np.int16 if np.issubdtype(ids.dtype, np.integer) else ids.dtype), np.asarray(fields)


def simulate_density_rk4(
    initial_density: np.ndarray,
    *,
    start_year: int = 2016,
    end_year: int = 2030,
    diffusion: float = 0.0015,
    diffusion_y: float | None = None,
    reaction: float = 0.70,
    steps_per_year: int = 80,
    land_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a normalized 2D Fisher-KPP RK4 simulation on a square grid."""

    if end_year < start_year:
        raise ValueError("end_year must be >= start_year.")
    u = np.asarray(initial_density, dtype=np.float64).copy()
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("initial_density must be a square 2D field.")
    mask = None
    if land_mask is not None:
        mask = np.asarray(land_mask, dtype=bool)
        if mask.shape != u.shape:
            raise ValueError(f"land_mask shape {mask.shape} does not match density shape {u.shape}.")
        u = np.where(mask, u, 0.0)
    dx = 1.0 / max(u.shape[0] - 1, 1)
    dt = 1.0 / float(steps_per_year)
    diffusion_x = float(diffusion)
    diffusion_y_value = diffusion_x if diffusion_y is None else float(diffusion_y)
    dt_diff_limit = 0.69 * dx**2 / (2.0 * max(diffusion_x + diffusion_y_value, 1.0e-12))
    if dt > 0.95 * min(dt_diff_limit, 1.0 / max(reaction, 1.0e-12)):
        raise ValueError(
            "RK4 Korea simulation is outside the practical stability estimate; "
            f"dt={dt:.3e}, diffusion_limit={dt_diff_limit:.3e}."
        )

    years = np.arange(start_year, end_year + 1, dtype=np.int16)
    fields = [np.where(mask, np.clip(u, 0.0, 1.0), 0.0) if mask is not None else np.clip(u, 0.0, 1.0)]
    for _year in years[:-1]:
        for _ in range(steps_per_year):
            u = _rk4_step_2d(u, dt, dx, diffusion_x, reaction, mask, diffusion_y=diffusion_y_value)
        fields.append(u.copy())
    return years, np.asarray(fields)


def compare_observed_and_simulated(
    observed_grid: KoreaPineWiltGrid,
    sim_years: np.ndarray,
    sim_fields: np.ndarray,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    sim_lookup = {int(year): idx for idx, year in enumerate(sim_years.tolist())}
    time_values = korea_grid_time_values(observed_grid)
    for obs_idx, year in enumerate(observed_grid.years.tolist()):
        if int(year) not in sim_lookup:
            continue
        obs = observed_grid.density[obs_idx]
        pred = sim_fields[sim_lookup[int(year)]]
        if observed_grid.land_mask is not None:
            mask = observed_grid.land_mask
            obs_eval = obs[mask]
            pred_eval = pred[mask]
        else:
            obs_eval = obs.ravel()
            pred_eval = pred.ravel()
        denom = float(np.linalg.norm(obs_eval)) + 1.0e-12
        rel_l2 = float(np.linalg.norm(pred_eval - obs_eval) / denom)
        mae = float(np.mean(np.abs(pred_eval - obs_eval)))
        observed_mass = float(np.mean(obs_eval))
        simulated_mass = float(np.mean(pred_eval))
        mass_absolute_error = abs(simulated_mass - observed_mass)
        mass_relative_error = mass_absolute_error / (abs(observed_mass) + 1.0e-12)
        if np.std(obs_eval) <= 1.0e-12 or np.std(pred_eval) <= 1.0e-12:
            corr = np.nan
        else:
            corr = float(np.corrcoef(obs_eval, pred_eval)[0, 1])
        row = {
            "year": int(year),
            "time_id": int(year),
            "time_label": korea_grid_time_label(observed_grid, int(year)),
            "elapsed_years": float(time_values[obs_idx]),
            "observed_mean": observed_mass,
            "simulated_mean": simulated_mass,
            "mean_absolute_error": mae,
            "mass_absolute_error": mass_absolute_error,
            "mass_relative_error": mass_relative_error,
            "relative_l2": rel_l2,
            "correlation": corr,
        }
        for threshold, suffix in [(0.05, "005"), (0.10, "010")]:
            obs_pos = obs_eval >= threshold
            pred_pos = pred_eval >= threshold
            tp = float(np.logical_and(obs_pos, pred_pos).sum())
            fp = float(np.logical_and(~obs_pos, pred_pos).sum())
            fn = float(np.logical_and(obs_pos, ~pred_pos).sum())
            tn = float(np.logical_and(~obs_pos, ~pred_pos).sum())
            obs_count = tp + fn
            pred_count = tp + fp
            row[f"support_tp_{suffix}"] = tp
            row[f"support_fp_{suffix}"] = fp
            row[f"support_fn_{suffix}"] = fn
            row[f"support_tn_{suffix}"] = tn
            row[f"support_observed_fraction_{suffix}"] = float(obs_count / max(len(obs_eval), 1))
            row[f"support_simulated_fraction_{suffix}"] = float(pred_count / max(len(pred_eval), 1))
            row[f"support_fnr_{suffix}"] = float(fn / obs_count) if obs_count > 0.0 else np.nan
            row[f"support_fpr_{suffix}"] = float(fp / (fp + tn)) if (fp + tn) > 0.0 else np.nan
            row[f"support_dice_{suffix}"] = float(2.0 * tp / (2.0 * tp + fp + fn)) if (tp + fp + fn) > 0.0 else np.nan
            row[f"support_tversky_{suffix}"] = float(tp / (tp + 0.3 * fp + 0.7 * fn)) if (tp + fp + fn) > 0.0 else np.nan
        rows.append(row)
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
    time_values = korea_grid_time_values(grid)
    xyt_rows = []
    value_rows = []
    for idx, _year in enumerate(grid.years.tolist()):
        t = np.full((xy.shape[0], 1), float(time_values[idx]), dtype=np.float32)
        xyt_rows.append(np.concatenate([xy, t], axis=1))
        value_rows.append(grid.density[idx].reshape(-1, 1).astype(np.float32))
    xyt = torch.tensor(np.concatenate(xyt_rows, axis=0), dtype=torch.float32, device=device)
    values = torch.tensor(np.concatenate(value_rows, axis=0), dtype=torch.float32, device=device)
    return xyt, values


def _korea_anisotropic_pde_residual(
    model: OriginPINN,
    xy: torch.Tensor,
    t: torch.Tensor,
    *,
    diffusion_x_scale: float,
    diffusion_y_scale: float,
) -> torch.Tensor:
    xy_req = xy.detach().clone().requires_grad_(True)
    t_req = t.detach().clone().requires_grad_(True)
    u = model(xy_req, t_req)
    ones = torch.ones_like(u)
    u_t = torch.autograd.grad(u, t_req, ones, create_graph=True)[0]
    u_xy = torch.autograd.grad(u, xy_req, ones, create_graph=True)[0]
    u_x = u_xy[:, 0:1]
    u_y = u_xy[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xy_req, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, xy_req, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    diffusion_base = model.diffusion_coefficient(xy_req).clamp_min(1.0e-12)
    if model.has_spatial_coefficients():
        diffusion_grad = torch.autograd.grad(
            diffusion_base,
            xy_req,
            torch.ones_like(diffusion_base),
            create_graph=True,
        )[0]
    else:
        diffusion_grad = torch.zeros_like(xy_req)
    diffusion_flux = (
        float(diffusion_x_scale) * (diffusion_base * u_xx + diffusion_grad[:, 0:1] * u_x)
        + float(diffusion_y_scale) * (diffusion_base * u_yy + diffusion_grad[:, 1:2] * u_y)
    )
    reaction = model.reaction_coefficient(xy_req)
    return u_t - diffusion_flux - reaction * u * (1.0 - u)


def predict_korea_pinn_fields(
    model: OriginPINN,
    grid: KoreaPineWiltGrid,
    years: np.ndarray,
    *,
    start_year: int,
    device: torch.device,
    batch_size: int = 8192,
    time_values: np.ndarray | None = None,
) -> np.ndarray:
    xy = torch.tensor(_normalized_grid_xy(grid), dtype=torch.float32, device=device)
    if time_values is None:
        grid_lookup = {int(year): float(korea_grid_time_values(grid)[idx]) for idx, year in enumerate(grid.years.tolist())}
        eval_times = np.asarray([grid_lookup.get(int(year), float(int(year) - start_year)) for year in years.tolist()], dtype=np.float64)
    else:
        eval_times = np.asarray(time_values, dtype=np.float64)
        if len(eval_times) != len(years):
            raise ValueError("time_values must match years length when predicting Korea PINN fields.")
    fields = []
    land_mask = grid.land_mask
    model.eval()
    with torch.no_grad():
        for time_value in eval_times.tolist():
            t = torch.full((len(xy), 1), float(time_value), dtype=torch.float32, device=device)
            preds = []
            for start in range(0, len(xy), batch_size):
                preds.append(model(xy[start : start + batch_size], t[start : start + batch_size]).detach().cpu())
            field = torch.cat(preds, dim=0).numpy().reshape(grid.density.shape[1:])
            if land_mask is not None:
                field = np.where(land_mask, field, 0.0)
            fields.append(np.clip(field, 0.0, 1.0))
    return np.asarray(fields, dtype=np.float64)


def fit_korea_pine_wilt_pinn(
    grid: KoreaPineWiltGrid,
    *,
    end_year: int = 2030,
    end_time_years: float | None = None,
    epochs: int = 120,
    batch_size: int = 4096,
    collocation_points: int = 768,
    boundary_points: int = 128,
    lr: float = 2.0e-3,
    data_weight: float = 8.0,
    pde_weight: float = 0.05,
    boundary_weight: float = 0.01,
    initial_condition_weight: float = 16.0,
    initial_condition_points: int = 2048,
    sea_weight: float = 2.0,
    diffusion: float = 0.0015,
    reaction: float = 0.20,
    physics_anchor_weight: float = 0.08,
    coefficient_field_weight: float = 0.02,
    physics_length_scale_mode: str = "max_extent",
    seed: int = 7,
    device: str | torch.device | None = None,
) -> KoreaPineWiltPINNResult:
    """Fit the repository's forward PINN architecture to Korea pine-wilt density grids.

    This is a baseline diagnostic, not a calibrated epidemiological model. It uses the
    same compact gridded observations as the RK4 baseline, adds a weak Fisher-KPP PDE
    residual, and reports the same observed-year field metrics.
    """

    start_year = int(grid.years[0])
    grid_times = korea_grid_time_values(grid)
    if end_time_years is None:
        if grid.time_values is not None:
            t_end = float(grid_times[-1])
            model_end_year = int(grid.years[-1])
        else:
            model_end_year = max(int(end_year), int(grid.years[-1]))
            t_end = float(model_end_year - start_year)
    else:
        t_end = float(end_time_years)
        model_end_year = int(grid.years[-1])
    if t_end <= 0.0:
        raise ValueError("Korea PINN end time must be after the first observed time.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)

    domain = DomainConfig(box=1.0, t_end=t_end, grid=int(grid.density.shape[-1]), truth_steps=0)
    pde = PDEConfig(diffusion=diffusion, reaction=reaction, velocity_x=0.0, velocity_y=0.0, include_advection=False)
    physics_prior = korea_physics_prior_from_normalized(
        grid,
        normalized_diffusion=diffusion,
        normalized_reaction=reaction,
        length_scale_mode=physics_length_scale_mode,
    )
    diffusion_x_scale = physics_prior.normalized_diffusion_x / max(float(diffusion), 1.0e-12)
    diffusion_y_scale = physics_prior.normalized_diffusion_y / max(float(diffusion), 1.0e-12)
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
        use_spatial_coefficients=True,
        spatial_coefficient_features=8,
        spatial_coefficient_sigma=0.65,
        spatial_coefficient_hidden=32,
        spatial_coefficient_log_scale=0.35,
    )
    model = OriginPINN(domain, pde, seed_cfg, model_cfg).to(device_obj)
    grid_xy_np = _normalized_grid_xy(grid)
    xyt, values = _korea_training_tensors(grid, start_year=start_year, device=device_obj)
    initial_xy = torch.tensor(grid_xy_np, dtype=torch.float32, device=device_obj)
    initial_values = torch.tensor(grid.density[0].reshape(-1, 1), dtype=torch.float32, device=device_obj)
    initial_density_weight = 1.0 + 12.0 * initial_values.detach()
    sea_flags = None
    land_xy = None
    if grid.land_mask is not None:
        flat_land = grid.land_mask.reshape(-1)
        sea_one_year = (~flat_land).astype(np.float32).reshape(-1, 1)
        sea_flags = torch.tensor(
            np.tile(sea_one_year, (len(grid.years), 1)),
            dtype=torch.float32,
            device=device_obj,
        )
        land_xy = torch.tensor(grid_xy_np[flat_land], dtype=torch.float32, device=device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_obs = len(xyt)
    history: list[dict[str, float]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        idx = torch.randint(0, n_obs, (min(batch_size, n_obs),), device=device_obj)
        pred = model(xyt[idx, :2], xyt[idx, 2:3])
        density_weight = 1.0 + 4.0 * values[idx].detach()
        data_loss = torch.mean(density_weight * (pred - values[idx]) ** 2)
        if initial_condition_weight > 0.0 and initial_condition_points > 0:
            ic_idx = torch.randint(0, len(initial_xy), (min(initial_condition_points, len(initial_xy)),), device=device_obj)
            t0 = torch.zeros((len(ic_idx), 1), dtype=torch.float32, device=device_obj)
            pred0 = model(initial_xy[ic_idx], t0)
            ic_loss = torch.mean(initial_density_weight[ic_idx] * (pred0 - initial_values[ic_idx]) ** 2)
        else:
            ic_loss = torch.zeros((), device=device_obj)
        if sea_flags is not None:
            sea_batch = sea_flags[idx]
            sea_loss = torch.sum(sea_batch * pred**2) / torch.clamp(sea_batch.sum(), min=1.0)
        else:
            sea_loss = torch.zeros((), device=device_obj)

        if land_xy is not None and len(land_xy) > 0:
            col_idx = torch.randint(0, len(land_xy), (collocation_points,), device=device_obj)
            xy_col = land_xy[col_idx]
        else:
            xy_col = torch.rand(collocation_points, 2, device=device_obj)
        t_col = torch.rand(collocation_points, 1, device=device_obj) * domain.t_end
        residual = _korea_anisotropic_pde_residual(
            model,
            xy_col,
            t_col,
            diffusion_x_scale=diffusion_x_scale,
            diffusion_y_scale=diffusion_y_scale,
        )
        pde_loss = torch.mean(residual**2)
        if boundary_points > 0:
            bc_loss = boundary_neumann_loss(model, boundary_points, device_obj)
        else:
            bc_loss = torch.zeros((), device=device_obj)
        if physics_anchor_weight > 0.0:
            ref_diffusion = torch.tensor(float(diffusion), dtype=torch.float32, device=device_obj).clamp_min(1.0e-12)
            ref_reaction = torch.tensor(float(reaction), dtype=torch.float32, device=device_obj).clamp_min(1.0e-12)
            physics_anchor = (
                torch.log(model.pde.diffusion().clamp_min(1.0e-12) / ref_diffusion).pow(2)
                + torch.log(model.pde.reaction().clamp_min(1.0e-12) / ref_reaction).pow(2)
            )
        else:
            physics_anchor = torch.zeros((), device=device_obj)
        if coefficient_field_weight > 0.0 and model.has_spatial_coefficients():
            coefficient_field_loss = spatial_coefficient_regularization_loss(
                model,
                xy_col[: min(len(xy_col), 512)],
            )
        else:
            coefficient_field_loss = torch.zeros((), device=device_obj)

        loss = (
            data_weight * data_loss
            + initial_condition_weight * ic_loss
            + pde_weight * pde_loss
            + boundary_weight * bc_loss
            + sea_weight * sea_loss
            + physics_anchor_weight * physics_anchor
            + coefficient_field_weight * coefficient_field_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "total": float(loss.detach().cpu()),
                    "data": float(data_loss.detach().cpu()),
                    "initial_condition": float(ic_loss.detach().cpu()),
                    "pde": float(pde_loss.detach().cpu()),
                    "boundary": float(bc_loss.detach().cpu()),
                    "sea": float(sea_loss.detach().cpu()),
                    "physics_anchor": float(physics_anchor.detach().cpu()),
                    "coefficient_field": float(coefficient_field_loss.detach().cpu()),
                    "diffusion": float(model.pde.diffusion().detach().cpu()),
                    "diffusion_x": float(model.pde.diffusion().detach().cpu()) * float(diffusion_x_scale),
                    "diffusion_y": float(model.pde.diffusion().detach().cpu()) * float(diffusion_y_scale),
                    "reaction": float(model.pde.reaction().detach().cpu()),
                    **model.coefficient_stats(xy_col[: min(len(xy_col), 512)]),
                }
            )

    if grid.time_values is not None:
        pred_years = grid.years.astype(np.int16)
        pred_times = grid_times
    else:
        pred_years = np.arange(start_year, model_end_year + 1, dtype=np.int16)
        pred_times = None
    fields = predict_korea_pinn_fields(
        model,
        grid,
        pred_years,
        start_year=start_year,
        device=device_obj,
        time_values=pred_times,
    )
    metrics = compare_observed_and_simulated(grid, pred_years, fields)
    mean_l2 = float(np.nanmean([row["relative_l2"] for row in metrics]))
    status = "diagnostic_baseline"
    if epochs < 50:
        status = "diagnostic_only_low_epoch"
    elif mean_l2 > 1.0:
        status = "diagnostic_high_error"
    normalized_physics = model.physics_dict()
    diffusion_phys = korea_normalized_to_physical_diffusion(
        normalized_physics["diffusion"],
        grid,
        length_scale_mode=physics_length_scale_mode,
    )
    reaction_phys = normalized_physics["reaction"]
    speed_phys = 2.0 * np.sqrt(max(diffusion_phys * reaction_phys, 0.0))
    diffusion_x_norm = float(normalized_physics["diffusion"]) * float(diffusion_x_scale)
    diffusion_y_norm = float(normalized_physics["diffusion"]) * float(diffusion_y_scale)
    physics = {
        **normalized_physics,
        **model.coefficient_stats(initial_xy[: min(len(initial_xy), 2048)]),
        "normalized_diffusion": float(normalized_physics["diffusion"]),
        "normalized_diffusion_x": diffusion_x_norm,
        "normalized_diffusion_y": diffusion_y_norm,
        "normalized_reaction": float(normalized_physics["reaction"]),
        "diffusion_km2_per_year": float(diffusion_phys),
        "reaction_per_year": float(reaction_phys),
        "front_speed_km_per_year": float(speed_phys),
        "prior_normalized_diffusion": float(physics_prior.normalized_diffusion),
        "prior_normalized_diffusion_x": float(physics_prior.normalized_diffusion_x),
        "prior_normalized_diffusion_y": float(physics_prior.normalized_diffusion_y),
        "prior_normalized_reaction": float(physics_prior.normalized_reaction),
        "prior_diffusion_km2_per_year": float(physics_prior.diffusion_km2_per_year),
        "prior_reaction_per_year": float(physics_prior.reaction_per_year),
        "prior_front_speed_km_per_year": float(physics_prior.front_speed_km_per_year),
        "physics_anchor_weight": float(physics_anchor_weight),
        "coefficient_field_weight": float(coefficient_field_weight),
        "length_scale_km": float(physics_prior.scale.length_scale_km),
        "length_scale_mode": physics_prior.scale.length_scale_mode,
    }
    return KoreaPineWiltPINNResult(
        years=pred_years,
        fields=fields,
        history=history,
        metrics=metrics,
        physics=physics,
        status=status,
    )
