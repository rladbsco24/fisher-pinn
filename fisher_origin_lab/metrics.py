from __future__ import annotations

import numpy as np
import torch

from .config import SeedConfig


def origin_error(center: tuple[float, float] | np.ndarray, seed: SeedConfig) -> float:
    c = np.asarray(center, dtype=np.float64)
    truth = np.array([seed.center_x, seed.center_y], dtype=np.float64)
    return float(np.linalg.norm(c - truth))


def centroid_from_field(xs: np.ndarray, field: np.ndarray) -> tuple[float, float]:
    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")
    mass = field.sum()
    if mass <= 1.0e-12:
        return (float("nan"), float("nan"))
    return (float((x_grid * field).sum() / mass), float((y_grid * field).sum() / mass))


def relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth) / max(np.linalg.norm(truth), 1.0e-12))


def _boundary_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def threshold_boundary_points(xs: np.ndarray, field: np.ndarray, level: float) -> np.ndarray:
    """Approximate an iso-level contour with threshold boundary cell centers."""

    field = np.asarray(field, dtype=np.float64)
    xs = np.asarray(xs, dtype=np.float64)
    if field.ndim != 2 or len(xs) != field.shape[0] or field.shape[0] != field.shape[1]:
        return np.empty((0, 2), dtype=np.float64)
    boundary = _boundary_mask(field >= float(level))
    if not np.any(boundary):
        return np.empty((0, 2), dtype=np.float64)
    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([x_grid[boundary], y_grid[boundary]], axis=1)


def symmetric_boundary_distances(points_a: np.ndarray, points_b: np.ndarray) -> tuple[float, float]:
    """Return symmetric mean nearest-neighbor distance and Hausdorff distance."""

    points_a = np.asarray(points_a, dtype=np.float64)
    points_b = np.asarray(points_b, dtype=np.float64)
    if len(points_a) == 0 or len(points_b) == 0:
        return float("nan"), float("nan")
    diff = points_a[:, None, :] - points_b[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    a_to_b = np.min(distances, axis=1)
    b_to_a = np.min(distances, axis=0)
    mean_distance = 0.5 * (float(np.mean(a_to_b)) + float(np.mean(b_to_a)))
    hausdorff = max(float(np.max(a_to_b)), float(np.max(b_to_a)))
    return mean_distance, hausdorff


def front_boundary_metrics(
    xs: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    level: float = 0.10,
) -> dict[str, float]:
    pred_points = threshold_boundary_points(xs, pred, level)
    truth_points = threshold_boundary_points(xs, truth, level)
    front_mae, hausdorff = symmetric_boundary_distances(pred_points, truth_points)
    return {
        "front_mae": front_mae,
        "hausdorff": hausdorff,
        "pred_boundary_points": float(len(pred_points)),
        "truth_boundary_points": float(len(truth_points)),
    }


def _front_coordinate(
    xs: np.ndarray,
    field: np.ndarray,
    level: float,
    *,
    mode: str,
    center: tuple[float, float],
) -> float:
    points = threshold_boundary_points(xs, field, level)
    if len(points) == 0:
        return float("nan")
    if mode == "x":
        return float(np.mean(points[:, 0]))
    center_arr = np.asarray(center, dtype=np.float64).reshape(1, 2)
    return float(np.mean(np.linalg.norm(points - center_arr, axis=1)))


def front_speed_error_from_fields(
    xs: np.ndarray,
    times: np.ndarray,
    pred_fields: list[np.ndarray] | np.ndarray,
    truth_fields: list[np.ndarray] | np.ndarray,
    level: float = 0.10,
    *,
    mode: str = "radius",
    center: tuple[float, float] = (0.5, 0.5),
) -> dict[str, object]:
    """Compare observed threshold-front speeds from predicted and truth fields."""

    times = np.asarray(times, dtype=np.float64)
    pred_arr = np.asarray(pred_fields, dtype=np.float64)
    truth_arr = np.asarray(truth_fields, dtype=np.float64)
    if len(times) < 2 or pred_arr.shape[0] != len(times) or truth_arr.shape[0] != len(times):
        return {"mae": float("nan"), "pred_speed": [], "truth_speed": [], "times": []}

    pred_coord = np.asarray(
        [_front_coordinate(xs, field, level, mode=mode, center=center) for field in pred_arr],
        dtype=np.float64,
    )
    truth_coord = np.asarray(
        [_front_coordinate(xs, field, level, mode=mode, center=center) for field in truth_arr],
        dtype=np.float64,
    )
    valid = np.isfinite(pred_coord) & np.isfinite(truth_coord) & np.isfinite(times)
    if int(np.sum(valid)) < 2:
        return {
            "mae": float("nan"),
            "pred_coordinate": pred_coord.tolist(),
            "truth_coordinate": truth_coord.tolist(),
            "pred_speed": [],
            "truth_speed": [],
            "times": times.tolist(),
        }
    pred_speed = np.gradient(pred_coord[valid], times[valid])
    truth_speed = np.gradient(truth_coord[valid], times[valid])
    return {
        "mae": float(np.mean(np.abs(pred_speed - truth_speed))),
        "pred_coordinate": pred_coord.tolist(),
        "truth_coordinate": truth_coord.tolist(),
        "pred_speed": pred_speed.tolist(),
        "truth_speed": truth_speed.tolist(),
        "times": times[valid].tolist(),
    }


def tensor_center(model: torch.nn.Module) -> tuple[float, float]:
    center = model.source.center().detach().cpu().numpy()
    return float(center[0]), float(center[1])
