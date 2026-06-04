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


def tensor_center(model: torch.nn.Module) -> tuple[float, float]:
    center = model.source.center().detach().cpu().numpy()
    return float(center[0]), float(center[1])
