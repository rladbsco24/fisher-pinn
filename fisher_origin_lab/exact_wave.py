from __future__ import annotations

import math

import numpy as np
import torch


AZ_DIFFUSION = 1.0
AZ_REACTION = 1.0
AZ_SPEED = 5.0 / math.sqrt(6.0)
AZ_ALPHA = 1.0 / math.sqrt(6.0)
AZ_X_LEFT_1D = -20.0
AZ_X_RIGHT_1D = 20.0
AZ_T_1D = 10.0
AZ_NX_1D = 201
AZ_DT_1D = 0.005
AZ_X_LEFT_2D = -15.0
AZ_X_RIGHT_2D = 15.0
AZ_Y_BOTTOM_2D = -15.0
AZ_Y_TOP_2D = 15.0
AZ_T_2D = 3.0
AZ_NX_2D = 61
AZ_DT_2D = 0.01


def az_normalized_diffusion(x_left: float = AZ_X_LEFT_1D, x_right: float = AZ_X_RIGHT_1D) -> float:
    length = float(x_right) - float(x_left)
    if length <= 0.0:
        raise ValueError("x_right must be greater than x_left.")
    return AZ_DIFFUSION / (length * length)


def az_physical_x_from_unit(
    x_unit: np.ndarray | torch.Tensor,
    x_left: float = AZ_X_LEFT_1D,
    x_right: float = AZ_X_RIGHT_1D,
) -> np.ndarray | torch.Tensor:
    return float(x_left) + (float(x_right) - float(x_left)) * x_unit


def az_exact_numpy(x: np.ndarray, t: float | np.ndarray, x0: float = 0.0) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - AZ_SPEED * np.asarray(t, dtype=np.float64) - float(x0)) / math.sqrt(6.0)
    return np.power(1.0 + np.exp(np.clip(z, -80.0, 80.0)), -2.0)


def az_exact_torch(x: torch.Tensor, t: torch.Tensor, x0: float = 0.0) -> torch.Tensor:
    z = (x - AZ_SPEED * t - float(x0)) / math.sqrt(6.0)
    return torch.pow(1.0 + torch.exp(torch.clamp(z, -80.0, 80.0)), -2.0)


def az_exact_unit_numpy(
    x_unit: np.ndarray,
    t: float | np.ndarray,
    *,
    x_left: float = AZ_X_LEFT_1D,
    x_right: float = AZ_X_RIGHT_1D,
    x0: float = 0.0,
) -> np.ndarray:
    return az_exact_numpy(az_physical_x_from_unit(np.asarray(x_unit, dtype=np.float64), x_left, x_right), t, x0=x0)


def az_exact_unit_torch(
    x_unit: torch.Tensor,
    t: torch.Tensor,
    *,
    x_left: float = AZ_X_LEFT_1D,
    x_right: float = AZ_X_RIGHT_1D,
    x0: float = 0.0,
) -> torch.Tensor:
    return az_exact_torch(az_physical_x_from_unit(x_unit, x_left, x_right), t, x0=x0)
