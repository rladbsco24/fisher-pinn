from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DomainConfig, PDEConfig, SeedConfig
from .simulate import TruthData, gaussian_seed_numpy


@dataclass(frozen=True)
class RK4StabilityInfo:
    dt: float
    dt_diff_limit: float
    dt_reaction_scale: float
    dt_practical: float
    is_practically_safe: bool


def check_rk4_stability(dx: float, dt: float, pde: PDEConfig, dim: int = 2, safety: float = 0.95) -> RK4StabilityInfo:
    if pde.diffusion <= 0.0 or pde.reaction <= 0.0:
        raise ValueError("diffusion and reaction must be positive for RK4 stability estimation.")
    dt_diff_limit = 0.69 * dx * dx / (dim * pde.diffusion)
    dt_reaction_scale = 1.0 / pde.reaction
    dt_practical = safety * min(dt_diff_limit, dt_reaction_scale)
    return RK4StabilityInfo(
        dt=dt,
        dt_diff_limit=dt_diff_limit,
        dt_reaction_scale=dt_reaction_scale,
        dt_practical=dt_practical,
        is_practically_safe=dt <= dt_practical,
    )


def _apply_neumann_copy(u: np.ndarray) -> np.ndarray:
    out = np.asarray(u, dtype=np.float64).copy()
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def _laplacian_neumann(u: np.ndarray, dx: float) -> np.ndarray:
    padded = np.pad(u, 1, mode="edge")
    return (
        padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
        - 4.0 * padded[1:-1, 1:-1]
    ) / dx**2


def _upwind_advection_neumann(u: np.ndarray, dx: float, pde: PDEConfig) -> np.ndarray:
    padded = np.pad(u, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    if pde.velocity_x >= 0.0:
        ux = (center - padded[:-2, 1:-1]) / dx
    else:
        ux = (padded[2:, 1:-1] - center) / dx
    if pde.velocity_y >= 0.0:
        uy = (center - padded[1:-1, :-2]) / dx
    else:
        uy = (padded[1:-1, 2:] - center) / dx
    return pde.velocity_x * ux + pde.velocity_y * uy


def fisher_kpp_rhs_2d(u: np.ndarray, dx: float, pde: PDEConfig) -> np.ndarray:
    u_bc = _apply_neumann_copy(u)
    advection = _upwind_advection_neumann(u_bc, dx, pde) if pde.include_advection else 0.0
    return (
        -advection
        + pde.diffusion * _laplacian_neumann(u_bc, dx)
        + pde.reaction * u_bc * (1.0 - u_bc)
    )


def rk4_step_2d(u: np.ndarray, dt: float, dx: float, pde: PDEConfig) -> np.ndarray:
    u0 = _apply_neumann_copy(u)
    k1 = fisher_kpp_rhs_2d(u0, dx, pde)
    k2 = fisher_kpp_rhs_2d(u0 + 0.5 * dt * k1, dx, pde)
    k3 = fisher_kpp_rhs_2d(u0 + 0.5 * dt * k2, dx, pde)
    k4 = fisher_kpp_rhs_2d(u0 + dt * k3, dx, pde)
    u_next = u0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(_apply_neumann_copy(u_next), 0.0, 1.0)


def forward_fisher_kpp_rk4(
    domain: DomainConfig,
    pde: PDEConfig,
    seed: SeedConfig,
    snapshots: int = 80,
    steps: int | None = None,
) -> TruthData:
    xs = np.linspace(0.0, domain.box, domain.grid)
    dx = xs[1] - xs[0]
    step_count = int(steps if steps is not None else domain.truth_steps)
    dt = domain.t_end / step_count
    info = check_rk4_stability(dx, dt, pde, dim=2)
    if not info.is_practically_safe:
        raise ValueError(
            "RK4 solver is outside the practical explicit stability estimate; "
            f"dt={dt:.3e}, practical_limit={info.dt_practical:.3e}."
        )

    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")
    u = gaussian_seed_numpy(x_grid, y_grid, seed).astype(np.float64)
    u = _apply_neumann_copy(np.clip(u, 0.0, 1.0))
    fields = [u.copy()]
    times = [0.0]
    stride = max(1, step_count // snapshots)

    for step in range(step_count):
        u = rk4_step_2d(u, dt, dx, pde)
        if (step + 1) % stride == 0 or step == step_count - 1:
            fields.append(u.copy())
            times.append((step + 1) * dt)

    return TruthData(xs=xs, times=np.array(times), fields=np.array(fields))
