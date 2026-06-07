from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RK4StabilityInfo:
    dt: float
    dt_diff_limit: float
    dt_reaction_scale: float
    dt_practical: float
    is_practically_safe: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "dt": self.dt,
            "dt_diff_limit": self.dt_diff_limit,
            "dt_reaction_scale": self.dt_reaction_scale,
            "dt_practical": self.dt_practical,
            "is_practically_safe": self.is_practically_safe,
        }


def apply_dirichlet_bc(u: np.ndarray, left_bc: float, right_bc: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64).copy()
    u[0] = left_bc
    u[-1] = right_bc
    return u


def apply_neumann_bc_2d(u: np.ndarray) -> np.ndarray:
    out = np.asarray(u, dtype=np.float64).copy()
    if out.ndim != 2:
        raise ValueError("2D Neumann boundary conditions require a 2D array.")
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    out[:, 0] = out[:, 1]
    out[:, -1] = out[:, -2]
    return out


def fisher_kpp_rhs(u: np.ndarray, dx: float, D: float, r: float) -> np.ndarray:
    """Semi-discrete 1D RHS: u_t = D u_xx + r u(1-u)."""
    dudt = np.zeros_like(u, dtype=np.float64)
    lap = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
    reaction = r * u[1:-1] * (1.0 - u[1:-1])
    dudt[1:-1] = D * lap + reaction
    return dudt


def fisher_kpp_rhs_2d(u: np.ndarray, dx: float, D: float, r: float) -> np.ndarray:
    """Semi-discrete 2D RHS with no-flux boundaries."""
    u_bc = apply_neumann_bc_2d(u)
    padded = np.pad(u_bc, 1, mode="edge")
    lap = (
        padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
        - 4.0 * padded[1:-1, 1:-1]
    ) / dx**2
    return D * lap + r * u_bc * (1.0 - u_bc)


def rk4_step(u: np.ndarray, dt: float, dx: float, D: float, r: float, left_bc: float, right_bc: float) -> np.ndarray:
    """One classical RK4 step for the 1D MOL-FDM Fisher-KPP system."""
    u0 = apply_dirichlet_bc(u, left_bc, right_bc)
    k1 = fisher_kpp_rhs(u0, dx, D, r)
    k2 = fisher_kpp_rhs(apply_dirichlet_bc(u0 + 0.5 * dt * k1, left_bc, right_bc), dx, D, r)
    k3 = fisher_kpp_rhs(apply_dirichlet_bc(u0 + 0.5 * dt * k2, left_bc, right_bc), dx, D, r)
    k4 = fisher_kpp_rhs(apply_dirichlet_bc(u0 + dt * k3, left_bc, right_bc), dx, D, r)
    u_next = u0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(apply_dirichlet_bc(u_next, left_bc, right_bc), 0.0, 1.0)


def rk4_step_2d(u: np.ndarray, dt: float, dx: float, D: float, r: float) -> np.ndarray:
    """One classical RK4 step for the 2D no-flux MOL-FDM Fisher-KPP system."""
    u0 = apply_neumann_bc_2d(u)
    k1 = fisher_kpp_rhs_2d(u0, dx, D, r)
    k2 = fisher_kpp_rhs_2d(u0 + 0.5 * dt * k1, dx, D, r)
    k3 = fisher_kpp_rhs_2d(u0 + 0.5 * dt * k2, dx, D, r)
    k4 = fisher_kpp_rhs_2d(u0 + dt * k3, dx, D, r)
    u_next = u0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(apply_neumann_bc_2d(u_next), 0.0, 1.0)


def check_rk4_stability(dx: float, dt: float, D: float, r: float, dim: int = 1, safety: float = 0.95) -> dict[str, float | bool]:
    """Practical explicit RK4 stability estimate for diffusion-reaction systems."""
    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")
    if dim not in (1, 2):
        raise ValueError("Only dim=1 or dim=2 is supported.")
    dt_diff_limit = 0.69 * dx**2 / (dim * D)
    dt_reaction_scale = 1.0 / r
    dt_practical = safety * min(dt_diff_limit, dt_reaction_scale)
    return RK4StabilityInfo(
        dt=dt,
        dt_diff_limit=dt_diff_limit,
        dt_reaction_scale=dt_reaction_scale,
        dt_practical=dt_practical,
        is_practically_safe=dt <= dt_practical,
    ).as_dict()


def front_position(x: np.ndarray, u: np.ndarray, level: float = 0.5) -> float:
    """Linear interpolation of the first 1D crossing u=level."""
    diff = np.asarray(u) - level
    idx = np.where(diff[:-1] * diff[1:] <= 0.0)[0]
    if len(idx) == 0:
        return np.nan
    i = int(idx[0])
    x0, x1 = x[i], x[i + 1]
    u0, u1 = u[i], u[i + 1]
    if abs(u1 - u0) < 1.0e-14:
        return float(x0)
    return float(x0 + (level - u0) * (x1 - x0) / (u1 - u0))


def area_fraction_2d(u: np.ndarray, level: float = 0.1) -> float:
    """Fraction of 2D cells whose concentration exceeds a level."""
    return float(np.mean(np.asarray(u) >= level))


def mean_mass(u: np.ndarray) -> float:
    return float(np.mean(np.asarray(u)))


def relative_l2(u_num: np.ndarray, u_ref: np.ndarray) -> float:
    denom = np.linalg.norm(u_ref)
    if denom == 0.0:
        return np.nan
    return float(np.linalg.norm(u_num - u_ref) / denom)


def solve_rk4(
    x: np.ndarray,
    dt: float,
    Nt: int,
    D: float,
    r: float,
    initial_condition,
    left_bc: float,
    right_bc: float,
    save_interval: float = 5.0,
) -> dict[str, np.ndarray]:
    """Solve the 1D Fisher-KPP equation with Dirichlet boundaries."""
    x = np.asarray(x, dtype=np.float64)
    dx = float(x[1] - x[0])
    u = np.clip(apply_dirichlet_bc(initial_condition(x), left_bc, right_bc), 0.0, 1.0)

    snapshots: list[np.ndarray] = []
    times: list[float] = []
    fronts: list[float] = []
    masses: list[float] = []
    next_save_t = 0.0

    for n in range(Nt + 1):
        t = n * dt
        if t >= next_save_t - 1.0e-12 or n == Nt:
            snapshots.append(u.copy())
            times.append(t)
            fronts.append(front_position(x, u, level=0.5))
            masses.append(mean_mass(u))
            next_save_t += save_interval
        if n == Nt:
            break
        u = rk4_step(u, dt, dx, D, r, left_bc, right_bc)

    return {
        "dimension": np.array(1),
        "x": x,
        "times": np.asarray(times),
        "snapshots": np.asarray(snapshots),
        "fronts": np.asarray(fronts),
        "mass": np.asarray(masses),
        "u_final": u,
    }


def solve_rk4_2d(
    x: np.ndarray,
    y: np.ndarray,
    dt: float,
    Nt: int,
    D: float,
    r: float,
    initial_condition,
    save_interval: float = 0.05,
    front_levels: tuple[float, ...] = (0.05, 0.10),
) -> dict[str, np.ndarray]:
    """Solve the 2D Fisher-KPP equation with no-flux boundaries."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("x and y must be one-dimensional grids with the same length.")
    dx = float(x[1] - x[0])
    if not np.allclose(np.diff(x), dx) or not np.allclose(np.diff(y), dx):
        raise ValueError("2D RK4 solver expects a uniform square grid.")

    xx, yy = np.meshgrid(x, y, indexing="ij")
    u = np.clip(apply_neumann_bc_2d(initial_condition(xx, yy)), 0.0, 1.0)

    snapshots: list[np.ndarray] = []
    times: list[float] = []
    masses: list[float] = []
    areas: dict[float, list[float]] = {level: [] for level in front_levels}
    next_save_t = 0.0

    for n in range(Nt + 1):
        t = n * dt
        if t >= next_save_t - 1.0e-12 or n == Nt:
            snapshots.append(u.copy())
            times.append(t)
            masses.append(mean_mass(u))
            for level in front_levels:
                areas[level].append(area_fraction_2d(u, level=level))
            next_save_t += save_interval
        if n == Nt:
            break
        u = rk4_step_2d(u, dt, dx, D, r)

    result: dict[str, np.ndarray] = {
        "dimension": np.array(2),
        "x": x,
        "y": y,
        "times": np.asarray(times),
        "snapshots": np.asarray(snapshots),
        "mass": np.asarray(masses),
        "u_final": u,
    }
    for level, values in areas.items():
        result[f"area_ge_{level:.2f}"] = np.asarray(values)
    return result


def estimate_front_speed(times: np.ndarray, fronts: np.ndarray, t_min: float | None = None, x_max: float | None = None) -> float:
    times = np.asarray(times)
    fronts = np.asarray(fronts)
    mask = np.isfinite(fronts)
    if t_min is not None:
        mask &= times >= t_min
    if x_max is not None:
        mask &= fronts <= x_max
    if np.count_nonzero(mask) < 2:
        return np.nan
    slope, _ = np.polyfit(times[mask], fronts[mask], deg=1)
    return float(slope)

