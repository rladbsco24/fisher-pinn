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


def forward_euler_step(
    u: np.ndarray,
    dt: float,
    dx: float,
    D: float,
    r: float,
    left_bc: float,
    right_bc: float,
) -> np.ndarray:
    """One forward-Euler step for the 1D MOL-FDM Fisher-KPP system."""
    u0 = apply_dirichlet_bc(u, left_bc, right_bc)
    u_next = u0 + dt * fisher_kpp_rhs(u0, dx, D, r)
    return np.clip(apply_dirichlet_bc(u_next, left_bc, right_bc), 0.0, 1.0)


def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm for a tridiagonal linear system."""
    n = len(diag)
    if n == 0:
        return np.asarray([], dtype=np.float64)
    a = np.asarray(lower, dtype=np.float64).copy()
    b = np.asarray(diag, dtype=np.float64).copy()
    c = np.asarray(upper, dtype=np.float64).copy()
    d = np.asarray(rhs, dtype=np.float64).copy()
    for i in range(1, n):
        factor = a[i - 1] / b[i - 1]
        b[i] -= factor * c[i - 1]
        d[i] -= factor * d[i - 1]
    x = np.empty(n, dtype=np.float64)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


def _theta_implicit_step(
    u: np.ndarray,
    dt: float,
    dx: float,
    D: float,
    r: float,
    left_bc: float,
    right_bc: float,
    theta: float,
    tol: float = 1.0e-10,
    max_iter: int = 30,
) -> tuple[np.ndarray, int, float]:
    """One theta-method step with Newton solves for the nonlinear logistic term."""
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must satisfy 0 < theta <= 1.")
    u_old = apply_dirichlet_bc(u, left_bc, right_bc)
    f_old = fisher_kpp_rhs(u_old, dx, D, r)
    v = apply_dirichlet_bc(u_old + dt * f_old, left_bc, right_bc)
    alpha = theta * dt * D / dx**2
    explicit_part = (1.0 - theta) * f_old[1:-1]
    residual_norm = np.inf
    iterations = 0

    for iterations in range(1, max_iter + 1):
        v = apply_dirichlet_bc(v, left_bc, right_bc)
        f_v = fisher_kpp_rhs(v, dx, D, r)
        g = v[1:-1] - u_old[1:-1] - dt * (explicit_part + theta * f_v[1:-1])
        residual_norm = float(np.linalg.norm(g, ord=np.inf))
        if residual_norm <= tol:
            break
        diag = 1.0 + 2.0 * alpha - theta * dt * r * (1.0 - 2.0 * v[1:-1])
        off = -alpha * np.ones(max(len(diag) - 1, 0), dtype=np.float64)
        delta = _solve_tridiagonal(off, diag, off, -g)
        v[1:-1] += delta
        if float(np.linalg.norm(delta, ord=np.inf)) <= tol:
            v = apply_dirichlet_bc(v, left_bc, right_bc)
            f_v = fisher_kpp_rhs(v, dx, D, r)
            g = v[1:-1] - u_old[1:-1] - dt * (explicit_part + theta * f_v[1:-1])
            residual_norm = float(np.linalg.norm(g, ord=np.inf))
            break

    return np.clip(apply_dirichlet_bc(v, left_bc, right_bc), 0.0, 1.0), int(iterations), residual_norm


def backward_euler_step(
    u: np.ndarray,
    dt: float,
    dx: float,
    D: float,
    r: float,
    left_bc: float,
    right_bc: float,
    tol: float = 1.0e-10,
    max_iter: int = 30,
) -> tuple[np.ndarray, int, float]:
    """One backward-Euler step for the nonlinear 1D Fisher-KPP system."""
    return _theta_implicit_step(u, dt, dx, D, r, left_bc, right_bc, theta=1.0, tol=tol, max_iter=max_iter)


def trapezoidal_step(
    u: np.ndarray,
    dt: float,
    dx: float,
    D: float,
    r: float,
    left_bc: float,
    right_bc: float,
    tol: float = 1.0e-10,
    max_iter: int = 30,
) -> tuple[np.ndarray, int, float]:
    """One Crank-Nicolson/trapezoidal step for the nonlinear 1D Fisher-KPP system."""
    return _theta_implicit_step(u, dt, dx, D, r, left_bc, right_bc, theta=0.5, tol=tol, max_iter=max_iter)


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


def check_forward_euler_stability(
    dx: float,
    dt: float,
    D: float,
    r: float,
    dim: int = 1,
    safety: float = 0.95,
) -> dict[str, float | bool]:
    """Practical forward-Euler stability estimate for diffusion plus logistic growth."""
    if D <= 0.0 or r <= 0.0:
        raise ValueError("D and r must be positive.")
    if dim not in (1, 2):
        raise ValueError("Only dim=1 or dim=2 is supported.")
    dt_diff_limit = dx**2 / (2.0 * dim * D)
    dt_reaction_scale = 1.0 / r
    dt_practical = safety * min(dt_diff_limit, dt_reaction_scale)
    return {
        "dt": float(dt),
        "dt_diff_limit": float(dt_diff_limit),
        "dt_reaction_scale": float(dt_reaction_scale),
        "dt_practical": float(dt_practical),
        "is_practically_safe": bool(dt <= dt_practical),
    }


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


def solve_1d_method(
    method: str,
    x: np.ndarray,
    dt: float,
    Nt: int,
    D: float,
    r: float,
    initial_condition,
    left_bc: float,
    right_bc: float,
    save_interval: float = 1.0,
    probe_x: float | None = None,
    tol: float = 1.0e-10,
    max_iter: int = 30,
) -> dict[str, np.ndarray]:
    """Solve the same 1D Fisher-KPP problem with FE, BE, trapezoidal, or RK4."""
    normalized = method.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fe": "forward_euler",
        "forward": "forward_euler",
        "forward_euler": "forward_euler",
        "be": "backward_euler",
        "backward": "backward_euler",
        "backward_euler": "backward_euler",
        "tr": "trapezoidal",
        "trap": "trapezoidal",
        "trapezoid": "trapezoidal",
        "trapezoidal": "trapezoidal",
        "cn": "trapezoidal",
        "crank_nicolson": "trapezoidal",
        "rk4": "rk4",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported method {method!r}.")
    method_name = aliases[normalized]
    x = np.asarray(x, dtype=np.float64)
    dx = float(x[1] - x[0])
    u = np.clip(apply_dirichlet_bc(initial_condition(x), left_bc, right_bc), 0.0, 1.0)
    if probe_x is None:
        probe_x = float(0.5 * (x[0] + x[-1]))

    snapshots: list[np.ndarray] = []
    times: list[float] = []
    fronts: list[float] = []
    masses: list[float] = []
    probes: list[float] = []
    newton_iters: list[int] = []
    newton_residuals: list[float] = []
    next_save_t = 0.0

    for n in range(Nt + 1):
        t = n * dt
        if t >= next_save_t - 1.0e-12 or n == Nt:
            snapshots.append(u.copy())
            times.append(t)
            fronts.append(front_position(x, u, level=0.5))
            masses.append(mean_mass(u))
            probes.append(float(np.interp(probe_x, x, u)))
            next_save_t += save_interval
        if n == Nt:
            break
        if method_name == "forward_euler":
            u = forward_euler_step(u, dt, dx, D, r, left_bc, right_bc)
            newton_iters.append(0)
            newton_residuals.append(0.0)
        elif method_name == "backward_euler":
            u, iters, residual = backward_euler_step(u, dt, dx, D, r, left_bc, right_bc, tol=tol, max_iter=max_iter)
            newton_iters.append(iters)
            newton_residuals.append(residual)
        elif method_name == "trapezoidal":
            u, iters, residual = trapezoidal_step(u, dt, dx, D, r, left_bc, right_bc, tol=tol, max_iter=max_iter)
            newton_iters.append(iters)
            newton_residuals.append(residual)
        else:
            u = rk4_step(u, dt, dx, D, r, left_bc, right_bc)
            newton_iters.append(0)
            newton_residuals.append(0.0)

    return {
        "dimension": np.array(1),
        "method": np.array(method_name),
        "x": x,
        "times": np.asarray(times),
        "snapshots": np.asarray(snapshots),
        "fronts": np.asarray(fronts),
        "mass": np.asarray(masses),
        "rho": np.asarray(probes),
        "probe_x": np.array(float(probe_x)),
        "newton_iterations": np.asarray(newton_iters, dtype=np.int16),
        "newton_residual": np.asarray(newton_residuals, dtype=np.float64),
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
