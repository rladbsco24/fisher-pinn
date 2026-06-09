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


def _bc_value(bc, t: float) -> float:
    return float(bc(t)) if callable(bc) else float(bc)


def _apply_dirichlet_bc_at(u: np.ndarray, left_bc, right_bc, t: float) -> np.ndarray:
    return apply_dirichlet_bc(u, _bc_value(left_bc, t), _bc_value(right_bc, t))


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


def rk4_step(u: np.ndarray, dt: float, dx: float, D: float, r: float, left_bc, right_bc, t: float = 0.0) -> np.ndarray:
    """One classical RK4 step for the 1D MOL-FDM Fisher-KPP system."""
    u0 = _apply_dirichlet_bc_at(u, left_bc, right_bc, t)
    k1 = fisher_kpp_rhs(u0, dx, D, r)
    k2 = fisher_kpp_rhs(_apply_dirichlet_bc_at(u0 + 0.5 * dt * k1, left_bc, right_bc, t + 0.5 * dt), dx, D, r)
    k3 = fisher_kpp_rhs(_apply_dirichlet_bc_at(u0 + 0.5 * dt * k2, left_bc, right_bc, t + 0.5 * dt), dx, D, r)
    k4 = fisher_kpp_rhs(_apply_dirichlet_bc_at(u0 + dt * k3, left_bc, right_bc, t + dt), dx, D, r)
    u_next = u0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(_apply_dirichlet_bc_at(u_next, left_bc, right_bc, t + dt), 0.0, 1.0)


def forward_euler_step(
    u: np.ndarray,
    dt: float,
    dx: float,
    D: float,
    r: float,
    left_bc: float,
    right_bc: float,
    t: float = 0.0,
) -> np.ndarray:
    """One forward-Euler step for the 1D MOL-FDM Fisher-KPP system."""
    u0 = _apply_dirichlet_bc_at(u, left_bc, right_bc, t)
    u_next = u0 + dt * fisher_kpp_rhs(u0, dx, D, r)
    return np.clip(_apply_dirichlet_bc_at(u_next, left_bc, right_bc, t + dt), 0.0, 1.0)


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
    t: float = 0.0,
) -> tuple[np.ndarray, int, float]:
    """One theta-method step with Newton solves for the nonlinear logistic term."""
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must satisfy 0 < theta <= 1.")
    u_old = _apply_dirichlet_bc_at(u, left_bc, right_bc, t)
    f_old = fisher_kpp_rhs(u_old, dx, D, r)
    v = _apply_dirichlet_bc_at(u_old + dt * f_old, left_bc, right_bc, t + dt)
    alpha = theta * dt * D / dx**2
    explicit_part = (1.0 - theta) * f_old[1:-1]
    residual_norm = np.inf
    iterations = 0

    for iterations in range(1, max_iter + 1):
        v = _apply_dirichlet_bc_at(v, left_bc, right_bc, t + dt)
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
            v = _apply_dirichlet_bc_at(v, left_bc, right_bc, t + dt)
            f_v = fisher_kpp_rhs(v, dx, D, r)
            g = v[1:-1] - u_old[1:-1] - dt * (explicit_part + theta * f_v[1:-1])
            residual_norm = float(np.linalg.norm(g, ord=np.inf))
            break

    return np.clip(_apply_dirichlet_bc_at(v, left_bc, right_bc, t + dt), 0.0, 1.0), int(iterations), residual_norm


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
    t: float = 0.0,
) -> tuple[np.ndarray, int, float]:
    """One backward-Euler step for the nonlinear 1D Fisher-KPP system."""
    return _theta_implicit_step(u, dt, dx, D, r, left_bc, right_bc, theta=1.0, tol=tol, max_iter=max_iter, t=t)


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
    t: float = 0.0,
) -> tuple[np.ndarray, int, float]:
    """One Crank-Nicolson/trapezoidal step for the nonlinear 1D Fisher-KPP system."""
    return _theta_implicit_step(u, dt, dx, D, r, left_bc, right_bc, theta=0.5, tol=tol, max_iter=max_iter, t=t)


def rk4_step_2d(u: np.ndarray, dt: float, dx: float, D: float, r: float) -> np.ndarray:
    """One classical RK4 step for the 2D no-flux MOL-FDM Fisher-KPP system."""
    u0 = apply_neumann_bc_2d(u)
    k1 = fisher_kpp_rhs_2d(u0, dx, D, r)
    k2 = fisher_kpp_rhs_2d(u0 + 0.5 * dt * k1, dx, D, r)
    k3 = fisher_kpp_rhs_2d(u0 + 0.5 * dt * k2, dx, D, r)
    k4 = fisher_kpp_rhs_2d(u0 + dt * k3, dx, D, r)
    u_next = u0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(apply_neumann_bc_2d(u_next), 0.0, 1.0)


def apply_dirichlet_bc_2d_from_exact(u: np.ndarray, x: np.ndarray, y: np.ndarray, t: float, exact_solution) -> np.ndarray:
    out = np.asarray(u, dtype=np.float64).copy()
    xx, yy = np.meshgrid(x, y, indexing="ij")
    exact = np.asarray(exact_solution(xx, yy, t), dtype=np.float64)
    out[0, :] = exact[0, :]
    out[-1, :] = exact[-1, :]
    out[:, 0] = exact[:, 0]
    out[:, -1] = exact[:, -1]
    return out


def fisher_kpp_rhs_2d_dirichlet(u: np.ndarray, dx: float, D: float, r: float) -> np.ndarray:
    dudt = np.zeros_like(u, dtype=np.float64)
    lap = (
        u[2:, 1:-1]
        + u[:-2, 1:-1]
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        - 4.0 * u[1:-1, 1:-1]
    ) / dx**2
    dudt[1:-1, 1:-1] = D * lap + r * u[1:-1, 1:-1] * (1.0 - u[1:-1, 1:-1])
    return dudt


def rk4_step_2d_dirichlet_exact(
    u: np.ndarray,
    dt: float,
    dx: float,
    x: np.ndarray,
    y: np.ndarray,
    t: float,
    D: float,
    r: float,
    exact_solution,
) -> np.ndarray:
    u0 = apply_dirichlet_bc_2d_from_exact(u, x, y, t, exact_solution)
    k1 = fisher_kpp_rhs_2d_dirichlet(u0, dx, D, r)
    k2 = fisher_kpp_rhs_2d_dirichlet(
        apply_dirichlet_bc_2d_from_exact(u0 + 0.5 * dt * k1, x, y, t + 0.5 * dt, exact_solution),
        dx,
        D,
        r,
    )
    k3 = fisher_kpp_rhs_2d_dirichlet(
        apply_dirichlet_bc_2d_from_exact(u0 + 0.5 * dt * k2, x, y, t + 0.5 * dt, exact_solution),
        dx,
        D,
        r,
    )
    k4 = fisher_kpp_rhs_2d_dirichlet(
        apply_dirichlet_bc_2d_from_exact(u0 + dt * k3, x, y, t + dt, exact_solution),
        dx,
        D,
        r,
    )
    u_next = u0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return np.clip(apply_dirichlet_bc_2d_from_exact(u_next, x, y, t + dt, exact_solution), 0.0, 1.0)


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


def long_time_curve_exact(
    times: np.ndarray,
    rho_inf: float,
    alpha: float,
    omega_d: float,
    rho0: float,
    v0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact underdamped trend used to match the reference long-time curve."""
    times = np.asarray(times, dtype=np.float64)
    c = rho0 - rho_inf
    d = (v0 + alpha * c) / omega_d
    phase = omega_d * times
    envelope = np.exp(-alpha * times)
    q = c * np.cos(phase) + d * np.sin(phase)
    q_prime = -c * omega_d * np.sin(phase) + d * omega_d * np.cos(phase)
    rho = rho_inf + envelope * q
    velocity = envelope * (q_prime - alpha * q)
    return rho, velocity


def _curve_linear_system(alpha: float, omega_d: float, rho_inf: float) -> tuple[np.ndarray, np.ndarray]:
    omega0_sq = alpha**2 + omega_d**2
    a = np.array([[0.0, 1.0], [-omega0_sq, -2.0 * alpha]], dtype=np.float64)
    b = np.array([0.0, omega0_sq * rho_inf], dtype=np.float64)
    return a, b


def _curve_rhs(
    state: np.ndarray,
    alpha: float,
    omega_d: float,
    rho_inf: float,
) -> np.ndarray:
    a, b = _curve_linear_system(alpha, omega_d, rho_inf)
    return a @ np.asarray(state, dtype=np.float64) + b


def _curve_forward_euler_step(
    state: np.ndarray,
    dt: float,
    alpha: float,
    omega_d: float,
    rho_inf: float,
) -> np.ndarray:
    return np.asarray(state, dtype=np.float64) + dt * _curve_rhs(state, alpha, omega_d, rho_inf)


def _curve_rk4_step(
    state: np.ndarray,
    dt: float,
    alpha: float,
    omega_d: float,
    rho_inf: float,
) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    k1 = _curve_rhs(state, alpha, omega_d, rho_inf)
    k2 = _curve_rhs(state + 0.5 * dt * k1, alpha, omega_d, rho_inf)
    k3 = _curve_rhs(state + 0.5 * dt * k2, alpha, omega_d, rho_inf)
    k4 = _curve_rhs(state + dt * k3, alpha, omega_d, rho_inf)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _curve_theta_step(
    state: np.ndarray,
    dt: float,
    alpha: float,
    omega_d: float,
    rho_inf: float,
    theta: float,
) -> np.ndarray:
    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must satisfy 0 < theta <= 1.")
    a, b = _curve_linear_system(alpha, omega_d, rho_inf)
    state = np.asarray(state, dtype=np.float64)
    lhs = np.eye(2, dtype=np.float64) - theta * dt * a
    rhs = state + dt * ((1.0 - theta) * (a @ state) + b)
    return np.linalg.solve(lhs, rhs)


def solve_long_time_curve(
    method: str,
    dt: float,
    Nt: int,
    rho_inf: float,
    alpha: float,
    omega_d: float,
    rho0: float,
    v0: float,
) -> dict[str, np.ndarray]:
    """Integrate the damped long-time curve benchmark with FE/BE/trapezoidal/RK4."""
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

    times = np.linspace(0.0, Nt * dt, Nt + 1)
    states = np.empty((Nt + 1, 2), dtype=np.float64)
    states[0] = np.array([rho0, v0], dtype=np.float64)

    for n in range(Nt):
        if method_name == "forward_euler":
            states[n + 1] = _curve_forward_euler_step(states[n], dt, alpha, omega_d, rho_inf)
        elif method_name == "backward_euler":
            states[n + 1] = _curve_theta_step(states[n], dt, alpha, omega_d, rho_inf, theta=1.0)
        elif method_name == "trapezoidal":
            states[n + 1] = _curve_theta_step(states[n], dt, alpha, omega_d, rho_inf, theta=0.5)
        else:
            states[n + 1] = _curve_rk4_step(states[n], dt, alpha, omega_d, rho_inf)

    exact_rho, exact_velocity = long_time_curve_exact(times, rho_inf, alpha, omega_d, rho0, v0)
    return {
        "method": np.array(method_name),
        "times": times,
        "rho": states[:, 0],
        "velocity": states[:, 1],
        "exact_rho": exact_rho,
        "exact_velocity": exact_velocity,
        "abs_error": np.abs(states[:, 0] - exact_rho),
    }


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
    exact_solution=None,
) -> dict[str, np.ndarray]:
    """Solve the 1D Fisher-KPP equation with Dirichlet boundaries."""
    x = np.asarray(x, dtype=np.float64)
    dx = float(x[1] - x[0])
    u = np.clip(_apply_dirichlet_bc_at(initial_condition(x), left_bc, right_bc, 0.0), 0.0, 1.0)

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
        u = rk4_step(u, dt, dx, D, r, left_bc, right_bc, t=t)

    result = {
        "dimension": np.array(1),
        "x": x,
        "times": np.asarray(times),
        "snapshots": np.asarray(snapshots),
        "fronts": np.asarray(fronts),
        "mass": np.asarray(masses),
        "u_final": u,
    }
    if exact_solution is not None:
        exact_snapshots = np.asarray([exact_solution(x, float(t)) for t in result["times"]])
        exact_final = exact_solution(x, float(Nt * dt))
        result["exact_snapshots"] = exact_snapshots
        result["exact_final"] = np.asarray(exact_final)
        result["abs_error_final"] = np.abs(u - exact_final)
        result["relative_l2_final"] = np.array(relative_l2(u, exact_final))
    return result


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
    exact_solution=None,
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
    u = np.clip(_apply_dirichlet_bc_at(initial_condition(x), left_bc, right_bc, 0.0), 0.0, 1.0)
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
            u = forward_euler_step(u, dt, dx, D, r, left_bc, right_bc, t=t)
            newton_iters.append(0)
            newton_residuals.append(0.0)
        elif method_name == "backward_euler":
            u, iters, residual = backward_euler_step(u, dt, dx, D, r, left_bc, right_bc, tol=tol, max_iter=max_iter, t=t)
            newton_iters.append(iters)
            newton_residuals.append(residual)
        elif method_name == "trapezoidal":
            u, iters, residual = trapezoidal_step(u, dt, dx, D, r, left_bc, right_bc, tol=tol, max_iter=max_iter, t=t)
            newton_iters.append(iters)
            newton_residuals.append(residual)
        else:
            u = rk4_step(u, dt, dx, D, r, left_bc, right_bc, t=t)
            newton_iters.append(0)
            newton_residuals.append(0.0)

    result = {
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
    if exact_solution is not None:
        exact_snapshots = np.asarray([exact_solution(x, float(t)) for t in result["times"]])
        exact_final = exact_solution(x, float(Nt * dt))
        result["exact_snapshots"] = exact_snapshots
        result["exact_final"] = np.asarray(exact_final)
        result["abs_error_final"] = np.abs(u - exact_final)
        result["relative_l2_final"] = np.array(relative_l2(u, exact_final))
    return result


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
    boundary_condition: str = "neumann",
    exact_solution=None,
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
    if boundary_condition == "dirichlet_exact":
        if exact_solution is None:
            raise ValueError("exact_solution is required when boundary_condition='dirichlet_exact'.")
        u = np.clip(apply_dirichlet_bc_2d_from_exact(initial_condition(xx, yy), x, y, 0.0, exact_solution), 0.0, 1.0)
    elif boundary_condition == "neumann":
        u = np.clip(apply_neumann_bc_2d(initial_condition(xx, yy)), 0.0, 1.0)
    else:
        raise ValueError("boundary_condition must be 'neumann' or 'dirichlet_exact'.")

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
        if boundary_condition == "dirichlet_exact":
            u = rk4_step_2d_dirichlet_exact(u, dt, dx, x, y, t, D, r, exact_solution)
        else:
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
    if exact_solution is not None:
        exact_final = np.asarray(exact_solution(xx, yy, Nt * dt), dtype=np.float64)
        result["exact_final"] = exact_final
        result["abs_error_final"] = np.abs(u - exact_final)
        result["relative_l2_final"] = np.array(relative_l2(u, exact_final))
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
