from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class CurveTrendConfig:
    """Fair long-time scalar trend matching the reference right-panel curve."""

    t_end: float = 30.0
    dt: float = 0.05
    rho_inf: float = 0.34
    alpha: float = 0.24
    omega_d: float = 1.0
    rho0: float = 0.0
    v0: float = 0.60

    @property
    def steps(self) -> int:
        return int(round(self.t_end / self.dt))

    @property
    def omega0_sq(self) -> float:
        return self.alpha**2 + self.omega_d**2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CurvePINNConfig:
    epochs: int = 1600
    lr: float = 2.0e-3
    hidden: int = 64
    layers: int = 3
    collocation_points: int = 192
    observation_points: int = 80
    physics_weight: float = 1.0
    observation_weight: float = 8.0
    initial_weight: float = 10.0
    correction_scale: float = 0.05
    print_every: int = 200

    def quick(self) -> "CurvePINNConfig":
        return CurvePINNConfig(
            epochs=min(self.epochs, 250),
            lr=self.lr,
            hidden=min(self.hidden, 48),
            layers=min(self.layers, 2),
            collocation_points=min(self.collocation_points, 128),
            observation_points=min(self.observation_points, 48),
            physics_weight=self.physics_weight,
            observation_weight=self.observation_weight,
            initial_weight=self.initial_weight,
            correction_scale=self.correction_scale,
            print_every=100,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def curve_exact(times: np.ndarray, cfg: CurveTrendConfig = CurveTrendConfig()) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=np.float64)
    c = cfg.rho0 - cfg.rho_inf
    d = (cfg.v0 + cfg.alpha * c) / cfg.omega_d
    phase = cfg.omega_d * times
    envelope = np.exp(-cfg.alpha * times)
    q = c * np.cos(phase) + d * np.sin(phase)
    q_prime = -c * cfg.omega_d * np.sin(phase) + d * cfg.omega_d * np.cos(phase)
    rho = cfg.rho_inf + envelope * q
    velocity = envelope * (q_prime - cfg.alpha * q)
    return rho, velocity


def _linear_system(cfg: CurveTrendConfig) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[0.0, 1.0], [-cfg.omega0_sq, -2.0 * cfg.alpha]], dtype=np.float64)
    b = np.array([0.0, cfg.omega0_sq * cfg.rho_inf], dtype=np.float64)
    return a, b


def _rhs(state: np.ndarray, cfg: CurveTrendConfig) -> np.ndarray:
    a, b = _linear_system(cfg)
    return a @ np.asarray(state, dtype=np.float64) + b


def _theta_step(state: np.ndarray, dt: float, cfg: CurveTrendConfig, theta: float) -> np.ndarray:
    a, b = _linear_system(cfg)
    state = np.asarray(state, dtype=np.float64)
    lhs = np.eye(2, dtype=np.float64) - theta * dt * a
    rhs = state + dt * ((1.0 - theta) * (a @ state) + b)
    return np.linalg.solve(lhs, rhs)


def _rk4_step(state: np.ndarray, dt: float, cfg: CurveTrendConfig) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    k1 = _rhs(state, cfg)
    k2 = _rhs(state + 0.5 * dt * k1, cfg)
    k3 = _rhs(state + 0.5 * dt * k2, cfg)
    k4 = _rhs(state + dt * k3, cfg)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_curve(method: str, cfg: CurveTrendConfig = CurveTrendConfig()) -> dict[str, np.ndarray]:
    aliases = {
        "fe": "forward_euler",
        "forward": "forward_euler",
        "forward_euler": "forward_euler",
        "be": "backward_euler",
        "backward": "backward_euler",
        "backward_euler": "backward_euler",
        "tr": "trapezoidal",
        "trap": "trapezoidal",
        "trapezoidal": "trapezoidal",
        "cn": "trapezoidal",
        "rk4": "rk4",
    }
    normalized = method.lower().replace("-", "_").replace(" ", "_")
    if normalized not in aliases:
        raise ValueError(f"Unsupported curve method {method!r}.")
    method_name = aliases[normalized]
    times = np.linspace(0.0, cfg.steps * cfg.dt, cfg.steps + 1)
    states = np.empty((cfg.steps + 1, 2), dtype=np.float64)
    states[0] = np.array([cfg.rho0, cfg.v0], dtype=np.float64)
    for n in range(cfg.steps):
        if method_name == "forward_euler":
            states[n + 1] = states[n] + cfg.dt * _rhs(states[n], cfg)
        elif method_name == "backward_euler":
            states[n + 1] = _theta_step(states[n], cfg.dt, cfg, theta=1.0)
        elif method_name == "trapezoidal":
            states[n + 1] = _theta_step(states[n], cfg.dt, cfg, theta=0.5)
        else:
            states[n + 1] = _rk4_step(states[n], cfg.dt, cfg)
    exact_rho, exact_velocity = curve_exact(times, cfg)
    return {
        "method": np.array(method_name),
        "times": times,
        "rho": states[:, 0],
        "velocity": states[:, 1],
        "exact_rho": exact_rho,
        "exact_velocity": exact_velocity,
        "abs_error": np.abs(states[:, 0] - exact_rho),
    }


class CurveTrendPINN(nn.Module):
    """Small ODE-PINN for the damped long-time rho(t) trend."""

    def __init__(
        self,
        cfg: CurveTrendConfig = CurveTrendConfig(),
        hidden: int = 64,
        layers: int = 3,
        correction_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.correction_scale = correction_scale
        width = max(8, hidden)
        modules: list[nn.Module] = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(max(0, layers - 1)):
            modules.extend([nn.Linear(width, width), nn.Tanh()])
        final = nn.Linear(width, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        modules.append(final)
        self.net = nn.Sequential(*modules)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.reshape(-1, 1)
        tau = 2.0 * t / self.cfg.t_end - 1.0
        c = self.cfg.rho0 - self.cfg.rho_inf
        d = (self.cfg.v0 + self.cfg.alpha * c) / self.cfg.omega_d
        base = self.cfg.rho_inf + torch.exp(-self.cfg.alpha * t) * (
            c * torch.cos(self.cfg.omega_d * t) + d * torch.sin(self.cfg.omega_d * t)
        )
        raw = self.net(tau)
        correction = self.correction_scale * (t / self.cfg.t_end) ** 2 * raw
        return base + correction


def curve_pinn_residual(model: CurveTrendPINN, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    t_req = t.detach().clone().reshape(-1, 1).requires_grad_(True)
    rho = model(t_req)
    drho = torch.autograd.grad(rho, t_req, torch.ones_like(rho), create_graph=True)[0]
    d2rho = torch.autograd.grad(drho, t_req, torch.ones_like(drho), create_graph=True)[0]
    cfg = model.cfg
    residual = d2rho + 2.0 * cfg.alpha * drho + cfg.omega0_sq * (rho - cfg.rho_inf)
    return residual, rho


def train_curve_pinn(
    trend_cfg: CurveTrendConfig = CurveTrendConfig(),
    pinn_cfg: CurvePINNConfig = CurvePINNConfig(),
    *,
    device: torch.device | str | None = None,
    seed: int = 7,
) -> dict[str, Any]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    torch.manual_seed(seed)
    model = CurveTrendPINN(
        trend_cfg,
        hidden=pinn_cfg.hidden,
        layers=pinn_cfg.layers,
        correction_scale=pinn_cfg.correction_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=pinn_cfg.lr)

    colloc_t = torch.linspace(0.0, trend_cfg.t_end, pinn_cfg.collocation_points, device=device).reshape(-1, 1)
    obs_t_np = np.linspace(0.0, trend_cfg.t_end, pinn_cfg.observation_points)
    obs_rho_np, _ = curve_exact(obs_t_np, trend_cfg)
    obs_t = torch.tensor(obs_t_np, dtype=torch.float32, device=device).reshape(-1, 1)
    obs_rho = torch.tensor(obs_rho_np, dtype=torch.float32, device=device).reshape(-1, 1)
    t0 = torch.zeros(1, 1, device=device, requires_grad=True)

    history: list[dict[str, float]] = []
    for epoch in range(1, pinn_cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        residual, _ = curve_pinn_residual(model, colloc_t)
        physics_loss = torch.mean(residual**2)
        pred_obs = model(obs_t)
        obs_loss = torch.mean((pred_obs - obs_rho) ** 2)
        rho0 = model(t0)
        drho0 = torch.autograd.grad(rho0, t0, torch.ones_like(rho0), create_graph=True)[0]
        ic_loss = (rho0 - trend_cfg.rho0).pow(2).mean() + (drho0 - trend_cfg.v0).pow(2).mean()
        loss = (
            pinn_cfg.physics_weight * physics_loss
            + pinn_cfg.observation_weight * obs_loss
            + pinn_cfg.initial_weight * ic_loss
        )
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch == pinn_cfg.epochs or epoch % max(1, pinn_cfg.print_every) == 0:
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": float(loss.detach().cpu()),
                    "physics_loss": float(physics_loss.detach().cpu()),
                    "observation_loss": float(obs_loss.detach().cpu()),
                    "initial_loss": float(ic_loss.detach().cpu()),
                }
            )

    times = np.linspace(0.0, trend_cfg.t_end, trend_cfg.steps + 1)
    with torch.no_grad():
        t_eval = torch.tensor(times, dtype=torch.float32, device=device).reshape(-1, 1)
        pred = model(t_eval).detach().cpu().numpy().reshape(-1)
    exact_rho, _ = curve_exact(times, trend_cfg)
    abs_error = np.abs(pred - exact_rho)
    return {
        "model": model,
        "times": times,
        "rho": pred,
        "exact_rho": exact_rho,
        "abs_error": abs_error,
        "history": history,
        "metrics": {
            "max_abs_error": float(abs_error.max()),
            "relative_l2_to_exact": float(np.linalg.norm(pred - exact_rho) / (np.linalg.norm(exact_rho) + 1.0e-12)),
            "final_rho": float(pred[-1]),
        },
        "trend_config": trend_cfg.to_dict(),
        "pinn_config": pinn_cfg.to_dict(),
    }


def save_curve_pinn_outputs(
    out_dir: Path,
    pinn_result: dict[str, Any],
    baselines: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(pinn_result["times"])
    exact = np.asarray(pinn_result["exact_rho"])

    plt.figure(figsize=(9.4, 4.8))
    plt.plot(times, exact, color="black", linestyle="--", linewidth=2.4, label="target exact")
    for method, result in baselines.items():
        plt.plot(result["times"], result["rho"], linewidth=1.2, alpha=0.75, label=method)
    plt.plot(times, pinn_result["rho"], color="tab:purple", linewidth=2.0, label="PINN")
    plt.xlabel("t")
    plt.ylabel("rho(t)")
    plt.title("PINN long-time damped curve trend")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3)
    plt.tight_layout()
    curve_path = out_dir / "pinn_long_time_curve_trend.png"
    plt.savefig(curve_path, dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)
    history = pinn_result["history"]
    if history:
        epochs = [row["epoch"] for row in history]
        axes[0].semilogy(epochs, [row["loss"] for row in history], label="total")
        axes[0].semilogy(epochs, [row["physics_loss"] for row in history], label="physics")
        axes[0].semilogy(epochs, [row["observation_loss"] for row in history], label="obs")
        axes[0].set_title("PINN training losses")
        axes[0].set_xlabel("epoch")
        axes[0].grid(alpha=0.25, which="both")
        axes[0].legend()
    axes[1].semilogy(times, np.maximum(pinn_result["abs_error"], 1.0e-14), label="PINN")
    for method, result in baselines.items():
        axes[1].semilogy(result["times"], np.maximum(result["abs_error"], 1.0e-14), alpha=0.7, label=method)
    axes[1].set_title("absolute error")
    axes[1].set_xlabel("t")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend()
    diagnostics_path = out_dir / "pinn_long_time_curve_diagnostics.png"
    fig.savefig(diagnostics_path, dpi=220)
    plt.close(fig)

    payload: dict[str, np.ndarray] = {
        "times": times,
        "exact_rho": exact,
        "pinn_rho": np.asarray(pinn_result["rho"]),
        "pinn_abs_error": np.asarray(pinn_result["abs_error"]),
    }
    for method, result in baselines.items():
        payload[f"{method}_rho"] = result["rho"]
        payload[f"{method}_abs_error"] = result["abs_error"]
    np.savez(out_dir / "pinn_long_time_curve_results.npz", **payload)
    return {
        "curve_png": str(curve_path),
        "diagnostics_png": str(diagnostics_path),
        "npz": str(out_dir / "pinn_long_time_curve_results.npz"),
    }
