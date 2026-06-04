from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import DomainConfig, ObservationConfig, PDEConfig, SeedConfig
from .metrics import centroid_from_field, origin_error
from .simulate import ObservationData, TruthData, truth_field_at


@dataclass(frozen=True)
class BaselineResult:
    name: str
    center: tuple[float, float] | None
    error: float | None
    extra: dict[str, float]


def late_centroid_baseline(truth: TruthData, domain: DomainConfig, seed: SeedConfig) -> BaselineResult:
    xs, field = truth_field_at(truth, domain.t_end)
    center = centroid_from_field(xs, field)
    return BaselineResult(
        name="late_centroid",
        center=center,
        error=origin_error(center, seed),
        extra={},
    )


def observation_weighted_centroid(observations: ObservationData) -> tuple[float, float]:
    xyt = observations.xyt
    values = observations.values[:, 0]
    latest = np.max(xyt[:, 2])
    mask = np.isclose(xyt[:, 2], latest)
    xy = xyt[mask, :2]
    weights = np.clip(values[mask], 0.0, None)
    if weights.sum() <= 1.0e-12:
        center = xy.mean(axis=0)
    else:
        center = (xy * weights[:, None]).sum(axis=0) / weights.sum()
    return (float(center[0]), float(center[1]))


def observation_centroid_baseline(
    observations: ObservationData,
    seed: SeedConfig,
) -> BaselineResult:
    center = observation_weighted_centroid(observations)
    return BaselineResult(
        name="observation_late_weighted_centroid",
        center=center,
        error=origin_error(center, seed),
        extra={"uses_truth_grid": 0.0},
    )


def drift_corrected_observation_centroid_baseline(
    observations: ObservationData,
    domain: DomainConfig,
    pde: PDEConfig,
    seed: SeedConfig,
) -> BaselineResult:
    centroid = np.array(observation_weighted_centroid(observations))
    latest = float(np.max(observations.xyt[:, 2]))
    corrected = centroid - np.array([pde.velocity_x, pde.velocity_y]) * latest
    corrected = np.clip(corrected, 0.0, domain.box)
    center = (float(corrected[0]), float(corrected[1]))
    return BaselineResult(
        name="observation_drift_corrected_centroid",
        center=center,
        error=origin_error(center, seed),
        extra={"uses_known_drift": 1.0, "uses_truth_grid": 0.0},
    )


def drift_corrected_centroid_baseline(
    truth: TruthData,
    domain: DomainConfig,
    pde: PDEConfig,
    seed: SeedConfig,
) -> BaselineResult:
    xs, field = truth_field_at(truth, domain.t_end)
    centroid = np.array(centroid_from_field(xs, field))
    corrected = centroid - np.array([pde.velocity_x, pde.velocity_y]) * domain.t_end
    corrected = np.clip(corrected, 0.0, domain.box)
    center = (float(corrected[0]), float(corrected[1]))
    return BaselineResult(
        name="drift_corrected_centroid",
        center=center,
        error=origin_error(center, seed),
        extra={"uses_known_drift": 1.0},
    )


def naive_backward_blowup(
    truth: TruthData,
    domain: DomainConfig,
    pde: PDEConfig,
    n: int = 64,
    steps: int = 120,
) -> BaselineResult:
    xs, u = truth_field_at(truth, domain.t_end, n=n)
    u = u.copy()
    dx = xs[1] - xs[0]
    dt = domain.t_end / steps
    blowup = float(np.nanmax(np.abs(u)))

    for _ in range(steps):
        ux = np.zeros_like(u)
        uy = np.zeros_like(u)
        if pde.velocity_x >= 0:
            ux[1:, :] = (u[1:, :] - u[:-1, :]) / dx
        else:
            ux[:-1, :] = (u[1:, :] - u[:-1, :]) / dx
        if pde.velocity_y >= 0:
            uy[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
        else:
            uy[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx
        adv = pde.velocity_x * ux + pde.velocity_y * uy
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = (
            u[2:, 1:-1]
            + u[:-2, 1:-1]
            + u[1:-1, 2:]
            + u[1:-1, :-2]
            - 4.0 * u[1:-1, 1:-1]
        ) / dx**2
        u = u - dt * (-adv + pde.diffusion * lap + pde.reaction * u * (1.0 - u))
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        blowup = max(blowup, float(np.nanmax(np.abs(u))))
        if not np.isfinite(blowup) or blowup > 1.0e12:
            break

    return BaselineResult(
        name="naive_backward_fd",
        center=None,
        error=None,
        extra={"max_abs_u": blowup},
    )


class TorchSeedSolver(nn.Module):
    def __init__(self, domain: DomainConfig, pde: PDEConfig, init_seed: SeedConfig, grid: int) -> None:
        super().__init__()
        self.domain = domain
        self.pde = pde
        self.grid = grid
        # Neutral initialization; the true synthetic origin is only used for scoring.
        cx = torch.tensor(0.5).clamp(1.0e-4, 1.0 - 1.0e-4)
        cy = torch.tensor(0.5).clamp(1.0e-4, 1.0 - 1.0e-4)
        self.center_logits = nn.Parameter(torch.logit(torch.stack([cx, cy])))
        init_sigma = max(0.08 * domain.box, init_seed.sigma)
        self.raw_sigma = nn.Parameter(torch.tensor(np.log(np.exp(init_sigma) - 1.0), dtype=torch.float32))
        self.raw_amp = nn.Parameter(torch.tensor(np.log(np.exp(0.35) - 1.0), dtype=torch.float32))

    def center(self) -> torch.Tensor:
        return torch.sigmoid(self.center_logits) * self.domain.box

    def _initial(self, device: torch.device) -> torch.Tensor:
        xs = torch.linspace(0.0, self.domain.box, self.grid, device=device)
        x, y = torch.meshgrid(xs, xs, indexing="ij")
        center = self.center()
        sigma = F.softplus(self.raw_sigma).clamp(0.01, 0.25)
        amp = F.softplus(self.raw_amp).clamp(0.01, 1.0)
        dist2 = (x - center[0]) ** 2 + (y - center[1]) ** 2
        return amp * torch.exp(-dist2 / (2.0 * sigma**2))

    def simulate(self, times: torch.Tensor, steps: int, device: torch.device) -> dict[int, torch.Tensor]:
        dx = self.domain.box / (self.grid - 1)
        dt = self.domain.t_end / steps
        u = self._initial(device)
        wanted = {int(torch.argmin(torch.abs(torch.linspace(0, self.domain.t_end, steps + 1, device=device) - t))): t for t in times}
        snapshots: dict[int, torch.Tensor] = {0: u}
        for step in range(1, steps + 1):
            ux = torch.zeros_like(u)
            uy = torch.zeros_like(u)
            if self.pde.velocity_x >= 0:
                ux[1:, :] = (u[1:, :] - u[:-1, :]) / dx
            else:
                ux[:-1, :] = (u[1:, :] - u[:-1, :]) / dx
            if self.pde.velocity_y >= 0:
                uy[:, 1:] = (u[:, 1:] - u[:, :-1]) / dx
            else:
                uy[:, :-1] = (u[:, 1:] - u[:, :-1]) / dx
            lap = torch.zeros_like(u)
            lap[1:-1, 1:-1] = (
                u[2:, 1:-1]
                + u[:-2, 1:-1]
                + u[1:-1, 2:]
                + u[1:-1, :-2]
                - 4.0 * u[1:-1, 1:-1]
            ) / dx**2
            rhs = (
                -(self.pde.velocity_x * ux + self.pde.velocity_y * uy)
                + self.pde.diffusion * lap
                + self.pde.reaction * u * (1.0 - u)
            )
            u = u + dt * rhs
            u = torch.cat([u[1:2, :], u[1:-1, :], u[-2:-1, :]], dim=0)
            u = torch.cat([u[:, 1:2], u[:, 1:-1], u[:, -2:-1]], dim=1)
            if step in wanted:
                snapshots[step] = u
        return snapshots


def differentiable_seed_baseline(
    observations: ObservationData,
    domain: DomainConfig,
    pde: PDEConfig,
    seed: SeedConfig,
    epochs: int,
    device: torch.device,
    grid: int = 45,
    steps: int = 140,
) -> BaselineResult:
    model = TorchSeedSolver(domain, pde, seed, grid).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3.0e-2)
    xyt = torch.tensor(observations.xyt, dtype=torch.float32, device=device)
    values = torch.tensor(observations.values, dtype=torch.float32, device=device)
    xs_idx = torch.clamp((xyt[:, 0] / domain.box * (grid - 1)).round().long(), 0, grid - 1)
    ys_idx = torch.clamp((xyt[:, 1] / domain.box * (grid - 1)).round().long(), 0, grid - 1)
    unique_times = torch.unique(xyt[:, 2])
    step_grid = torch.linspace(0.0, domain.t_end, steps + 1, device=device)
    time_steps = torch.argmin(torch.abs(step_grid.view(1, -1) - xyt[:, 2:3]), dim=1)

    for _ in range(epochs):
        opt.zero_grad()
        snapshots = model.simulate(unique_times, steps=steps, device=device)
        preds = torch.empty_like(values)
        for step in torch.unique(time_steps):
            mask = time_steps == step
            snap = snapshots[int(step.item())]
            preds[mask, 0] = snap[xs_idx[mask], ys_idx[mask]]
        loss = torch.mean((preds - values) ** 2)
        loss.backward()
        opt.step()

    center_tensor = model.center().detach().cpu().numpy()
    center = (float(center_tensor[0]), float(center_tensor[1]))
    return BaselineResult(
        name="differentiable_fd_seed_fit",
        center=center,
        error=origin_error(center, seed),
        extra={"grid": float(grid), "steps": float(steps), "epochs": float(epochs)},
    )
