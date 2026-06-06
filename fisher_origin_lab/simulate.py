from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import DomainConfig, ObservationConfig, PDEConfig, SeedConfig


@dataclass(frozen=True)
class TruthData:
    xs: np.ndarray
    times: np.ndarray
    fields: np.ndarray


@dataclass(frozen=True)
class ObservationData:
    xyt: np.ndarray
    values: np.ndarray


def gaussian_seed_numpy(x: np.ndarray, y: np.ndarray, seed: SeedConfig) -> np.ndarray:
    dist2 = (x - seed.center_x) ** 2 + (y - seed.center_y) ** 2
    return seed.amplitude * np.exp(-dist2 / (2.0 * seed.sigma**2))


def _upwind_advection(u: np.ndarray, dx: float, pde: PDEConfig) -> np.ndarray:
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
    return pde.velocity_x * ux + pde.velocity_y * uy


def _laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1]
        + u[:-2, 1:-1]
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        - 4.0 * u[1:-1, 1:-1]
    ) / dx**2
    return lap


def _apply_neumann(u: np.ndarray) -> None:
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]


def forward_fisher_kpp(
    domain: DomainConfig,
    pde: PDEConfig,
    seed: SeedConfig,
    snapshots: int = 80,
) -> TruthData:
    xs = np.linspace(0.0, domain.box, domain.grid)
    dx = xs[1] - xs[0]
    dt = domain.t_end / domain.truth_steps
    if dt >= dx * dx / (4.0 * pde.diffusion):
        raise ValueError("Explicit solver is unstable; increase truth_steps or lower grid.")

    x_grid, y_grid = np.meshgrid(xs, xs, indexing="ij")
    u = gaussian_seed_numpy(x_grid, y_grid, seed).astype(np.float64)
    fields = [u.copy()]
    times = [0.0]
    stride = max(1, domain.truth_steps // snapshots)

    for step in range(domain.truth_steps):
        advection = _upwind_advection(u, dx, pde) if pde.include_advection else 0.0
        rhs = (
            -advection
            + pde.diffusion * _laplacian(u, dx)
            + pde.reaction * u * (1.0 - u)
        )
        u = u + dt * rhs
        _apply_neumann(u)
        u = np.clip(u, 0.0, 1.0)
        if (step + 1) % stride == 0 or step == domain.truth_steps - 1:
            fields.append(u.copy())
            times.append((step + 1) * dt)

    return TruthData(xs=xs, times=np.array(times), fields=np.array(fields))


def interpolate_truth(truth: TruthData, xyt: np.ndarray) -> np.ndarray:
    xs = truth.xs
    out = np.empty((len(xyt),), dtype=np.float64)
    for i, (x, y, t) in enumerate(xyt):
        ti = int(np.argmin(np.abs(truth.times - t)))
        f = truth.fields[ti]
        gx = np.clip(x / xs[-1] * (len(xs) - 1), 0, len(xs) - 1)
        gy = np.clip(y / xs[-1] * (len(xs) - 1), 0, len(xs) - 1)
        i0 = int(np.floor(gx))
        j0 = int(np.floor(gy))
        i1 = min(i0 + 1, len(xs) - 1)
        j1 = min(j0 + 1, len(xs) - 1)
        wx = gx - i0
        wy = gy - j0
        out[i] = (
            (1.0 - wx) * (1.0 - wy) * f[i0, j0]
            + wx * (1.0 - wy) * f[i1, j0]
            + (1.0 - wx) * wy * f[i0, j1]
            + wx * wy * f[i1, j1]
        )
    return out


def sample_observations(
    truth: TruthData,
    domain: DomainConfig,
    obs: ObservationConfig,
    rng: np.random.Generator,
) -> ObservationData:
    rows: list[list[float]] = []
    for t in np.linspace(obs.start_time, domain.t_end, obs.frames):
        uniform_n = int(round(obs.samples_per_frame * (1.0 - obs.focus_fraction)))
        focus_n = obs.samples_per_frame - uniform_n
        xy_parts = []
        if uniform_n > 0:
            xy_parts.append(rng.uniform(0.0, domain.box, size=(uniform_n, 2)))
        if focus_n > 0:
            ti = int(np.argmin(np.abs(truth.times - t)))
            field = truth.fields[ti]
            probs = field.ravel() + 1.0e-4
            probs = probs / probs.sum()
            flat_idx = rng.choice(field.size, size=focus_n, replace=True, p=probs)
            ix, iy = np.unravel_index(flat_idx, field.shape)
            dx = domain.box / (len(truth.xs) - 1)
            jitter = rng.uniform(-0.45 * dx, 0.45 * dx, size=(focus_n, 2))
            focused = np.stack([truth.xs[ix], truth.xs[iy]], axis=1) + jitter
            xy_parts.append(np.clip(focused, 0.0, domain.box))
        xy = np.concatenate(xy_parts, axis=0)
        tt = np.full((obs.samples_per_frame, 1), t)
        rows.extend(np.concatenate([xy, tt], axis=1).tolist())
    xyt = np.array(rows, dtype=np.float64)
    values = interpolate_truth(truth, xyt)
    if obs.noise_std > 0.0:
        values = values + rng.normal(0.0, obs.noise_std, size=values.shape)
    values = np.clip(values, 0.0, 1.0)
    return ObservationData(xyt=xyt, values=values[:, None])


def observation_tensors(
    observations: ObservationData,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    xyt = torch.tensor(observations.xyt, dtype=torch.float32, device=device)
    values = torch.tensor(observations.values, dtype=torch.float32, device=device)
    return xyt, values


def split_observations(
    observations: ObservationData,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[ObservationData, ObservationData]:
    n = len(observations.xyt)
    if n == 0:
        raise ValueError("Cannot split an empty observation set.")
    validation_fraction = float(np.clip(validation_fraction, 0.0, 0.9))
    if validation_fraction <= 0.0:
        empty = ObservationData(
            xyt=np.empty((0, 3), dtype=observations.xyt.dtype),
            values=np.empty((0, 1), dtype=observations.values.dtype),
        )
        return observations, empty
    indices = rng.permutation(n)
    val_n = max(1, int(round(n * validation_fraction)))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]
    if len(train_idx) == 0:
        raise ValueError("Validation fraction leaves no training observations.")
    train = ObservationData(xyt=observations.xyt[train_idx], values=observations.values[train_idx])
    validation = ObservationData(xyt=observations.xyt[val_idx], values=observations.values[val_idx])
    return train, validation


def truth_field_at(truth: TruthData, time: float, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    ti = int(np.argmin(np.abs(truth.times - time)))
    if n is None or n == len(truth.xs):
        return truth.xs, truth.fields[ti]
    xs = np.linspace(0.0, truth.xs[-1], n)
    src = truth.xs
    scale = (len(src) - 1) / max(float(src[-1]), 1.0e-12)
    gx = np.clip(xs * scale, 0.0, len(src) - 1)
    gy = gx
    i0 = np.floor(gx).astype(int)
    j0 = np.floor(gy).astype(int)
    i1 = np.minimum(i0 + 1, len(src) - 1)
    j1 = np.minimum(j0 + 1, len(src) - 1)
    wx = (gx - i0)[:, None]
    wy = (gy - j0)[None, :]
    field = truth.fields[ti]
    f00 = field[np.ix_(i0, j0)]
    f10 = field[np.ix_(i1, j0)]
    f01 = field[np.ix_(i0, j1)]
    f11 = field[np.ix_(i1, j1)]
    resampled = (1.0 - wx) * (1.0 - wy) * f00 + wx * (1.0 - wy) * f10 + (1.0 - wx) * wy * f01 + wx * wy * f11
    return xs, resampled
