from __future__ import annotations

import torch

from .models import OriginPINN


def _central_advection(u: torch.Tensor, dx: float, velocity: torch.Tensor) -> torch.Tensor:
    ux = torch.zeros_like(u)
    uy = torch.zeros_like(u)
    ux[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * dx)
    uy[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dx)
    ux[0, :] = (u[1, :] - u[0, :]) / dx
    ux[-1, :] = (u[-1, :] - u[-2, :]) / dx
    uy[:, 0] = (u[:, 1] - u[:, 0]) / dx
    uy[:, -1] = (u[:, -1] - u[:, -2]) / dx
    return velocity[0] * ux + velocity[1] * uy


def _laplacian(u: torch.Tensor, dx: float) -> torch.Tensor:
    lap = torch.zeros_like(u)
    lap[1:-1, 1:-1] = (
        u[2:, 1:-1]
        + u[:-2, 1:-1]
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        - 4.0 * u[1:-1, 1:-1]
    ) / dx**2
    return lap


def _apply_neumann(u: torch.Tensor) -> torch.Tensor:
    u = torch.cat([u[1:2, :], u[1:-1, :], u[-2:-1, :]], dim=0)
    return torch.cat([u[:, 1:2], u[:, 1:-1], u[:, -2:-1]], dim=1)


def source_shooting_loss(
    model: OriginPINN,
    xyt: torch.Tensor,
    values: torch.Tensor,
    grid: int,
    steps: int,
    max_points: int,
) -> torch.Tensor:
    """Coarse differentiable source-to-observation consistency loss.

    The neural field still handles continuous reconstruction and PDE residuals. This loss
    adds a separate shooting constraint that directly connects the trainable source head to
    late observations through a coarse finite-difference solver.
    """

    if len(xyt) == 0:
        return torch.zeros((), dtype=values.dtype, device=values.device)

    if max_points > 0 and len(xyt) > max_points:
        idx = torch.randperm(len(xyt), device=xyt.device)[:max_points]
        xyt = xyt[idx]
        values = values[idx]

    dx = model.domain.box / (grid - 1)
    dt = model.domain.t_end / steps
    diffusion = model.pde.diffusion()
    if dt >= dx * dx / (4.0 * float(diffusion.detach().cpu())):
        raise ValueError("Shooting solver is unstable; increase shooting_steps or lower shooting_grid.")

    xs = torch.linspace(0.0, model.domain.box, grid, dtype=xyt.dtype, device=xyt.device)
    x_grid, y_grid = torch.meshgrid(xs, xs, indexing="ij")
    xy_grid = torch.stack([x_grid.reshape(-1), y_grid.reshape(-1)], dim=1)
    u = model.source.profile(xy_grid).reshape(grid, grid)

    sample_i = torch.clamp((xyt[:, 0] / model.domain.box * (grid - 1)).round().long(), 0, grid - 1)
    sample_j = torch.clamp((xyt[:, 1] / model.domain.box * (grid - 1)).round().long(), 0, grid - 1)
    target_steps = torch.clamp((xyt[:, 2] / model.domain.t_end * steps).round().long(), 0, steps)
    wanted_steps = set(int(step.item()) for step in torch.unique(target_steps))
    preds = torch.empty_like(values[:, 0])

    for step in range(steps + 1):
        if step in wanted_steps:
            mask = target_steps == step
            preds[mask] = u[sample_i[mask], sample_j[mask]]
        if step == steps:
            break
        advection = _central_advection(u, dx, model.pde.velocity) if model.pde.include_advection else torch.zeros_like(u)
        rhs = (
            -advection
            + diffusion * _laplacian(u, dx)
            + model.pde.reaction() * u * (1.0 - u)
        )
        u = _apply_neumann(u + dt * rhs)

    return torch.mean((preds[:, None] - values) ** 2)
