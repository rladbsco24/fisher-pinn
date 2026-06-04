from __future__ import annotations

import torch

from .models import OriginPINN
from .config import SeedConfig


def pde_residual(
    model: OriginPINN,
    xy: torch.Tensor,
    t: torch.Tensor,
    return_inputs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual, _, _, xy_req, t_req = pde_residual_terms(model, xy, t)
    if return_inputs:
        return residual, xy_req, t_req
    return residual


def pde_residual_terms(
    model: OriginPINN,
    xy: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xy_req = xy.detach().clone().requires_grad_(True)
    t_req = t.detach().clone().requires_grad_(True)
    u = model(xy_req, t_req)
    grad = torch.autograd.grad
    u_t = grad(u, t_req, torch.ones_like(u), create_graph=True)[0]
    u_xy = grad(u, xy_req, torch.ones_like(u), create_graph=True)[0]
    lap = torch.zeros_like(u)
    for dim in range(2):
        component = u_xy[:, dim : dim + 1]
        second = grad(component, xy_req, torch.ones_like(component), create_graph=True)[0]
        lap = lap + second[:, dim : dim + 1]
    if model.pde.include_advection:
        adv = model.pde.velocity[0] * u_xy[:, 0:1] + model.pde.velocity[1] * u_xy[:, 1:2]
    else:
        adv = torch.zeros_like(u)
    residual = u_t + adv - model.pde.diffusion() * lap - model.pde.reaction() * u * (1.0 - u)
    return residual, u, u_xy, xy_req, t_req


def front_indicator_weights(
    u: torch.Tensor,
    u_xy: torch.Tensor,
    alpha: float,
    gradient_beta: float,
) -> torch.Tensor:
    weights = torch.ones_like(u)
    if alpha > 0.0:
        front = (u.detach() * (1.0 - u.detach())).clamp_min(0.0)
        weights = weights + alpha * front
    if gradient_beta > 0.0:
        grad_norm = torch.linalg.norm(u_xy.detach(), dim=-1, keepdim=True)
        weights = weights + gradient_beta * grad_norm
    return weights.clamp(0.25, 8.0)


def front_speed_kinematics(
    model: OriginPINN,
    xy: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1.0e-3,
) -> dict[str, torch.Tensor]:
    """KPP moving-front kinematics for the implicit level sets of u.

    In the leading-edge approximation of Fisher-KPP, the asymptotic front speed
    relative to the medium is c*=2*sqrt(D*r). With outward normal n, a level set
    should satisfy u_t + (v.n + c*) grad(u).n = 0 near the active front.
    """

    xy_req = xy.detach().clone().requires_grad_(True)
    t_req = t.detach().clone().requires_grad_(True)
    u = model(xy_req, t_req)
    grad = torch.autograd.grad
    u_t = grad(u, t_req, torch.ones_like(u), create_graph=True)[0]
    u_xy = grad(u, xy_req, torch.ones_like(u), create_graph=True)[0]

    center = model.seed_center.to(dtype=xy_req.dtype, device=xy_req.device).view(1, 2)
    radial = xy_req - center
    radius = torch.linalg.norm(radial, dim=-1, keepdim=True).clamp_min(eps)
    normal = radial / radius
    directional_grad = torch.sum(u_xy * normal, dim=-1, keepdim=True)
    advective_speed = torch.sum(model.pde.velocity.view(1, 2) * normal, dim=-1, keepdim=True)
    kpp_speed = 2.0 * torch.sqrt(
        model.pde.diffusion().clamp_min(1.0e-10) * model.pde.reaction().clamp_min(1.0e-10)
    )
    target_normal_speed = advective_speed + kpp_speed
    signed_residual = u_t + target_normal_speed * directional_grad
    abs_denom = directional_grad.detach().abs().clamp_min(eps)
    signed_denom = torch.where(
        directional_grad.detach().abs() < eps,
        -torch.full_like(directional_grad, eps),
        directional_grad,
    )
    speed_error = signed_residual / abs_denom
    observed_normal_speed = -u_t / signed_denom
    front_indicator = (u.detach() * (1.0 - u.detach())).clamp_min(0.0)
    return {
        "u": u,
        "u_t": u_t,
        "u_xy": u_xy,
        "normal": normal,
        "directional_grad": directional_grad,
        "target_normal_speed": target_normal_speed,
        "observed_normal_speed": observed_normal_speed,
        "signed_residual": signed_residual,
        "speed_error": speed_error,
        "front_indicator": front_indicator,
    }


def boundary_neumann_loss(model: OriginPINN, n: int, device: torch.device) -> torch.Tensor:
    t = torch.rand(n, 1, device=device) * model.domain.t_end
    xy = torch.rand(n, 2, device=device) * model.domain.box
    faces = torch.randint(0, 4, (n,), device=device)
    dims = faces // 2
    xy[torch.arange(n, device=device), dims] = (faces % 2).float() * model.domain.box
    xy.requires_grad_(True)
    u = model(xy, t)
    du = torch.autograd.grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    return (du[torch.arange(n, device=device), dims] ** 2).mean()


def seed_regularization_loss(
    model: OriginPINN,
    n: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    xy = torch.rand(n, 2, device=device) * model.domain.box
    t0 = torch.zeros(n, 1, device=device)
    predicted = model(xy, t0)
    source = model.source.profile(xy)
    match = torch.mean((predicted - source) ** 2)
    mass = predicted.mean() + 0.25 * source.mean()
    return match, mass


def known_initial_condition_loss(
    model: OriginPINN,
    seed: SeedConfig,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    uniform_n = n // 2
    focused_n = n - uniform_n
    xy_parts = []
    if uniform_n > 0:
        xy_parts.append(torch.rand(uniform_n, 2, device=device) * model.domain.box)
    if focused_n > 0:
        center = torch.tensor([seed.center_x, seed.center_y], dtype=torch.float32, device=device)
        jitter = torch.randn(focused_n, 2, device=device) * (3.0 * seed.sigma)
        xy_parts.append((center + jitter).clamp(0.0, model.domain.box))
    xy = torch.cat(xy_parts, dim=0)
    t0 = torch.zeros(len(xy), 1, device=device)
    pred = model(xy, t0)
    dist2 = (xy[:, 0:1] - seed.center_x) ** 2 + (xy[:, 1:2] - seed.center_y) ** 2
    target = seed.amplitude * torch.exp(-dist2 / (2.0 * seed.sigma**2))
    return torch.mean((pred - target) ** 2)


def parabolic_mass_balance_loss(
    model: OriginPINN,
    n_times: int,
    grid: int,
    device: torch.device,
) -> torch.Tensor:
    """No-flux Fisher-KPP integral balance over the square domain.

    For u_t = D Laplacian(u) + r u(1-u) with Neumann boundaries, integrating
    over the domain removes the diffusion term, leaving d mean(u)/dt =
    r mean(u(1-u)). This low-dimensional parabolic constraint is cheap and
    discourages sparse-data fits that create the right local blobs but the
    wrong global growth trajectory.
    """

    if n_times <= 0 or grid <= 1 or model.pde.include_advection:
        return torch.zeros((), device=device)
    xs = torch.linspace(0.0, model.domain.box, grid, device=device)
    x, y = torch.meshgrid(xs, xs, indexing="ij")
    xy_one = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    times = torch.linspace(
        0.0,
        model.domain.t_end,
        n_times + 2,
        device=device,
    )[1:-1].reshape(-1, 1)
    xy = xy_one.repeat(n_times, 1)
    t = times.repeat_interleave(grid * grid, dim=0).requires_grad_(True)
    u = model(xy, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_by_time = u.reshape(n_times, grid * grid, 1)
    ut_by_time = u_t.reshape(n_times, grid * grid, 1)
    lhs = ut_by_time.mean(dim=1)
    rhs = model.pde.reaction() * (u_by_time * (1.0 - u_by_time)).mean(dim=1)
    return torch.mean((lhs - rhs) ** 2)


def front_speed_consistency_loss(
    model: OriginPINN,
    xy: torch.Tensor,
    t: torch.Tensor,
    low: float = 0.05,
    high: float = 0.95,
    max_points: int = 128,
) -> torch.Tensor:
    kin = front_speed_kinematics(model, xy, t)
    u = kin["u"]
    front_mask = ((u.detach() > low) & (u.detach() < high)).flatten()
    if not torch.any(front_mask):
        indicator = kin["front_indicator"].flatten()
        keep = min(max_points, len(indicator))
        if keep == 0:
            return torch.zeros((), dtype=xy.dtype, device=xy.device)
        front_mask = torch.zeros_like(indicator, dtype=torch.bool)
        front_mask[torch.topk(indicator, k=keep).indices] = True
    elif max_points > 0 and int(front_mask.sum()) > max_points:
        idx = torch.where(front_mask)[0]
        chosen = idx[torch.randperm(len(idx), device=idx.device)[:max_points]]
        front_mask = torch.zeros_like(front_mask)
        front_mask[chosen] = True

    selected_error = kin["speed_error"][front_mask]
    selected_weight = (4.0 * kin["front_indicator"][front_mask]).clamp(0.1, 1.0)
    if len(selected_error) == 0:
        return torch.zeros((), dtype=xy.dtype, device=xy.device)
    return torch.mean(selected_weight * selected_error.clamp(-2.0, 2.0).pow(2))


def gradient_residual_loss(
    model: OriginPINN,
    xy: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    residual, xy_req, t_req = pde_residual(model, xy, t, return_inputs=True)
    grad_xy = torch.autograd.grad(
        residual,
        xy_req,
        torch.ones_like(residual),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_t = torch.autograd.grad(
        residual,
        t_req,
        torch.ones_like(residual),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad_xy.pow(2).mean() + grad_t.pow(2).mean()


def front_local_gradient_residual_loss(
    model: OriginPINN,
    xy: torch.Tensor,
    t: torch.Tensor,
    low: float = 0.05,
    high: float = 0.95,
    max_points: int = 128,
) -> torch.Tensor:
    residual, u, _, xy_req, t_req = pde_residual_terms(model, xy, t)
    front_mask = ((u.detach() > low) & (u.detach() < high)).flatten()
    if not torch.any(front_mask):
        indicator = (u.detach() * (1.0 - u.detach())).flatten()
        keep = min(max_points, len(indicator))
        if keep == 0:
            return torch.zeros((), dtype=xy.dtype, device=xy.device)
        front_mask = torch.zeros_like(indicator, dtype=torch.bool)
        front_mask[torch.topk(indicator, k=keep).indices] = True
    elif max_points > 0 and int(front_mask.sum()) > max_points:
        idx = torch.where(front_mask)[0]
        chosen = idx[torch.randperm(len(idx), device=idx.device)[:max_points]]
        front_mask = torch.zeros_like(front_mask)
        front_mask[chosen] = True

    selected = residual[front_mask]
    if len(selected) == 0:
        return torch.zeros((), dtype=xy.dtype, device=xy.device)
    grad_xy = torch.autograd.grad(
        selected,
        xy_req,
        torch.ones_like(selected),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad_t = torch.autograd.grad(
        selected,
        t_req,
        torch.ones_like(selected),
        create_graph=True,
        retain_graph=True,
    )[0]
    return grad_xy[front_mask].pow(2).mean() + grad_t[front_mask].pow(2).mean()


def bounded_residual_weights(residual_sq: torch.Tensor, exponent: float = 0.5) -> torch.Tensor:
    scale = residual_sq.detach().mean().clamp_min(1.0e-12)
    weights = (residual_sq.detach() / scale).pow(exponent)
    return weights.clamp(0.25, 4.0)


class BalancedDecayWeights:
    """Bounded time-bin weights for slow-decaying residual regions."""

    def __init__(self, bins: int, beta: float, device: torch.device) -> None:
        self.beta = beta
        self.initial: torch.Tensor | None = None
        self.ema = torch.ones(bins, device=device)

    def update(self, losses: torch.Tensor) -> torch.Tensor:
        detached = losses.detach().clamp_min(1.0e-12)
        if self.initial is None:
            self.initial = detached.clone()
            self.ema = detached.clone()
        else:
            self.ema = self.beta * self.ema + (1.0 - self.beta) * detached
        rate = self.ema / self.initial.clamp_min(1.0e-12)
        weights = rate / rate.mean().clamp_min(1.0e-12)
        return weights.clamp(0.25, 4.0).detach()


def causal_weights(bin_losses: torch.Tensor, eps: float) -> torch.Tensor:
    if len(bin_losses) == 1:
        return torch.ones_like(bin_losses)
    zero = torch.zeros(1, device=bin_losses.device, dtype=bin_losses.dtype)
    cumulative = torch.cumsum(torch.cat([zero, bin_losses[:-1].detach()]), dim=0)
    weights = torch.exp(-eps * cumulative)
    return weights / weights.mean().clamp_min(1.0e-12)
