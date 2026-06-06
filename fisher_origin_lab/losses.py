from __future__ import annotations

import math

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
    min_grad: float = 1.0e-2,
) -> torch.Tensor:
    kin = front_speed_kinematics(model, xy, t)
    u = kin["u"]
    grad_norm = torch.linalg.norm(kin["u_xy"].detach(), dim=-1, keepdim=True)
    front_mask = ((u.detach() > low) & (u.detach() < high) & (grad_norm > min_grad)).flatten()
    if not torch.any(front_mask):
        indicator = (kin["front_indicator"] * grad_norm).flatten()
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


def expected_front_samples(
    model: OriginPINN,
    n: int,
    device: torch.device,
    *,
    width: float = 0.08,
    speed_factor: float = 0.45,
    level: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample collocation points in the conservative Fisher-KPP front corridor."""

    if n <= 0:
        empty_xy = torch.empty(0, 2, device=device)
        empty_t = torch.empty(0, 1, device=device)
        return empty_xy, empty_t

    dtype = next(model.parameters()).dtype
    t = torch.rand(n, 1, device=device, dtype=dtype) * model.domain.t_end
    angle = 2.0 * math.pi * torch.rand(n, 1, device=device, dtype=dtype)
    direction = torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)
    diffusion = torch.as_tensor(model.reference_diffusion, dtype=dtype, device=device).clamp_min(1.0e-10)
    reaction = torch.as_tensor(model.reference_reaction, dtype=dtype, device=device).clamp_min(1.0e-10)
    c_star = 2.0 * torch.sqrt(diffusion * reaction)
    amplitude = max(float(model.seed_amplitude), 1.0e-8)
    ratio = min(max(float(level) / amplitude, 1.0e-8), 1.0 - 1.0e-7)
    base_radius = float(model.seed_sigma) * math.sqrt(max(0.0, -2.0 * math.log(ratio)))
    radial_jitter = torch.randn(n, 1, device=device, dtype=dtype) * float(width)
    min_radius = max(0.25 * float(model.seed_sigma), 1.0e-4)
    radius = (base_radius + float(speed_factor) * c_star * t + radial_jitter).clamp(min_radius, model.domain.box)
    center = model.seed_center.to(dtype=dtype, device=device).view(1, 2)
    if model.pde.include_advection:
        center = center + model.pde.velocity.detach().view(1, 2).to(dtype=dtype, device=device) * t
    xy = (center + radius * direction).clamp(0.0, model.domain.box)
    return xy, t


def expected_front_pde_loss(
    model: OriginPINN,
    n: int,
    device: torch.device,
    *,
    width: float = 0.08,
    speed_factor: float = 0.45,
    level: float = 0.1,
    front_alpha: float = 1.0,
    front_gradient: float = 0.1,
) -> torch.Tensor:
    xy, t = expected_front_samples(
        model,
        n,
        device,
        width=width,
        speed_factor=speed_factor,
        level=level,
    )
    if len(xy) == 0:
        return torch.zeros((), device=device)
    residual, u, u_xy, _, _ = pde_residual_terms(model, xy, t)
    weights = front_indicator_weights(u, u_xy, front_alpha, front_gradient).flatten()
    return torch.mean(weights * residual.pow(2).flatten())


def linearized_kpp_gaussian(
    model: OriginPINN,
    xy: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """Leading-edge Fisher-KPP approximation from the known Gaussian initial seed.

    In the pulled-front regime u is small, so u(1-u) is well approximated by u.
    The resulting heat-kernel Gaussian times exp(r t) is used only as a weak
    one-sided floor, not as a supervised truth field.
    """

    center = model.seed_center.to(device=xy.device, dtype=xy.dtype).view(1, 2)
    if model.pde.include_advection:
        center = center + model.pde.velocity.detach().view(1, 2).to(device=xy.device, dtype=xy.dtype) * t
    diffusion = torch.as_tensor(model.reference_diffusion, dtype=xy.dtype, device=xy.device).clamp_min(1.0e-10)
    reaction = torch.as_tensor(model.reference_reaction, dtype=xy.dtype, device=xy.device).clamp_min(1.0e-10)
    sigma2 = torch.as_tensor(model.seed_sigma**2, dtype=xy.dtype, device=xy.device)
    spread = sigma2 + 2.0 * diffusion * t
    dist2 = ((xy - center) ** 2).sum(dim=-1, keepdim=True)
    heat = model.seed_amplitude * sigma2 / spread.clamp_min(1.0e-8) * torch.exp(-dist2 / (2.0 * spread))
    return (heat * torch.exp(reaction * t)).clamp(0.0, 1.0)


def leading_edge_floor_loss(
    model: OriginPINN,
    n: int,
    device: torch.device,
    *,
    width: float = 0.08,
    speed_factor: float = 0.45,
    level: float = 0.1,
    floor_fraction: float = 0.65,
    max_floor: float = 0.18,
) -> torch.Tensor:
    xy, t = expected_front_samples(
        model,
        n,
        device,
        width=width,
        speed_factor=speed_factor,
        level=level,
    )
    if len(xy) == 0:
        return torch.zeros((), device=device)
    pred = model(xy, t)
    floor = (float(floor_fraction) * linearized_kpp_gaussian(model, xy, t)).clamp(0.0, float(max_floor))
    active = (floor.detach() > 1.0e-4).float()
    return torch.mean(active * torch.relu(floor - pred).pow(2))


def linearized_kpp_area_targets(
    model: OriginPINN,
    times: torch.Tensor,
    levels: tuple[float, ...] = (0.05, 0.10),
) -> torch.Tensor:
    diffusion = torch.as_tensor(model.reference_diffusion, dtype=times.dtype, device=times.device).clamp_min(1.0e-10)
    reaction = torch.as_tensor(model.reference_reaction, dtype=times.dtype, device=times.device).clamp_min(1.0e-10)
    sigma2 = torch.as_tensor(model.seed_sigma**2, dtype=times.dtype, device=times.device)
    amplitude = torch.as_tensor(model.seed_amplitude, dtype=times.dtype, device=times.device)
    spread = sigma2 + 2.0 * diffusion * times
    leading_amplitude = amplitude * sigma2 / spread.clamp_min(1.0e-8) * torch.exp(reaction * times)
    targets = []
    for level in levels:
        level_tensor = torch.as_tensor(float(level), dtype=times.dtype, device=times.device)
        log_ratio = torch.log(leading_amplitude.clamp_min(1.0e-12) / level_tensor)
        radius_sq = torch.where(log_ratio > 0.0, 2.0 * spread * log_ratio, torch.zeros_like(spread))
        area = math.pi * radius_sq / float(model.domain.box**2)
        targets.append(area.clamp(0.0, 1.0))
    return torch.cat(targets, dim=1)


def _linearized_kpp_level_radius(
    model: OriginPINN,
    times: torch.Tensor,
    levels: torch.Tensor,
) -> torch.Tensor:
    diffusion = torch.as_tensor(model.reference_diffusion, dtype=times.dtype, device=times.device).clamp_min(1.0e-10)
    reaction = torch.as_tensor(model.reference_reaction, dtype=times.dtype, device=times.device).clamp_min(1.0e-10)
    sigma2 = torch.as_tensor(model.seed_sigma**2, dtype=times.dtype, device=times.device)
    amplitude = torch.as_tensor(model.seed_amplitude, dtype=times.dtype, device=times.device)
    spread = sigma2 + 2.0 * diffusion * times
    leading_amplitude = amplitude * sigma2 / spread.clamp_min(1.0e-8) * torch.exp(reaction * times)
    log_ratio = torch.log(leading_amplitude.clamp_min(1.0e-12) / levels.clamp_min(1.0e-12))
    radius_sq = torch.where(log_ratio > 0.0, 2.0 * spread * log_ratio, torch.zeros_like(spread))
    return torch.sqrt(radius_sq.clamp_min(0.0))


def front_level_set_alignment_loss(
    model: OriginPINN,
    n: int,
    device: torch.device,
    *,
    levels: tuple[float, ...] = (0.05, 0.10),
    width: float = 0.025,
    sign_margin_fraction: float = 0.15,
    t_low: float = 0.0,
    t_high: float | None = None,
) -> torch.Tensor:
    """Align predicted Fisher-KPP level sets with the leading-edge front.

    The existing soft area loss matches integrated front coverage. This loss is
    more local: on the expected KPP level-set ring it drives u toward the target
    level, and it adds weak inside/outside hinge terms so a broad low-amplitude
    haze cannot satisfy the area metric by itself.
    """

    if n <= 0 or not levels:
        return torch.zeros((), device=device)
    dtype = next(model.parameters()).dtype
    levels_t = torch.tensor(levels, dtype=dtype, device=device)
    level_idx = torch.randint(0, len(levels), (n, 1), device=device)
    level = levels_t[level_idx]
    low = max(0.0, min(float(t_low), model.domain.t_end))
    high = model.domain.t_end if t_high is None else max(0.0, min(float(t_high), model.domain.t_end))
    if high < low:
        low, high = high, low
    t = low + torch.rand(n, 1, dtype=dtype, device=device) * max(high - low, 1.0e-8)
    radius = _linearized_kpp_level_radius(model, t, level)
    angle = 2.0 * math.pi * torch.rand(n, 1, dtype=dtype, device=device)
    normal = torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)
    center = model.seed_center.to(dtype=dtype, device=device).view(1, 2)
    if model.pde.include_advection:
        center = center + model.pde.velocity.detach().view(1, 2).to(dtype=dtype, device=device) * t
    ring = center + radius * normal
    offset = max(float(width), 1.0e-4) * model.domain.box
    xy_on = ring.clamp(0.0, model.domain.box)
    xy_in = (ring - offset * normal).clamp(0.0, model.domain.box)
    xy_out = (ring + offset * normal).clamp(0.0, model.domain.box)

    pred_on = model(xy_on, t)
    pred_in = model(xy_in, t)
    pred_out = model(xy_out, t)
    valid = (radius > 1.0e-6).detach().float()
    on_loss = valid * (pred_on - level).pow(2)
    margin = (float(sign_margin_fraction) * level).clamp_min(1.0e-4)
    order_loss = valid * (
        torch.relu(level + margin - pred_in).pow(2)
        + torch.relu(pred_out - (level - margin).clamp_min(0.0)).pow(2)
    )
    denom = valid.mean().clamp_min(1.0e-6)
    return (on_loss.mean() + 0.5 * order_loss.mean()) / denom


def leading_edge_area_loss(
    model: OriginPINN,
    n_times: int,
    grid: int,
    device: torch.device,
    *,
    levels: tuple[float, ...] = (0.05, 0.10),
    temperature: float = 0.015,
) -> torch.Tensor:
    if n_times <= 0 or grid <= 1:
        return torch.zeros((), device=device)
    dtype = next(model.parameters()).dtype
    xs = torch.linspace(0.0, model.domain.box, grid, dtype=dtype, device=device)
    x, y = torch.meshgrid(xs, xs, indexing="ij")
    xy_one = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    times = torch.linspace(0.0, model.domain.t_end, n_times, dtype=dtype, device=device).reshape(-1, 1)
    xy = xy_one.repeat(n_times, 1)
    t = times.repeat_interleave(grid * grid, dim=0)
    pred = model(xy, t).reshape(n_times, grid * grid, 1)
    soft_areas = []
    temp = max(float(temperature), 1.0e-4)
    for level in levels:
        soft = torch.sigmoid((pred - float(level)) / temp).mean(dim=1)
        soft_areas.append(soft)
    soft_area = torch.cat(soft_areas, dim=1)
    target_area = linearized_kpp_area_targets(model, times, levels)
    area_loss = torch.mean((soft_area - target_area).pow(2))

    sorted_pred, _ = torch.sort(pred.squeeze(-1), dim=1, descending=True)
    n_points = sorted_pred.shape[1]
    hinge_terms = []
    for level_index, level in enumerate(levels):
        target_fraction = target_area[:, level_index].detach().clamp(1.0 / n_points, 1.0)
        rank = torch.ceil(target_fraction * n_points).long().clamp(1, n_points) - 1
        kth_values = sorted_pred[torch.arange(n_times, device=device), rank]
        hinge_terms.append(torch.relu(float(level) - kth_values).pow(2).mean())
    quantile_hinge = torch.stack(hinge_terms).mean() if hinge_terms else torch.zeros((), device=device)
    return area_loss + 4.0 * quantile_hinge


def front_area_contrast_loss(
    model: OriginPINN,
    n_times: int,
    grid: int,
    device: torch.device,
    *,
    levels: tuple[float, float] = (0.05, 0.10),
    temperature: float = 0.015,
) -> torch.Tensor:
    """Match low/high front-area ratio to discourage diffuse halos."""

    if n_times <= 0 or grid <= 1 or len(levels) != 2:
        return torch.zeros((), device=device)
    dtype = next(model.parameters()).dtype
    xs = torch.linspace(0.0, model.domain.box, grid, dtype=dtype, device=device)
    x, y = torch.meshgrid(xs, xs, indexing="ij")
    xy_one = torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)
    times = torch.linspace(0.0, model.domain.t_end, n_times, dtype=dtype, device=device).reshape(-1, 1)
    xy = xy_one.repeat(n_times, 1)
    t = times.repeat_interleave(grid * grid, dim=0)
    pred = model(xy, t).reshape(n_times, grid * grid, 1)
    temp = max(float(temperature), 1.0e-4)
    low_level, high_level = float(levels[0]), float(levels[1])
    soft_low = torch.sigmoid((pred - low_level) / temp).mean(dim=1).clamp_min(1.0e-6)
    soft_high = torch.sigmoid((pred - high_level) / temp).mean(dim=1).clamp_min(1.0e-6)
    targets = linearized_kpp_area_targets(model, times, levels).clamp_min(1.0e-6)
    pred_ratio = (soft_high / soft_low).clamp(1.0e-6, 1.0)
    target_ratio = (targets[:, 1:2] / targets[:, 0:1]).clamp(1.0e-6, 1.0)
    return torch.mean((torch.log(pred_ratio) - torch.log(target_ratio.detach())).pow(2))


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
