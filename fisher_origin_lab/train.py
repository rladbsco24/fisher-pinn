from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .baselines import (
    BaselineResult,
    differentiable_seed_baseline,
    drift_corrected_observation_centroid_baseline,
    naive_backward_blowup,
    observation_centroid_baseline,
)
from .config import ExperimentConfig
from .losses import (
    BalancedDecayWeights,
    bounded_residual_weights,
    boundary_neumann_loss,
    causal_weights,
    expected_front_pde_loss,
    front_area_contrast_loss,
    front_indicator_weights,
    front_local_gradient_residual_loss,
    front_profile_alignment_loss,
    front_level_set_alignment_loss,
    front_speed_consistency_loss,
    gradient_residual_loss,
    known_initial_condition_loss,
    leading_edge_floor_loss,
    leading_edge_area_loss,
    parabolic_mass_balance_loss,
    pde_residual,
    pde_residual_terms,
    seed_regularization_loss,
)
from .metrics import origin_error, relative_l2, tensor_center
from .models import OriginPINN
from .plotting import (
    generated_figure_names,
    predict_field,
    save_observation_coverage_figure,
    save_pinn_evolution_gif,
    save_pinn_rk4_comparison_figure,
    save_reconstruction_figure,
    save_residual_front_diagnostics_figure,
    save_spacetime_error_figure,
    save_training_diagnostics_figure,
)
from .rk4 import forward_fisher_kpp_rk4
from .samplers import SobolCollocation
from .shooting import source_shooting_loss
from .simulate import (
    forward_fisher_kpp,
    interpolate_truth,
    observation_tensors,
    sample_observations,
    split_observations,
    truth_field_at,
)
from .utils import default_device, seed_everything, write_json


@dataclass
class SingleRunResult:
    seed: int
    origin: tuple[float, float]
    origin_error: float
    physics: dict[str, float]
    history: list[dict[str, float]]
    model: OriginPINN


def _bin_losses(
    residual_sq: torch.Tensor,
    times: torch.Tensor,
    time_bins: int,
    t_end: float,
) -> torch.Tensor:
    ids = torch.clamp((times[:, 0] / t_end * time_bins).long(), 0, time_bins - 1)
    losses = []
    for idx in range(time_bins):
        mask = ids == idx
        if torch.any(mask):
            losses.append(residual_sq[mask].mean())
        else:
            losses.append(residual_sq.mean() * 0.0)
    return torch.stack(losses)


def _weighted_data_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    density_gain: float,
) -> torch.Tensor:
    err_sq = (pred - target) ** 2
    if density_gain <= 0.0:
        return err_sq.mean()
    weights = 1.0 + density_gain * target.detach().clamp_min(0.0)
    return torch.mean(weights * err_sq)


def _sample_rk4_teacher_points(
    rk4_truth,
    cfg: ExperimentConfig,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_n = max(4 * n, n + 1024)
    xyt = np.empty((candidate_n, 3), dtype=np.float64)
    xyt[:, 0] = rng.uniform(0.0, cfg.domain.box, size=candidate_n)
    xyt[:, 1] = rng.uniform(0.0, cfg.domain.box, size=candidate_n)
    xyt[:, 2] = rng.uniform(0.0, cfg.domain.t_end, size=candidate_n)
    late_fraction = float(np.clip(cfg.train.rk4_teacher_late_fraction, 0.0, 1.0))
    late_n = int(round(candidate_n * late_fraction))
    if late_n > 0:
        start = min(max(cfg.observations.start_time, 0.0), cfg.domain.t_end)
        xyt[:late_n, 2] = rng.uniform(start, cfg.domain.t_end, size=late_n)
    values = interpolate_truth(rk4_truth, xyt)
    activity = np.clip(values, 0.0, 1.0) * (1.0 - np.clip(values, 0.0, 1.0))
    low_front = ((values > 0.02) & (values < 0.30)).astype(np.float64)
    weights = 0.05 + activity + 0.50 * low_front
    weights = weights / np.sum(weights)
    replace_points = n > candidate_n
    idx = rng.choice(candidate_n, size=n, replace=replace_points, p=weights)
    return xyt[idx].astype(np.float32), values[idx, None].astype(np.float32)


def _rk4_teacher_batch_loss(
    model: OriginPINN,
    teacher_xyt: torch.Tensor | None,
    teacher_values: torch.Tensor | None,
    batch_size: int,
    density_gain: float,
    device: torch.device,
    *,
    t_low: float = 0.0,
    t_high: float | None = None,
) -> torch.Tensor:
    if teacher_xyt is None or teacher_values is None or len(teacher_xyt) == 0 or batch_size <= 0:
        return torch.zeros((), device=device)
    teacher_batch = min(max(1, int(batch_size)), len(teacher_xyt))
    teacher_idx = _masked_batch_indices(teacher_xyt, teacher_batch, device, t_low=t_low, t_high=t_high)
    if teacher_idx is None:
        return torch.zeros((), device=device)
    teacher_pred = model(teacher_xyt[teacher_idx, :2], teacher_xyt[teacher_idx, 2:3])
    return _weighted_data_mse(teacher_pred, teacher_values[teacher_idx], density_gain)


def _front_geometry_summary(
    truth,
    model: OriginPINN,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    levels: tuple[float, ...] = (0.05, 0.10),
    n: int = 96,
    max_times: int = 14,
) -> dict[str, object]:
    if len(truth.times) <= max_times:
        times = truth.times
    else:
        idx = np.linspace(0, len(truth.times) - 1, max_times).astype(int)
        times = truth.times[idx]

    summary: dict[str, object] = {
        "levels": list(levels),
        "times": [float(t) for t in times],
        "truth_mass": [],
        "pinn_mass": [],
        "truth_active_band": [],
        "pinn_active_band": [],
        "area_above": {f"{level:.2f}": {"truth": [], "pinn": []} for level in levels},
    }
    for time_value in times:
        _, true_field = truth_field_at(truth, float(time_value), n=n)
        _, pred = predict_field(model, float(time_value), n=n, device=device)
        summary["truth_mass"].append(float(true_field.mean()))
        summary["pinn_mass"].append(float(pred.mean()))
        summary["truth_active_band"].append(float(np.mean((true_field > 0.1) & (true_field < 0.9))))
        summary["pinn_active_band"].append(float(np.mean((pred > 0.1) & (pred < 0.9))))
        for level in levels:
            bucket = summary["area_above"][f"{level:.2f}"]
            bucket["truth"].append(float(np.mean(true_field > level)))
            bucket["pinn"].append(float(np.mean(pred > level)))

    mass_truth = np.asarray(summary["truth_mass"], dtype=np.float64)
    mass_pred = np.asarray(summary["pinn_mass"], dtype=np.float64)
    active_truth = np.asarray(summary["truth_active_band"], dtype=np.float64)
    active_pred = np.asarray(summary["pinn_active_band"], dtype=np.float64)
    summary["mass_mae"] = float(np.mean(np.abs(mass_pred - mass_truth)))
    summary["active_band_mae"] = float(np.mean(np.abs(active_pred - active_truth)))
    summary["area_mae"] = {}
    for level in levels:
        bucket = summary["area_above"][f"{level:.2f}"]
        truth_area = np.asarray(bucket["truth"], dtype=np.float64)
        pred_area = np.asarray(bucket["pinn"], dtype=np.float64)
        summary["area_mae"][f"{level:.2f}"] = float(np.mean(np.abs(pred_area - truth_area)))
        summary[f"area_above_{int(round(level * 1000)):03d}_mae"] = summary["area_mae"][f"{level:.2f}"]
    return summary


class _AdaptiveLossBalancer:
    """Relative-progress loss balancer for multi-term PINN objectives."""

    def __init__(self, momentum: float, min_weight: float, max_weight: float, eps: float = 1.0e-8) -> None:
        self.momentum = float(momentum)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.eps = float(eps)
        self.initial: dict[str, float] = {}
        self.weights: dict[str, float] = {}

    def update(self, losses: dict[str, torch.Tensor]) -> dict[str, float]:
        if not losses:
            return {}
        ratios: dict[str, float] = {}
        for name, loss in losses.items():
            value = max(float(loss.detach().cpu()), self.eps)
            self.initial.setdefault(name, value)
            ratios[name] = value / max(self.initial[name], self.eps)
        mean_ratio = max(float(np.mean(list(ratios.values()))), self.eps)
        next_weights: dict[str, float] = {}
        for name, ratio in ratios.items():
            target = float(np.clip(ratio / mean_ratio, self.min_weight, self.max_weight))
            previous = self.weights.get(name, target)
            next_weights[name] = self.momentum * previous + (1.0 - self.momentum) * target
        mean_weight = max(float(np.mean(list(next_weights.values()))), self.eps)
        self.weights = {
            name: float(np.clip(weight / mean_weight, self.min_weight, self.max_weight))
            for name, weight in next_weights.items()
        }
        return dict(self.weights)


def _residual_weight_exponent(cfg: ExperimentConfig, epoch: int) -> float:
    if cfg.train.residual_curriculum_epochs <= 0:
        return float(cfg.train.residual_weight_exponent_end)
    progress = min(1.0, max(0.0, epoch / float(cfg.train.residual_curriculum_epochs)))
    start = float(cfg.train.residual_weight_exponent_start)
    end = float(cfg.train.residual_weight_exponent_end)
    return start + progress * (end - start)


def _time_window(cfg: ExperimentConfig, epoch: int) -> tuple[float, float]:
    t_end = float(cfg.domain.t_end)
    if t_end <= 0.0:
        return 0.0, 0.0
    high = t_end
    if cfg.train.time_marching:
        ramp_epochs = cfg.train.time_marching_epochs or cfg.train.epochs
        progress = min(1.0, max(0.0, epoch / float(max(1, ramp_epochs))))
        start = min(max(float(cfg.train.time_marching_start_fraction), 0.0), 1.0) * t_end
        high = start + progress * (t_end - start)
    low = 0.0
    if cfg.train.time_slabs > 1:
        slabs = max(1, int(cfg.train.time_slabs))
        if cfg.train.time_slab_curriculum:
            progress = min(1.0, max(0.0, (epoch - 1) / float(max(1, cfg.train.epochs - 1))))
            slab = min(slabs - 1, int(progress * slabs))
            edge1 = t_end * (slab + 1) / slabs
            overlap = max(0.0, float(cfg.train.time_slab_overlap)) * t_end
            high = min(high, min(t_end, edge1 + overlap))
        else:
            slab = (epoch - 1) % slabs
            edge0 = t_end * slab / slabs
            edge1 = t_end * (slab + 1) / slabs
            overlap = max(0.0, float(cfg.train.time_slab_overlap)) * t_end
            low = max(0.0, edge0 - overlap)
            high = min(high, min(t_end, edge1 + overlap))
            if high <= low:
                high = min(t_end, edge1 + overlap)
    return low, max(low + 1.0e-8, min(high, t_end))


def _masked_batch_indices(
    xyt: torch.Tensor,
    n: int,
    device: torch.device,
    *,
    t_low: float = 0.0,
    t_high: float | None = None,
    fallback_all: bool = True,
) -> torch.Tensor | None:
    if t_high is None:
        candidates = torch.arange(len(xyt), device=device)
    else:
        mask = (xyt[:, 2] >= float(t_low) - 1.0e-8) & (xyt[:, 2] <= float(t_high) + 1.0e-8)
        candidates = torch.where(mask)[0]
        if len(candidates) == 0:
            if not fallback_all:
                return None
            candidates = torch.arange(len(xyt), device=device)
    if len(candidates) == 0:
        return None
    return candidates[torch.randint(0, len(candidates), (n,), device=device)]


def _sample_curriculum_collocation(
    sampler: SobolCollocation,
    n: int,
    *,
    t_low: float,
    t_high: float,
    focus_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if n <= 0:
        return sampler.sample(0)
    focus_fraction = float(np.clip(focus_fraction, 0.0, 1.0))
    if focus_fraction >= 1.0 - 1.0e-8:
        return sampler.sample(n, t_low=t_low, t_high=t_high)
    if focus_fraction <= 1.0e-8:
        return sampler.sample(n)

    focus_n = min(n, max(1, int(round(n * focus_fraction))))
    global_n = n - focus_n
    focus_xy, focus_t = sampler.sample(focus_n, t_low=t_low, t_high=t_high)
    if global_n <= 0:
        return focus_xy, focus_t
    global_xy, global_t = sampler.sample(global_n)
    return torch.cat([focus_xy, global_xy], dim=0), torch.cat([focus_t, global_t], dim=0)


def _combine_loss_terms(
    terms: list[tuple[str, float, torch.Tensor]],
    balancer: _AdaptiveLossBalancer | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    balanceable = {
        "data",
        "rk4_teacher",
        "pde",
        "bc",
        "seed_match",
        "seed_mass",
        "source_anchor",
        "shooting",
        "grad",
        "front_grad",
        "mass",
        "expected_front_pde",
        "leading_edge",
        "front_contrast",
        "front_profile",
        "level_set",
    }
    active = [(name, weight, loss) for name, weight, loss in terms if weight > 0.0]
    if not active:
        device = terms[0][2].device if terms else torch.device("cpu")
        return torch.zeros((), device=device), {}
    adaptive_losses = {
        name: loss
        for name, _, loss in active
        if name in balanceable and float(loss.detach().cpu()) > 1.0e-10
    }
    adaptive = balancer.update(adaptive_losses) if balancer is not None else {}
    total = torch.zeros((), dtype=active[0][2].dtype, device=active[0][2].device)
    for name, weight, loss in active:
        total = total + float(weight) * float(adaptive.get(name, 1.0)) * loss
    return total, adaptive


def _pretrain_on_rk4_teacher(
    model: OriginPINN,
    cfg: ExperimentConfig,
    teacher_xyt: torch.Tensor | None,
    teacher_values: torch.Tensor | None,
    device: torch.device,
) -> float | None:
    if (
        cfg.train.rk4_pretrain_steps <= 0
        or cfg.train.rk4_pretrain_batch <= 0
        or teacher_xyt is None
        or teacher_values is None
        or len(teacher_xyt) == 0
    ):
        return None
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.rk4_pretrain_lr)
    last_loss = None
    report_every = max(1, cfg.train.rk4_pretrain_steps // 4)
    for step in range(1, cfg.train.rk4_pretrain_steps + 1):
        optimizer.zero_grad()
        loss = _rk4_teacher_batch_loss(
            model,
            teacher_xyt,
            teacher_values,
            cfg.train.rk4_pretrain_batch,
            cfg.weights.data_density_gain,
            device,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step == 1 or step % report_every == 0 or step == cfg.train.rk4_pretrain_steps:
            print(f"rk4_pretrain step={step:4d} teacher_mse={last_loss:.3e}")
    return last_loss


def train_single(
    cfg: ExperimentConfig,
    observations_xyt: torch.Tensor,
    observations_values: torch.Tensor,
    validation_xyt: torch.Tensor | None,
    validation_values: torch.Tensor | None,
    teacher_xyt: torch.Tensor | None,
    teacher_values: torch.Tensor | None,
    run_seed: int,
    device: torch.device,
) -> SingleRunResult:
    seed_everything(run_seed)
    model = OriginPINN(cfg.domain, cfg.pde, cfg.seed, cfg.model).to(device)
    if cfg.warm_start.mode == "shooting_prefit":
        centroid_cfg = replace(cfg, warm_start=replace(cfg.warm_start, mode="centroid"))
        initial_center = _warm_start_center_from_observations(centroid_cfg, observations_xyt, observations_values)
        model.source.set_center(initial_center)
        _prefit_source_with_shooting(model, cfg, observations_xyt, observations_values)
        warm_start = tensor_center(model)
    else:
        warm_start = _warm_start_center_from_observations(cfg, observations_xyt, observations_values)
    print(f"warm_start mode={cfg.warm_start.mode} center=({warm_start[0]:.3f}, {warm_start[1]:.3f})")
    model.source.set_center(warm_start)
    warm_start_tensor = torch.tensor(warm_start, dtype=torch.float32, device=device)
    source_ids = {id(param) for param in model.source.parameters()}
    source_params = [param for param in model.parameters() if id(param) in source_ids]
    other_params = [param for param in model.parameters() if id(param) not in source_ids]
    rk4_pretrain_final_loss = _pretrain_on_rk4_teacher(model, cfg, teacher_xyt, teacher_values, device)
    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": cfg.train.lr},
            {"params": source_params, "lr": cfg.train.lr * cfg.train.source_lr_multiplier},
        ]
    )
    mask_kind = cfg.geo.mask_kind if cfg.geo.enabled else "box"
    sampler = SobolCollocation(cfg.domain.box, cfg.domain.t_end, device, seed=run_seed, mask_kind=mask_kind)
    decay = BalancedDecayWeights(cfg.train.time_bins, cfg.train.decay_beta, device)
    loss_balancer = (
        _AdaptiveLossBalancer(
            cfg.train.adaptive_loss_momentum,
            cfg.train.adaptive_loss_min,
            cfg.train.adaptive_loss_max,
        )
        if cfg.train.adaptive_loss_balancing
        else None
    )
    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    start = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        window_t_low, window_t_high = _time_window(cfg, epoch)
        optimizer.zero_grad()
        pred = model(observations_xyt[:, :2], observations_xyt[:, 2:3])
        data_loss = _weighted_data_mse(pred, observations_values, cfg.weights.data_density_gain)
        if cfg.weights.rk4_teacher > 0.0:
            rk4_teacher_loss = _rk4_teacher_batch_loss(
                model,
                teacher_xyt,
                teacher_values,
                cfg.train.rk4_teacher_batch,
                cfg.weights.data_density_gain,
                device,
                t_low=window_t_low,
                t_high=(
                    window_t_high
                    if cfg.train.time_window_teacher and (cfg.train.time_marching or cfg.train.time_slabs > 1)
                    else None
                ),
            )
        else:
            rk4_teacher_loss = torch.zeros((), device=device)

        if cfg.train.time_marching or cfg.train.time_slabs > 1:
            xy_col, t_col = _sample_curriculum_collocation(
                sampler,
                cfg.train.collocation_points,
                t_low=window_t_low,
                t_high=window_t_high,
                focus_fraction=cfg.train.time_window_focus_fraction,
            )
        else:
            xy_col, t_col = sampler.sample(cfg.train.collocation_points)
        residual, u_col, u_xy_col, _, _ = pde_residual_terms(model, xy_col, t_col)
        residual_sq = residual.pow(2).flatten()
        residual_exponent = _residual_weight_exponent(cfg, epoch)
        front_weights = front_indicator_weights(
            u_col,
            u_xy_col,
            cfg.weights.front_pde_alpha,
            cfg.weights.front_pde_gradient,
        ).flatten()
        point_weights = bounded_residual_weights(residual_sq, exponent=residual_exponent) * front_weights
        weighted_residual_sq = residual_sq * point_weights
        bins = _bin_losses(weighted_residual_sq, t_col, cfg.train.time_bins, cfg.domain.t_end)
        time_weights = causal_weights(bins, cfg.train.causal_eps) * decay.update(bins)
        pde_loss_value = torch.sum(time_weights * bins) / time_weights.sum().clamp_min(1.0e-12)

        if cfg.weights.boundary > 0.0 and cfg.train.boundary_points > 0:
            bc_loss = boundary_neumann_loss(model, cfg.train.boundary_points, device)
        else:
            bc_loss = torch.zeros((), device=device)
        if cfg.weights.initial_condition > 0.0 and cfg.train.seed_points > 0:
            ic_loss = known_initial_condition_loss(model, cfg.seed, cfg.train.seed_points, device)
        else:
            ic_loss = torch.zeros((), device=device)
        if (cfg.weights.seed_match > 0.0 or cfg.weights.seed_mass > 0.0) and cfg.train.seed_points > 0:
            seed_match, seed_mass = seed_regularization_loss(model, cfg.train.seed_points, device)
        else:
            seed_match = torch.zeros((), device=device)
            seed_mass = torch.zeros((), device=device)
        if cfg.weights.source_anchor > 0.0:
            source_anchor = torch.mean(((model.source.center() - warm_start_tensor) / cfg.domain.box) ** 2)
        else:
            source_anchor = torch.zeros((), device=device)
        if cfg.weights.shooting > 0.0:
            shooting_loss = source_shooting_loss(
                model,
                observations_xyt,
                observations_values,
                grid=cfg.train.shooting_grid,
                steps=cfg.train.shooting_steps,
                max_points=cfg.train.shooting_points,
            )
        else:
            shooting_loss = torch.zeros((), device=device)
        if cfg.weights.front_gradient > 0.0:
            subset = min(128, len(xy_col))
            front_grad_loss = front_local_gradient_residual_loss(model, xy_col[:subset], t_col[:subset])
            grad_loss = torch.zeros((), device=device)
        elif cfg.weights.gradient > 0.0:
            subset = min(128, len(xy_col))
            grad_loss = gradient_residual_loss(model, xy_col[:subset], t_col[:subset])
            front_grad_loss = torch.zeros((), device=device)
        else:
            grad_loss = torch.zeros((), device=device)
            front_grad_loss = torch.zeros((), device=device)
        if cfg.weights.front_speed > 0.0 and cfg.train.front_speed_points > 0:
            fs_subset = min(cfg.train.front_speed_points, len(xy_col))
            front_speed_loss = front_speed_consistency_loss(
                model,
                xy_col[:fs_subset],
                t_col[:fs_subset],
                max_points=cfg.train.front_speed_max_points,
                min_grad=cfg.train.front_speed_min_grad,
            )
        else:
            front_speed_loss = torch.zeros((), device=device)
        if cfg.weights.expected_front_pde > 0.0 and cfg.train.expected_front_points > 0:
            expected_front_loss = expected_front_pde_loss(
                model,
                cfg.train.expected_front_points,
                device,
                width=cfg.train.expected_front_width,
                speed_factor=cfg.train.expected_front_speed_factor,
                level=cfg.train.expected_front_level,
                front_alpha=cfg.weights.front_pde_alpha,
                front_gradient=cfg.weights.front_pde_gradient,
            )
        else:
            expected_front_loss = torch.zeros((), device=device)
        if cfg.weights.leading_edge > 0.0 and cfg.train.expected_front_points > 0:
            leading_edge_loss = leading_edge_floor_loss(
                model,
                cfg.train.expected_front_points,
                device,
                width=cfg.train.expected_front_width,
                speed_factor=cfg.train.expected_front_speed_factor,
                level=cfg.train.expected_front_level,
            )
        else:
            leading_edge_loss = torch.zeros((), device=device)
        if cfg.weights.leading_edge_area > 0.0 and cfg.train.leading_edge_area_times > 0:
            leading_edge_area = leading_edge_area_loss(
                model,
                cfg.train.leading_edge_area_times,
                cfg.train.leading_edge_area_grid,
                device,
                temperature=cfg.train.leading_edge_area_temperature,
            )
        else:
            leading_edge_area = torch.zeros((), device=device)
        if cfg.weights.front_contrast > 0.0 and cfg.train.front_contrast_times > 0:
            front_contrast = front_area_contrast_loss(
                model,
                cfg.train.front_contrast_times,
                cfg.train.front_contrast_grid,
                device,
                temperature=cfg.train.leading_edge_area_temperature,
            )
        else:
            front_contrast = torch.zeros((), device=device)
        if cfg.weights.front_profile > 0.0 and cfg.train.front_profile_points > 0:
            front_profile = front_profile_alignment_loss(
                model,
                cfg.train.front_profile_points,
                device,
                width=cfg.train.front_profile_width,
                t_low=window_t_low,
                t_high=window_t_high,
            )
        else:
            front_profile = torch.zeros((), device=device)
        if cfg.weights.level_set_alignment > 0.0 and cfg.train.level_set_points > 0:
            level_set_loss = front_level_set_alignment_loss(
                model,
                cfg.train.level_set_points,
                device,
                width=cfg.train.level_set_width,
                t_low=window_t_low,
                t_high=window_t_high,
            )
        else:
            level_set_loss = torch.zeros((), device=device)
        if cfg.weights.mass_balance > 0.0:
            mass_loss = parabolic_mass_balance_loss(
                model,
                cfg.train.mass_balance_times,
                cfg.train.mass_balance_grid,
                device,
            )
        else:
            mass_loss = torch.zeros((), device=device)
        sparse_loss = model.sparse_last_layer_l1() if cfg.weights.sparse > 0.0 else torch.zeros((), device=device)

        total, adaptive_weights = _combine_loss_terms(
            [
                ("data", cfg.weights.data, data_loss),
                ("rk4_teacher", cfg.weights.rk4_teacher, rk4_teacher_loss),
                ("pde", cfg.weights.pde, pde_loss_value),
                ("ic", cfg.weights.initial_condition, ic_loss),
                ("bc", cfg.weights.boundary, bc_loss),
                ("seed_match", cfg.weights.seed_match, seed_match),
                ("seed_mass", cfg.weights.seed_mass, seed_mass),
                ("source_anchor", cfg.weights.source_anchor, source_anchor),
                ("shooting", cfg.weights.shooting, shooting_loss),
                ("grad", cfg.weights.gradient, grad_loss),
                ("front_grad", cfg.weights.front_gradient, front_grad_loss),
                ("front_speed", cfg.weights.front_speed, front_speed_loss),
                ("expected_front_pde", cfg.weights.expected_front_pde, expected_front_loss),
                ("leading_edge", cfg.weights.leading_edge, leading_edge_loss),
                ("leading_edge_area", cfg.weights.leading_edge_area, leading_edge_area),
                ("front_contrast", cfg.weights.front_contrast, front_contrast),
                ("front_profile", cfg.weights.front_profile, front_profile),
                ("level_set", cfg.weights.level_set_alignment, level_set_loss),
                ("mass", cfg.weights.mass_balance, mass_loss),
                ("sparse", cfg.weights.sparse, sparse_loss),
            ],
            loss_balancer,
        )
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if cfg.train.rar_interval > 0 and epoch % cfg.train.rar_interval == 0:
            sampler.refresh(
                model,
                cfg.train.rar_candidates,
                cfg.train.rar_keep,
                front_alpha=cfg.weights.front_pde_alpha,
                front_gradient=cfg.weights.front_pde_gradient,
                residual_weight=cfg.train.rar_residual_weight,
                gradient_weight=cfg.train.rar_gradient_weight,
                activity_weight=cfg.train.rar_activity_weight,
                t_low=window_t_low,
                t_high=window_t_high,
            )

        if epoch == 1 or epoch % cfg.train.print_every == 0 or epoch == cfg.train.epochs:
            validation_loss = None
            should_validate = (
                validation_xyt is not None
                and validation_values is not None
                and len(validation_xyt) > 0
                and (
                    epoch == 1
                    or epoch == cfg.train.epochs
                    or cfg.train.validation_every <= 0
                    or epoch % cfg.train.validation_every == 0
                )
            )
            if should_validate:
                with torch.no_grad():
                    validation_pred = model(validation_xyt[:, :2], validation_xyt[:, 2:3])
                    validation_loss = float(torch.mean((validation_pred - validation_values) ** 2).detach().cpu())
                if cfg.train.restore_best_validation and validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            center = tensor_center(model)
            row = {
                "epoch": float(epoch),
                "total": float(total.detach().cpu()),
                "data": float(data_loss.detach().cpu()),
                "rk4_teacher": float(rk4_teacher_loss.detach().cpu()),
                "pde": float(pde_loss_value.detach().cpu()),
                "ic": float(ic_loss.detach().cpu()),
                "bc": float(bc_loss.detach().cpu()),
                "seed_match": float(seed_match.detach().cpu()),
                "seed_mass": float(seed_mass.detach().cpu()),
                "source_anchor": float(source_anchor.detach().cpu()),
                "shooting": float(shooting_loss.detach().cpu()),
                "grad": float(grad_loss.detach().cpu()),
                "front_grad": float(front_grad_loss.detach().cpu()),
                "front_speed": float(front_speed_loss.detach().cpu()),
                "expected_front_pde": float(expected_front_loss.detach().cpu()),
                "leading_edge": float(leading_edge_loss.detach().cpu()),
                "leading_edge_area": float(leading_edge_area.detach().cpu()),
                "front_contrast": float(front_contrast.detach().cpu()),
                "front_profile": float(front_profile.detach().cpu()),
                "level_set": float(level_set_loss.detach().cpu()),
                "mass": float(mass_loss.detach().cpu()),
                "sparse": float(sparse_loss.detach().cpu()),
                "residual_exponent": float(residual_exponent),
                "time_window_low": float(window_t_low),
                "time_window_high": float(window_t_high),
                "front_weight_mean": float(front_weights.detach().mean().cpu()),
                "diffusion": float(model.pde.diffusion().detach().cpu()),
                "reaction": float(model.pde.reaction().detach().cpu()),
                "velocity_x": float(model.pde.velocity[0].detach().cpu()),
                "velocity_y": float(model.pde.velocity[1].detach().cpu()),
                "origin_error": origin_error(center, cfg.seed),
                "elapsed_sec": time.time() - start,
            }
            if validation_loss is not None:
                row["validation_data"] = validation_loss
                row["best_validation_data"] = best_validation_loss
                row["best_validation_epoch"] = float(best_epoch)
            if rk4_pretrain_final_loss is not None:
                row["rk4_pretrain_final"] = rk4_pretrain_final_loss
            for name, value in adaptive_weights.items():
                row[f"aw_{name}"] = float(value)
            history.append(row)
            print(
                f"seed={run_seed} ep={epoch:5d} "
                f"loss={row['total']:.3e} data={row['data']:.2e} "
                f"pde={row['pde']:.2e} rw_exp={row['residual_exponent']:.2f} "
                f"origin_err={row['origin_error']:.3f}"
            )

    if cfg.train.adam_to_lbfgs and cfg.train.lbfgs_steps > 0:
        _lbfgs_polish(model, cfg, observations_xyt, observations_values, sampler, device)

    if best_state is not None:
        model.load_state_dict({name: value.to(device) for name, value in best_state.items()})
        print(f"restored best validation checkpoint: epoch={best_epoch} mse={best_validation_loss:.3e}")

    center = tensor_center(model)
    return SingleRunResult(
        seed=run_seed,
        origin=center,
        origin_error=origin_error(center, cfg.seed),
        physics=model.physics_dict(),
        history=history,
        model=model,
    )


def _warm_start_center_from_observations(
    cfg: ExperimentConfig,
    observations_xyt: torch.Tensor,
    observations_values: torch.Tensor,
) -> tuple[float, float]:
    if cfg.warm_start.mode == "neutral":
        return (0.5 * cfg.domain.box, 0.5 * cfg.domain.box)

    latest = torch.max(observations_xyt[:, 2])
    mask = torch.isclose(observations_xyt[:, 2], latest)
    xy = observations_xyt[mask, :2]
    weights = observations_values[mask, 0].clamp_min(0.0)
    if torch.sum(weights) <= 1.0e-12:
        centroid = xy.mean(dim=0)
    else:
        centroid = torch.sum(xy * weights[:, None], dim=0) / torch.sum(weights)

    if cfg.warm_start.mode == "centroid":
        corrected = centroid.clamp(0.0, cfg.domain.box)
        return (float(corrected[0].detach().cpu()), float(corrected[1].detach().cpu()))

    if cfg.warm_start.mode != "drift_corrected":
        raise ValueError(
            f"Unknown warm_start.mode={cfg.warm_start.mode!r}; "
            "expected 'drift_corrected', 'centroid', 'neutral', or 'shooting_prefit'."
        )

    drift = torch.tensor(
        [cfg.pde.velocity_x, cfg.pde.velocity_y],
        dtype=centroid.dtype,
        device=centroid.device,
    )
    corrected = (centroid - drift * latest).clamp(0.0, cfg.domain.box)
    return (float(corrected[0].detach().cpu()), float(corrected[1].detach().cpu()))


def _prefit_source_with_shooting(
    model: OriginPINN,
    cfg: ExperimentConfig,
    observations_xyt: torch.Tensor,
    observations_values: torch.Tensor,
) -> None:
    if cfg.train.shooting_prefit_steps <= 0:
        return
    params = list(model.source.parameters())
    if cfg.model.learn_drift:
        params.append(model.pde.velocity)
    optimizer = torch.optim.Adam(params, lr=3.0e-2)
    for _ in range(cfg.train.shooting_prefit_steps):
        optimizer.zero_grad()
        loss = source_shooting_loss(
            model,
            observations_xyt,
            observations_values,
            grid=cfg.train.shooting_grid,
            steps=cfg.train.shooting_steps,
            max_points=cfg.train.shooting_points,
        )
        loss.backward()
        optimizer.step()


def _lbfgs_polish(
    model: OriginPINN,
    cfg: ExperimentConfig,
    observations_xyt: torch.Tensor,
    observations_values: torch.Tensor,
    sampler: SobolCollocation,
    device: torch.device,
) -> None:
    opt = torch.optim.LBFGS(model.parameters(), max_iter=cfg.train.lbfgs_steps, tolerance_grad=1.0e-7)
    xy_col, t_col = sampler.sample(min(cfg.train.collocation_points, 2048))

    def closure() -> torch.Tensor:
        opt.zero_grad()
        pred = model(observations_xyt[:, :2], observations_xyt[:, 2:3])
        data_loss = _weighted_data_mse(pred, observations_values, cfg.weights.data_density_gain)
        residual, u_col, u_xy_col, _, _ = pde_residual_terms(model, xy_col, t_col)
        front_weights = front_indicator_weights(
            u_col,
            u_xy_col,
            cfg.weights.front_pde_alpha,
            cfg.weights.front_pde_gradient,
        )
        residual_sq = residual.pow(2).flatten()
        point_weights = bounded_residual_weights(
            residual_sq,
            exponent=cfg.train.residual_weight_exponent_end,
        ) * front_weights.flatten()
        pde_loss_value = torch.mean(point_weights * residual_sq)
        if cfg.weights.boundary > 0.0 and cfg.train.boundary_points > 0:
            bc_loss = boundary_neumann_loss(model, cfg.train.boundary_points, device)
        else:
            bc_loss = torch.zeros((), device=device)
        if cfg.weights.initial_condition > 0.0 and cfg.train.seed_points > 0:
            ic_loss = known_initial_condition_loss(model, cfg.seed, cfg.train.seed_points, device)
        else:
            ic_loss = torch.zeros((), device=device)
        if cfg.weights.front_speed > 0.0 and cfg.train.front_speed_points > 0:
            fs_subset = min(cfg.train.front_speed_points, len(xy_col))
            front_speed_loss = front_speed_consistency_loss(
                model,
                xy_col[:fs_subset],
                t_col[:fs_subset],
                max_points=cfg.train.front_speed_max_points,
                min_grad=cfg.train.front_speed_min_grad,
            )
        else:
            front_speed_loss = torch.zeros((), device=device)
        if cfg.weights.expected_front_pde > 0.0 and cfg.train.expected_front_points > 0:
            expected_front_loss = expected_front_pde_loss(
                model,
                min(cfg.train.expected_front_points, 256),
                device,
                width=cfg.train.expected_front_width,
                speed_factor=cfg.train.expected_front_speed_factor,
                level=cfg.train.expected_front_level,
                front_alpha=cfg.weights.front_pde_alpha,
                front_gradient=cfg.weights.front_pde_gradient,
            )
        else:
            expected_front_loss = torch.zeros((), device=device)
        if cfg.weights.leading_edge > 0.0 and cfg.train.expected_front_points > 0:
            leading_edge_loss = leading_edge_floor_loss(
                model,
                min(cfg.train.expected_front_points, 256),
                device,
                width=cfg.train.expected_front_width,
                speed_factor=cfg.train.expected_front_speed_factor,
                level=cfg.train.expected_front_level,
            )
        else:
            leading_edge_loss = torch.zeros((), device=device)
        if cfg.weights.leading_edge_area > 0.0 and cfg.train.leading_edge_area_times > 0:
            leading_edge_area = leading_edge_area_loss(
                model,
                cfg.train.leading_edge_area_times,
                cfg.train.leading_edge_area_grid,
                device,
                temperature=cfg.train.leading_edge_area_temperature,
            )
        else:
            leading_edge_area = torch.zeros((), device=device)
        if cfg.weights.front_contrast > 0.0 and cfg.train.front_contrast_times > 0:
            front_contrast = front_area_contrast_loss(
                model,
                cfg.train.front_contrast_times,
                cfg.train.front_contrast_grid,
                device,
                temperature=cfg.train.leading_edge_area_temperature,
            )
        else:
            front_contrast = torch.zeros((), device=device)
        if cfg.weights.front_profile > 0.0 and cfg.train.front_profile_points > 0:
            front_profile = front_profile_alignment_loss(
                model,
                cfg.train.front_profile_points,
                device,
                width=cfg.train.front_profile_width,
            )
        else:
            front_profile = torch.zeros((), device=device)
        if cfg.weights.mass_balance > 0.0:
            mass_loss = parabolic_mass_balance_loss(
                model,
                cfg.train.mass_balance_times,
                cfg.train.mass_balance_grid,
                device,
            )
        else:
            mass_loss = torch.zeros((), device=device)
        if (cfg.weights.seed_match > 0.0 or cfg.weights.seed_mass > 0.0) and cfg.train.seed_points > 0:
            seed_match, seed_mass = seed_regularization_loss(model, cfg.train.seed_points, device)
        else:
            seed_match = torch.zeros((), device=device)
            seed_mass = torch.zeros((), device=device)
        loss = (
            cfg.weights.data * data_loss
            + cfg.weights.pde * pde_loss_value
            + cfg.weights.initial_condition * ic_loss
            + cfg.weights.boundary * bc_loss
            + cfg.weights.seed_match * seed_match
            + cfg.weights.seed_mass * seed_mass
            + cfg.weights.front_speed * front_speed_loss
            + cfg.weights.expected_front_pde * expected_front_loss
            + cfg.weights.leading_edge * leading_edge_loss
            + cfg.weights.leading_edge_area * leading_edge_area
            + cfg.weights.front_contrast * front_contrast
            + cfg.weights.front_profile * front_profile
            + cfg.weights.mass_balance * mass_loss
            + cfg.weights.sparse * model.sparse_last_layer_l1()
        )
        loss.backward()
        return loss

    opt.step(closure)


def run_experiment(cfg: ExperimentConfig) -> dict[str, object]:
    device = default_device()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} out_dir={cfg.out_dir}")

    rng = np.random.default_rng(cfg.base_seed)
    truth = forward_fisher_kpp(cfg.domain, cfg.pde, cfg.seed)
    observations = sample_observations(truth, cfg.domain, cfg.observations, rng)
    train_observations, validation_observations = split_observations(
        observations,
        cfg.observations.validation_fraction,
        rng,
    )
    obs_xyt, obs_values = observation_tensors(train_observations, device)
    val_xyt, val_values = observation_tensors(validation_observations, device)
    teacher_xyt = None
    teacher_values = None
    rk4_truth = None
    rk4_runtime_sec = None
    needs_rk4_teacher = cfg.train.rk4_teacher_pool > 0 and (
        (cfg.weights.rk4_teacher > 0.0 and cfg.train.rk4_teacher_batch > 0)
        or (cfg.train.rk4_pretrain_steps > 0 and cfg.train.rk4_pretrain_batch > 0)
    )
    if needs_rk4_teacher:
        rk4_start = time.time()
        rk4_truth = forward_fisher_kpp_rk4(cfg.domain, cfg.pde, cfg.seed)
        rk4_runtime_sec = time.time() - rk4_start
        teacher_np, teacher_values_np = _sample_rk4_teacher_points(
            rk4_truth,
            cfg,
            cfg.train.rk4_teacher_pool,
            rng,
        )
        teacher_xyt = torch.tensor(teacher_np, dtype=torch.float32, device=device)
        teacher_values = torch.tensor(teacher_values_np, dtype=torch.float32, device=device)

    baseline_results: list[BaselineResult] = [
        observation_centroid_baseline(train_observations, cfg.seed),
        drift_corrected_observation_centroid_baseline(train_observations, cfg.domain, cfg.pde, cfg.seed),
        naive_backward_blowup(truth, cfg.domain, cfg.pde),
    ]
    if cfg.run_classical_baseline:
        baseline_results.append(
            differentiable_seed_baseline(
                train_observations,
                cfg.domain,
                cfg.pde,
                cfg.seed,
                epochs=cfg.baseline_epochs,
                device=device,
            )
        )

    runs = []
    for member in range(cfg.ensemble):
        runs.append(
            train_single(
                cfg,
                obs_xyt,
                obs_values,
                val_xyt,
                val_values,
                teacher_xyt,
                teacher_values,
                cfg.base_seed + member,
                device,
            )
        )

    best = min(runs, key=lambda r: r.origin_error)
    with torch.no_grad():
        train_pred = best.model(obs_xyt[:, :2], obs_xyt[:, 2:3])
        train_observation_mse = float(torch.mean((train_pred - obs_values) ** 2).detach().cpu())
        if len(val_xyt) > 0:
            val_pred = best.model(val_xyt[:, :2], val_xyt[:, 2:3])
            validation_observation_mse = float(torch.mean((val_pred - val_values) ** 2).detach().cpu())
        else:
            validation_observation_mse = None
    _, final_truth = truth_field_at(truth, cfg.domain.t_end, n=96)
    _, final_pred = predict_field(best.model, cfg.domain.t_end, n=96, device=device)
    pinn_final_time_relative_l2 = relative_l2(final_pred, final_truth)
    front_geometry = _front_geometry_summary(truth, best.model, cfg, device)
    if rk4_truth is None:
        rk4_start = time.time()
        rk4_truth = forward_fisher_kpp_rk4(cfg.domain, cfg.pde, cfg.seed)
        rk4_runtime_sec = time.time() - rk4_start
    else:
        rk4_runtime_sec = float(rk4_runtime_sec or 0.0)
    _, final_rk4 = truth_field_at(rk4_truth, cfg.domain.t_end, n=96)
    rk4_final_time_relative_l2 = relative_l2(final_rk4, final_truth)
    pinn_vs_rk4_final_relative_l2 = relative_l2(final_pred, final_rk4)
    rk4_train_values = interpolate_truth(rk4_truth, train_observations.xyt)[:, None]
    rk4_train_observation_mse = float(np.mean((rk4_train_values - train_observations.values) ** 2))
    if len(validation_observations.xyt) > 0:
        rk4_val_values = interpolate_truth(rk4_truth, validation_observations.xyt)[:, None]
        rk4_validation_observation_mse = float(np.mean((rk4_val_values - validation_observations.values) ** 2))
    else:
        rk4_validation_observation_mse = None
    ensemble_centers = [run.origin for run in runs]
    save_reconstruction_figure(
        cfg.out_dir / "reconstruction.png",
        truth,
        best.model,
        cfg.domain,
        cfg.observations,
        cfg.seed,
        device,
        ensemble_centers=ensemble_centers,
    )
    save_observation_coverage_figure(
        cfg.out_dir / "observation_coverage.png",
        train_observations,
        validation_observations,
        cfg.domain,
    )
    save_spacetime_error_figure(
        cfg.out_dir / "spacetime_error.png",
        truth,
        best.model,
        cfg.domain,
        device,
    )
    save_residual_front_diagnostics_figure(
        cfg.out_dir / "residual_front_diagnostics.png",
        truth,
        best.model,
        cfg.domain,
        device,
        time_value=cfg.domain.t_end,
        front_alpha=cfg.weights.front_pde_alpha,
        front_gradient=cfg.weights.front_pde_gradient,
    )
    rk4_compare_metrics = {
        "pinn_final_relative_l2": pinn_final_time_relative_l2,
        "rk4_final_relative_l2": rk4_final_time_relative_l2,
        "pinn_vs_rk4_final_relative_l2": pinn_vs_rk4_final_relative_l2,
        "pinn_validation_mse": validation_observation_mse,
        "rk4_validation_mse": rk4_validation_observation_mse,
    }
    save_pinn_rk4_comparison_figure(
        cfg.out_dir / "pinn_vs_rk4_comparison.png",
        truth,
        rk4_truth,
        best.model,
        cfg.domain,
        device,
        rk4_compare_metrics,
        time_value=cfg.domain.t_end,
    )
    save_training_diagnostics_figure(
        cfg.out_dir / "training_diagnostics.png",
        best.history,
        true_diffusion=cfg.pde.diffusion,
        true_reaction=cfg.pde.reaction,
    )
    save_pinn_evolution_gif(
        cfg.out_dir / "pinn_evolution.gif",
        truth,
        best.model,
        cfg.domain,
        device,
    )

    centers = np.array(ensemble_centers, dtype=np.float64)
    figure_paths = [str(cfg.out_dir / name) for name in generated_figure_names()]
    metrics: dict[str, object] = {
        "config": cfg.to_dict(),
        "truth_origin": [cfg.seed.center_x, cfg.seed.center_y],
        "ensemble_origin_mean": centers.mean(axis=0).tolist(),
        "ensemble_origin_std": centers.std(axis=0).tolist(),
        "best_origin": list(best.origin),
        "best_origin_error": best.origin_error,
        "final_time_relative_l2": pinn_final_time_relative_l2,
        "pinn_final_time_relative_l2": pinn_final_time_relative_l2,
        "rk4_final_time_relative_l2": rk4_final_time_relative_l2,
        "pinn_vs_rk4_final_relative_l2": pinn_vs_rk4_final_relative_l2,
        "front_area_005_mae": front_geometry.get("area_above_050_mae"),
        "front_area_010_mae": front_geometry.get("area_above_100_mae"),
        "active_front_area_mae": front_geometry.get("active_band_mae"),
        "mass_mae": front_geometry.get("mass_mae"),
        "train_observation_mse": train_observation_mse,
        "validation_observation_mse": validation_observation_mse,
        "rk4_train_observation_mse": rk4_train_observation_mse,
        "rk4_validation_observation_mse": rk4_validation_observation_mse,
        "rk4_runtime_sec": rk4_runtime_sec,
        "train_observation_count": int(len(train_observations.xyt)),
        "validation_observation_count": int(len(validation_observations.xyt)),
        "warm_start_mode": cfg.warm_start.mode,
        "front_geometry": front_geometry,
        "figures": figure_paths,
        "runs": [
            {
                "seed": run.seed,
                "origin": list(run.origin),
                "origin_error": run.origin_error,
                "physics": run.physics,
                "history": run.history,
            }
            for run in runs
        ],
        "baselines": [
            {
                "name": result.name,
                "center": list(result.center) if result.center is not None else None,
                "error": result.error,
                "extra": result.extra,
            }
            for result in baseline_results
        ],
    }
    write_json(cfg.out_dir / "metrics.json", metrics)
    print(f"wrote {cfg.out_dir / 'metrics.json'}")
    for figure_path in figure_paths:
        print(f"wrote {figure_path}")
    return metrics
