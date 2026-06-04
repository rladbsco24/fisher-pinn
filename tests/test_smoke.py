from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from scripts.run_inverse_origin import build_config
from scripts.run_forward_ablation import aggregate as aggregate_forward_ablation
from scripts.run_forward_ablation import make_forward_cases

from fisher_origin_lab.config import DomainConfig, ExperimentConfig, ModelConfig, PDEConfig, SeedConfig, WarmStartConfig
from fisher_origin_lab.losses import (
    expected_front_pde_loss,
    expected_front_samples,
    front_local_gradient_residual_loss,
    front_speed_consistency_loss,
    front_speed_kinematics,
    known_initial_condition_loss,
    leading_edge_area_loss,
    leading_edge_floor_loss,
    parabolic_mass_balance_loss,
    pde_residual,
)
from fisher_origin_lab.models import OriginPINN
from fisher_origin_lab.plotting import (
    save_observation_coverage_figure,
    save_pinn_rk4_comparison_figure,
    save_residual_front_diagnostics_figure,
    save_spacetime_error_figure,
    save_training_diagnostics_figure,
)
from fisher_origin_lab.rk4 import forward_fisher_kpp_rk4
from fisher_origin_lab.shooting import source_shooting_loss
from fisher_origin_lab.simulate import forward_fisher_kpp, sample_observations, split_observations
from fisher_origin_lab.train import _warm_start_center_from_observations


def test_forward_solver_shape_and_bounds() -> None:
    truth = forward_fisher_kpp(
        DomainConfig(grid=25, truth_steps=80),
        PDEConfig(),
        SeedConfig(),
        snapshots=8,
    )
    assert truth.fields.ndim == 3
    assert truth.fields.shape[1:] == (25, 25)
    assert np.isfinite(truth.fields).all()
    assert truth.fields.min() >= 0.0
    assert truth.fields.max() <= 1.0


def test_rk4_solver_matches_current_problem_shape_and_bounds() -> None:
    truth = forward_fisher_kpp_rk4(
        DomainConfig(grid=25, truth_steps=80),
        PDEConfig(include_advection=False),
        SeedConfig(),
        snapshots=8,
    )
    assert truth.fields.ndim == 3
    assert truth.fields.shape[1:] == (25, 25)
    assert np.isfinite(truth.fields).all()
    assert truth.fields.min() >= 0.0
    assert truth.fields.max() <= 1.0


def test_pde_residual_shape() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    model = OriginPINN(domain, PDEConfig(), SeedConfig(), ModelConfig(hidden=16, layers=1, fourier_features=8))
    xy = torch.rand(10, 2)
    t = torch.rand(10, 1) * domain.t_end
    residual = pde_residual(model, xy, t)
    assert residual.shape == (10, 1)
    assert torch.isfinite(residual).all()


def test_advection_flag_removes_velocity_from_residual() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    model_cfg = ModelConfig(hidden=16, layers=1, fourier_features=8, use_source_envelope=False)
    torch.manual_seed(123)
    no_adv = OriginPINN(
        domain,
        PDEConfig(velocity_x=50.0, velocity_y=-40.0, include_advection=False),
        SeedConfig(),
        model_cfg,
    )
    torch.manual_seed(123)
    zero_velocity = OriginPINN(
        domain,
        PDEConfig(velocity_x=0.0, velocity_y=0.0, include_advection=True),
        SeedConfig(),
        model_cfg,
    )
    xy = torch.rand(10, 2)
    t = torch.rand(10, 1) * domain.t_end
    assert torch.allclose(pde_residual(no_adv, xy, t), pde_residual(zero_velocity, xy, t), atol=1.0e-6)


def test_source_envelope_can_be_disabled() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    model = OriginPINN(
        domain,
        PDEConfig(),
        SeedConfig(),
        ModelConfig(hidden=16, layers=1, fourier_features=8, use_source_envelope=False),
    )
    xy = torch.rand(10, 2)
    t0 = torch.zeros(10, 1)
    before = model(xy, t0).detach()
    model.source.set_center((0.1, 0.1))
    after = model(xy, t0).detach()
    assert torch.allclose(before, after)


def test_geo_spectral_forward_model_terms_are_finite() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    cfg = ExperimentConfig(domain=domain).geo_spectral_forward()
    model = OriginPINN(domain, cfg.pde, cfg.seed, cfg.model)
    xy = torch.rand(12, 2)
    t = torch.rand(12, 1) * domain.t_end
    pred = model(xy, t)
    residual = pde_residual(model, xy, t)
    front_grad = front_local_gradient_residual_loss(model, xy, t, max_points=6)
    sparse = model.sparse_last_layer_l1()
    assert pred.shape == (12, 1)
    assert residual.shape == (12, 1)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(residual).all()
    assert torch.isfinite(front_grad)
    assert torch.isfinite(sparse)


def test_parabolic_mass_balance_loss_is_finite() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    cfg = ExperimentConfig(domain=domain).geo_spectral_forward()
    model = OriginPINN(domain, cfg.pde, cfg.seed, ModelConfig(hidden=16, layers=1, fourier_features=8))
    loss = parabolic_mass_balance_loss(model, n_times=2, grid=8, device=torch.device("cpu"))
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_front_speed_consistency_loss_is_finite() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    cfg = ExperimentConfig(domain=domain).geo_spectral_forward()
    model = OriginPINN(domain, cfg.pde, cfg.seed, cfg.model)
    xy = torch.rand(16, 2)
    t = torch.rand(16, 1) * domain.t_end
    kin = front_speed_kinematics(model, xy, t)
    loss = front_speed_consistency_loss(model, xy, t, max_points=8)
    assert kin["speed_error"].shape == (16, 1)
    assert loss.shape == ()
    assert torch.isfinite(kin["speed_error"]).all()
    assert torch.isfinite(loss)


def test_expected_front_losses_are_finite() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    cfg = ExperimentConfig(domain=domain).geo_spectral_forward()
    model = OriginPINN(domain, cfg.pde, cfg.seed, cfg.model)
    xy, t = expected_front_samples(model, 16, torch.device("cpu"))
    pde_loss = expected_front_pde_loss(model, 16, torch.device("cpu"))
    floor_loss = leading_edge_floor_loss(model, 16, torch.device("cpu"))
    area_loss = leading_edge_area_loss(model, n_times=3, grid=8, device=torch.device("cpu"))
    assert xy.shape == (16, 2)
    assert t.shape == (16, 1)
    assert torch.isfinite(xy).all()
    assert torch.isfinite(t).all()
    assert torch.isfinite(pde_loss)
    assert torch.isfinite(floor_loss)
    assert torch.isfinite(area_loss)


def test_korea_pine_style_matches_forward_pinn_setup() -> None:
    cfg = ExperimentConfig().korea_pine_style()
    assert cfg.pde.include_advection is False
    assert cfg.pde.velocity_x == 0.0
    assert cfg.pde.velocity_y == 0.0
    assert cfg.model.learn_diffusion is True
    assert cfg.model.learn_reaction is True
    assert cfg.model.learn_drift is False
    assert cfg.model.use_source_envelope is False
    assert cfg.weights.boundary == 0.0
    assert cfg.weights.initial_condition == 0.0
    assert cfg.weights.seed_match == 0.0
    assert cfg.weights.shooting == 0.0
    assert cfg.weights.data_density_gain == 4.0


def test_geo_spectral_forward_profile_extends_korea_setup() -> None:
    cfg = ExperimentConfig().geo_spectral_forward()
    assert cfg.pde.include_advection is False
    assert cfg.geo.enabled is True
    assert cfg.geo.mask_kind == "box"
    assert cfg.model.use_geo_features is True
    assert cfg.model.spatial_fourier_only is True
    assert cfg.model.use_source_envelope is False
    assert cfg.model.use_seed_front_features is True
    assert cfg.model.hard_initial_condition is True
    assert cfg.model.use_kpp_front_envelope is True
    assert cfg.weights.initial_condition > 0.0
    assert cfg.weights.boundary > 0.0
    assert cfg.weights.front_pde_alpha > 0.0
    assert cfg.weights.front_gradient > 0.0
    assert cfg.weights.front_speed > 0.0
    assert cfg.weights.mass_balance > 0.0
    assert cfg.weights.expected_front_pde == 0.0
    assert cfg.weights.leading_edge == 0.0
    assert cfg.weights.leading_edge_area > 0.0
    assert cfg.weights.sparse > 0.0
    assert cfg.train.adaptive_loss_balancing is True
    assert cfg.train.mass_balance_times > 0
    assert cfg.train.mass_balance_grid > 1
    assert cfg.train.front_speed_points > 0
    assert cfg.train.front_speed_max_points > 0
    assert cfg.train.front_speed_min_grad > 0.0
    assert cfg.train.expected_front_points > 0
    assert cfg.train.expected_front_width > 0.0
    assert cfg.train.expected_front_speed_factor > 0.0
    assert cfg.train.leading_edge_area_times > 0
    assert cfg.train.leading_edge_area_grid > 1
    assert cfg.train.residual_curriculum_epochs > 0
    assert cfg.train.residual_weight_exponent_start < cfg.train.residual_weight_exponent_end


def test_cli_quick_epochs_preserves_geo_training_stabilizers() -> None:
    args = argparse.Namespace(
        out_dir=Path("runs/test"),
        quick=True,
        epochs=12,
        ensemble=1,
        seed=7,
        grid=101,
        truth_steps=500,
        obs_samples=500,
        noise=0.02,
        focus_fraction=0.5,
        validation_fraction=0.2,
        learn_drift=False,
        learn_diffusion=False,
        learn_reaction=False,
        run_classical_baseline=False,
        baseline_epochs=250,
        gradient_weight=0.01,
        shooting_weight=5.0,
        data_density_gain=None,
        korea_pine_style=False,
        geo_spectral_forward=True,
        warm_start="drift_corrected",
    )
    cfg = build_config(args)
    assert cfg.train.epochs == 12
    assert cfg.train.adaptive_loss_balancing is True
    assert cfg.train.residual_curriculum_epochs > 0
    assert cfg.train.residual_weight_exponent_start < cfg.train.residual_weight_exponent_end


def test_forward_ablation_cases_report_front_metrics() -> None:
    cfg = ExperimentConfig(domain=DomainConfig(grid=25, truth_steps=80)).geo_spectral_forward().quick()
    cases = make_forward_cases(cfg)
    names = [case["name"] for case in cases]
    assert "korea_style_forward" in names
    assert "geo_front_area" in names
    assert any(case["cfg"].weights.leading_edge_area > 0.0 for case in cases)
    summary = aggregate_forward_ablation(
        [
            {
                "case": "geo_front_area",
                "seed": 7,
                "validation_observation_mse": 1.0e-3,
                "final_time_relative_l2": 0.4,
                "front_area_005_mae": 0.03,
                "front_area_010_mae": 0.02,
                "active_front_area_mae": 0.02,
                "mass_mae": 0.01,
                "note": "test",
            }
        ]
    )
    assert summary["cases"][0]["final_time_relative_l2_mean"] == 0.4
    assert summary["cases"][0]["front_area_010_mae_mean"] == 0.02


def test_warm_start_modes() -> None:
    xyt = torch.tensor(
        [
            [0.6, 0.4, 0.5],
            [0.8, 0.4, 0.5],
            [0.2, 0.9, 0.35],
        ],
        dtype=torch.float32,
    )
    values = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)
    base = ExperimentConfig()

    neutral = _warm_start_center_from_observations(replace(base, warm_start=WarmStartConfig("neutral")), xyt, values)
    centroid = _warm_start_center_from_observations(replace(base, warm_start=WarmStartConfig("centroid")), xyt, values)
    drift = _warm_start_center_from_observations(replace(base, warm_start=WarmStartConfig("drift_corrected")), xyt, values)

    assert neutral == (0.5, 0.5)
    assert np.allclose(centroid, (0.6, 0.4))
    assert np.allclose(drift, (0.3, 0.625))


def test_source_shooting_loss_is_finite() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    model = OriginPINN(domain, PDEConfig(), SeedConfig(), ModelConfig(hidden=16, layers=1, fourier_features=8))
    xyt = torch.tensor([[0.3, 0.68, 0.0], [0.6, 0.45, 0.5]], dtype=torch.float32)
    values = torch.tensor([[0.5], [0.2]], dtype=torch.float32)
    loss = source_shooting_loss(model, xyt, values, grid=17, steps=40, max_points=0)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_known_initial_condition_loss_is_finite() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    seed = SeedConfig()
    model = OriginPINN(domain, PDEConfig(include_advection=False), seed, ModelConfig(hidden=16, layers=1, fourier_features=8, use_source_envelope=False))
    loss = known_initial_condition_loss(model, seed, n=64, device=torch.device("cpu"))
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_hard_initial_condition_matches_seed_profile() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    seed = SeedConfig()
    model = OriginPINN(
        domain,
        PDEConfig(include_advection=False),
        seed,
        ModelConfig(
            hidden=16,
            layers=1,
            fourier_features=8,
            use_source_envelope=False,
            hard_initial_condition=True,
            use_kpp_front_envelope=True,
        ),
    )
    xy = torch.tensor([[seed.center_x, seed.center_y], [0.0, 0.0], [0.5, 0.5]], dtype=torch.float32)
    t0 = torch.zeros(len(xy), 1)
    pred = model(xy, t0).detach()
    center = torch.tensor([seed.center_x, seed.center_y], dtype=torch.float32).view(1, 2)
    dist2 = ((xy - center) ** 2).sum(dim=-1, keepdim=True)
    target = seed.amplitude * torch.exp(-dist2 / (2.0 * seed.sigma**2))
    assert torch.allclose(pred, target, atol=1.0e-6)


def test_visualization_exports_are_created(tmp_path) -> None:
    domain = DomainConfig(grid=21, truth_steps=60)
    cfg = ExperimentConfig(domain=domain).geo_spectral_forward().quick()
    model = OriginPINN(cfg.domain, cfg.pde, cfg.seed, ModelConfig(hidden=16, layers=1, fourier_features=8, use_geo_features=True, spatial_fourier_only=True, use_source_envelope=False))
    truth = forward_fisher_kpp(cfg.domain, cfg.pde, cfg.seed, snapshots=6)
    rng = np.random.default_rng(123)
    observations = sample_observations(truth, cfg.domain, cfg.observations, rng)
    train_obs, val_obs = split_observations(observations, 0.2, rng)
    history = [
        {
            "epoch": 1.0,
            "total": 1.0,
            "data": 0.2,
            "validation_data": 0.25,
            "pde": 0.5,
            "bc": 0.1,
            "ic": 0.2,
            "mass": 0.04,
            "front_speed": 0.08,
            "front_grad": 0.3,
            "residual_exponent": 0.5,
            "front_weight_mean": 1.5,
            "aw_data": 0.8,
            "aw_pde": 1.2,
            "aw_ic": 1.0,
            "aw_mass": 1.1,
            "aw_front_speed": 1.0,
            "sparse": 0.05,
            "origin_error": 0.1,
            "elapsed_sec": 0.2,
            "diffusion": 0.02,
            "reaction": 3.0,
        }
    ]

    save_observation_coverage_figure(tmp_path / "coverage.png", train_obs, val_obs, cfg.domain)
    save_spacetime_error_figure(tmp_path / "spacetime.png", truth, model, cfg.domain, torch.device("cpu"), n=16)
    save_residual_front_diagnostics_figure(
        tmp_path / "residual.png",
        truth,
        model,
        cfg.domain,
        torch.device("cpu"),
        time_value=cfg.domain.t_end,
        front_alpha=cfg.weights.front_pde_alpha,
        front_gradient=cfg.weights.front_pde_gradient,
        n=16,
    )
    rk4_truth = forward_fisher_kpp_rk4(cfg.domain, cfg.pde, cfg.seed, snapshots=6)
    save_pinn_rk4_comparison_figure(
        tmp_path / "pinn_vs_rk4.png",
        truth,
        rk4_truth,
        model,
        cfg.domain,
        torch.device("cpu"),
        {
            "pinn_final_relative_l2": 1.0,
            "rk4_final_relative_l2": 0.01,
            "pinn_vs_rk4_final_relative_l2": 0.99,
            "pinn_validation_mse": 0.2,
            "rk4_validation_mse": 0.001,
        },
        time_value=cfg.domain.t_end,
        n=16,
    )
    save_training_diagnostics_figure(tmp_path / "training.png", history, true_diffusion=0.02, true_reaction=3.0)

    for name in ["coverage.png", "spacetime.png", "residual.png", "pinn_vs_rk4.png", "training.png"]:
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 1000
