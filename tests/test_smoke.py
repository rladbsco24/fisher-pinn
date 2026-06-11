from __future__ import annotations

import argparse
import ast
import base64
import csv
import io
import json
import re
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from scripts.run_inverse_origin import build_config
from scripts.run_forward_ablation import aggregate as aggregate_forward_ablation
from scripts.run_forward_ablation import make_forward_cases
from scripts.run_feature_validation_ablation import make_feature_validation_pairs, summarize_feature_validation_rows
from scripts.run_korea_pine_wilt_simulation import _save_korea_map_baseline_gif

from PIL import Image

from fisher_origin_lab.ablation_visuals import save_feature_pair_error_map, save_feature_pair_evolution_gif
from fisher_origin_lab.config import (
    DomainConfig,
    ExperimentConfig,
    ModelConfig,
    PDEConfig,
    SeedConfig,
    WarmStartConfig,
    shared_geo_forward_model_config,
)
from fisher_origin_lab.exact_wave import az_exact_unit_torch
from fisher_origin_lab.curve_trend import (
    CurvePINNConfig,
    CurveTrendConfig,
    CurveTrendPINN,
    curve_exact,
    curve_pinn_residual,
    integrate_curve,
    train_curve_pinn,
)
from fisher_origin_lab.korea_data import (
    build_density_grid,
    compare_observed_and_simulated,
    estimate_korea_action_start,
    fit_korea_pine_wilt_pinn,
    korea_grid_time_values,
    korea_physics_prior_from_normalized,
    korea_physics_prior_from_physical,
    load_korea_pine_wilt_action_time_points,
    load_korea_pine_wilt_points,
    load_manifest,
    simulate_density_rk4,
    simulate_density_rk4_at_times,
)
from fisher_origin_lab.losses import (
    ablowitz_zeppetella_dirichlet_loss,
    ablowitz_zeppetella_front_band_samples,
    ablowitz_zeppetella_front_phase_loss,
    ablowitz_zeppetella_initial_condition_loss,
    discrete_rk4_consistency_loss,
    expected_front_gradient_residual_loss,
    expected_front_pde_loss,
    expected_front_samples,
    front_area_contrast_loss,
    front_level_set_alignment_loss,
    front_local_gradient_residual_loss,
    front_profile_alignment_loss,
    front_support_tversky_loss,
    front_speed_consistency_loss,
    front_speed_kinematics,
    known_initial_condition_loss,
    leading_edge_area_loss,
    leading_edge_distribution_loss,
    leading_edge_floor_loss,
    mass_floor_trajectory_loss,
    observed_support_tversky_loss,
    parabolic_mass_balance_loss,
    pde_residual,
    radial_symmetry_loss,
    spatial_coefficient_regularization_loss,
    time_slab_interface_loss,
)
from fisher_origin_lab.models import OriginPINN
from fisher_origin_lab.rk4 import forward_ablowitz_zeppetella_rk4
from fisher_origin_lab.plotting import (
    save_observation_coverage_figure,
    save_pinn_evolution_gif,
    save_pinn_rk4_comparison_figure,
    save_residual_front_diagnostics_figure,
    save_spacetime_error_figure,
    save_training_diagnostics_figure,
)
from fisher_origin_lab.rk4 import forward_fisher_kpp_rk4
from fisher_origin_lab.samplers import SobolCollocation
from fisher_origin_lab.shooting import source_shooting_loss
from fisher_origin_lab.simulate import gaussian_seed_numpy, forward_fisher_kpp, sample_observations, split_observations, truth_field_at
from fisher_origin_lab.train import _masked_batch_indices, _time_window, _warm_start_center_from_observations


REPO_ROOT = Path(__file__).resolve().parents[1]


def _notebook_ast_source(source: str) -> str:
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return ""
    return "\n".join(line for line in lines if not line.lstrip().startswith(("%", "!")))


def test_notebooks_and_embedded_archives_are_parseable() -> None:
    notebooks = [
        REPO_ROOT / "fisher_kpp_origin_lab.ipynb",
        REPO_ROOT / "fisher_kpp_origin_lab_colab.ipynb",
        REPO_ROOT / "korea_pine_wilt_fisher_kpp_lab.ipynb",
    ]
    embedded_notebooks = {
        REPO_ROOT / "fisher_kpp_origin_lab.ipynb",
        REPO_ROOT / "fisher_kpp_origin_lab_colab.ipynb",
    }
    archive_pattern = re.compile(r'_EMBEDDED_PROJECT_ZIP_B64 = """\n(.*?)\n"""', re.S)
    broken_table_pattern = re.compile(r'"md = [^\\n]*\|[^\\]*\n",\n\s+"\|---')

    for notebook in notebooks:
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        full_source = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])
        assert not broken_table_pattern.search(notebook.read_text(encoding="utf-8"))
        if notebook.name == "korea_pine_wilt_fisher_kpp_lab.ipynb":
            assert "???" not in full_source
            assert "Korea Forest Service" in full_source
            assert "_prepare_colab_repo" in full_source

        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = _notebook_ast_source("".join(cell.get("source", [])))
            ast.parse(source, filename=f"{notebook.name}:cell{idx}")

        match = archive_pattern.search(full_source)
        if notebook not in embedded_notebooks:
            assert match is None
            continue
        assert match is not None
        raw = base64.b64decode("".join(match.group(1).split()))
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            assert "fisher_origin_lab/losses.py" in zf.namelist()
            assert "fisher_origin_lab/plotting.py" in zf.namelist()
            for name in zf.namelist():
                data = zf.read(name)
                if name.endswith(".py"):
                    ast.parse(data.decode("utf-8"), filename=f"{notebook.name}:embedded:{name}")
                repo_file = REPO_ROOT / name
                if repo_file.exists():
                    assert data == repo_file.read_bytes(), f"{notebook.name} embedded {name} differs from repo"


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


def test_truth_field_at_uses_interpolation_for_resampling() -> None:
    domain = DomainConfig(grid=51, truth_steps=160)
    seed = SeedConfig()
    truth = forward_fisher_kpp(domain, PDEConfig(), seed, snapshots=8)
    xs, field = truth_field_at(truth, 0.0, n=96)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    analytic = gaussian_seed_numpy(x, y, seed)

    denom = np.sqrt(np.mean(analytic**2)) + 1.0e-12
    rel = np.sqrt(np.mean((field - analytic) ** 2)) / denom
    assert rel < 3.0e-2


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


def test_long_time_curve_reference_shape_and_integrators() -> None:
    cfg = CurveTrendConfig()
    times = np.linspace(0.0, cfg.t_end, cfg.steps + 1)
    exact, _ = curve_exact(times, cfg)
    peak_window = times <= 8.0
    peak_idx = int(np.argmax(exact[peak_window]))
    trough_mask = (times >= 3.0) & (times <= 7.0)
    trough = exact[trough_mask].min()

    assert 0.68 <= exact[peak_window][peak_idx] <= 0.76
    assert 1.5 <= times[peak_window][peak_idx] <= 2.3
    assert 0.13 <= trough <= 0.22
    assert 0.32 <= exact[-1] <= 0.36

    errors = {
        method: float(integrate_curve(method, cfg)["abs_error"].max())
        for method in ("forward_euler", "backward_euler", "trapezoidal", "rk4")
    }
    assert errors["forward_euler"] < 3.5e-2
    assert errors["backward_euler"] < 3.5e-2
    assert errors["trapezoidal"] < 1.0e-3
    assert errors["rk4"] < 1.0e-6


def test_curve_trend_pinn_residual_and_short_training_are_finite() -> None:
    trend_cfg = CurveTrendConfig()
    model = CurveTrendPINN(trend_cfg, hidden=16, layers=1)
    t = torch.linspace(0.0, trend_cfg.t_end, 8).reshape(-1, 1)
    residual, rho = curve_pinn_residual(model, t)
    assert residual.shape == (8, 1)
    assert rho.shape == (8, 1)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(rho).all()

    result = train_curve_pinn(
        trend_cfg,
        CurvePINNConfig(epochs=2, hidden=16, layers=1, collocation_points=16, observation_points=12, print_every=1),
        device=torch.device("cpu"),
        seed=11,
    )
    assert np.isfinite(result["rho"]).all()
    assert np.isfinite(result["abs_error"]).all()
    assert result["metrics"]["max_abs_error"] >= 0.0
    assert len(result["history"]) >= 1


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
    coefficient_loss = spatial_coefficient_regularization_loss(model, xy)
    sparse = model.sparse_last_layer_l1()
    assert pred.shape == (12, 1)
    assert residual.shape == (12, 1)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(residual).all()
    assert torch.isfinite(front_grad)
    assert torch.isfinite(coefficient_loss)
    assert torch.isfinite(sparse)
    assert model.has_spatial_coefficients()
    assert model.coefficient_stats(xy)["coefficient_log_abs_mean"] >= 0.0


def test_ablowitz_zeppetella_forward_preset_matches_exact_wave() -> None:
    cfg = ExperimentConfig().ablowitz_zeppetella_forward().quick()
    assert cfg.benchmark.kind == "ablowitz_zeppetella"
    assert cfg.pde.include_advection is False
    assert np.isclose(cfg.pde.reaction, 1.0)
    assert np.isclose(cfg.pde.diffusion, 1.0 / 40.0**2)
    assert cfg.weights.level_set_alignment > 0.0
    assert cfg.model.use_az_hard_constraints is True
    model = OriginPINN(cfg.domain, cfg.pde, cfg.seed, cfg.model)
    xy = torch.rand(16, 2)
    t = torch.rand(16, 1) * cfg.domain.t_end
    residual = pde_residual(model, xy, t)
    ic = ablowitz_zeppetella_initial_condition_loss(model, 16, torch.device("cpu"))
    bc = ablowitz_zeppetella_dirichlet_loss(model, 16, torch.device("cpu"))
    front_xy, front_t = ablowitz_zeppetella_front_band_samples(model, 24, torch.device("cpu"))
    phase = ablowitz_zeppetella_front_phase_loss(model, 24, torch.device("cpu"))
    assert residual.shape == (16, 1)
    assert torch.isfinite(residual).all()
    assert torch.isfinite(ic)
    assert torch.isfinite(bc)
    assert front_xy.shape[1] == 2
    assert front_t.shape[1] == 1
    assert torch.isfinite(front_xy).all()
    assert torch.isfinite(front_t).all()
    assert torch.isfinite(phase)


def test_ablowitz_zeppetella_hard_constraints_match_ic_and_boundaries() -> None:
    cfg = ExperimentConfig().ablowitz_zeppetella_forward().quick()
    model = OriginPINN(cfg.domain, cfg.pde, cfg.seed, cfg.model)
    x = torch.linspace(0.0, 1.0, 9).view(-1, 1)
    y = torch.rand_like(x)
    xy = torch.cat([x, y], dim=1)
    t0 = torch.zeros_like(x)
    pred_initial = model(xy, t0)
    target_initial = az_exact_unit_torch(
        x,
        t0,
        x_left=cfg.benchmark.x_left,
        x_right=cfg.benchmark.x_right,
        x0=cfg.benchmark.wave_x0,
    )
    assert torch.allclose(pred_initial, target_initial, atol=1.0e-6)

    t = torch.linspace(0.0, cfg.domain.t_end, 9).view(-1, 1)
    left_xy = torch.cat([torch.zeros_like(t), y], dim=1)
    right_xy = torch.cat([torch.ones_like(t), y], dim=1)
    left_target = az_exact_unit_torch(
        torch.zeros_like(t),
        t,
        x_left=cfg.benchmark.x_left,
        x_right=cfg.benchmark.x_right,
        x0=cfg.benchmark.wave_x0,
    )
    right_target = az_exact_unit_torch(
        torch.ones_like(t),
        t,
        x_left=cfg.benchmark.x_left,
        x_right=cfg.benchmark.x_right,
        x0=cfg.benchmark.wave_x0,
    )
    assert torch.allclose(model(left_xy, t), left_target, atol=1.0e-6)
    assert torch.allclose(model(right_xy, t), right_target, atol=1.0e-6)


def test_ablowitz_zeppetella_rk4_truth_shape_and_error() -> None:
    cfg = ExperimentConfig().ablowitz_zeppetella_forward().quick()
    truth = forward_ablowitz_zeppetella_rk4(cfg.domain, cfg.pde, snapshots=3, steps=80)
    assert truth.fields.shape[1:] == (cfg.domain.grid, cfg.domain.grid)
    assert np.isfinite(truth.fields).all()
    assert truth.fields.min() >= 0.0
    assert truth.fields.max() <= 1.0


def test_nif_pirate_model_terms_are_finite() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    cfg = ExperimentConfig(domain=domain).geo_spectral_forward()
    model_cfg = replace(cfg.model, architecture="nif_pirate", hidden=16, layers=1, fourier_features=8, nif_rank=6)
    model = OriginPINN(domain, cfg.pde, cfg.seed, model_cfg)
    xy = torch.rand(12, 2)
    t = torch.rand(12, 1) * domain.t_end
    pred = model(xy, t)
    residual = pde_residual(model, xy, t)
    sparse = model.sparse_last_layer_l1()
    assert pred.shape == (12, 1)
    assert residual.shape == (12, 1)
    assert torch.isfinite(pred).all()
    assert torch.isfinite(residual).all()
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
    distribution_loss = leading_edge_distribution_loss(model, n_times=3, grid=8, device=torch.device("cpu"))
    symmetry_loss = radial_symmetry_loss(model, groups=4, angles=5, device=torch.device("cpu"))
    observed_support = observed_support_tversky_loss(
        torch.tensor([[0.02], [0.12], [0.20]], dtype=torch.float32),
        torch.tensor([[0.00], [0.10], [0.25]], dtype=torch.float32),
    )
    support_loss = front_support_tversky_loss(model, n_times=3, grid=8, device=torch.device("cpu"))
    contrast_loss = front_area_contrast_loss(model, n_times=3, grid=8, device=torch.device("cpu"))
    profile_loss = front_profile_alignment_loss(model, n=24, device=torch.device("cpu"))
    level_set_loss = front_level_set_alignment_loss(model, n=16, device=torch.device("cpu"))
    mass_floor_loss = mass_floor_trajectory_loss(model, n_times=3, grid=8, device=torch.device("cpu"))
    front_gpinn_loss = expected_front_gradient_residual_loss(model, n=16, device=torch.device("cpu"))
    interface_loss = time_slab_interface_loss(model, n=16, device=torch.device("cpu"), slabs=3)
    discrete_loss = discrete_rk4_consistency_loss(model, n_times=1, grid=8, device=torch.device("cpu"))
    assert xy.shape == (16, 2)
    assert t.shape == (16, 1)
    assert torch.isfinite(xy).all()
    assert torch.isfinite(t).all()
    assert torch.isfinite(pde_loss)
    assert torch.isfinite(floor_loss)
    assert torch.isfinite(area_loss)
    assert torch.isfinite(distribution_loss)
    assert torch.isfinite(symmetry_loss)
    assert torch.isfinite(observed_support)
    assert torch.isfinite(support_loss)
    assert torch.isfinite(contrast_loss)
    assert torch.isfinite(profile_loss)
    assert torch.isfinite(level_set_loss)
    assert torch.isfinite(mass_floor_loss)
    assert torch.isfinite(front_gpinn_loss)
    assert torch.isfinite(interface_loss)
    assert torch.isfinite(discrete_loss)


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


def test_korea_pine_wilt_compact_dataset_and_rk4_smoke(tmp_path) -> None:
    manifest = load_manifest()
    compact = manifest["compact_files"]
    processed_dir = REPO_ROOT / "data" / "korea_pine_wilt" / "processed"
    csv_gz = processed_dir / "infected_points_2016_2023.csv.gz"
    npz = processed_dir / "infected_points_2016_2023.npz"

    assert len(manifest["raw_files"]) == 8
    assert compact["records"] == 3_183_376
    assert csv_gz.exists()
    assert npz.exists()
    assert csv_gz.stat().st_size < 100_000_000
    assert npz.stat().st_size < 100_000_000

    points = load_korea_pine_wilt_points()
    assert len(points.year) == compact["records"]
    assert int(points.year.min()) == 2016
    assert int(points.year.max()) == 2023
    assert np.isfinite(points.x).all()
    assert np.isfinite(points.y).all()

    grid = build_density_grid(points, years=(2016, 2017), grid_size=24, pad_m=5_000.0, smooth_passes=0)
    assert grid.density.shape == (2, 24, 24)
    assert grid.density.min() >= 0.0
    assert grid.density.max() <= 1.0
    assert grid.land_mask is not None
    assert grid.land_mask.shape == (24, 24)
    assert grid.land_mask.any()
    assert (~grid.land_mask).any()
    assert np.all(grid.density[:, ~grid.land_mask] == 0.0)
    physical_prior = korea_physics_prior_from_physical(
        grid,
        diffusion_km2_per_year=15.5,
        reaction_per_year=0.70,
    )
    assert physical_prior.normalized_diffusion > 0.0
    assert physical_prior.normalized_diffusion_x > 0.0
    assert physical_prior.normalized_diffusion_y > 0.0
    assert physical_prior.normalized_reaction == 0.70
    assert np.isclose(
        physical_prior.diffusion_km2_per_year,
        korea_physics_prior_from_normalized(
            grid,
            normalized_diffusion=physical_prior.normalized_diffusion,
            normalized_reaction=physical_prior.normalized_reaction,
        ).diffusion_km2_per_year,
    )

    years, fields = simulate_density_rk4(
        grid.density[0],
        start_year=2016,
        end_year=2017,
        diffusion=1.0e-4,
        reaction=0.10,
        steps_per_year=4,
        land_mask=grid.land_mask,
    )
    assert years.tolist() == [2016, 2017]
    assert fields.shape == (2, 24, 24)
    assert np.isfinite(fields).all()
    assert fields.min() >= 0.0
    assert fields.max() <= 1.0
    assert np.all(fields[:, ~grid.land_mask] == 0.0)

    pinn = fit_korea_pine_wilt_pinn(
        grid,
        end_year=2017,
        epochs=1,
        batch_size=128,
        collocation_points=8,
        boundary_points=4,
        initial_condition_points=32,
        device=torch.device("cpu"),
    )
    assert pinn.years.tolist() == [2016, 2017]
    assert pinn.fields.shape == (2, 24, 24)
    assert len(pinn.metrics) == 2
    assert pinn.status == "diagnostic_only_low_epoch"
    assert np.isfinite(pinn.fields).all()
    assert np.all(pinn.fields[:, ~grid.land_mask] == 0.0)
    assert "initial_condition" in pinn.history[-1]
    assert np.isfinite(pinn.history[-1]["initial_condition"])
    assert "physics_anchor" in pinn.history[-1]
    assert np.isfinite(pinn.history[-1]["physics_anchor"])
    assert "coefficient_field" in pinn.history[-1]
    assert np.isfinite(pinn.history[-1]["coefficient_field"])
    assert "support" in pinn.history[-1]
    assert np.isfinite(pinn.history[-1]["support"])
    assert "mass_trajectory" in pinn.history[-1]
    assert np.isfinite(pinn.history[-1]["mass_trajectory"])
    assert "diffusion_km2_per_year" in pinn.physics
    assert "normalized_diffusion_x" in pinn.physics
    assert "normalized_diffusion_y" in pinn.physics
    assert "coefficient_field_weight" in pinn.physics
    assert "prior_diffusion_km2_per_year" in pinn.physics
    assert "support_weight" in pinn.physics
    assert "mass_trajectory_weight" in pinn.physics

    gif_info = _save_korea_map_baseline_gif(
        tmp_path / "korea_map_baselines.gif",
        grid,
        years,
        fields,
        pinn_years=pinn.years,
        pinn_fields=pinn.fields,
        fps=2.0,
    )
    assert (tmp_path / "korea_map_baselines.gif").read_bytes()[:6] in {b"GIF87a", b"GIF89a"}
    assert (tmp_path / "korea_map_baselines_preview.png").exists()
    assert gif_info["frames"] == 2
    assert gif_info["panels"] == ["observed", "rk4", "pinn"]


def test_korea_action_time_points_and_rk4_smoke(tmp_path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    csv_path = raw_dir / "병해충발생정보관리_2016.csv"
    rows = [
        ["5420000", "1000", "2000", "A", "1", "1", "30", "피해고사목", "감염목", "2016-01-05", "완료"],
        ["5420000", "1050", "2050", "A", "1", "1", "30", "피해고사목", "감염목", "2016-02-05", "완료"],
        ["5420000", "1100", "2100", "A", "1", "1", "30", "피해고사목", "감염목", "2016-03-05", "완료"],
        ["5420000", "1150", "2150", "A", "1", "1", "30", "피해고사목", "감염목", "2016-04-05", "완료"],
    ]
    with csv_path.open("w", encoding="cp949", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["발생관리기관코드", "지역X좌표", "지역Y좌표", "국가지점번호", "PNU코드", "법정동코드", "발생경급수치", "고사목구분", "감염목구분", "조사일자", "방제완료여부"])
        writer.writerows(rows)

    action = estimate_korea_action_start(raw_dir, cumulative_infected_complete_threshold=3)
    assert action["large_scale_action_start_date"] == "2016-03-05"
    action_data = load_korea_pine_wilt_action_time_points(raw_dir, cumulative_infected_complete_threshold=3)
    assert action_data.time_labels == ("2016-01", "2016-02")
    assert action_data.metadata["kept_records"] == 2
    grid = build_density_grid(
        action_data.points,
        years=action_data.period_ids,
        grid_size=12,
        pad_m=10.0,
        smooth_passes=0,
        enforce_land_mask=False,
        time_values=action_data.time_values,
        time_labels=action_data.time_labels,
        time_unit="pre_action_month",
        action_metadata=action_data.metadata,
    )
    assert np.allclose(korea_grid_time_values(grid), [0.0, 1.0 / 12.0])
    sim_ids, sim_fields = simulate_density_rk4_at_times(
        grid.density[0],
        output_ids=grid.years,
        output_times=korea_grid_time_values(grid),
        diffusion=1.0e-4,
        reaction=0.1,
        steps_per_year=12,
    )
    assert sim_ids.tolist() == [0, 1]
    assert sim_fields.shape[0] == 2
    rows_out = compare_observed_and_simulated(grid, sim_ids, sim_fields)
    assert rows_out[0]["time_label"] == "2016-01"
    assert rows_out[1]["elapsed_years"] == 1.0 / 12.0


def test_geo_spectral_forward_profile_extends_korea_setup() -> None:
    cfg = ExperimentConfig().geo_spectral_forward()
    assert cfg.pde.include_advection is False
    assert cfg.geo.enabled is True
    assert cfg.geo.mask_kind == "box"
    assert cfg.model.use_geo_features is True
    assert cfg.model.architecture == "pirate"
    assert cfg.model.nif_rank > 0
    assert cfg.model.use_random_weight_factorization is True
    assert cfg.model.spatial_fourier_only is True
    assert cfg.model.use_source_envelope is False
    assert cfg.model.use_seed_front_features is True
    assert cfg.model.use_traveling_wave_features is True
    assert cfg.model.use_front_fourier_features is True
    assert cfg.model.front_fourier_features > 0
    assert cfg.model.hard_initial_condition is True
    assert cfg.model.use_spatial_coefficients is True
    assert cfg.model.spatial_coefficient_log_scale > 0.0
    assert cfg.model.initial_envelope_tau > 0.10
    assert cfg.model.use_kpp_front_envelope is True
    assert cfg.weights.initial_condition > 0.0
    assert cfg.weights.boundary > 0.0
    assert cfg.weights.front_pde_alpha > 0.0
    assert cfg.weights.front_gradient > 0.0
    assert cfg.weights.front_speed > 0.0
    assert cfg.weights.mass_balance > 0.0
    assert cfg.weights.mass_floor > 0.0
    assert cfg.weights.observation_support == 0.0
    assert cfg.weights.expected_front_pde == 0.0
    assert cfg.weights.leading_edge > 0.0
    assert cfg.weights.leading_edge_area > 0.0
    assert cfg.weights.leading_edge_distribution == 0.0
    assert cfg.weights.radial_symmetry == 0.0
    assert cfg.weights.front_support_tversky > 0.0
    assert cfg.weights.level_set_alignment > 0.0
    assert cfg.weights.time_interface > 0.0
    assert cfg.weights.discrete_rk4 > 0.0
    assert cfg.weights.rk4_teacher == 0.0
    assert cfg.weights.physics_parameter_anchor > 0.0
    assert cfg.weights.coefficient_field > 0.0
    assert cfg.weights.sparse > 0.0
    assert cfg.train.adaptive_loss_balancing is True
    assert cfg.train.gradient_norm_balancing is True
    assert cfg.train.gradient_norm_balance_every > 0
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
    assert cfg.train.leading_edge_distribution_times > 0
    assert cfg.train.leading_edge_distribution_grid > 1
    assert cfg.train.radial_symmetry_groups > 0
    assert cfg.train.radial_symmetry_angles >= 4
    assert cfg.train.front_gradient_expected_points > 0
    assert cfg.train.time_marching is True
    assert cfg.train.time_slabs > 1
    assert cfg.train.time_slab_curriculum is True
    assert cfg.train.time_window_focus_fraction < 1.0
    assert cfg.train.time_window_teacher is False
    assert cfg.train.time_window_observations is True
    assert cfg.train.observation_batch > 0
    assert cfg.train.time_interface_points > 0
    assert cfg.train.discrete_rk4_times > 0
    assert cfg.train.discrete_rk4_grid > 2
    assert cfg.train.discrete_rk4_dt_fraction > 0.0
    assert cfg.train.rk4_teacher_pool == 0
    assert cfg.train.rk4_teacher_batch == 0
    assert cfg.train.rk4_pretrain_steps == 0
    assert cfg.train.residual_curriculum_epochs > 0
    assert cfg.train.residual_weight_exponent_start < cfg.train.residual_weight_exponent_end
    assert cfg.train.pde_loss_warmup_fraction > 0.0
    assert cfg.train.front_loss_start_fraction > 0.0
    assert cfg.train.front_loss_start_fraction <= cfg.train.pde_loss_warmup_fraction
    assert cfg.train.front_loss_warmup_fraction > 0.0
    assert cfg.train.time_interface_start_fraction > cfg.train.front_loss_start_fraction
    assert cfg.train.time_interface_warmup_fraction > 0.0
    assert cfg.train.rar_residual_weight > 0.0
    assert cfg.train.rar_gradient_weight > 0.0
    assert cfg.train.rar_activity_weight > 0.0


def test_shared_forward_model_factory_specializes_korea_without_changing_backbone() -> None:
    baseline = ExperimentConfig().geo_spectral_forward().model
    korea = shared_geo_forward_model_config(
        hidden=48,
        layers=3,
        fourier_features=16,
        front_fourier_features=0,
        use_seed_front_features=False,
        use_traveling_wave_features=False,
        use_front_fourier_features=False,
        hard_initial_condition=False,
        use_kpp_front_envelope=False,
        spatial_coefficient_sigma=0.65,
        spatial_coefficient_log_scale=0.35,
    )
    shared_attrs = [
        "architecture",
        "use_random_weight_factorization",
        "learn_diffusion",
        "learn_reaction",
        "learn_drift",
        "use_source_envelope",
        "use_geo_features",
        "spatial_fourier_only",
        "use_spatial_coefficients",
        "spatial_coefficient_features",
        "spatial_coefficient_hidden",
    ]
    for attr in shared_attrs:
        assert getattr(korea, attr) == getattr(baseline, attr)
    assert baseline.use_seed_front_features is True
    assert korea.use_seed_front_features is False
    assert baseline.hard_initial_condition is True
    assert korea.hard_initial_condition is False
    assert baseline.use_kpp_front_envelope is True
    assert korea.use_kpp_front_envelope is False


def test_front_aware_adaptive_sampler_refreshes_anchors() -> None:
    domain = DomainConfig(grid=25, truth_steps=80)
    cfg = ExperimentConfig(domain=domain).geo_spectral_forward()
    model = OriginPINN(
        domain,
        cfg.pde,
        cfg.seed,
        replace(cfg.model, hidden=16, layers=1, fourier_features=8),
    )
    sampler = SobolCollocation(domain.box, domain.t_end, torch.device("cpu"), seed=3)
    sampler.refresh(
        model,
        candidate_n=40,
        keep=8,
        chunk=20,
        front_alpha=cfg.weights.front_pde_alpha,
        front_gradient=cfg.weights.front_pde_gradient,
        residual_weight=cfg.train.rar_residual_weight,
        gradient_weight=cfg.train.rar_gradient_weight,
        activity_weight=cfg.train.rar_activity_weight,
    )
    assert sampler.anchors is not None
    assert sampler.anchors.shape == (8, 3)
    assert torch.isfinite(sampler.anchors).all()


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
    assert "geo_levelset_time_slab" in names
    assert "geo_no_mass_floor" in names
    assert "geo_no_support_tversky" in names
    assert "geo_no_physics_anchor" in names
    assert "geo_no_spatial_coefficients" in names
    assert "geo_no_discrete_rk4" in names
    assert "geo_no_radial_symmetry" in names
    assert "geo_no_collapse_guards" in names
    assert "geo_no_tw_front_area" in names
    assert "geo_rk4_teacher_front_area" in names
    assert "geo_rk4_late_teacher_front_area" in names
    assert "geo_rk4_pretrain_front_area" in names
    assert "geo_nif_front_area" in names
    assert "geo_gated_front_area" in names
    assert any(case["cfg"].weights.leading_edge_area > 0.0 for case in cases)
    assert any(case["cfg"].weights.radial_symmetry > 0.0 for case in cases)
    assert any(case["cfg"].weights.front_contrast > 0.0 for case in cases)
    assert any(case["cfg"].weights.front_profile > 0.0 for case in cases)
    assert any(case["cfg"].weights.level_set_alignment > 0.0 for case in cases)
    assert any(case["cfg"].weights.time_interface > 0.0 for case in cases)
    assert any(case["cfg"].model.use_front_fourier_features is True for case in cases)
    assert any(case["cfg"].train.gradient_norm_balancing is True for case in cases)
    assert any(case["cfg"].model.architecture == "nif_pirate" for case in cases)
    assert any(case["cfg"].model.architecture == "pirate" for case in cases)
    assert any(case["cfg"].model.architecture == "gated_mlp" for case in cases)
    assert any(case["cfg"].model.use_traveling_wave_features is False for case in cases)
    assert any(case["cfg"].weights.rk4_teacher > 0.0 for case in cases)
    assert any(case["cfg"].weights.mass_floor == 0.0 for case in cases)
    assert any(case["cfg"].weights.front_support_tversky == 0.0 for case in cases)
    assert any(case["cfg"].weights.physics_parameter_anchor == 0.0 for case in cases)
    assert any(case["cfg"].model.use_spatial_coefficients is False for case in cases)
    assert any(case["cfg"].weights.discrete_rk4 == 0.0 for case in cases)
    assert any(case["cfg"].weights.radial_symmetry == 0.0 for case in cases)
    assert any(case["cfg"].train.rk4_teacher_late_fraction > 0.0 for case in cases)
    assert any(case["cfg"].train.rk4_pretrain_steps > 0 for case in cases)
    assert any(case["cfg"].train.time_marching for case in cases)
    assert any(case["cfg"].train.time_slabs > 1 for case in cases)
    assert any(case["cfg"].train.time_window_focus_fraction < 1.0 for case in cases)
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


def test_feature_validation_pairs_and_visual_exports(tmp_path) -> None:
    cfg = ExperimentConfig(domain=DomainConfig(grid=21, truth_steps=60)).geo_spectral_forward().quick()
    pairs = make_feature_validation_pairs(cfg)
    names = [pair["name"] for pair in pairs]
    assert "level_set_alignment" in names
    assert "time_marching_curriculum" in names
    assert "moving_front_features" in names
    assert "mass_support_guards" in names
    assert "front_speed_gpinn" in names
    assert "adaptive_balancing" in names
    assert "spatial_coefficients" in names
    assert "discrete_rk4_consistency" in names
    assert "leading_edge_distribution" in names
    assert "radial_symmetry" in names
    assert "rk4_teacher" in names
    assert all("without" in pair and "with" in pair for pair in pairs)

    xs = np.linspace(0.0, 1.0, 16)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    truth = np.exp(-((x - 0.35) ** 2 + (y - 0.55) ** 2) / 0.04)
    without_pred = np.clip(truth + 0.12 * np.sin(2 * np.pi * x), 0.0, 1.0)
    with_pred = np.clip(truth + 0.05 * np.sin(2 * np.pi * x), 0.0, 1.0)
    without_fields = tmp_path / "without_fields.npz"
    with_fields = tmp_path / "with_fields.npz"
    np.savez_compressed(
        without_fields,
        truth_final=truth,
        pinn_final=without_pred,
        rk4_final=truth,
        pinn_abs_error=np.abs(without_pred - truth),
    )
    np.savez_compressed(
        with_fields,
        truth_final=truth,
        pinn_final=with_pred,
        rk4_final=truth,
        pinn_abs_error=np.abs(with_pred - truth),
    )
    error_map = save_feature_pair_error_map(
        tmp_path / "feature_error_map_comparison.png",
        feature_name="test_feature",
        without_label="without",
        with_label="with",
        without_fields=without_fields,
        with_fields=with_fields,
        without_metrics={"final_time_relative_l2": 0.5, "validation_observation_mse": 0.02},
        with_metrics={"final_time_relative_l2": 0.2, "validation_observation_mse": 0.01},
        domain=cfg.domain,
    )
    assert (tmp_path / "feature_error_map_comparison.png").exists()
    assert error_map["with_max_abs_error"] < error_map["without_max_abs_error"]

    def _dummy_gif(path: Path, color: tuple[int, int, int]) -> None:
        frames = [Image.new("RGB", (30, 24), tuple(min(255, channel + idx * 20) for channel in color)) for idx in range(3)]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)

    _dummy_gif(tmp_path / "without.gif", (120, 80, 80))
    _dummy_gif(tmp_path / "with.gif", (80, 120, 80))
    gif_info = save_feature_pair_evolution_gif(
        tmp_path / "feature_evolution_comparison.gif",
        feature_name="test_feature",
        without_label="without",
        with_label="with",
        without_gif=tmp_path / "without.gif",
        with_gif=tmp_path / "with.gif",
    )
    assert (tmp_path / "feature_evolution_comparison.gif").read_bytes()[:6] in {b"GIF87a", b"GIF89a"}
    assert Path(str(gif_info["preview"])).exists()
    assert gif_info["frames"] == 3

    summary = summarize_feature_validation_rows(
        [
            {"feature": "test", "variant": "without", "final_time_relative_l2": 0.5},
            {"feature": "test", "variant": "with", "final_time_relative_l2": 0.2},
        ]
    )
    assert summary["features"][0]["improvement_final_time_relative_l2_mean"] == 0.3


def test_time_slab_curriculum_uses_cumulative_windows() -> None:
    cfg = replace(
        ExperimentConfig(domain=DomainConfig(t_end=0.5)),
        train=replace(
            ExperimentConfig().train,
            epochs=40,
            time_marching=True,
            time_marching_start_fraction=0.25,
            time_marching_epochs=20,
            time_slabs=4,
            time_slab_overlap=0.05,
            time_slab_curriculum=True,
        ),
    )

    early = _time_window(cfg, 1)
    late = _time_window(cfg, 40)
    assert early[0] == 0.0
    assert late[0] == 0.0
    assert early[1] < late[1]
    assert np.isclose(late[1], cfg.domain.t_end)


def test_masked_batch_indices_can_skip_empty_time_window() -> None:
    xyt = torch.tensor([[0.1, 0.2, 0.35], [0.2, 0.2, 0.5]], dtype=torch.float32)
    assert _masked_batch_indices(xyt, 4, torch.device("cpu"), t_low=0.0, t_high=0.2, fallback_all=False) is None

    idx = _masked_batch_indices(xyt, 4, torch.device("cpu"), t_low=0.0, t_high=0.2)
    assert idx is not None
    assert idx.shape == (4,)


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
    gif_diag = save_pinn_evolution_gif(
        tmp_path / "pinn_evolution.gif",
        truth,
        model,
        cfg.domain,
        torch.device("cpu"),
        n=12,
        max_frames=3,
        fps=2,
        caption="test diagnostic",
        warning="DIAGNOSTIC ONLY: smoke visualization.",
    )

    for name in ["coverage.png", "spacetime.png", "residual.png", "pinn_vs_rk4.png", "training.png", "pinn_evolution.gif"]:
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 1000
    assert (tmp_path / "pinn_evolution.gif").read_bytes()[:6] in {b"GIF87a", b"GIF89a"}
    assert gif_diag["frames"] == 3
    assert gif_diag["warning"] == "DIAGNOSTIC ONLY: smoke visualization."
    assert gif_diag["final_frame_relative_l2"] >= 0.0
