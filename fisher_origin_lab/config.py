from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DomainConfig:
    box: float = 1.0
    t_end: float = 0.5
    grid: int = 101
    truth_steps: int = 500


@dataclass(frozen=True)
class PDEConfig:
    diffusion: float = 2.0e-2
    reaction: float = 3.0
    velocity_x: float = 0.6
    velocity_y: float = -0.45
    include_advection: bool = True


@dataclass(frozen=True)
class SeedConfig:
    center_x: float = 0.30
    center_y: float = 0.68
    sigma: float = 0.05
    amplitude: float = 0.50


@dataclass(frozen=True)
class ObservationConfig:
    start_time: float = 0.35
    frames: int = 5
    samples_per_frame: int = 500
    noise_std: float = 0.02
    focus_fraction: float = 0.5
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class GeoConfig:
    enabled: bool = False
    # "box" preserves the current synthetic square-domain problem. The sampler
    # interface is explicit so a real land mask can replace this later.
    mask_kind: str = "box"


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "gated_mlp"
    fourier_features: int = 32
    fourier_sigma: float = 3.5
    front_fourier_features: int = 0
    front_fourier_sigma: float = 2.0
    hidden: int = 96
    layers: int = 5
    nif_rank: int = 16
    use_random_weight_factorization: bool = False
    learn_diffusion: bool = False
    learn_reaction: bool = False
    learn_drift: bool = False
    use_source_envelope: bool = True
    use_geo_features: bool = False
    spatial_fourier_only: bool = False
    use_seed_front_features: bool = False
    use_traveling_wave_features: bool = False
    use_front_fourier_features: bool = False
    hard_initial_condition: bool = False
    initial_envelope_tau: float = 0.06
    use_kpp_front_envelope: bool = False
    front_envelope_level: float = 0.03
    front_envelope_margin: float = 0.08
    front_envelope_width: float = 0.04


@dataclass(frozen=True)
class LossWeights:
    data: float = 60.0
    pde: float = 1.0
    initial_condition: float = 0.0
    boundary: float = 0.5
    seed_match: float = 4.0
    seed_mass: float = 0.25
    source_anchor: float = 2.0
    shooting: float = 5.0
    gradient: float = 0.01
    data_density_gain: float = 0.0
    front_pde_alpha: float = 0.0
    front_pde_gradient: float = 0.0
    front_gradient: float = 0.0
    front_speed: float = 0.0
    mass_balance: float = 0.0
    expected_front_pde: float = 0.0
    leading_edge: float = 0.0
    leading_edge_area: float = 0.0
    front_contrast: float = 0.0
    front_profile: float = 0.0
    level_set_alignment: float = 0.0
    time_interface: float = 0.0
    rk4_teacher: float = 0.0
    sparse: float = 0.0


@dataclass(frozen=True)
class WarmStartConfig:
    # Options: "drift_corrected", "centroid", "neutral", "shooting_prefit".
    # "drift_corrected" is valid only when drift is treated as known.
    mode: str = "drift_corrected"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1200
    lr: float = 2.0e-3
    source_lr_multiplier: float = 2.0
    collocation_points: int = 2048
    boundary_points: int = 256
    seed_points: int = 1024
    time_bins: int = 8
    causal_eps: float = 8.0
    decay_beta: float = 0.95
    rar_interval: int = 200
    rar_candidates: int = 4096
    rar_keep: int = 512
    rar_residual_weight: float = 1.0
    rar_gradient_weight: float = 0.25
    rar_activity_weight: float = 0.25
    shooting_grid: int = 33
    shooting_steps: int = 80
    shooting_points: int = 512
    shooting_prefit_steps: int = 40
    residual_weight_exponent_start: float = 0.5
    residual_weight_exponent_end: float = 0.5
    residual_curriculum_epochs: int = 0
    adaptive_loss_balancing: bool = False
    gradient_norm_balancing: bool = False
    gradient_norm_balance_every: int = 25
    adaptive_loss_momentum: float = 0.9
    adaptive_loss_min: float = 0.25
    adaptive_loss_max: float = 4.0
    validation_every: int = 50
    restore_best_validation: bool = True
    front_speed_points: int = 256
    front_speed_max_points: int = 128
    front_speed_min_grad: float = 1.0e-2
    expected_front_points: int = 256
    expected_front_width: float = 0.08
    expected_front_speed_factor: float = 0.45
    expected_front_level: float = 0.1
    level_set_points: int = 256
    level_set_width: float = 0.025
    leading_edge_area_times: int = 5
    leading_edge_area_grid: int = 32
    leading_edge_area_temperature: float = 0.015
    front_contrast_times: int = 5
    front_contrast_grid: int = 32
    front_profile_points: int = 256
    front_profile_width: float = 0.06
    front_gradient_expected_points: int = 128
    mass_balance_times: int = 4
    mass_balance_grid: int = 18
    time_interface_points: int = 256
    time_interface_width: float = 0.01
    rk4_teacher_pool: int = 0
    rk4_teacher_batch: int = 0
    rk4_teacher_late_fraction: float = 0.0
    rk4_pretrain_steps: int = 0
    rk4_pretrain_batch: int = 0
    rk4_pretrain_lr: float = 1.0e-3
    time_marching: bool = False
    time_marching_start_fraction: float = 0.35
    time_marching_epochs: int = 0
    time_slabs: int = 1
    time_slab_overlap: float = 0.05
    time_slab_curriculum: bool = False
    time_window_focus_fraction: float = 1.0
    time_window_teacher: bool = False
    time_window_observations: bool = False
    observation_batch: int = 0
    print_every: int = 100
    adam_to_lbfgs: bool = False
    lbfgs_steps: int = 100


@dataclass(frozen=True)
class ExperimentConfig:
    domain: DomainConfig = DomainConfig()
    pde: PDEConfig = PDEConfig()
    seed: SeedConfig = SeedConfig()
    observations: ObservationConfig = ObservationConfig()
    geo: GeoConfig = GeoConfig()
    model: ModelConfig = ModelConfig()
    weights: LossWeights = LossWeights()
    warm_start: WarmStartConfig = WarmStartConfig()
    train: TrainConfig = TrainConfig()
    ensemble: int = 1
    base_seed: int = 7
    out_dir: Path = Path("runs/default")
    run_classical_baseline: bool = False
    baseline_epochs: int = 250

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        return data

    def quick(self) -> "ExperimentConfig":
        return ExperimentConfig(
            domain=DomainConfig(grid=51, truth_steps=160),
            pde=self.pde,
            seed=self.seed,
            observations=ObservationConfig(
                start_time=self.observations.start_time,
                frames=4,
                samples_per_frame=160,
                noise_std=self.observations.noise_std,
                focus_fraction=self.observations.focus_fraction,
                validation_fraction=self.observations.validation_fraction,
            ),
            geo=self.geo,
            model=ModelConfig(
                architecture=self.model.architecture,
                fourier_features=16,
                fourier_sigma=self.model.fourier_sigma,
                front_fourier_features=min(self.model.front_fourier_features, 8)
                if self.model.front_fourier_features > 0
                else 0,
                front_fourier_sigma=self.model.front_fourier_sigma,
                hidden=48,
                layers=3,
                nif_rank=self.model.nif_rank,
                use_random_weight_factorization=self.model.use_random_weight_factorization,
                learn_diffusion=self.model.learn_diffusion,
                learn_reaction=self.model.learn_reaction,
                learn_drift=self.model.learn_drift,
                use_source_envelope=self.model.use_source_envelope,
                use_geo_features=self.model.use_geo_features,
                spatial_fourier_only=self.model.spatial_fourier_only,
                use_seed_front_features=self.model.use_seed_front_features,
                use_traveling_wave_features=self.model.use_traveling_wave_features,
                use_front_fourier_features=self.model.use_front_fourier_features,
                hard_initial_condition=self.model.hard_initial_condition,
                initial_envelope_tau=self.model.initial_envelope_tau,
                use_kpp_front_envelope=self.model.use_kpp_front_envelope,
                front_envelope_level=self.model.front_envelope_level,
                front_envelope_margin=self.model.front_envelope_margin,
                front_envelope_width=self.model.front_envelope_width,
            ),
            weights=LossWeights(
                data=self.weights.data,
                pde=self.weights.pde,
                initial_condition=self.weights.initial_condition,
                boundary=self.weights.boundary,
                seed_match=self.weights.seed_match,
                seed_mass=self.weights.seed_mass,
                source_anchor=self.weights.source_anchor,
                shooting=self.weights.shooting,
                gradient=0.0,
                data_density_gain=self.weights.data_density_gain,
                front_pde_alpha=self.weights.front_pde_alpha,
                front_pde_gradient=self.weights.front_pde_gradient,
                front_gradient=self.weights.front_gradient,
                front_speed=self.weights.front_speed,
                mass_balance=self.weights.mass_balance,
                expected_front_pde=self.weights.expected_front_pde,
                leading_edge=self.weights.leading_edge,
                leading_edge_area=self.weights.leading_edge_area,
                front_contrast=self.weights.front_contrast,
                front_profile=self.weights.front_profile,
                level_set_alignment=self.weights.level_set_alignment,
                time_interface=self.weights.time_interface,
                rk4_teacher=self.weights.rk4_teacher,
                sparse=self.weights.sparse,
            ),
            warm_start=self.warm_start,
            train=TrainConfig(
                epochs=120,
                lr=self.train.lr,
                source_lr_multiplier=self.train.source_lr_multiplier,
                collocation_points=512,
                boundary_points=96,
                seed_points=256,
                time_bins=6,
                causal_eps=self.train.causal_eps,
                decay_beta=self.train.decay_beta,
                rar_interval=60,
                rar_candidates=1024,
                rar_keep=128,
                rar_residual_weight=self.train.rar_residual_weight,
                rar_gradient_weight=self.train.rar_gradient_weight,
                rar_activity_weight=self.train.rar_activity_weight,
                shooting_grid=31,
                shooting_steps=80,
                shooting_points=256,
                shooting_prefit_steps=25,
                residual_weight_exponent_start=self.train.residual_weight_exponent_start,
                residual_weight_exponent_end=self.train.residual_weight_exponent_end,
                residual_curriculum_epochs=(
                    min(self.train.residual_curriculum_epochs, 60)
                    if self.train.residual_curriculum_epochs > 0
                    else 0
                ),
                adaptive_loss_balancing=self.train.adaptive_loss_balancing,
                gradient_norm_balancing=self.train.gradient_norm_balancing,
                gradient_norm_balance_every=max(1, self.train.gradient_norm_balance_every),
                adaptive_loss_momentum=self.train.adaptive_loss_momentum,
                adaptive_loss_min=self.train.adaptive_loss_min,
                adaptive_loss_max=self.train.adaptive_loss_max,
                validation_every=30,
                restore_best_validation=self.train.restore_best_validation,
                front_speed_points=min(self.train.front_speed_points, 128),
                front_speed_max_points=min(self.train.front_speed_max_points, 64),
                front_speed_min_grad=self.train.front_speed_min_grad,
                expected_front_points=min(self.train.expected_front_points, 128),
                expected_front_width=self.train.expected_front_width,
                expected_front_speed_factor=self.train.expected_front_speed_factor,
                expected_front_level=self.train.expected_front_level,
                level_set_points=min(self.train.level_set_points, 128),
                level_set_width=self.train.level_set_width,
                leading_edge_area_times=min(self.train.leading_edge_area_times, 4),
                leading_edge_area_grid=min(self.train.leading_edge_area_grid, 24),
                leading_edge_area_temperature=self.train.leading_edge_area_temperature,
                front_contrast_times=min(self.train.front_contrast_times, 4),
                front_contrast_grid=min(self.train.front_contrast_grid, 24),
                front_profile_points=min(self.train.front_profile_points, 128),
                front_profile_width=self.train.front_profile_width,
                front_gradient_expected_points=min(self.train.front_gradient_expected_points, 96),
                mass_balance_times=min(self.train.mass_balance_times, 4),
                mass_balance_grid=min(self.train.mass_balance_grid, 18),
                time_interface_points=min(self.train.time_interface_points, 128),
                time_interface_width=self.train.time_interface_width,
                rk4_teacher_pool=min(self.train.rk4_teacher_pool, 2048) if self.train.rk4_teacher_pool > 0 else 0,
                rk4_teacher_batch=min(self.train.rk4_teacher_batch, 256) if self.train.rk4_teacher_batch > 0 else 0,
                rk4_teacher_late_fraction=self.train.rk4_teacher_late_fraction,
                rk4_pretrain_steps=min(self.train.rk4_pretrain_steps, 80)
                if self.train.rk4_pretrain_steps > 0
                else 0,
                rk4_pretrain_batch=min(self.train.rk4_pretrain_batch, 256)
                if self.train.rk4_pretrain_batch > 0
                else 0,
                rk4_pretrain_lr=self.train.rk4_pretrain_lr,
                time_marching=self.train.time_marching,
                time_marching_start_fraction=self.train.time_marching_start_fraction,
                time_marching_epochs=(
                    min(self.train.time_marching_epochs, 60)
                    if self.train.time_marching_epochs > 0
                    else 0
                ),
                time_slabs=min(max(1, self.train.time_slabs), 4),
                time_slab_overlap=self.train.time_slab_overlap,
                time_slab_curriculum=self.train.time_slab_curriculum,
                time_window_focus_fraction=self.train.time_window_focus_fraction,
                time_window_teacher=self.train.time_window_teacher,
                time_window_observations=self.train.time_window_observations,
                observation_batch=min(self.train.observation_batch, 384) if self.train.observation_batch > 0 else 0,
                print_every=30,
                adam_to_lbfgs=False,
            ),
            ensemble=self.ensemble,
            base_seed=self.base_seed,
            out_dir=self.out_dir,
            run_classical_baseline=self.run_classical_baseline,
            baseline_epochs=min(self.baseline_epochs, 60),
        )

    def korea_pine_style(self) -> "ExperimentConfig":
        """Match the PINN problem setup used in the Korea pine-wilt notebook.

        The referenced notebook trains a forward Fisher-KPP PINN with diffusion
        and logistic growth only. It has no explicit PINN boundary-condition loss;
        land/coast masking is used in data preparation and plotting rather than as
        a Neumann or Dirichlet penalty in the neural loss.
        """

        return ExperimentConfig(
            domain=self.domain,
            pde=PDEConfig(
                diffusion=self.pde.diffusion,
                reaction=self.pde.reaction,
                velocity_x=0.0,
                velocity_y=0.0,
                include_advection=False,
            ),
            seed=self.seed,
            observations=self.observations,
            geo=self.geo,
            model=ModelConfig(
                architecture=self.model.architecture,
                fourier_features=self.model.fourier_features,
                fourier_sigma=self.model.fourier_sigma,
                front_fourier_features=self.model.front_fourier_features,
                front_fourier_sigma=self.model.front_fourier_sigma,
                hidden=self.model.hidden,
                layers=self.model.layers,
                nif_rank=self.model.nif_rank,
                use_random_weight_factorization=self.model.use_random_weight_factorization,
                learn_diffusion=True,
                learn_reaction=True,
                learn_drift=False,
                use_source_envelope=False,
                use_geo_features=self.model.use_geo_features,
                spatial_fourier_only=self.model.spatial_fourier_only,
                use_seed_front_features=self.model.use_seed_front_features,
                use_traveling_wave_features=self.model.use_traveling_wave_features,
                use_front_fourier_features=self.model.use_front_fourier_features,
                hard_initial_condition=False,
                initial_envelope_tau=self.model.initial_envelope_tau,
                use_kpp_front_envelope=False,
                front_envelope_level=self.model.front_envelope_level,
                front_envelope_margin=self.model.front_envelope_margin,
                front_envelope_width=self.model.front_envelope_width,
            ),
            weights=LossWeights(
                data=1.0,
                pde=1.0,
                initial_condition=0.0,
                boundary=0.0,
                seed_match=0.0,
                seed_mass=0.0,
                source_anchor=0.0,
                shooting=0.0,
                gradient=0.0,
                data_density_gain=4.0,
                front_pde_alpha=0.0,
                front_pde_gradient=0.0,
                front_gradient=0.0,
                front_speed=0.0,
                mass_balance=0.0,
                expected_front_pde=0.0,
                leading_edge=0.0,
                leading_edge_area=0.0,
                front_contrast=0.0,
                front_profile=0.0,
                level_set_alignment=0.0,
                time_interface=0.0,
                rk4_teacher=0.0,
                sparse=0.0,
            ),
            warm_start=WarmStartConfig(mode="neutral"),
            train=self.train,
            ensemble=self.ensemble,
            base_seed=self.base_seed,
            out_dir=self.out_dir,
            run_classical_baseline=self.run_classical_baseline,
            baseline_epochs=self.baseline_epochs,
        )

    def geo_spectral_forward(self) -> "ExperimentConfig":
        """Forward Fisher-KPP PINN with geo features and front-aware losses."""

        base = self.korea_pine_style()
        return ExperimentConfig(
            domain=base.domain,
            pde=base.pde,
            seed=base.seed,
            observations=base.observations,
            geo=GeoConfig(enabled=True, mask_kind="box"),
            model=ModelConfig(
                architecture="pirate",
                fourier_features=base.model.fourier_features,
                fourier_sigma=1.0,
                front_fourier_features=16,
                front_fourier_sigma=1.5,
                hidden=base.model.hidden,
                layers=base.model.layers,
                nif_rank=24,
                use_random_weight_factorization=True,
                learn_diffusion=True,
                learn_reaction=True,
                learn_drift=False,
                use_source_envelope=False,
                use_geo_features=True,
                spatial_fourier_only=True,
                use_seed_front_features=True,
                use_traveling_wave_features=True,
                use_front_fourier_features=True,
                hard_initial_condition=True,
                initial_envelope_tau=0.06,
                use_kpp_front_envelope=True,
                front_envelope_level=0.03,
                front_envelope_margin=0.05,
                front_envelope_width=0.025,
            ),
            weights=LossWeights(
                data=1.0,
                pde=1.0,
                initial_condition=10.0,
                boundary=0.2,
                seed_match=0.0,
                seed_mass=0.0,
                source_anchor=0.0,
                shooting=0.0,
                gradient=0.0,
                data_density_gain=4.0,
                front_pde_alpha=2.0,
                front_pde_gradient=0.5,
                front_gradient=0.005,
                front_speed=0.01,
                mass_balance=0.20,
                expected_front_pde=0.0,
                leading_edge=0.0,
                leading_edge_area=1.0,
                front_contrast=0.10,
                front_profile=0.20,
                level_set_alignment=0.06,
                time_interface=0.03,
                rk4_teacher=0.0,
                sparse=1.0e-5,
            ),
            warm_start=base.warm_start,
            train=replace(
                base.train,
                residual_weight_exponent_start=0.0,
                residual_weight_exponent_end=0.5,
                residual_curriculum_epochs=max(1, base.train.epochs // 4),
                adaptive_loss_balancing=True,
                gradient_norm_balancing=True,
                gradient_norm_balance_every=25,
                adaptive_loss_momentum=0.9,
                adaptive_loss_min=0.33,
                adaptive_loss_max=3.0,
                front_speed_min_grad=1.0e-2,
                expected_front_points=256,
                expected_front_width=0.08,
                expected_front_speed_factor=0.45,
                expected_front_level=0.1,
                leading_edge_area_times=5,
                leading_edge_area_grid=32,
                leading_edge_area_temperature=0.015,
                front_contrast_times=5,
                front_contrast_grid=32,
                front_profile_points=256,
                front_profile_width=0.06,
                front_gradient_expected_points=128,
                time_marching=True,
                time_marching_start_fraction=0.30,
                time_marching_epochs=max(1, base.train.epochs // 2),
                time_slabs=4,
                time_slab_overlap=0.08,
                time_slab_curriculum=True,
                time_window_focus_fraction=0.65,
                time_window_teacher=True,
                time_window_observations=True,
                observation_batch=512,
                time_interface_points=256,
                time_interface_width=0.015,
            ),
            ensemble=base.ensemble,
            base_seed=base.base_seed,
            out_dir=base.out_dir,
            run_classical_baseline=base.run_classical_baseline,
            baseline_epochs=base.baseline_epochs,
        )
