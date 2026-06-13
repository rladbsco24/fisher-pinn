from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .exact_wave import AZ_DT_1D, AZ_NX_1D, AZ_T_1D, AZ_X_LEFT_1D, AZ_X_RIGHT_1D, az_normalized_diffusion


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
class BenchmarkConfig:
    kind: str = "gaussian_seed"
    x_left: float = AZ_X_LEFT_1D
    x_right: float = AZ_X_RIGHT_1D
    wave_x0: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "gated_mlp"
    fourier_features: int = 32
    fourier_sigma: float = 3.5
    front_fourier_features: int = 0
    front_fourier_sigma: float = 2.0
    use_intrinsic_front_phase: bool = False
    intrinsic_front_phase_fourier_features: int = 8
    intrinsic_front_phase_fourier_sigma: float = 1.0
    intrinsic_front_phase_hidden: int = 32
    intrinsic_front_phase_layers: int = 2
    intrinsic_front_phase_feature_frequencies: int = 4
    intrinsic_front_phase_correction_scale: float = 0.25
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
    use_planar_wave_features: bool = False
    planar_wave_direction_x: float = 1.0
    planar_wave_direction_y: float = 0.0
    use_az_hard_constraints: bool = False
    az_x_left: float = AZ_X_LEFT_1D
    az_x_right: float = AZ_X_RIGHT_1D
    az_wave_x0: float = 0.0
    use_front_fourier_features: bool = False
    hard_initial_condition: bool = False
    initial_envelope_tau: float = 0.06
    use_kpp_front_envelope: bool = False
    front_envelope_level: float = 0.03
    front_envelope_margin: float = 0.08
    front_envelope_width: float = 0.04
    use_spatial_coefficients: bool = False
    spatial_coefficient_features: int = 8
    spatial_coefficient_sigma: float = 1.0
    spatial_coefficient_hidden: int = 32
    spatial_coefficient_log_scale: float = 0.20


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
    observation_support: float = 0.0
    observation_support_area: float = 0.0
    gradient: float = 0.01
    data_density_gain: float = 0.0
    phase_pde: float = 0.0
    residual_cvar: float = 0.0
    intrinsic_phase_initial: float = 0.0
    intrinsic_phase_gradient_alignment: float = 0.0
    intrinsic_phase_monotonicity: float = 0.0
    front_pde_alpha: float = 0.0
    front_pde_gradient: float = 0.0
    front_gradient: float = 0.0
    front_speed: float = 0.0
    mass_balance: float = 0.0
    mass_floor: float = 0.0
    expected_front_pde: float = 0.0
    leading_edge: float = 0.0
    leading_edge_area: float = 0.0
    leading_edge_distribution: float = 0.0
    radial_symmetry: float = 0.0
    front_support_tversky: float = 0.0
    front_contrast: float = 0.0
    front_profile: float = 0.0
    level_set_alignment: float = 0.0
    transverse_invariance: float = 0.0
    time_interface: float = 0.0
    discrete_rk4: float = 0.0
    rk4_teacher: float = 0.0
    physics_parameter_anchor: float = 0.0
    coefficient_field: float = 0.0
    sparse: float = 0.0


@dataclass(frozen=True)
class WarmStartConfig:
    # Options: "drift_corrected", "centroid", "neutral", "shooting_prefit".
    # "drift_corrected" is valid only when drift is treated as known.
    mode: str = "drift_corrected"


def shared_geo_forward_model_config(
    *,
    hidden: int = 96,
    layers: int = 5,
    fourier_features: int = 32,
    fourier_sigma: float = 1.0,
    front_fourier_features: int = 16,
    front_fourier_sigma: float = 1.5,
    nif_rank: int = 24,
    learn_diffusion: bool = True,
    learn_reaction: bool = True,
    learn_drift: bool = False,
    use_seed_front_features: bool = True,
    use_traveling_wave_features: bool = True,
    use_planar_wave_features: bool = False,
    planar_wave_direction_x: float = 1.0,
    planar_wave_direction_y: float = 0.0,
    use_az_hard_constraints: bool = False,
    az_x_left: float = AZ_X_LEFT_1D,
    az_x_right: float = AZ_X_RIGHT_1D,
    az_wave_x0: float = 0.0,
    use_front_fourier_features: bool = True,
    use_intrinsic_front_phase: bool = False,
    intrinsic_front_phase_fourier_features: int = 8,
    intrinsic_front_phase_fourier_sigma: float = 1.0,
    intrinsic_front_phase_hidden: int = 32,
    intrinsic_front_phase_layers: int = 2,
    intrinsic_front_phase_feature_frequencies: int = 4,
    intrinsic_front_phase_correction_scale: float = 0.25,
    hard_initial_condition: bool = True,
    initial_envelope_tau: float = 0.18,
    use_kpp_front_envelope: bool = True,
    front_envelope_level: float = 0.03,
    front_envelope_margin: float = 0.05,
    front_envelope_width: float = 0.025,
    use_spatial_coefficients: bool = True,
    spatial_coefficient_features: int = 8,
    spatial_coefficient_sigma: float = 0.85,
    spatial_coefficient_hidden: int = 32,
    spatial_coefficient_log_scale: float = 0.12,
) -> ModelConfig:
    """Shared forward PINN backbone for synthetic and Korea pine-wilt runs."""

    return ModelConfig(
        architecture="pirate",
        fourier_features=int(fourier_features),
        fourier_sigma=float(fourier_sigma),
        front_fourier_features=int(front_fourier_features),
        front_fourier_sigma=float(front_fourier_sigma),
        hidden=int(hidden),
        layers=int(layers),
        nif_rank=int(nif_rank),
        use_random_weight_factorization=True,
        learn_diffusion=bool(learn_diffusion),
        learn_reaction=bool(learn_reaction),
        learn_drift=bool(learn_drift),
        use_source_envelope=False,
        use_geo_features=True,
        spatial_fourier_only=True,
        use_seed_front_features=bool(use_seed_front_features),
        use_traveling_wave_features=bool(use_traveling_wave_features),
        use_planar_wave_features=bool(use_planar_wave_features),
        planar_wave_direction_x=float(planar_wave_direction_x),
        planar_wave_direction_y=float(planar_wave_direction_y),
        use_az_hard_constraints=bool(use_az_hard_constraints),
        az_x_left=float(az_x_left),
        az_x_right=float(az_x_right),
        az_wave_x0=float(az_wave_x0),
        use_front_fourier_features=bool(use_front_fourier_features),
        use_intrinsic_front_phase=bool(use_intrinsic_front_phase),
        intrinsic_front_phase_fourier_features=int(intrinsic_front_phase_fourier_features),
        intrinsic_front_phase_fourier_sigma=float(intrinsic_front_phase_fourier_sigma),
        intrinsic_front_phase_hidden=int(intrinsic_front_phase_hidden),
        intrinsic_front_phase_layers=int(intrinsic_front_phase_layers),
        intrinsic_front_phase_feature_frequencies=int(intrinsic_front_phase_feature_frequencies),
        intrinsic_front_phase_correction_scale=float(intrinsic_front_phase_correction_scale),
        hard_initial_condition=bool(hard_initial_condition),
        initial_envelope_tau=float(initial_envelope_tau),
        use_kpp_front_envelope=bool(use_kpp_front_envelope),
        front_envelope_level=float(front_envelope_level),
        front_envelope_margin=float(front_envelope_margin),
        front_envelope_width=float(front_envelope_width),
        use_spatial_coefficients=bool(use_spatial_coefficients),
        spatial_coefficient_features=int(spatial_coefficient_features),
        spatial_coefficient_sigma=float(spatial_coefficient_sigma),
        spatial_coefficient_hidden=int(spatial_coefficient_hidden),
        spatial_coefficient_log_scale=float(spatial_coefficient_log_scale),
    )


def korea_pine_model_config(
    base: ModelConfig | None = None,
    *,
    hidden: int | None = None,
    layers: int | None = None,
    fourier_features: int | None = None,
    fourier_sigma: float | None = None,
    front_fourier_features: int | None = 0,
    front_fourier_sigma: float | None = None,
    nif_rank: int | None = None,
    use_seed_front_features: bool = False,
    use_traveling_wave_features: bool = False,
    use_front_fourier_features: bool = False,
    use_intrinsic_front_phase: bool | None = None,
    intrinsic_front_phase_fourier_features: int | None = None,
    intrinsic_front_phase_fourier_sigma: float | None = None,
    intrinsic_front_phase_hidden: int | None = None,
    intrinsic_front_phase_layers: int | None = None,
    intrinsic_front_phase_feature_frequencies: int | None = None,
    intrinsic_front_phase_correction_scale: float | None = None,
    hard_initial_condition: bool = False,
    use_kpp_front_envelope: bool = False,
    use_spatial_coefficients: bool = True,
    spatial_coefficient_features: int | None = None,
    spatial_coefficient_sigma: float | None = None,
    spatial_coefficient_hidden: int | None = None,
    spatial_coefficient_log_scale: float | None = None,
) -> ModelConfig:
    """Korea pine-wilt specialization of the shared forward PINN backbone."""

    base = base or ModelConfig()
    return shared_geo_forward_model_config(
        hidden=base.hidden if hidden is None else hidden,
        layers=base.layers if layers is None else layers,
        fourier_features=base.fourier_features if fourier_features is None else fourier_features,
        fourier_sigma=base.fourier_sigma if fourier_sigma is None else fourier_sigma,
        front_fourier_features=(
            base.front_fourier_features if front_fourier_features is None else front_fourier_features
        ),
        front_fourier_sigma=base.front_fourier_sigma if front_fourier_sigma is None else front_fourier_sigma,
        nif_rank=base.nif_rank if nif_rank is None else nif_rank,
        learn_diffusion=True,
        learn_reaction=True,
        learn_drift=False,
        use_seed_front_features=use_seed_front_features,
        use_traveling_wave_features=use_traveling_wave_features,
        use_front_fourier_features=use_front_fourier_features,
        use_intrinsic_front_phase=(
            base.use_intrinsic_front_phase if use_intrinsic_front_phase is None else bool(use_intrinsic_front_phase)
        ),
        intrinsic_front_phase_fourier_features=(
            base.intrinsic_front_phase_fourier_features
            if intrinsic_front_phase_fourier_features is None
            else intrinsic_front_phase_fourier_features
        ),
        intrinsic_front_phase_fourier_sigma=(
            base.intrinsic_front_phase_fourier_sigma
            if intrinsic_front_phase_fourier_sigma is None
            else intrinsic_front_phase_fourier_sigma
        ),
        intrinsic_front_phase_hidden=(
            base.intrinsic_front_phase_hidden
            if intrinsic_front_phase_hidden is None
            else intrinsic_front_phase_hidden
        ),
        intrinsic_front_phase_layers=(
            base.intrinsic_front_phase_layers
            if intrinsic_front_phase_layers is None
            else intrinsic_front_phase_layers
        ),
        intrinsic_front_phase_feature_frequencies=(
            base.intrinsic_front_phase_feature_frequencies
            if intrinsic_front_phase_feature_frequencies is None
            else intrinsic_front_phase_feature_frequencies
        ),
        intrinsic_front_phase_correction_scale=(
            base.intrinsic_front_phase_correction_scale
            if intrinsic_front_phase_correction_scale is None
            else intrinsic_front_phase_correction_scale
        ),
        hard_initial_condition=hard_initial_condition,
        use_kpp_front_envelope=use_kpp_front_envelope,
        use_spatial_coefficients=use_spatial_coefficients,
        spatial_coefficient_features=(
            base.spatial_coefficient_features
            if spatial_coefficient_features is None
            else spatial_coefficient_features
        ),
        spatial_coefficient_sigma=(
            base.spatial_coefficient_sigma if spatial_coefficient_sigma is None else spatial_coefficient_sigma
        ),
        spatial_coefficient_hidden=(
            base.spatial_coefficient_hidden
            if spatial_coefficient_hidden is None
            else spatial_coefficient_hidden
        ),
        spatial_coefficient_log_scale=(
            base.spatial_coefficient_log_scale
            if spatial_coefficient_log_scale is None
            else spatial_coefficient_log_scale
        ),
    )


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
    phase_residual_low: float = 1.0e-3
    phase_residual_high: float = 0.995
    phase_residual_temperature: float = 0.02
    phase_residual_clip: float = 25.0
    residual_cvar_fraction: float = 0.10
    intrinsic_phase_anchor_points: int = 512
    intrinsic_phase_anchor_level: float = 0.10
    intrinsic_phase_anchor_band: float = 0.025
    intrinsic_phase_anchor_sign_margin: float = 0.015
    intrinsic_phase_compatibility_points: int = 256
    intrinsic_phase_compatibility_low: float = 0.02
    intrinsic_phase_compatibility_high: float = 0.98
    intrinsic_phase_compatibility_temperature: float = 0.03
    intrinsic_phase_compatibility_min_grad: float = 1.0e-4
    pde_loss_warmup_fraction: float = 0.0
    front_loss_start_fraction: float = 0.0
    front_loss_warmup_fraction: float = 0.0
    time_interface_start_fraction: float = 0.0
    time_interface_warmup_fraction: float = 0.0
    adaptive_loss_balancing: bool = False
    gradient_norm_balancing: bool = False
    gradient_norm_balance_every: int = 25
    adaptive_loss_momentum: float = 0.9
    adaptive_loss_min: float = 0.25
    adaptive_loss_max: float = 4.0
    validation_every: int = 50
    observation_support_temperature: float = 0.025
    observation_support_false_positive_weight: float = 0.20
    observation_support_false_negative_weight: float = 0.80
    observation_support_focal_gamma: float = 1.0
    restore_best_validation: bool = True
    resume_from_checkpoint: bool = True
    training_checkpoint_every: int = 1
    checkpoint_min_epoch_fraction: float = 0.10
    checkpoint_validation_weight: float = 0.45
    checkpoint_teacher_weight: float = 0.20
    checkpoint_pde_weight: float = 0.20
    checkpoint_initial_condition_weight: float = 0.05
    checkpoint_front_weight: float = 0.07
    checkpoint_mass_weight: float = 0.03
    front_speed_points: int = 256
    front_speed_max_points: int = 128
    front_speed_min_grad: float = 1.0e-2
    use_front_curvature_correction: bool = False
    front_curvature_correction_weight: float = 0.05
    expected_front_points: int = 256
    expected_front_width: float = 0.08
    expected_front_speed_factor: float = 0.45
    expected_front_level: float = 0.1
    level_set_points: int = 256
    level_set_width: float = 0.025
    leading_edge_area_times: int = 5
    leading_edge_area_grid: int = 32
    leading_edge_area_temperature: float = 0.015
    leading_edge_distribution_times: int = 5
    leading_edge_distribution_grid: int = 32
    radial_symmetry_groups: int = 0
    radial_symmetry_angles: int = 8
    front_contrast_times: int = 5
    front_contrast_grid: int = 32
    front_profile_points: int = 256
    front_profile_width: float = 0.06
    transverse_invariance_points: int = 0
    front_gradient_expected_points: int = 128
    mass_balance_times: int = 4
    mass_balance_grid: int = 18
    time_interface_points: int = 256
    time_interface_width: float = 0.01
    discrete_rk4_times: int = 0
    discrete_rk4_grid: int = 0
    discrete_rk4_dt_fraction: float = 0.05
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
    benchmark: BenchmarkConfig = BenchmarkConfig()
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

    def flagship(self, epochs: int = 20_000) -> "ExperimentConfig":
        """Return a high-budget, paper-style training preset without RK4 labels."""

        epochs = int(epochs)
        return replace(
            self,
            train=replace(
                self.train,
                epochs=epochs,
                lr=min(self.train.lr, 1.0e-3),
                collocation_points=max(self.train.collocation_points, 8192),
                boundary_points=max(self.train.boundary_points, 2048),
                seed_points=max(self.train.seed_points, 4096),
                observation_batch=max(self.train.observation_batch, 4096),
                validation_every=max(250, epochs // 40),
                print_every=max(250, epochs // 40),
                rar_interval=100,
                rar_candidates=max(self.train.rar_candidates, 32768),
                rar_keep=max(self.train.rar_keep, 4096),
                residual_curriculum_epochs=max(self.train.residual_curriculum_epochs, epochs // 4),
                adaptive_loss_balancing=True,
                gradient_norm_balancing=True,
                gradient_norm_balance_every=25,
                level_set_points=max(self.train.level_set_points, 1024),
                leading_edge_area_times=max(self.train.leading_edge_area_times, 8),
                leading_edge_area_grid=max(self.train.leading_edge_area_grid, 48),
                front_contrast_times=max(self.train.front_contrast_times, 8),
                front_contrast_grid=max(self.train.front_contrast_grid, 48),
                front_profile_points=max(self.train.front_profile_points, 1024),
                transverse_invariance_points=(
                    max(self.train.transverse_invariance_points, 2048)
                    if self.weights.transverse_invariance > 0.0
                    else self.train.transverse_invariance_points
                ),
                mass_balance_times=max(self.train.mass_balance_times, 8),
                mass_balance_grid=max(self.train.mass_balance_grid, 32),
                time_marching=True,
                time_marching_epochs=max(self.train.time_marching_epochs, epochs // 2),
                time_slabs=max(self.train.time_slabs, 8),
                time_slab_overlap=max(self.train.time_slab_overlap, 0.08),
                time_slab_curriculum=True,
                time_window_focus_fraction=min(self.train.time_window_focus_fraction, 0.75),
                time_window_teacher=False,
                time_window_observations=True,
                discrete_rk4_times=max(self.train.discrete_rk4_times, 4),
                discrete_rk4_grid=max(self.train.discrete_rk4_grid, 32),
                rk4_teacher_pool=0,
                rk4_teacher_batch=0,
                rk4_pretrain_steps=0,
                adam_to_lbfgs=True,
                lbfgs_steps=max(self.train.lbfgs_steps, 500),
                restore_best_validation=True,
                resume_from_checkpoint=True,
                training_checkpoint_every=25,
                checkpoint_min_epoch_fraction=max(self.train.checkpoint_min_epoch_fraction, 0.15),
            ),
        )

    def quick(self) -> "ExperimentConfig":
        return ExperimentConfig(
            domain=DomainConfig(box=self.domain.box, t_end=self.domain.t_end, grid=51, truth_steps=160),
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
            benchmark=self.benchmark,
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
                use_planar_wave_features=self.model.use_planar_wave_features,
                planar_wave_direction_x=self.model.planar_wave_direction_x,
                planar_wave_direction_y=self.model.planar_wave_direction_y,
                use_az_hard_constraints=self.model.use_az_hard_constraints,
                az_x_left=self.model.az_x_left,
                az_x_right=self.model.az_x_right,
                az_wave_x0=self.model.az_wave_x0,
                use_front_fourier_features=self.model.use_front_fourier_features,
                use_intrinsic_front_phase=self.model.use_intrinsic_front_phase,
                intrinsic_front_phase_fourier_features=min(
                    self.model.intrinsic_front_phase_fourier_features,
                    6,
                ),
                intrinsic_front_phase_fourier_sigma=self.model.intrinsic_front_phase_fourier_sigma,
                intrinsic_front_phase_hidden=min(self.model.intrinsic_front_phase_hidden, 24),
                intrinsic_front_phase_layers=self.model.intrinsic_front_phase_layers,
                intrinsic_front_phase_feature_frequencies=min(
                    self.model.intrinsic_front_phase_feature_frequencies,
                    3,
                ),
                intrinsic_front_phase_correction_scale=self.model.intrinsic_front_phase_correction_scale,
                hard_initial_condition=self.model.hard_initial_condition,
                initial_envelope_tau=self.model.initial_envelope_tau,
                use_kpp_front_envelope=self.model.use_kpp_front_envelope,
                front_envelope_level=self.model.front_envelope_level,
                front_envelope_margin=self.model.front_envelope_margin,
                front_envelope_width=self.model.front_envelope_width,
                use_spatial_coefficients=self.model.use_spatial_coefficients,
                spatial_coefficient_features=min(self.model.spatial_coefficient_features, 8),
                spatial_coefficient_sigma=self.model.spatial_coefficient_sigma,
                spatial_coefficient_hidden=min(self.model.spatial_coefficient_hidden, 24),
                spatial_coefficient_log_scale=self.model.spatial_coefficient_log_scale,
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
                observation_support=self.weights.observation_support,
                observation_support_area=self.weights.observation_support_area,
                gradient=0.0,
                data_density_gain=self.weights.data_density_gain,
                phase_pde=self.weights.phase_pde,
                residual_cvar=self.weights.residual_cvar,
                intrinsic_phase_initial=self.weights.intrinsic_phase_initial,
                intrinsic_phase_gradient_alignment=self.weights.intrinsic_phase_gradient_alignment,
                intrinsic_phase_monotonicity=self.weights.intrinsic_phase_monotonicity,
                front_pde_alpha=self.weights.front_pde_alpha,
                front_pde_gradient=self.weights.front_pde_gradient,
                front_gradient=self.weights.front_gradient,
                front_speed=self.weights.front_speed,
                mass_balance=self.weights.mass_balance,
                mass_floor=self.weights.mass_floor,
                expected_front_pde=self.weights.expected_front_pde,
                leading_edge=self.weights.leading_edge,
                leading_edge_area=self.weights.leading_edge_area,
                leading_edge_distribution=self.weights.leading_edge_distribution,
                radial_symmetry=self.weights.radial_symmetry,
                front_support_tversky=self.weights.front_support_tversky,
                front_contrast=self.weights.front_contrast,
                front_profile=self.weights.front_profile,
                level_set_alignment=self.weights.level_set_alignment,
                transverse_invariance=self.weights.transverse_invariance,
                time_interface=self.weights.time_interface,
                discrete_rk4=self.weights.discrete_rk4,
                rk4_teacher=self.weights.rk4_teacher,
                physics_parameter_anchor=self.weights.physics_parameter_anchor,
                coefficient_field=self.weights.coefficient_field,
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
                phase_residual_low=self.train.phase_residual_low,
                phase_residual_high=self.train.phase_residual_high,
                phase_residual_temperature=self.train.phase_residual_temperature,
                phase_residual_clip=self.train.phase_residual_clip,
                residual_cvar_fraction=self.train.residual_cvar_fraction,
                intrinsic_phase_anchor_points=min(self.train.intrinsic_phase_anchor_points, 128),
                intrinsic_phase_anchor_level=self.train.intrinsic_phase_anchor_level,
                intrinsic_phase_anchor_band=self.train.intrinsic_phase_anchor_band,
                intrinsic_phase_anchor_sign_margin=self.train.intrinsic_phase_anchor_sign_margin,
                intrinsic_phase_compatibility_points=min(self.train.intrinsic_phase_compatibility_points, 128),
                intrinsic_phase_compatibility_low=self.train.intrinsic_phase_compatibility_low,
                intrinsic_phase_compatibility_high=self.train.intrinsic_phase_compatibility_high,
                intrinsic_phase_compatibility_temperature=self.train.intrinsic_phase_compatibility_temperature,
                intrinsic_phase_compatibility_min_grad=self.train.intrinsic_phase_compatibility_min_grad,
                pde_loss_warmup_fraction=self.train.pde_loss_warmup_fraction,
                front_loss_start_fraction=self.train.front_loss_start_fraction,
                front_loss_warmup_fraction=self.train.front_loss_warmup_fraction,
                time_interface_start_fraction=self.train.time_interface_start_fraction,
                time_interface_warmup_fraction=self.train.time_interface_warmup_fraction,
                adaptive_loss_balancing=self.train.adaptive_loss_balancing,
                gradient_norm_balancing=self.train.gradient_norm_balancing,
                gradient_norm_balance_every=max(1, self.train.gradient_norm_balance_every),
                adaptive_loss_momentum=self.train.adaptive_loss_momentum,
                adaptive_loss_min=self.train.adaptive_loss_min,
                adaptive_loss_max=self.train.adaptive_loss_max,
                validation_every=30,
                observation_support_temperature=self.train.observation_support_temperature,
                observation_support_false_positive_weight=self.train.observation_support_false_positive_weight,
                observation_support_false_negative_weight=self.train.observation_support_false_negative_weight,
                observation_support_focal_gamma=self.train.observation_support_focal_gamma,
                restore_best_validation=self.train.restore_best_validation,
                front_speed_points=min(self.train.front_speed_points, 128),
                front_speed_max_points=min(self.train.front_speed_max_points, 64),
                front_speed_min_grad=self.train.front_speed_min_grad,
                use_front_curvature_correction=self.train.use_front_curvature_correction,
                front_curvature_correction_weight=self.train.front_curvature_correction_weight,
                expected_front_points=min(self.train.expected_front_points, 128),
                expected_front_width=self.train.expected_front_width,
                expected_front_speed_factor=self.train.expected_front_speed_factor,
                expected_front_level=self.train.expected_front_level,
                level_set_points=min(self.train.level_set_points, 128),
                level_set_width=self.train.level_set_width,
                leading_edge_area_times=min(self.train.leading_edge_area_times, 4),
                leading_edge_area_grid=min(self.train.leading_edge_area_grid, 24),
                leading_edge_area_temperature=self.train.leading_edge_area_temperature,
                leading_edge_distribution_times=min(self.train.leading_edge_distribution_times, 4),
                leading_edge_distribution_grid=min(self.train.leading_edge_distribution_grid, 24),
                radial_symmetry_groups=min(self.train.radial_symmetry_groups, 64)
                if self.train.radial_symmetry_groups > 0
                else 0,
                radial_symmetry_angles=min(max(self.train.radial_symmetry_angles, 4), 8),
                front_contrast_times=min(self.train.front_contrast_times, 4),
                front_contrast_grid=min(self.train.front_contrast_grid, 24),
                front_profile_points=min(self.train.front_profile_points, 128),
                front_profile_width=self.train.front_profile_width,
                transverse_invariance_points=(
                    min(self.train.transverse_invariance_points, 128)
                    if self.train.transverse_invariance_points > 0
                    else 0
                ),
                front_gradient_expected_points=min(self.train.front_gradient_expected_points, 96),
                mass_balance_times=min(self.train.mass_balance_times, 4),
                mass_balance_grid=min(self.train.mass_balance_grid, 18),
                time_interface_points=min(self.train.time_interface_points, 128),
                time_interface_width=self.train.time_interface_width,
                discrete_rk4_times=min(self.train.discrete_rk4_times, 2)
                if self.train.discrete_rk4_times > 0
                else 0,
                discrete_rk4_grid=min(self.train.discrete_rk4_grid, 16)
                if self.train.discrete_rk4_grid > 0
                else 0,
                discrete_rk4_dt_fraction=self.train.discrete_rk4_dt_fraction,
                rk4_teacher_pool=min(self.train.rk4_teacher_pool, 2048) if self.train.rk4_teacher_pool > 0 else 0,
                rk4_teacher_batch=min(self.train.rk4_teacher_batch, 256) if self.train.rk4_teacher_batch > 0 else 0,
                rk4_teacher_late_fraction=self.train.rk4_teacher_late_fraction,
                rk4_pretrain_steps=min(self.train.rk4_pretrain_steps, 20)
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

        The Korea specialization keeps the same phase-capable geo-spectral
        backbone as the synthetic flagship model, while disabling synthetic
        travelling-wave and seed-front priors. Its phase loss is intentionally
        weak because the observed Korea front is a data-derived support contour,
        not an exact analytic level set.
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
            benchmark=self.benchmark,
            model=korea_pine_model_config(
                self.model,
                use_intrinsic_front_phase=True,
                intrinsic_front_phase_fourier_features=8,
                intrinsic_front_phase_fourier_sigma=1.0,
                intrinsic_front_phase_hidden=32,
                intrinsic_front_phase_layers=2,
                intrinsic_front_phase_feature_frequencies=4,
                intrinsic_front_phase_correction_scale=1.0,
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
                phase_pde=0.03,
                residual_cvar=0.05,
                intrinsic_phase_initial=0.03,
                intrinsic_phase_gradient_alignment=0.01,
                intrinsic_phase_monotonicity=0.01,
                front_pde_alpha=0.0,
                front_pde_gradient=0.0,
                front_gradient=0.0,
                front_speed=0.0,
                mass_balance=0.0,
                expected_front_pde=0.0,
                leading_edge=0.0,
                leading_edge_area=0.0,
                leading_edge_distribution=0.0,
                radial_symmetry=0.0,
                front_contrast=0.0,
                front_profile=0.0,
                level_set_alignment=0.0,
                transverse_invariance=0.0,
                time_interface=0.0,
                discrete_rk4=0.0,
                rk4_teacher=0.0,
                physics_parameter_anchor=self.weights.physics_parameter_anchor,
                coefficient_field=self.weights.coefficient_field,
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
            benchmark=base.benchmark,
            model=shared_geo_forward_model_config(
                hidden=base.model.hidden,
                layers=base.model.layers,
                fourier_features=base.model.fourier_features,
                fourier_sigma=1.0,
                front_fourier_features=16,
                front_fourier_sigma=1.5,
                nif_rank=24,
                learn_diffusion=True,
                learn_reaction=True,
                learn_drift=False,
                use_seed_front_features=True,
                use_traveling_wave_features=True,
                use_planar_wave_features=False,
                use_front_fourier_features=True,
                use_intrinsic_front_phase=True,
                intrinsic_front_phase_fourier_features=8,
                intrinsic_front_phase_fourier_sigma=1.0,
                intrinsic_front_phase_hidden=32,
                intrinsic_front_phase_layers=2,
                intrinsic_front_phase_feature_frequencies=4,
                intrinsic_front_phase_correction_scale=0.25,
                hard_initial_condition=True,
                initial_envelope_tau=0.18,
                use_kpp_front_envelope=True,
                front_envelope_level=0.03,
                front_envelope_margin=0.05,
                front_envelope_width=0.025,
                use_spatial_coefficients=True,
                spatial_coefficient_features=8,
                spatial_coefficient_sigma=0.85,
                spatial_coefficient_hidden=32,
                spatial_coefficient_log_scale=0.12,
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
                observation_support=0.0,
                observation_support_area=0.15,
                gradient=0.0,
                data_density_gain=4.0,
                phase_pde=0.18,
                residual_cvar=0.12,
                intrinsic_phase_initial=0.20,
                intrinsic_phase_gradient_alignment=0.02,
                intrinsic_phase_monotonicity=0.01,
                front_pde_alpha=2.0,
                front_pde_gradient=0.5,
                front_gradient=0.005,
                front_speed=0.01,
                mass_balance=0.35,
                mass_floor=0.25,
                expected_front_pde=0.0,
                leading_edge=0.25,
                leading_edge_area=0.90,
                leading_edge_distribution=0.0,
                radial_symmetry=0.0,
                front_support_tversky=0.45,
                front_contrast=0.12,
                front_profile=0.25,
                level_set_alignment=0.05,
                transverse_invariance=0.0,
                time_interface=0.01,
                discrete_rk4=0.08,
                rk4_teacher=0.0,
                physics_parameter_anchor=0.20,
                coefficient_field=0.05,
                sparse=1.0e-5,
            ),
            warm_start=base.warm_start,
            train=replace(
                base.train,
                residual_weight_exponent_start=0.0,
                residual_weight_exponent_end=0.5,
                residual_curriculum_epochs=max(1, base.train.epochs // 4),
                phase_residual_low=7.5e-4,
                phase_residual_high=0.997,
                phase_residual_temperature=0.018,
                phase_residual_clip=20.0,
                residual_cvar_fraction=0.12,
                intrinsic_phase_anchor_points=512,
                intrinsic_phase_anchor_level=0.10,
                intrinsic_phase_anchor_band=0.025,
                intrinsic_phase_anchor_sign_margin=0.015,
                intrinsic_phase_compatibility_points=256,
                intrinsic_phase_compatibility_low=0.02,
                intrinsic_phase_compatibility_high=0.98,
                intrinsic_phase_compatibility_temperature=0.03,
                intrinsic_phase_compatibility_min_grad=1.0e-4,
                pde_loss_warmup_fraction=0.05,
                front_loss_start_fraction=0.05,
                front_loss_warmup_fraction=0.20,
                time_interface_start_fraction=0.25,
                time_interface_warmup_fraction=0.20,
                adaptive_loss_balancing=True,
                gradient_norm_balancing=True,
                gradient_norm_balance_every=25,
                adaptive_loss_momentum=0.9,
                adaptive_loss_min=0.33,
                adaptive_loss_max=3.0,
                front_speed_min_grad=1.0e-2,
                use_front_curvature_correction=True,
                front_curvature_correction_weight=0.05,
                expected_front_points=256,
                expected_front_width=0.08,
                expected_front_speed_factor=0.45,
                expected_front_level=0.1,
                leading_edge_area_times=5,
                leading_edge_area_grid=32,
                leading_edge_area_temperature=0.015,
                leading_edge_distribution_times=5,
                leading_edge_distribution_grid=32,
                radial_symmetry_groups=128,
                radial_symmetry_angles=8,
                front_contrast_times=5,
                front_contrast_grid=32,
                front_profile_points=256,
                front_profile_width=0.06,
                transverse_invariance_points=0,
                front_gradient_expected_points=128,
                time_marching=True,
                time_marching_start_fraction=0.30,
                time_marching_epochs=max(1, base.train.epochs // 2),
                time_slabs=4,
                time_slab_overlap=0.08,
                time_slab_curriculum=True,
                time_window_focus_fraction=0.65,
                time_window_teacher=False,
                time_window_observations=True,
                observation_batch=512,
                time_interface_points=256,
                time_interface_width=0.015,
                discrete_rk4_times=3,
                discrete_rk4_grid=20,
                discrete_rk4_dt_fraction=0.05,
                rk4_teacher_pool=0,
                rk4_teacher_batch=0,
                rk4_teacher_late_fraction=0.50,
                rk4_pretrain_steps=0,
                rk4_pretrain_batch=0,
                rk4_pretrain_lr=1.0e-3,
            ),
            ensemble=base.ensemble,
            base_seed=base.base_seed,
            out_dir=base.out_dir,
            run_classical_baseline=base.run_classical_baseline,
            baseline_epochs=base.baseline_epochs,
        )

    def ablowitz_zeppetella_forward(self) -> "ExperimentConfig":
        """Exact-wave Fisher-KPP benchmark matching the attached parameters.

        The physical equation is u_t = u_xx + u(1-u) on x in [-20, 20] with
        c = 5/sqrt(6). The PINN domain remains [0, 1]^2, so the diffusion is
        normalized as D/L^2 while exact initial and boundary targets are
        evaluated in the physical x-coordinate.
        """

        truth_steps = int(round(AZ_T_1D / AZ_DT_1D))
        base = self.geo_spectral_forward()
        return ExperimentConfig(
            domain=DomainConfig(box=1.0, t_end=AZ_T_1D, grid=AZ_NX_1D, truth_steps=truth_steps),
            pde=PDEConfig(
                diffusion=az_normalized_diffusion(AZ_X_LEFT_1D, AZ_X_RIGHT_1D),
                reaction=1.0,
                velocity_x=0.0,
                velocity_y=0.0,
                include_advection=False,
            ),
            seed=SeedConfig(center_x=0.5, center_y=0.5, sigma=0.08, amplitude=0.5),
            observations=ObservationConfig(
                start_time=0.0,
                frames=7,
                samples_per_frame=800,
                noise_std=0.0,
                focus_fraction=0.25,
                validation_fraction=0.2,
            ),
            geo=GeoConfig(enabled=True, mask_kind="box"),
            benchmark=BenchmarkConfig(
                kind="ablowitz_zeppetella",
                x_left=AZ_X_LEFT_1D,
                x_right=AZ_X_RIGHT_1D,
                wave_x0=0.0,
            ),
            model=replace(
                base.model,
                hard_initial_condition=False,
                use_kpp_front_envelope=False,
                use_seed_front_features=False,
                use_traveling_wave_features=False,
                use_planar_wave_features=True,
                planar_wave_direction_x=1.0,
                planar_wave_direction_y=0.0,
                use_az_hard_constraints=True,
                az_x_left=AZ_X_LEFT_1D,
                az_x_right=AZ_X_RIGHT_1D,
                az_wave_x0=0.0,
                use_front_fourier_features=False,
                use_spatial_coefficients=False,
                learn_diffusion=False,
                learn_reaction=False,
                fourier_sigma=1.2,
            ),
            weights=replace(
                base.weights,
                data=2.0,
                pde=1.0,
                phase_pde=0.25,
                residual_cvar=0.10,
                intrinsic_phase_initial=0.40,
                intrinsic_phase_gradient_alignment=0.02,
                intrinsic_phase_monotonicity=0.01,
                initial_condition=8.0,
                boundary=4.0,
                front_speed=0.0,
                mass_balance=0.0,
                mass_floor=0.0,
                leading_edge=0.0,
                leading_edge_area=0.0,
                leading_edge_distribution=0.0,
                radial_symmetry=0.0,
                front_support_tversky=0.0,
                front_contrast=0.0,
                front_profile=0.0,
                level_set_alignment=1.0,
                transverse_invariance=3.0,
                time_interface=0.0,
                discrete_rk4=0.10,
                physics_parameter_anchor=0.0,
                coefficient_field=0.0,
            ),
            warm_start=WarmStartConfig(mode="neutral"),
            train=replace(
                base.train,
                epochs=1200,
                collocation_points=2048,
                boundary_points=512,
                seed_points=1024,
                observation_batch=2048,
                time_marching=True,
                time_slabs=4,
                time_slab_curriculum=True,
                time_window_focus_fraction=0.75,
                time_window_observations=True,
                pde_loss_warmup_fraction=0.05,
                front_loss_start_fraction=0.0,
                front_loss_warmup_fraction=0.0,
                discrete_rk4_times=3,
                discrete_rk4_grid=24,
                discrete_rk4_dt_fraction=0.025,
                transverse_invariance_points=512,
                rk4_teacher_pool=4096,
                rk4_teacher_batch=512,
                rk4_pretrain_steps=0,
                mass_balance_times=0,
            ),
            ensemble=self.ensemble,
            base_seed=self.base_seed,
            out_dir=Path("runs") / "ablowitz_zeppetella_pinn",
            run_classical_baseline=False,
            baseline_epochs=self.baseline_epochs,
        )
