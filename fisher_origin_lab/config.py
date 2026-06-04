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
    fourier_features: int = 32
    fourier_sigma: float = 3.5
    hidden: int = 96
    layers: int = 5
    learn_diffusion: bool = False
    learn_reaction: bool = False
    learn_drift: bool = False
    use_source_envelope: bool = True
    use_geo_features: bool = False
    spatial_fourier_only: bool = False


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
    shooting_grid: int = 33
    shooting_steps: int = 80
    shooting_points: int = 512
    shooting_prefit_steps: int = 40
    residual_weight_exponent_start: float = 0.5
    residual_weight_exponent_end: float = 0.5
    residual_curriculum_epochs: int = 0
    adaptive_loss_balancing: bool = False
    adaptive_loss_momentum: float = 0.9
    adaptive_loss_min: float = 0.25
    adaptive_loss_max: float = 4.0
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
                fourier_features=16,
                fourier_sigma=self.model.fourier_sigma,
                hidden=48,
                layers=3,
                learn_diffusion=self.model.learn_diffusion,
                learn_reaction=self.model.learn_reaction,
                learn_drift=self.model.learn_drift,
                use_source_envelope=self.model.use_source_envelope,
                use_geo_features=self.model.use_geo_features,
                spatial_fourier_only=self.model.spatial_fourier_only,
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
                adaptive_loss_momentum=self.train.adaptive_loss_momentum,
                adaptive_loss_min=self.train.adaptive_loss_min,
                adaptive_loss_max=self.train.adaptive_loss_max,
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
                fourier_features=self.model.fourier_features,
                fourier_sigma=self.model.fourier_sigma,
                hidden=self.model.hidden,
                layers=self.model.layers,
                learn_diffusion=True,
                learn_reaction=True,
                learn_drift=False,
                use_source_envelope=False,
                use_geo_features=self.model.use_geo_features,
                spatial_fourier_only=self.model.spatial_fourier_only,
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
                fourier_features=base.model.fourier_features,
                fourier_sigma=base.model.fourier_sigma,
                hidden=base.model.hidden,
                layers=base.model.layers,
                learn_diffusion=True,
                learn_reaction=True,
                learn_drift=False,
                use_source_envelope=False,
                use_geo_features=True,
                spatial_fourier_only=True,
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
                front_gradient=0.02,
                sparse=1.0e-5,
            ),
            warm_start=base.warm_start,
            train=replace(
                base.train,
                residual_weight_exponent_start=0.0,
                residual_weight_exponent_end=0.5,
                residual_curriculum_epochs=max(1, base.train.epochs // 4),
                adaptive_loss_balancing=True,
                adaptive_loss_momentum=0.9,
                adaptive_loss_min=0.33,
                adaptive_loss_max=3.0,
            ),
            ensemble=base.ensemble,
            base_seed=base.base_seed,
            out_dir=base.out_dir,
            run_classical_baseline=base.run_classical_baseline,
            baseline_epochs=base.baseline_epochs,
        )
