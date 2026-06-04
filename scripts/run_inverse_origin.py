from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_origin_lab.config import (
    DomainConfig,
    ExperimentConfig,
    LossWeights,
    ModelConfig,
    ObservationConfig,
    PDEConfig,
    SeedConfig,
    TrainConfig,
    WarmStartConfig,
)
from fisher_origin_lab.train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Fisher-KPP inverse origin inference.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/default"))
    parser.add_argument("--quick", action="store_true", help="Use a small CPU-friendly configuration.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--grid", type=int, default=101)
    parser.add_argument("--truth-steps", type=int, default=500)
    parser.add_argument("--obs-samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--focus-fraction", type=float, default=0.5)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--learn-drift", action="store_true")
    parser.add_argument("--learn-diffusion", action="store_true")
    parser.add_argument("--learn-reaction", action="store_true")
    parser.add_argument("--run-classical-baseline", action="store_true")
    parser.add_argument("--baseline-epochs", type=int, default=250)
    parser.add_argument("--gradient-weight", type=float, default=0.01)
    parser.add_argument("--shooting-weight", type=float, default=5.0)
    parser.add_argument(
        "--data-density-gain",
        type=float,
        default=None,
        help="Optional multiplier for density-weighted data MSE, e.g. 4.0 in the pine-wilt notebook.",
    )
    parser.add_argument(
        "--korea-pine-style",
        action="store_true",
        help="Use the Korea pine-wilt notebook's forward Fisher-KPP setup: no advection, no PINN BC loss.",
    )
    parser.add_argument(
        "--geo-spectral-forward",
        action="store_true",
        help="Use the Geo-Spectral Causal Adaptive gPINN forward profile on the same Fisher-KPP problem.",
    )
    parser.add_argument(
        "--warm-start",
        choices=["drift_corrected", "centroid", "neutral", "shooting_prefit"],
        default="drift_corrected",
        help="How to initialize the trainable source center from observations.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        domain=DomainConfig(grid=args.grid, truth_steps=args.truth_steps),
        pde=PDEConfig(),
        seed=SeedConfig(),
        observations=ObservationConfig(
            samples_per_frame=args.obs_samples,
            noise_std=args.noise,
            focus_fraction=args.focus_fraction,
            validation_fraction=args.validation_fraction,
        ),
        model=ModelConfig(
            learn_drift=args.learn_drift,
            learn_diffusion=args.learn_diffusion,
            learn_reaction=args.learn_reaction,
        ),
        weights=LossWeights(
            gradient=args.gradient_weight,
            shooting=args.shooting_weight,
            data_density_gain=args.data_density_gain if args.data_density_gain is not None else 0.0,
        ),
        warm_start=WarmStartConfig(mode=args.warm_start),
        train=TrainConfig(epochs=args.epochs if args.epochs is not None else TrainConfig().epochs),
        ensemble=args.ensemble,
        base_seed=args.seed,
        out_dir=args.out_dir,
        run_classical_baseline=args.run_classical_baseline,
        baseline_epochs=args.baseline_epochs,
    )
    if args.korea_pine_style:
        cfg = cfg.korea_pine_style()
    if args.geo_spectral_forward:
        cfg = cfg.geo_spectral_forward()
    if args.data_density_gain is not None:
        cfg = replace(cfg, weights=replace(cfg.weights, data_density_gain=args.data_density_gain))
    if args.quick:
        cfg = cfg.quick()
        if args.epochs is not None:
            cfg = ExperimentConfig(
                domain=cfg.domain,
                pde=cfg.pde,
                seed=cfg.seed,
                observations=cfg.observations,
                geo=cfg.geo,
                model=cfg.model,
                weights=cfg.weights,
                warm_start=cfg.warm_start,
                train=TrainConfig(
                    epochs=args.epochs,
                    lr=cfg.train.lr,
                    source_lr_multiplier=cfg.train.source_lr_multiplier,
                    collocation_points=cfg.train.collocation_points,
                    boundary_points=cfg.train.boundary_points,
                    seed_points=cfg.train.seed_points,
                    time_bins=cfg.train.time_bins,
                    causal_eps=cfg.train.causal_eps,
                    decay_beta=cfg.train.decay_beta,
                    rar_interval=cfg.train.rar_interval,
                    rar_candidates=cfg.train.rar_candidates,
                    rar_keep=cfg.train.rar_keep,
                    shooting_grid=cfg.train.shooting_grid,
                    shooting_steps=cfg.train.shooting_steps,
                    shooting_points=cfg.train.shooting_points,
                    shooting_prefit_steps=cfg.train.shooting_prefit_steps,
                    print_every=cfg.train.print_every,
                    adam_to_lbfgs=cfg.train.adam_to_lbfgs,
                    lbfgs_steps=cfg.train.lbfgs_steps,
                ),
                ensemble=cfg.ensemble,
                base_seed=cfg.base_seed,
                out_dir=cfg.out_dir,
                run_classical_baseline=cfg.run_classical_baseline,
                baseline_epochs=cfg.baseline_epochs,
            )
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
