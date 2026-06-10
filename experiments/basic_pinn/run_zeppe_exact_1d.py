from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_origin_lab.config import ExperimentConfig
from fisher_origin_lab.train import run_experiment


def _apply_preset(cfg: ExperimentConfig, preset: str, epochs: int | None) -> ExperimentConfig:
    if preset in {"smoke", "quick"}:
        cfg = cfg.quick()
    if preset == "smoke":
        cfg = replace(
            cfg,
            train=replace(
                cfg.train,
                epochs=epochs if epochs is not None else 8,
                collocation_points=min(cfg.train.collocation_points, 160),
                boundary_points=min(cfg.train.boundary_points, 64),
                seed_points=min(cfg.train.seed_points, 96),
                observation_batch=min(cfg.train.observation_batch, 160),
                rk4_teacher_pool=0,
                rk4_teacher_batch=0,
                rk4_pretrain_steps=0,
                print_every=4,
            ),
        )
    elif epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=epochs))
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the basic PINN Ablowitz-Zeppetella exact-wave benchmark. "
            "The 1D physical wave is represented as a y-invariant field in the shared 2D PINN backbone."
        )
    )
    parser.add_argument("--preset", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs") / "basic_pinn" / "zeppe_exact_1d")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        ensemble=args.ensemble,
        base_seed=args.seed,
        out_dir=args.out_dir,
    ).ablowitz_zeppetella_forward()
    cfg = replace(cfg, out_dir=args.out_dir, ensemble=args.ensemble, base_seed=args.seed)
    cfg = _apply_preset(cfg, args.preset, args.epochs)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
