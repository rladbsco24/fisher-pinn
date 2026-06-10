from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_origin_lab.config import ExperimentConfig, ObservationConfig  # noqa: E402
from fisher_origin_lab.train import run_experiment  # noqa: E402
from fisher_origin_lab.utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run forward Fisher-KPP method ablations scored by field, front, mass, and RK4 metrics."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/forward_ablations"))
    parser.add_argument("--preset", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seeds", default="7", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per run.")
    parser.add_argument("--no-resume", action="store_true", help="Disable training resume checkpoints.")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Save training state every N epochs.")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def _seed_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _preset_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        observations=ObservationConfig(samples_per_frame=500, noise_std=0.02, focus_fraction=0.5),
        ensemble=1,
        run_classical_baseline=False,
        baseline_epochs=60,
    ).geo_spectral_forward()
    if args.preset in {"smoke", "quick"}:
        cfg = cfg.quick()
    if args.preset == "smoke":
        cfg = replace(
            cfg,
            train=replace(cfg.train, epochs=12, print_every=6, validation_every=6, rar_interval=0),
            observations=replace(cfg.observations, samples_per_frame=80),
            baseline_epochs=2,
        )
    elif args.preset == "quick":
        cfg = replace(cfg, train=replace(cfg.train, epochs=120, print_every=30, validation_every=30))
    else:
        cfg = replace(cfg, train=replace(cfg.train, epochs=1200, print_every=100, validation_every=100))
    if args.epochs is not None:
        cfg = replace(
            cfg,
            train=replace(cfg.train, epochs=args.epochs, print_every=max(1, args.epochs // 4)),
        )
    cfg = replace(
        cfg,
        train=replace(
            cfg.train,
            resume_from_checkpoint=not args.no_resume,
            training_checkpoint_every=max(1, int(args.checkpoint_every)),
        ),
    )
    return cfg


def _case(name: str, cfg: ExperimentConfig, note: str) -> dict[str, Any]:
    return {"name": name, "cfg": cfg, "note": note}


def make_forward_cases(base: ExperimentConfig) -> list[dict[str, Any]]:
    korea = replace(
        base.korea_pine_style(),
        domain=base.domain,
        observations=base.observations,
        train=base.train,
        out_dir=base.out_dir,
        baseline_epochs=base.baseline_epochs,
    )
    geo_no_front = replace(
        base,
        weights=replace(
            base.weights,
            front_speed=0.0,
            mass_balance=0.0,
            leading_edge_area=0.0,
            front_contrast=0.0,
            front_profile=0.0,
            expected_front_pde=0.0,
            leading_edge=0.0,
            front_gradient=0.0,
            level_set_alignment=0.0,
            time_interface=0.0,
        ),
        train=replace(
            base.train,
            time_marching=False,
            time_slabs=1,
            time_slab_curriculum=False,
            time_window_focus_fraction=1.0,
            time_window_teacher=False,
            time_window_observations=False,
        ),
    )
    geo_speed_mass = replace(
        base,
        weights=replace(
            base.weights,
            leading_edge_area=0.0,
            front_contrast=0.0,
            front_profile=0.0,
            expected_front_pde=0.0,
            leading_edge=0.0,
            level_set_alignment=0.0,
            time_interface=0.0,
        ),
    )
    geo_front_area = base
    geo_area_strong = replace(base, weights=replace(base.weights, leading_edge_area=3.0))
    geo_no_mass_floor = replace(
        base,
        weights=replace(base.weights, mass_floor=0.0),
    )
    geo_no_support_tversky = replace(
        base,
        weights=replace(base.weights, front_support_tversky=0.0),
    )
    geo_no_physics_anchor = replace(
        base,
        weights=replace(base.weights, physics_parameter_anchor=0.0),
    )
    geo_no_spatial_coefficients = replace(
        base,
        model=replace(base.model, use_spatial_coefficients=False),
        weights=replace(base.weights, coefficient_field=0.0),
    )
    geo_no_collapse_guards = replace(
        base,
        weights=replace(
            base.weights,
            mass_floor=0.0,
            front_support_tversky=0.0,
            physics_parameter_anchor=0.0,
            leading_edge=0.0,
            leading_edge_area=0.0,
        ),
    )
    geo_levelset_time_slab = replace(
        base,
        weights=replace(
            base.weights,
            level_set_alignment=max(base.weights.level_set_alignment, 0.06),
            time_interface=max(base.weights.time_interface, 0.03),
            rk4_teacher=max(base.weights.rk4_teacher, 0.005),
        ),
        train=replace(
            base.train,
            level_set_points=max(base.train.level_set_points, 256),
            level_set_width=0.02,
            time_marching=True,
            time_marching_start_fraction=0.30,
            time_marching_epochs=max(base.train.time_marching_epochs, base.train.epochs // 2),
            time_slabs=4,
            time_slab_overlap=0.08,
            time_slab_curriculum=True,
            time_window_focus_fraction=0.5,
            time_window_teacher=False,
            rk4_teacher_pool=max(base.train.rk4_teacher_pool, 4096),
            rk4_teacher_batch=max(base.train.rk4_teacher_batch, 512),
        ),
    )
    geo_rk4_teacher_front_area = replace(
        base,
        weights=replace(base.weights, rk4_teacher=max(base.weights.rk4_teacher, 0.01)),
        train=replace(
            base.train,
            rk4_teacher_pool=max(base.train.rk4_teacher_pool, 4096),
            rk4_teacher_batch=max(base.train.rk4_teacher_batch, 512),
        ),
    )
    geo_rk4_late_teacher_front_area = replace(
        base,
        weights=replace(base.weights, rk4_teacher=0.75),
        train=replace(
            base.train,
            rk4_teacher_pool=max(base.train.rk4_teacher_pool, 4096),
            rk4_teacher_batch=max(base.train.rk4_teacher_batch, 512),
            rk4_teacher_late_fraction=0.65,
        ),
    )
    geo_rk4_pretrain_front_area = replace(
        base,
        weights=replace(base.weights, rk4_teacher=0.25),
        train=replace(
            base.train,
            rk4_teacher_pool=max(base.train.rk4_teacher_pool, 4096),
            rk4_teacher_batch=max(base.train.rk4_teacher_batch, 512),
            rk4_pretrain_steps=max(base.train.rk4_pretrain_steps, 80),
            rk4_pretrain_batch=max(base.train.rk4_pretrain_batch, 512),
        ),
    )
    geo_no_tw_front_area = replace(
        base,
        model=replace(base.model, use_traveling_wave_features=False),
    )
    geo_nif_front_area = replace(
        base,
        model=replace(
            base.model,
            architecture="nif_pirate",
            use_random_weight_factorization=True,
            use_traveling_wave_features=False,
        ),
    )
    geo_gated_front_area = replace(
        base,
        model=replace(base.model, architecture="gated_mlp", use_random_weight_factorization=False),
    )

    return [
        _case("korea_style_forward", korea, "Korea pine-wilt compatible forward PINN objective."),
        _case("geo_no_front_terms", geo_no_front, "Geo/spectral/hard-IC model without explicit front or mass losses."),
        _case("geo_speed_mass", geo_speed_mass, "Adds gradient-filtered front-speed and parabolic mass-balance losses."),
        _case("geo_front_area", geo_front_area, "Default full front-aware stack: level-set, gPINN, causal slabs, front Fourier, and balancing."),
        _case("geo_front_area_strong", geo_area_strong, "Stronger front-area ablation; useful for checking metric tradeoffs."),
        _case("geo_no_mass_floor", geo_no_mass_floor, "Ablates the minimum mass trajectory guard against near-zero collapse."),
        _case("geo_no_support_tversky", geo_no_support_tversky, "Ablates threshold-support Tversky loss for false-negative front support."),
        _case("geo_no_physics_anchor", geo_no_physics_anchor, "Ablates the weak D/r parameter anchor used during coefficient learning."),
        _case("geo_no_spatial_coefficients", geo_no_spatial_coefficients, "Ablates the smooth spatial D(x,y), r(x,y) coefficient correction field."),
        _case("geo_no_collapse_guards", geo_no_collapse_guards, "Removes mass floor, support Tversky, parameter anchor, and leading-edge guards together."),
        _case("geo_levelset_time_slab", geo_levelset_time_slab, "Front level-set alignment plus causal time-slab windowing."),
        _case("geo_rk4_teacher_front_area", geo_rk4_teacher_front_area, "Weak RK4 pseudo-label regularizer for solver-assisted PINN training."),
        _case("geo_rk4_late_teacher_front_area", geo_rk4_late_teacher_front_area, "RK4 teacher profile biased toward late moving-front snapshots."),
        _case("geo_rk4_pretrain_front_area", geo_rk4_pretrain_front_area, "RK4 teacher pretraining followed by PINN fine-tuning."),
        _case("geo_no_tw_front_area", geo_no_tw_front_area, "Ablates the scaled traveling-wave moving-frame features."),
        _case("geo_nif_front_area", geo_nif_front_area, "NIF-style last-layer parameterized ShapeNet/ParameterNet head."),
        _case("geo_gated_front_area", geo_gated_front_area, "Ablates PirateNet/RWF by using the previous gated MLP backbone."),
    ]


def _row(case: dict[str, Any], seed: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": case["name"],
        "seed": seed,
        "validation_observation_mse": metrics.get("validation_observation_mse"),
        "final_time_relative_l2": metrics.get("final_time_relative_l2"),
        "rk4_final_time_relative_l2": metrics.get("rk4_final_time_relative_l2"),
        "pinn_vs_rk4_final_relative_l2": metrics.get("pinn_vs_rk4_final_relative_l2"),
        "front_area_005_mae": metrics.get("front_area_005_mae"),
        "front_area_010_mae": metrics.get("front_area_010_mae"),
        "active_front_area_mae": metrics.get("active_front_area_mae"),
        "mass_mae": metrics.get("mass_mae"),
        "note": case["note"],
    }


def _finite(values: list[Any]) -> list[float]:
    out = []
    for value in values:
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            out.append(value)
    return out


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = []
    for case in sorted({row["case"] for row in rows}):
        case_rows = [row for row in rows if row["case"] == case]
        item: dict[str, Any] = {
            "case": case,
            "n": len(case_rows),
            "note": case_rows[0]["note"],
        }
        for key in [
            "validation_observation_mse",
            "final_time_relative_l2",
            "front_area_005_mae",
            "front_area_010_mae",
            "active_front_area_mae",
            "mass_mae",
        ]:
            vals = _finite([row.get(key) for row in case_rows])
            item[f"{key}_mean"] = mean(vals) if vals else None
            item[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
        summary.append(item)
    return {"cases": summary}


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, summary: dict[str, Any]) -> None:
    cases = summary["cases"]
    labels = [case["case"] for case in cases]
    metrics = [
        ("final_time_relative_l2_mean", "final relative L2"),
        ("front_area_010_mae_mean", "front area MAE, u>0.10"),
        ("mass_mae_mean", "mass MAE"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(max(11.0, 1.8 * len(labels)), 4.2), constrained_layout=True)
    for ax, (key, title) in zip(axes, metrics):
        vals = [case[key] or 0.0 for case in cases]
        ax.bar(range(len(labels)), vals, color="#4C78A8")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seeds = _seed_list(args.seeds)
    base = _preset_config(args)
    cases = make_forward_cases(base)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "manifest.json",
        {
            "preset": args.preset,
            "seeds": seeds,
            "cases": [{"name": case["name"], "note": case["note"]} for case in cases],
        },
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        for seed in seeds:
            cfg = replace(case["cfg"], base_seed=seed, out_dir=args.out_dir / case["name"] / f"seed_{seed}")
            print(f"\n=== forward case={case['name']} seed={seed} ===")
            rows.append(_row(case, seed, run_experiment(cfg)))

    write_rows_csv(args.out_dir / "results.csv", rows)
    summary = aggregate(rows)
    write_json(args.out_dir / "summary.json", summary)
    plot_summary(args.out_dir / "summary.png", summary)
    print(f"\nwrote {args.out_dir / 'results.csv'}")
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
