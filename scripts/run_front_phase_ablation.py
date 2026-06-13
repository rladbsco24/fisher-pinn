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

from fisher_origin_lab.config import ExperimentConfig, GeoConfig, ModelConfig, ObservationConfig  # noqa: E402
from fisher_origin_lab.train import run_experiment  # noqa: E402
from fisher_origin_lab.utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vanilla PINN, geo-spectral PINN, and explicit front-phase PINN "
            "on 1D AZ and 2D Gaussian Fisher-KPP moving-front benchmarks."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/front_phase_ablations"))
    parser.add_argument("--problem", choices=["both", "az1d", "gaussian2d"], default="both")
    parser.add_argument("--preset", choices=["smoke", "quick", "full", "flagship", "flagship_safe"], default="smoke")
    parser.add_argument("--seeds", default="7")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def _seed_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _with_preset(cfg: ExperimentConfig, preset: str, epochs: int | None) -> ExperimentConfig:
    if preset in {"flagship", "flagship_safe"}:
        cfg = cfg.flagship(20_000 if epochs is None else int(epochs))
        if preset == "flagship_safe":
            cfg = replace(
                cfg,
                train=replace(
                    cfg.train,
                    collocation_points=min(cfg.train.collocation_points, 4096),
                    boundary_points=min(cfg.train.boundary_points, 1024),
                    seed_points=min(cfg.train.seed_points, 2048),
                    observation_batch=min(cfg.train.observation_batch, 2048)
                    if cfg.train.observation_batch > 0
                    else 0,
                    rar_candidates=min(cfg.train.rar_candidates, 16384),
                    rar_keep=min(cfg.train.rar_keep, 2048),
                    level_set_points=min(cfg.train.level_set_points, 512),
                    leading_edge_area_grid=min(cfg.train.leading_edge_area_grid, 40),
                    front_contrast_grid=min(cfg.train.front_contrast_grid, 40),
                    front_profile_points=min(cfg.train.front_profile_points, 512),
                    transverse_invariance_points=min(cfg.train.transverse_invariance_points, 1024)
                    if cfg.train.transverse_invariance_points > 0
                    else 0,
                    mass_balance_grid=min(cfg.train.mass_balance_grid, 24),
                    discrete_rk4_grid=min(cfg.train.discrete_rk4_grid, 24),
                    lbfgs_steps=min(cfg.train.lbfgs_steps, 300),
                    intrinsic_phase_anchor_points=min(cfg.train.intrinsic_phase_anchor_points, 512),
                    intrinsic_phase_compatibility_points=min(cfg.train.intrinsic_phase_compatibility_points, 512),
                ),
            )
        return cfg
    if preset in {"smoke", "quick"}:
        cfg = cfg.quick()
    if preset == "smoke":
        cfg = replace(
            cfg,
            model=replace(
                cfg.model,
                hidden=min(cfg.model.hidden, 16),
                layers=min(cfg.model.layers, 2),
                fourier_features=min(cfg.model.fourier_features, 8),
                front_fourier_features=min(cfg.model.front_fourier_features, 4),
                intrinsic_front_phase_hidden=min(cfg.model.intrinsic_front_phase_hidden, 16),
                intrinsic_front_phase_layers=min(cfg.model.intrinsic_front_phase_layers, 2),
                intrinsic_front_phase_fourier_features=min(cfg.model.intrinsic_front_phase_fourier_features, 4),
                spatial_coefficient_features=min(cfg.model.spatial_coefficient_features, 4),
                spatial_coefficient_hidden=min(cfg.model.spatial_coefficient_hidden, 8),
                use_spatial_coefficients=False,
            ),
            weights=replace(
                cfg.weights,
                phase_pde=0.0,
                residual_cvar=0.0,
                front_pde_alpha=0.0,
                front_pde_gradient=0.0,
                front_gradient=0.0,
                front_speed=0.0,
                mass_balance=0.0,
                mass_floor=0.0,
                expected_front_pde=0.0,
                leading_edge=0.0,
                leading_edge_area=0.0,
                leading_edge_distribution=0.0,
                radial_symmetry=0.0,
                front_support_tversky=0.0,
                front_contrast=0.0,
                front_profile=0.0,
                level_set_alignment=0.0,
                transverse_invariance=0.0,
                time_interface=0.0,
                discrete_rk4=0.0,
                rk4_teacher=0.0,
                coefficient_field=0.0,
            ),
            train=replace(
                cfg.train,
                epochs=12,
                collocation_points=32,
                boundary_points=16,
                seed_points=32,
                print_every=6,
                validation_every=6,
                rar_interval=0,
                adaptive_loss_balancing=False,
                gradient_norm_balancing=False,
                observation_batch=64 if cfg.train.observation_batch > 0 else 0,
                time_slabs=min(cfg.train.time_slabs, 2),
                intrinsic_phase_anchor_points=min(cfg.train.intrinsic_phase_anchor_points, 24),
                intrinsic_phase_compatibility_points=min(cfg.train.intrinsic_phase_compatibility_points, 24),
                front_speed_points=min(cfg.train.front_speed_points, 16),
                expected_front_points=min(cfg.train.expected_front_points, 16),
                level_set_points=min(cfg.train.level_set_points, 16),
                leading_edge_area_times=min(cfg.train.leading_edge_area_times, 2),
                front_gradient_expected_points=min(cfg.train.front_gradient_expected_points, 12),
                mass_balance_times=min(cfg.train.mass_balance_times, 2),
                mass_balance_grid=min(cfg.train.mass_balance_grid, 16),
            ),
            observations=replace(cfg.observations, samples_per_frame=40),
            baseline_epochs=2,
            ensemble=1,
            run_classical_baseline=False,
        )
    elif preset == "quick":
        cfg = replace(
            cfg,
            train=replace(cfg.train, epochs=120, print_every=30, validation_every=30),
            ensemble=1,
            run_classical_baseline=False,
        )
    else:
        cfg = replace(
            cfg,
            train=replace(cfg.train, epochs=1200, print_every=100, validation_every=100),
            ensemble=1,
            run_classical_baseline=False,
        )
    if epochs is not None:
        interval = max(1, int(epochs) // 4)
        cfg = replace(cfg, train=replace(cfg.train, epochs=int(epochs), print_every=interval, validation_every=interval))
    return cfg


def _disable_intrinsic_phase(cfg: ExperimentConfig) -> ExperimentConfig:
    return replace(
        cfg,
        model=replace(cfg.model, use_intrinsic_front_phase=False),
        weights=replace(
            cfg.weights,
            intrinsic_phase_initial=0.0,
            intrinsic_phase_gradient_alignment=0.0,
            intrinsic_phase_monotonicity=0.0,
        ),
        train=replace(cfg.train, intrinsic_phase_anchor_points=0, intrinsic_phase_compatibility_points=0),
    )


def _vanilla_from(cfg: ExperimentConfig) -> ExperimentConfig:
    return replace(
        cfg,
        geo=GeoConfig(enabled=False, mask_kind="box"),
        model=ModelConfig(
            architecture="gated_mlp",
            hidden=cfg.model.hidden,
            layers=cfg.model.layers,
            fourier_features=cfg.model.fourier_features,
            fourier_sigma=cfg.model.fourier_sigma,
            use_source_envelope=False,
            hard_initial_condition=False,
        ),
        weights=replace(
            cfg.weights,
            pde=1.0,
            phase_pde=0.0,
            residual_cvar=0.0,
            intrinsic_phase_initial=0.0,
            intrinsic_phase_gradient_alignment=0.0,
            intrinsic_phase_monotonicity=0.0,
            front_pde_alpha=0.0,
            front_pde_gradient=0.0,
            front_gradient=0.0,
            front_speed=0.0,
            mass_balance=0.0,
            mass_floor=0.0,
            expected_front_pde=0.0,
            leading_edge=0.0,
            leading_edge_area=0.0,
            leading_edge_distribution=0.0,
            radial_symmetry=0.0,
            front_support_tversky=0.0,
            front_contrast=0.0,
            front_profile=0.0,
            level_set_alignment=0.0,
            transverse_invariance=0.0,
            time_interface=0.0,
            discrete_rk4=0.0,
            rk4_teacher=0.0,
            coefficient_field=0.0,
        ),
        train=replace(
            cfg.train,
            adaptive_loss_balancing=False,
            gradient_norm_balancing=False,
            time_marching=False,
            time_slabs=1,
            intrinsic_phase_anchor_points=0,
            intrinsic_phase_compatibility_points=0,
            front_speed_points=0,
            expected_front_points=0,
            level_set_points=0,
            radial_symmetry_groups=0,
            time_interface_points=0,
            discrete_rk4_times=0,
            discrete_rk4_grid=0,
        ),
    )


def _base_problem(problem: str, preset: str, epochs: int | None) -> ExperimentConfig:
    common = ExperimentConfig(
        observations=ObservationConfig(samples_per_frame=500, noise_std=0.02, focus_fraction=0.5),
        ensemble=1,
        run_classical_baseline=False,
        baseline_epochs=2,
    )
    cfg = common.ablowitz_zeppetella_forward() if problem == "az1d" else common.geo_spectral_forward()
    return _with_preset(cfg, preset, epochs)


def make_front_phase_ablation_cases(
    *,
    problem: str = "both",
    preset: str = "smoke",
    epochs: int | None = None,
) -> list[dict[str, Any]]:
    problems = ["az1d", "gaussian2d"] if problem == "both" else [problem]
    cases: list[dict[str, Any]] = []
    for problem_name in problems:
        base = _base_problem(problem_name, preset, epochs)
        phase_points = 24 if preset == "smoke" else (1024 if preset == "flagship" else 512 if preset == "flagship_safe" else 256)
        vanilla = _vanilla_from(base)
        geo = _disable_intrinsic_phase(base)
        front_phase = replace(
            base,
            model=replace(base.model, use_intrinsic_front_phase=True),
            weights=replace(
                base.weights,
                intrinsic_phase_initial=max(base.weights.intrinsic_phase_initial, 0.20),
                intrinsic_phase_gradient_alignment=max(base.weights.intrinsic_phase_gradient_alignment, 0.02),
                intrinsic_phase_monotonicity=max(base.weights.intrinsic_phase_monotonicity, 0.01),
            ),
            train=replace(
                base.train,
                intrinsic_phase_anchor_points=max(base.train.intrinsic_phase_anchor_points, phase_points),
                intrinsic_phase_compatibility_points=max(base.train.intrinsic_phase_compatibility_points, phase_points),
            ),
        )
        cases.extend(
            [
                {
                    "problem": problem_name,
                    "method": "vanilla_pinn",
                    "cfg": vanilla,
                    "note": "plain PINN backbone without geo/front-phase losses",
                },
                {
                    "problem": problem_name,
                    "method": "geo_spectral_pinn",
                    "cfg": geo,
                    "note": "existing geo-spectral moving-front stack without intrinsic phase head",
                },
                {
                    "problem": problem_name,
                    "method": "explicit_front_phase_pinn",
                    "cfg": front_phase,
                    "note": "geo-spectral stack plus FrontPhaseHead psi and weak compatibility losses",
                },
            ]
        )
    return cases


def _finite(values: list[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            out.append(value_f)
    return out


def _row(case: dict[str, Any], seed: int, out_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": case["problem"],
        "method": case["method"],
        "seed": seed,
        "out_dir": str(out_dir),
        "validation_observation_mse": metrics.get("validation_observation_mse"),
        "final_time_relative_l2": metrics.get("final_time_relative_l2"),
        "front_area_010_mae": metrics.get("front_area_010_mae"),
        "front_mae_010": metrics.get("front_mae_010"),
        "hausdorff_010": metrics.get("hausdorff_010"),
        "front_speed_mae_010": metrics.get("front_speed_mae_010"),
        "mass_mae": metrics.get("mass_mae"),
        "note": case["note"],
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for problem in sorted({row["problem"] for row in rows}):
        for method in sorted({row["method"] for row in rows if row["problem"] == problem}):
            subset = [row for row in rows if row["problem"] == problem and row["method"] == method]
            item: dict[str, Any] = {"problem": problem, "method": method, "n": len(subset), "note": subset[0]["note"]}
            for key in [
                "validation_observation_mse",
                "final_time_relative_l2",
                "front_area_010_mae",
                "front_mae_010",
                "hausdorff_010",
                "front_speed_mae_010",
                "mass_mae",
            ]:
                vals = _finite([row.get(key) for row in subset])
                item[f"{key}_mean"] = mean(vals) if vals else None
                item[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
            items.append(item)
    return {"cases": items}


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(path: Path, summary: dict[str, Any]) -> None:
    cases = summary["cases"]
    metrics = [
        ("final_time_relative_l2_mean", "relative L2"),
        ("front_mae_010_mean", "front MAE"),
        ("hausdorff_010_mean", "Hausdorff"),
        ("front_speed_mae_010_mean", "speed error"),
    ]
    problems = list(dict.fromkeys(str(item["problem"]) for item in cases))
    fig, axes = plt.subplots(len(problems), len(metrics), figsize=(15.0, max(3.6, 3.1 * len(problems))), squeeze=False)
    colors = {"vanilla_pinn": "#7A828F", "geo_spectral_pinn": "#5477C4", "explicit_front_phase_pinn": "#2A9D8F"}
    for row_idx, problem in enumerate(problems):
        subset = [item for item in cases if item["problem"] == problem]
        labels = [str(item["method"]).replace("_", "\n") for item in subset]
        for col_idx, (key, title) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            vals = [0.0 if item.get(key) is None else float(item[key]) for item in subset]
            ax.bar(range(len(vals)), vals, color=[colors.get(str(item["method"]), "#5477C4") for item in subset])
            ax.set_title(f"{problem}: {title}")
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.grid(axis="y", alpha=0.25)
            if key == "final_time_relative_l2_mean":
                ax.set_yscale("log")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _run_or_load(case: dict[str, Any], seed: int, out_dir: Path, *, skip_existing: bool) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    if skip_existing and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = replace(case["cfg"], base_seed=seed, out_dir=out_dir)
    return run_experiment(cfg)


def main() -> None:
    args = parse_args()
    seeds = _seed_list(args.seeds)
    cases = make_front_phase_ablation_cases(problem=args.problem, preset=args.preset, epochs=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "manifest.json",
        {
            "problem": args.problem,
            "preset": args.preset,
            "seeds": seeds,
            "cases": [
                {"problem": case["problem"], "method": case["method"], "note": case["note"]}
                for case in cases
            ],
        },
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        for seed in seeds:
            out_dir = args.out_dir / str(case["problem"]) / str(case["method"]) / f"seed_{seed}"
            print(f"\n=== problem={case['problem']} method={case['method']} seed={seed} ===")
            rows.append(_row(case, seed, out_dir, _run_or_load(case, seed, out_dir, skip_existing=args.skip_existing)))
    write_rows_csv(args.out_dir / "results.csv", rows)
    summary = aggregate(rows)
    write_json(args.out_dir / "summary.json", summary)
    plot_summary(args.out_dir / "summary.png", summary)
    print(f"wrote {args.out_dir / 'results.csv'}")
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
