from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fisher_origin_lab.ablation_visuals import save_feature_pair_error_map, save_feature_pair_evolution_gif  # noqa: E402
from fisher_origin_lab.config import ExperimentConfig  # noqa: E402
from fisher_origin_lab.train import run_experiment  # noqa: E402
from fisher_origin_lab.utils import write_json  # noqa: E402
from scripts.run_forward_ablation import _preset_config, _seed_list  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired Fisher-KPP PINN feature validations and export ON/OFF "
            "error-map and GIF comparisons."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/feature_validation_ablations"))
    parser.add_argument("--preset", choices=["smoke", "quick", "full"], default="smoke")
    parser.add_argument("--seeds", default="7", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per run.")
    parser.add_argument("--pairs", default="", help="Comma-separated feature pair names to run.")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse completed case metrics and diagnostics.")
    return parser.parse_args()


def _variant(name: str, cfg: ExperimentConfig, label: str, note: str) -> dict[str, Any]:
    return {"name": name, "cfg": cfg, "label": label, "note": note}


def _pair(name: str, description: str, without: dict[str, Any], with_feature: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "description": description, "without": without, "with": with_feature}


def make_feature_validation_pairs(base: ExperimentConfig) -> list[dict[str, Any]]:
    """Create isolated ON/OFF feature pairs from the current geo-aware PINN stack."""

    level_without = replace(
        base,
        weights=replace(base.weights, level_set_alignment=0.0),
        train=replace(base.train, level_set_points=0),
    )
    level_with = replace(
        base,
        weights=replace(base.weights, level_set_alignment=max(base.weights.level_set_alignment, 0.02)),
        train=replace(base.train, level_set_points=max(base.train.level_set_points, 256)),
    )

    time_without = replace(
        base,
        weights=replace(base.weights, time_interface=0.0),
        train=replace(
            base.train,
            time_marching=False,
            time_slabs=1,
            time_slab_curriculum=False,
            time_window_focus_fraction=1.0,
            time_window_teacher=False,
            time_window_observations=False,
            time_interface_points=0,
        ),
    )

    moving_without = replace(
        base,
        model=replace(
            base.model,
            front_fourier_features=0,
            use_seed_front_features=False,
            use_traveling_wave_features=False,
            use_front_fourier_features=False,
            use_kpp_front_envelope=False,
        ),
    )

    guards_without = replace(
        base,
        weights=replace(
            base.weights,
            mass_balance=0.0,
            mass_floor=0.0,
            leading_edge=0.0,
            leading_edge_area=0.0,
            front_support_tversky=0.0,
            front_contrast=0.0,
            front_profile=0.0,
        ),
    )

    speed_without = replace(
        base,
        weights=replace(
            base.weights,
            front_speed=0.0,
            front_gradient=0.0,
            front_pde_gradient=0.0,
        ),
        train=replace(base.train, front_speed_points=0, front_gradient_expected_points=0),
    )

    balancing_without = replace(
        base,
        train=replace(base.train, adaptive_loss_balancing=False, gradient_norm_balancing=False),
    )

    coefficients_without = replace(
        base,
        model=replace(base.model, use_spatial_coefficients=False),
        weights=replace(base.weights, coefficient_field=0.0),
    )

    teacher_without = replace(
        base,
        weights=replace(base.weights, rk4_teacher=0.0),
        train=replace(base.train, rk4_pretrain_steps=0, rk4_pretrain_batch=0),
    )
    teacher_with = replace(
        base,
        weights=replace(base.weights, rk4_teacher=max(base.weights.rk4_teacher, 0.005)),
        train=replace(
            base.train,
            rk4_teacher_pool=max(base.train.rk4_teacher_pool, 4096),
            rk4_teacher_batch=max(base.train.rk4_teacher_batch, 512),
        ),
    )

    return [
        _pair(
            "level_set_alignment",
            "Front phase/level-set alignment around the expected moving front.",
            _variant("without_level_set_alignment", level_without, "level-set loss off", "level_set_alignment=0"),
            _variant("with_level_set_alignment", level_with, "level-set loss on", "level_set_alignment enabled"),
        ),
        _pair(
            "time_marching_curriculum",
            "Causal time-window training and time-slab interface regularization.",
            _variant("without_time_marching", time_without, "time curriculum off", "single full time window"),
            _variant("with_time_marching", base, "time curriculum on", "cumulative time slabs enabled"),
        ),
        _pair(
            "moving_front_features",
            "Seed-relative, traveling-wave, front Fourier, and KPP front-envelope features.",
            _variant("without_moving_front_features", moving_without, "front features off", "plain spatial Fourier backbone"),
            _variant("with_moving_front_features", base, "front features on", "moving-frame feature stack enabled"),
        ),
        _pair(
            "mass_support_guards",
            "Mass balance, mass floor, leading-edge, support, contrast, and profile guards.",
            _variant("without_mass_support_guards", guards_without, "mass/support guards off", "collapse guards removed"),
            _variant("with_mass_support_guards", base, "mass/support guards on", "collapse guards enabled"),
        ),
        _pair(
            "front_speed_gpinn",
            "Gradient-filtered front-speed consistency and front-local gPINN penalties.",
            _variant("without_front_speed_gpinn", speed_without, "front speed/gPINN off", "front speed and front-gradient losses removed"),
            _variant("with_front_speed_gpinn", base, "front speed/gPINN on", "front-local derivative losses enabled"),
        ),
        _pair(
            "adaptive_balancing",
            "Adaptive loss balancing and gradient-norm balancing.",
            _variant("without_adaptive_balancing", balancing_without, "adaptive balancing off", "fixed manual loss weights"),
            _variant("with_adaptive_balancing", base, "adaptive balancing on", "adaptive and gradient-norm balancing enabled"),
        ),
        _pair(
            "spatial_coefficients",
            "Smooth spatial correction fields for learned diffusion/reaction.",
            _variant("without_spatial_coefficients", coefficients_without, "spatial coefficients off", "global D/r only"),
            _variant("with_spatial_coefficients", base, "spatial coefficients on", "smooth spatial D/r correction enabled"),
        ),
        _pair(
            "rk4_teacher",
            "Weak RK4 teacher samples used as solver-assisted regularization.",
            _variant("without_rk4_teacher", teacher_without, "RK4 teacher off", "no RK4 pseudo-label term"),
            _variant("with_rk4_teacher", teacher_with, "RK4 teacher on", "weak RK4 pseudo-label term enabled"),
        ),
    ]


def _metrics_path(out_dir: Path) -> Path:
    return out_dir / "metrics.json"


def _fields_path(out_dir: Path) -> Path:
    return out_dir / "diagnostic_fields.npz"


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_or_load(case: dict[str, Any], seed: int, out_dir: Path, *, skip_existing: bool) -> dict[str, Any]:
    metrics_path = _metrics_path(out_dir)
    fields_path = _fields_path(out_dir)
    if skip_existing and metrics_path.exists() and fields_path.exists():
        return _load_metrics(metrics_path)
    cfg = replace(case["cfg"], base_seed=seed, out_dir=out_dir)
    metrics = run_experiment(cfg)
    if not fields_path.exists():
        raise RuntimeError(f"{fields_path} was not produced; cannot render feature comparison maps.")
    return metrics


def _row(pair: dict[str, Any], variant_key: str, seed: int, out_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    variant = pair[variant_key]
    return {
        "feature": pair["name"],
        "variant": variant_key,
        "case": variant["name"],
        "label": variant["label"],
        "seed": seed,
        "out_dir": str(out_dir),
        "validation_observation_mse": metrics.get("validation_observation_mse"),
        "final_time_relative_l2": metrics.get("final_time_relative_l2"),
        "pinn_final_time_max_abs_error": metrics.get("pinn_final_time_max_abs_error"),
        "front_area_010_mae": metrics.get("front_area_010_mae"),
        "active_front_area_mae": metrics.get("active_front_area_mae"),
        "mass_mae": metrics.get("mass_mae"),
        "note": variant["note"],
    }


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def summarize_feature_validation_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "validation_observation_mse",
        "final_time_relative_l2",
        "pinn_final_time_max_abs_error",
        "front_area_010_mae",
        "active_front_area_mae",
        "mass_mae",
    ]
    features = sorted({row["feature"] for row in rows})
    summary: list[dict[str, Any]] = []
    for feature in features:
        item: dict[str, Any] = {"feature": feature}
        feature_rows = [row for row in rows if row["feature"] == feature]
        for metric in metrics:
            without_values = [_finite(row.get(metric)) for row in feature_rows if row["variant"] == "without"]
            with_values = [_finite(row.get(metric)) for row in feature_rows if row["variant"] == "with"]
            without_values = [value for value in without_values if value is not None]
            with_values = [value for value in with_values if value is not None]
            item[f"without_{metric}_mean"] = mean(without_values) if without_values else None
            item[f"with_{metric}_mean"] = mean(with_values) if with_values else None
            if without_values and with_values:
                item[f"delta_{metric}_mean"] = mean(with_values) - mean(without_values)
                item[f"improvement_{metric}_mean"] = mean(without_values) - mean(with_values)
            else:
                item[f"delta_{metric}_mean"] = None
                item[f"improvement_{metric}_mean"] = None
        summary.append(item)
    return {"features": summary}


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = _seed_list(args.seeds)
    base = _preset_config(args)
    pairs = make_feature_validation_pairs(base)
    if args.pairs.strip():
        requested = {name.strip() for name in args.pairs.split(",") if name.strip()}
        pairs = [pair for pair in pairs if pair["name"] in requested]
        missing = sorted(requested - {pair["name"] for pair in pairs})
        if missing:
            raise SystemExit(f"Unknown feature pair(s): {', '.join(missing)}")
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "manifest.json",
        {
            "preset": args.preset,
            "seeds": seeds,
            "pairs": [
                {
                    "name": pair["name"],
                    "description": pair["description"],
                    "without": pair["without"]["note"],
                    "with": pair["with"]["note"],
                }
                for pair in pairs
            ],
        },
    )

    rows: list[dict[str, Any]] = []
    visual_manifest: list[dict[str, Any]] = []
    for pair in pairs:
        for seed in seeds:
            pair_root = args.out_dir / pair["name"] / f"seed_{seed}"
            without_dir = pair_root / "without"
            with_dir = pair_root / "with"
            print(f"\n=== feature={pair['name']} seed={seed} without ===")
            without_metrics = _run_or_load(pair["without"], seed, without_dir, skip_existing=args.skip_existing)
            print(f"\n=== feature={pair['name']} seed={seed} with ===")
            with_metrics = _run_or_load(pair["with"], seed, with_dir, skip_existing=args.skip_existing)
            rows.append(_row(pair, "without", seed, without_dir, without_metrics))
            rows.append(_row(pair, "with", seed, with_dir, with_metrics))

            comparison_dir = pair_root / "comparison"
            error_map = save_feature_pair_error_map(
                comparison_dir / "feature_error_map_comparison.png",
                feature_name=pair["name"],
                without_label=pair["without"]["label"],
                with_label=pair["with"]["label"],
                without_fields=_fields_path(without_dir),
                with_fields=_fields_path(with_dir),
                without_metrics=without_metrics,
                with_metrics=with_metrics,
                domain=base.domain,
            )
            evolution = save_feature_pair_evolution_gif(
                comparison_dir / "feature_evolution_comparison.gif",
                feature_name=pair["name"],
                without_label=pair["without"]["label"],
                with_label=pair["with"]["label"],
                without_gif=without_dir / "pinn_evolution.gif",
                with_gif=with_dir / "pinn_evolution.gif",
            )
            visual_manifest.append(
                {
                    "feature": pair["name"],
                    "seed": seed,
                    "error_map": error_map,
                    "evolution_gif": evolution,
                }
            )
            write_json(comparison_dir / "comparison_manifest.json", visual_manifest[-1])

    write_rows_csv(args.out_dir / "results.csv", rows)
    summary = summarize_feature_validation_rows(rows)
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "visual_manifest.json", {"comparisons": visual_manifest})
    print(f"\nwrote {args.out_dir / 'results.csv'}")
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'visual_manifest.json'}")


if __name__ == "__main__":
    main()
