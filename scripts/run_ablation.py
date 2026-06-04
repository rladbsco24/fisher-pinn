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

from fisher_origin_lab.config import (  # noqa: E402
    ExperimentConfig,
    LossWeights,
    ModelConfig,
    ObservationConfig,
    TrainConfig,
    WarmStartConfig,
)
from fisher_origin_lab.train import run_experiment  # noqa: E402
from fisher_origin_lab.utils import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Fisher-KPP inverse-origin ablations over warm starts, anchors, sensors, and learned drift."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/ablations"))
    parser.add_argument("--preset", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--case-set", choices=["core", "anchor", "method", "stress", "all"], default="core")
    parser.add_argument("--seeds", default="7,8,9", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per run.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--run-classical-baseline", action="store_true")
    return parser.parse_args()


def _seed_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _preset_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig(
        observations=ObservationConfig(samples_per_frame=500, noise_std=0.02, focus_fraction=0.5, validation_fraction=0.2),
        weights=LossWeights(gradient=0.01, source_anchor=2.0, shooting=50.0),
        warm_start=WarmStartConfig(mode="drift_corrected"),
        ensemble=1,
        run_classical_baseline=args.run_classical_baseline,
        baseline_epochs=250,
    )
    if args.preset in {"smoke", "quick"}:
        cfg = cfg.quick()
    if args.preset == "smoke":
        cfg = replace(
            cfg,
            train=replace(cfg.train, epochs=8, print_every=4, rar_interval=0),
            observations=replace(cfg.observations, samples_per_frame=80),
            baseline_epochs=2,
        )
    elif args.preset == "quick":
        cfg = replace(cfg, train=replace(cfg.train, epochs=120, print_every=30))
    else:
        cfg = replace(cfg, train=replace(cfg.train, epochs=1200, print_every=100))

    if args.epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=args.epochs, print_every=max(1, args.epochs // 4)))
    return cfg


def _case(name: str, cfg: ExperimentConfig, group: str, note: str) -> dict[str, Any]:
    return {"name": name, "cfg": cfg, "group": group, "note": note}


def make_cases(base: ExperimentConfig) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    cases.append(
        _case(
            "known_drift_anchor2",
            replace(
                base,
                warm_start=WarmStartConfig("drift_corrected"),
                weights=replace(base.weights, source_anchor=2.0),
                model=replace(base.model, learn_drift=False),
            ),
            "core",
            "Known-drift observation warm start plus moderate source anchor.",
        )
    )
    cases.append(
        _case(
            "known_drift_no_anchor",
            replace(
                base,
                warm_start=WarmStartConfig("drift_corrected"),
                weights=replace(base.weights, source_anchor=0.0),
                model=replace(base.model, learn_drift=False),
            ),
            "anchor",
            "Tests whether the source estimate survives without the warm-start anchor.",
        )
    )
    cases.append(
        _case(
            "known_drift_no_shooting",
            replace(
                base,
                warm_start=WarmStartConfig("drift_corrected"),
                weights=replace(base.weights, source_anchor=2.0, shooting=0.0),
                model=replace(base.model, learn_drift=False),
            ),
            "method",
            "Ablates the differentiable source-shooting consistency loss.",
        )
    )
    cases.append(
        _case(
            "shooting_prefit_anchor2",
            replace(
                base,
                warm_start=WarmStartConfig("shooting_prefit"),
                weights=replace(base.weights, source_anchor=2.0, shooting=50.0),
                model=replace(base.model, learn_drift=False),
            ),
            "method",
            "Initializes the source by fitting the coarse differentiable shooting model before PINN training.",
        )
    )
    cases.append(
        _case(
            "centroid_anchor2",
            replace(
                base,
                warm_start=WarmStartConfig("centroid"),
                weights=replace(base.weights, source_anchor=2.0),
                model=replace(base.model, learn_drift=False),
            ),
            "core",
            "No drift correction in warm start; tests dependence on known advection.",
        )
    )
    cases.append(
        _case(
            "centroid_learn_drift_anchor05",
            replace(
                base,
                warm_start=WarmStartConfig("centroid"),
                weights=replace(base.weights, source_anchor=0.5),
                model=replace(base.model, learn_drift=True),
            ),
            "stress",
            "Harder setting: warm start does not know drift, and drift is trainable.",
        )
    )
    cases.append(
        _case(
            "neutral_no_anchor",
            replace(
                base,
                warm_start=WarmStartConfig("neutral"),
                weights=replace(base.weights, source_anchor=0.0),
                model=replace(base.model, learn_drift=False),
            ),
            "stress",
            "No observation-derived source initialization; tests identifiability.",
        )
    )
    cases.append(
        _case(
            "uniform_sensors_anchor2",
            replace(
                base,
                observations=replace(base.observations, focus_fraction=0.0),
                warm_start=WarmStartConfig("drift_corrected"),
                weights=replace(base.weights, source_anchor=2.0),
            ),
            "stress",
            "Uniform-only sensors; exposes near-zero-background data imbalance.",
        )
    )
    cases.append(
        _case(
            "sparse_noisy_anchor2",
            replace(
                base,
                observations=replace(
                    base.observations,
                    samples_per_frame=max(40, base.observations.samples_per_frame // 4),
                    noise_std=0.05,
                ),
                warm_start=WarmStartConfig("drift_corrected"),
                weights=replace(base.weights, source_anchor=2.0),
            ),
            "stress",
            "Sparse and noisier observations.",
        )
    )
    return cases


def select_cases(cases: list[dict[str, Any]], case_set: str, max_cases: int | None) -> list[dict[str, Any]]:
    if case_set == "core":
        selected = [case for case in cases if case["group"] == "core"]
    elif case_set == "anchor":
        selected = [case for case in cases if case["group"] in {"core", "anchor", "method"}]
    elif case_set == "method":
        selected = [case for case in cases if case["group"] in {"core", "method"}]
    elif case_set == "stress":
        selected = [case for case in cases if case["group"] in {"core", "stress"}]
    else:
        selected = cases
    if max_cases is not None:
        selected = selected[:max_cases]
    return selected


def _baseline_error(metrics: dict[str, Any], name: str) -> float | None:
    for baseline in metrics["baselines"]:
        if baseline["name"] == name:
            return baseline["error"]
    return None


def _row(case: dict[str, Any], seed: int, metrics: dict[str, Any]) -> dict[str, Any]:
    cfg: ExperimentConfig = case["cfg"]
    return {
        "case": case["name"],
        "group": case["group"],
        "seed": seed,
        "best_origin_error": metrics["best_origin_error"],
        "final_time_relative_l2": metrics["final_time_relative_l2"],
        "train_observation_mse": metrics["train_observation_mse"],
        "validation_observation_mse": metrics["validation_observation_mse"],
        "late_centroid_error": _baseline_error(metrics, "observation_late_weighted_centroid"),
        "drift_corrected_centroid_error": _baseline_error(metrics, "observation_drift_corrected_centroid"),
        "fd_seed_fit_error": _baseline_error(metrics, "differentiable_fd_seed_fit"),
        "warm_start": cfg.warm_start.mode,
        "source_anchor": cfg.weights.source_anchor,
        "shooting": cfg.weights.shooting,
        "focus_fraction": cfg.observations.focus_fraction,
        "samples_per_frame": cfg.observations.samples_per_frame,
        "noise_std": cfg.observations.noise_std,
        "learn_drift": cfg.model.learn_drift,
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
    cases = sorted({row["case"] for row in rows})
    summary = []
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        origin = _finite([row["best_origin_error"] for row in case_rows])
        l2 = _finite([row["final_time_relative_l2"] for row in case_rows])
        val_mse = _finite([row["validation_observation_mse"] for row in case_rows])
        drift = _finite([row["drift_corrected_centroid_error"] for row in case_rows])
        late = _finite([row["late_centroid_error"] for row in case_rows])
        summary.append(
            {
                "case": case,
                "n": len(case_rows),
                "origin_error_mean": mean(origin) if origin else None,
                "origin_error_std": pstdev(origin) if len(origin) > 1 else 0.0,
                "relative_l2_mean": mean(l2) if l2 else None,
                "relative_l2_std": pstdev(l2) if len(l2) > 1 else 0.0,
                "validation_observation_mse_mean": mean(val_mse) if val_mse else None,
                "validation_observation_mse_std": pstdev(val_mse) if len(val_mse) > 1 else 0.0,
                "late_centroid_error_mean": mean(late) if late else None,
                "drift_corrected_centroid_error_mean": mean(drift) if drift else None,
                "warm_start": case_rows[0]["warm_start"],
                "source_anchor": case_rows[0]["source_anchor"],
                "shooting": case_rows[0]["shooting"],
                "focus_fraction": case_rows[0]["focus_fraction"],
                "samples_per_frame": case_rows[0]["samples_per_frame"],
                "noise_std": case_rows[0]["noise_std"],
                "learn_drift": case_rows[0]["learn_drift"],
                "note": case_rows[0]["note"],
            }
        )
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
    means = [case["origin_error_mean"] or 0.0 for case in cases]
    stds = [case["origin_error_std"] or 0.0 for case in cases]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.5 * len(labels)), 4.5), constrained_layout=True)
    ax.bar(range(len(labels)), means, yerr=stds, capsize=4, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("origin error")
    ax.set_title("Ablation summary: lower origin error is better")
    ax.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seeds = _seed_list(args.seeds)
    base = _preset_config(args)
    cases = select_cases(make_cases(base), args.case_set, args.max_cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    manifest = {
        "preset": args.preset,
        "case_set": args.case_set,
        "seeds": seeds,
        "cases": [{"name": case["name"], "group": case["group"], "note": case["note"]} for case in cases],
    }
    write_json(args.out_dir / "manifest.json", manifest)

    for case in cases:
        for seed in seeds:
            cfg: ExperimentConfig = replace(
                case["cfg"],
                base_seed=seed,
                out_dir=args.out_dir / case["name"] / f"seed_{seed}",
            )
            print(f"\n=== case={case['name']} seed={seed} ===")
            metrics = run_experiment(cfg)
            rows.append(_row(case, seed, metrics))

    write_rows_csv(args.out_dir / "results.csv", rows)
    summary = aggregate(rows)
    write_json(args.out_dir / "summary.json", summary)
    plot_summary(args.out_dir / "summary.png", summary)
    print(f"\nwrote {args.out_dir / 'results.csv'}")
    print(f"wrote {args.out_dir / 'summary.json'}")
    print(f"wrote {args.out_dir / 'summary.png'}")


if __name__ == "__main__":
    main()
