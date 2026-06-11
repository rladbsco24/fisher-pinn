from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class LocalRun:
    problem: str
    method: str
    run_dir: str
    metric_source: str


LOCAL_RUNS = [
    LocalRun(
        problem="1D AZ traveling front",
        method="PINN vanilla moving-front proxy",
        run_dir="zeppe_exact_1d_quick40",
        metric_source="local metrics.json",
    ),
    LocalRun(
        problem="1D AZ traveling front",
        method="PINN planar + transverse invariant",
        run_dir="zeppe_exact_1d_planar_transverse_w3_quick40",
        metric_source="local metrics.json",
    ),
    LocalRun(
        problem="1D AZ traveling front",
        method="RK4 exact-wave reference",
        run_dir="zeppe_exact_1d_planar_transverse_w3_quick40",
        metric_source="same-problem RK4 metric",
    ),
    LocalRun(
        problem="2D Gaussian moving front",
        method="PINN vanilla moving-front proxy",
        run_dir="gaussian_front_2d_no_teacher_no_radial_quick40",
        metric_source="local metrics.json",
    ),
    LocalRun(
        problem="2D Gaussian moving front",
        method="PINN geo-spectral + mass/front guards",
        run_dir="gaussian_front_2d_mass_upper115_quick40",
        metric_source="local metrics.json",
    ),
    LocalRun(
        problem="2D Gaussian moving front",
        method="RK4 finite-difference reference",
        run_dir="gaussian_front_2d_mass_upper115_quick40",
        metric_source="same-problem RK4 metric",
    ),
]


BENCHMARK_FAMILIES = [
    {
        "family": "SciML NeuralPDE PINN optimizer benchmark",
        "moving_front_role": "standard PINN optimizer/loss construction reference",
        "representative_problem": "diffusion and related PDE PINN benchmarks",
        "local_mapping": "vanilla PINN proxy rows",
        "implemented_here": "partly",
        "source": "https://docs.sciml.ai/SciMLBenchmarksOutput/dev/PINNOptimizers/1d_diffusion/",
    },
    {
        "family": "SciML NeuralPDE level-set error-vs-time benchmark",
        "moving_front_role": "level-set/interface benchmark framing",
        "representative_problem": "2D level-set PDE",
        "local_mapping": "front-level-set alignment and front-band diagnostics",
        "implemented_here": "partly",
        "source": "https://docs.sciml.ai/SciMLBenchmarksOutput/dev/PINNErrorsVsTime/level_set_et/",
    },
    {
        "family": "PINNacle benchmark suite",
        "moving_front_role": "benchmark-model family comparison",
        "representative_problem": "PINN, PINN+L-BFGS, LRA, NTK, RAR, gPINN, FBPINN",
        "local_mapping": "adaptive balancing, front RAR, gPINN-like front losses",
        "implemented_here": "partly",
        "source": "https://github.com/i207M/PINNacle",
    },
    {
        "family": "Level-set moving-interface PINN / PirateNet line",
        "moving_front_role": "moving-interface accuracy target",
        "representative_problem": "level-set moving interface and vortex front",
        "local_mapping": "Pirate/RWF backbone, Fourier features, causal curriculum, level-set front losses",
        "implemented_here": "partly",
        "source": "https://arxiv.org/html/2502.02440v1",
    },
    {
        "family": "Classical RK4 method-of-lines reference",
        "moving_front_role": "same-regime numerical accuracy reference",
        "representative_problem": "1D AZ exact wave and 2D Fisher-KPP exact/finite-difference reference",
        "local_mapping": "RK4 rows in local accuracy table",
        "implemented_here": "yes",
        "source": "local fisher-kpp-rk4 and PINN metric outputs",
    },
]


def _load_metric(run_root: Path, run: LocalRun) -> dict[str, float | str]:
    metrics_path = run_root / run.run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if run.method.startswith("RK4"):
        final_l2 = metrics["rk4_final_time_relative_l2"]
        val_mse = None
        mass_mae = None
        front_mae = None
    else:
        final_l2 = metrics["final_time_relative_l2"]
        val_mse = metrics.get("validation_observation_mse")
        mass_mae = metrics.get("mass_mae")
        front_mae = metrics.get("front_area_010_mae") or metrics.get("active_front_area_mae")
    return {
        "problem": run.problem,
        "method": run.method,
        "run_dir": run.run_dir,
        "final_time_relative_l2": float(final_l2),
        "validation_observation_mse": "" if val_mse is None else float(val_mse),
        "front_area_010_mae": "" if front_mae is None else float(front_mae),
        "mass_mae": "" if mass_mae is None else float(mass_mae),
        "metric_source": run.metric_source,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_accuracy(rows: list[dict[str, object]], output: Path) -> None:
    problems = list(dict.fromkeys(str(row["problem"]) for row in rows))
    method_order = [
        "PINN vanilla moving-front proxy",
        "PINN planar + transverse invariant",
        "PINN geo-spectral + mass/front guards",
        "RK4 exact-wave reference",
        "RK4 finite-difference reference",
    ]
    colors = {
        "PINN vanilla moving-front proxy": "#6f7684",
        "PINN planar + transverse invariant": "#1f77b4",
        "PINN geo-spectral + mass/front guards": "#2ca02c",
        "RK4 exact-wave reference": "#d62728",
        "RK4 finite-difference reference": "#d62728",
    }

    fig, axes = plt.subplots(1, len(problems), figsize=(12, 4.2), constrained_layout=True)
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems):
        subset = [row for row in rows if row["problem"] == problem]
        subset.sort(key=lambda row: method_order.index(str(row["method"])))
        labels = [str(row["method"]).replace("PINN ", "").replace(" reference", "") for row in subset]
        values = [float(row["final_time_relative_l2"]) for row in subset]
        bars = ax.bar(
            np.arange(len(values)),
            values,
            color=[colors[str(row["method"])] for row in subset],
            width=0.68,
        )
        ax.set_yscale("log")
        ax.set_title(problem)
        ax.set_ylabel("Final-time relative L2")
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.28, which="both")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value * 1.12,
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("Moving-front benchmark accuracy: local PINN variants vs RK4", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    by_problem: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_problem.setdefault(str(row["problem"]), []).append(row)

    lines = [
        "# SciML-style Moving-front Comparison",
        "",
        "This file separates two comparisons. The first table is a local, same-regime",
        "accuracy comparison using metrics generated by this repository. The second table",
        "maps the local implementation to benchmark model families used in SciML,",
        "PINNacle, and recent level-set moving-interface PINN work. External benchmark",
        "families are not mixed into the numeric L2 table because their PDEs, domains,",
        "training budgets, and error definitions are not identical.",
        "",
        "## Local Accuracy",
        "",
        "| problem | method | final-time relative L2 | validation MSE | front MAE | mass MAE |",
        "|---|---:|---:|---:|---:|---:|".replace("---:|---", "---|---"),
    ]
    for problem, problem_rows in by_problem.items():
        rk4_rows = [r for r in problem_rows if str(r["method"]).startswith("RK4")]
        rk4_l2 = float(rk4_rows[0]["final_time_relative_l2"]) if rk4_rows else None
        for row in problem_rows:
            lines.append(
                "| {problem} | {method} | {l2:.6g} | {mse} | {front} | {mass} |".format(
                    problem=problem,
                    method=row["method"],
                    l2=float(row["final_time_relative_l2"]),
                    mse=_fmt_optional(row["validation_observation_mse"]),
                    front=_fmt_optional(row["front_area_010_mae"]),
                    mass=_fmt_optional(row["mass_mae"]),
                )
            )
        if rk4_l2 is not None:
            best_pinn = min(
                float(r["final_time_relative_l2"])
                for r in problem_rows
                if not str(r["method"]).startswith("RK4")
            )
            lines.extend(
                [
                    "",
                    f"For {problem}, the best local PINN is still {best_pinn / rk4_l2:.1f}x above the RK4 final-time relative L2.",
                    "",
                ]
            )

    lines.extend(
        [
            "## Benchmark-family Mapping",
            "",
            "| benchmark family | moving-front role | local mapping | implemented here | source |",
            "|---|---|---|---|---|",
        ]
    )
    for item in BENCHMARK_FAMILIES:
        lines.append(
            "| {family} | {role} | {mapping} | {impl} | {source} |".format(
                family=item["family"],
                role=item["moving_front_role"],
                mapping=item["local_mapping"],
                impl=item["implemented_here"],
                source=item["source"],
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_optional(value: object) -> str:
    if value == "" or value is None:
        return "-"
    return f"{float(value):.6g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs") / "goal_eval")
    parser.add_argument("--output-dir", type=Path, default=Path("docs") / "figure_previews")
    args = parser.parse_args()

    rows = [_load_metric(args.run_root, run) for run in LOCAL_RUNS]
    _write_csv(args.output_dir / "sciml_moving_front_accuracy_comparison.csv", rows)
    _write_csv(args.output_dir / "sciml_moving_front_benchmark_families.csv", BENCHMARK_FAMILIES)
    _plot_accuracy(rows, args.output_dir / "sciml_moving_front_accuracy_comparison.png")
    _write_markdown(args.output_dir / "sciml_moving_front_comparison.md", rows)
    print(f"wrote {args.output_dir / 'sciml_moving_front_accuracy_comparison.png'}")


if __name__ == "__main__":
    main()
