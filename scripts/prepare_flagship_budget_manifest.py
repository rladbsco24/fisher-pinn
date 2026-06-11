from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "figure_previews" / "flagship_budget_manifest"


def _command(parts: list[str]) -> str:
    return " ".join(parts)


def _case(
    *,
    name: str,
    problem: str,
    command: list[str],
    output_dir: Path,
    epochs: int,
    notes: str,
) -> dict[str, object]:
    return {
        "name": name,
        "problem": problem,
        "epochs": epochs,
        "command": _command(command),
        "output_dir": str(output_dir),
        "expected_outputs": [
            str(output_dir / "metrics.json"),
            str(output_dir / "pinn_vs_rk4_comparison.png"),
            str(output_dir / "prediction_evolution.gif"),
        ],
        "fairness_constraints": {
            "rk4_teacher_labels": False,
            "posthoc_field_correction": False,
            "checkpoint_resume": True,
            "same_metric_family": [
                "final-time relative L2",
                "validation observation MSE",
                "front-area/front-band error",
                "mass trajectory error",
            ],
        },
        "notes": notes,
    }


def build_manifest(epochs: int, root: Path) -> dict[str, object]:
    runs_root = Path("runs")
    basic_root = runs_root / "basic_pinn"
    cases = [
        _case(
            name="basic_1d_az_flagship",
            problem="Ablowitz-Zeppetella 1D traveling-front benchmark in the shared 2D PINN backbone",
            command=[
                "python",
                "experiments/basic_pinn/run_zeppe_exact_1d.py",
                "--preset",
                "flagship",
                "--epochs",
                str(epochs),
                "--out-dir",
                str(basic_root / "zeppe_exact_1d_flagship"),
            ],
            output_dir=basic_root / "zeppe_exact_1d_flagship",
            epochs=epochs,
            notes=(
                "Uses exact initial/boundary constraints and traveling-wave geometry, "
                "but no RK4 pseudo-labels or after-the-fact correction."
            ),
        ),
        _case(
            name="basic_2d_gaussian_flagship",
            problem="2D Gaussian moving-front Fisher-KPP benchmark",
            command=[
                "python",
                "experiments/basic_pinn/run_gaussian_moving_front_2d.py",
                "--preset",
                "flagship",
                "--epochs",
                str(epochs),
                "--run-classical-baseline",
                "--out-dir",
                str(basic_root / "gaussian_moving_front_2d_flagship"),
            ],
            output_dir=basic_root / "gaussian_moving_front_2d_flagship",
            epochs=epochs,
            notes=(
                "Uses the same geo-spectral/front-aware model family and logs the RK4 "
                "reference as an evaluation baseline, not as supervised training labels."
            ),
        ),
        _case(
            name="korea_pine_flagship",
            problem="Korea pine-wilt land-mask reaction-diffusion PINN/RK4 comparison",
            command=[
                "python",
                "experiments/korea_pine_pinn/run_pine_pinn.py",
                "--preset",
                "flagship",
                "--output-dir",
                str(runs_root / "korea_pine_pinn_flagship"),
            ],
            output_dir=runs_root / "korea_pine_pinn_flagship",
            epochs=epochs,
            notes=(
                "Uses the Korea-specific land mask, sea exclusion, and observation timeline "
                "while keeping the same PINN training budget scale."
            ),
        ),
    ]
    return {
        "purpose": (
            "Execution plan for high-budget PINN runs intended to close the accuracy gap "
            "to the same-problem RK4 references without RK4 teacher labels or postprocessing."
        ),
        "root": str(root),
        "default_epochs": epochs,
        "cases": cases,
    }


def write_markdown(manifest: dict[str, object], path: Path) -> None:
    lines = [
        "# Flagship PINN Budget Manifest",
        "",
        str(manifest["purpose"]),
        "",
        f"Default epochs: `{manifest['default_epochs']}`",
        "",
        "The listed commands are prepared for execution but this script does not run them.",
        "",
    ]
    for case in manifest["cases"]:
        lines.extend(
            [
                f"## {case['name']}",
                "",
                f"Problem: {case['problem']}",
                "",
                "Command:",
                "",
                "```powershell",
                str(case["command"]),
                "```",
                "",
                f"Output directory: `{case['output_dir']}`",
                "",
                f"Notes: {case['notes']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a non-executing manifest for flagship PINN budget runs."
    )
    parser.add_argument("--epochs", type=int, default=20_000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(epochs=args.epochs, root=ROOT)
    json_path = args.output_dir / "flagship_budget_manifest.json"
    md_path = args.output_dir / "flagship_budget_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(manifest, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
