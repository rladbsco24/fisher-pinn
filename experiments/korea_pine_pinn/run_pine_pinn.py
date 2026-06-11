from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_korea_pine_wilt_simulation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Korea pine-wilt Fisher-KPP PINN/RK4 comparison.")
    parser.add_argument("--preset", choices=["smoke", "quick", "full", "flagship"], default="quick")
    parser.add_argument("--output-dir", type=Path, default=Path("runs") / "korea_pine_pinn")
    parser.add_argument("--time-axis", choices=["year", "pre_action_month"], default="year")
    parser.add_argument("--raw-csv-dir", type=Path, default=None)
    parser.add_argument("--end-year", type=int, default=2030)
    parser.add_argument("--skip-pinn", action="store_true")
    parser.add_argument("--pinn-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def _preset_args(args: argparse.Namespace) -> list[str]:
    if args.preset == "smoke":
        grid_size = 24
        pinn_epochs = 1
        steps_per_year = 8
        max_frames = 4
    elif args.preset == "quick":
        grid_size = 48
        pinn_epochs = 30
        steps_per_year = 24
        max_frames = 8
    elif args.preset == "full":
        grid_size = 96
        pinn_epochs = 1200
        steps_per_year = 80
        max_frames = 15
    else:
        grid_size = 128
        pinn_epochs = 20_000
        steps_per_year = 120
        max_frames = 24
    if args.pinn_epochs is not None:
        pinn_epochs = int(args.pinn_epochs)

    cmd = [
        "--grid-size",
        str(grid_size),
        "--steps-per-year",
        str(steps_per_year),
        "--end-year",
        str(args.end_year),
        "--output-dir",
        str(args.output_dir),
        "--time-axis",
        str(args.time_axis),
        "--pinn-epochs",
        str(pinn_epochs),
        "--pinn-batch-size",
        str(8192 if args.preset == "flagship" else 4096),
        "--pinn-collocation-points",
        str(8192 if args.preset == "flagship" else 768),
        "--pinn-boundary-points",
        str(1024 if args.preset == "flagship" else 128),
        "--pinn-initial-condition-points",
        str(8192 if args.preset == "flagship" else 2048),
        "--pinn-support-area-weight",
        str(0.50 if args.preset == "flagship" else 0.35),
        "--pinn-mass-trajectory-points",
        str(8192 if args.preset == "flagship" else 2048),
        "--pinn-mass-trajectory-times",
        str(8 if args.preset == "flagship" else 4),
        "--pinn-phase-pde-weight",
        str(0.05 if args.preset == "flagship" else 0.02),
        "--pinn-residual-cvar-weight",
        str(0.06 if args.preset == "flagship" else 0.03),
        "--pinn-residual-cvar-fraction",
        str(0.12 if args.preset == "flagship" else 0.10),
        "--pinn-lr",
        str(1.0e-3 if args.preset == "flagship" else 2.0e-3),
        "--pinn-checkpoint-every",
        str(25 if args.preset == "flagship" else 1),
        "--map-gif-max-frames",
        str(max_frames),
        "--seed",
        str(args.seed),
    ]
    if args.raw_csv_dir is not None:
        cmd.extend(["--raw-csv-dir", str(args.raw_csv_dir)])
    if args.skip_pinn:
        cmd.append("--skip-pinn")
    return cmd


def main() -> None:
    args = parse_args()
    subprocess.run([sys.executable, str(SCRIPT), *_preset_args(args)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
