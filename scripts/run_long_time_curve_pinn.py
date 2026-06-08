from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fisher_origin_lab.curve_trend import (  # noqa: E402
    CurvePINNConfig,
    CurveTrendConfig,
    integrate_curve,
    save_curve_pinn_outputs,
    train_curve_pinn,
)
from fisher_origin_lab.utils import default_device, seed_everything, write_json  # noqa: E402


METHODS = ("forward_euler", "backward_euler", "trapezoidal", "rk4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PINN for the reference-style long-time rho(t) curve.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "runs" / "long_time_curve_pinn")
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--quick", action="store_true", help="Use a shorter Colab/smoke-friendly PINN run.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    trend_cfg = CurveTrendConfig()
    pinn_cfg = replace(CurvePINNConfig(), epochs=args.epochs)
    if args.quick:
        pinn_cfg = pinn_cfg.quick()
    device = args.device or default_device()

    baselines = {method: integrate_curve(method, trend_cfg) for method in METHODS}
    result = train_curve_pinn(trend_cfg, pinn_cfg, device=device, seed=args.seed)
    output_files = save_curve_pinn_outputs(args.out_dir, result, baselines)

    baseline_metrics = {
        method: {
            "max_abs_error": float(values["abs_error"].max()),
            "final_rho": float(values["rho"][-1]),
        }
        for method, values in baselines.items()
    }
    summary = {
        "trend_config": trend_cfg.to_dict(),
        "pinn_config": pinn_cfg.to_dict(),
        "device": str(device),
        "seed": args.seed,
        "pinn_metrics": result["metrics"],
        "baseline_metrics": baseline_metrics,
        "outputs": output_files,
    }
    write_json(args.out_dir / "metrics.json", summary)
    print("=== PINN long-time curve trend ===")
    print(f"out_dir={args.out_dir}")
    print(f"PINN max_abs_error={result['metrics']['max_abs_error']:.3e}")
    print(f"PINN relative_l2_to_exact={result['metrics']['relative_l2_to_exact']:.3e}")
    print(f"curve={output_files['curve_png']}")
    print(f"diagnostics={output_files['diagnostics_png']}")


if __name__ == "__main__":
    main()
