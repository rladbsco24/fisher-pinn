from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import relative_l2, solve_long_time_curve
from fisher_kpp_rk4.config import (
    CURVE_ALPHA,
    CURVE_DT,
    CURVE_NT,
    CURVE_OMEGA_D,
    CURVE_RHO0,
    CURVE_RHO_INF,
    CURVE_T,
    CURVE_V0,
)


METHODS = ("forward_euler", "backward_euler", "trapezoidal", "rk4")


def run_methods() -> dict[str, dict[str, np.ndarray]]:
    return {
        method: solve_long_time_curve(
            method,
            dt=CURVE_DT,
            Nt=CURVE_NT,
            rho_inf=CURVE_RHO_INF,
            alpha=CURVE_ALPHA,
            omega_d=CURVE_OMEGA_D,
            rho0=CURVE_RHO0,
            v0=CURVE_V0,
        )
        for method in METHODS
    }


def first_extrema(times: np.ndarray, rho: np.ndarray) -> tuple[float, float, float, float]:
    search = times <= 8.0
    local_times = times[search]
    local_rho = rho[search]
    peak_idx = int(np.argmax(local_rho))
    trough_candidates = np.arange(peak_idx + 1, len(local_rho))
    if len(trough_candidates) == 0:
        return float(local_times[peak_idx]), float(local_rho[peak_idx]), np.nan, np.nan
    trough_idx = int(trough_candidates[np.argmin(local_rho[trough_candidates])])
    return (
        float(local_times[peak_idx]),
        float(local_rho[peak_idx]),
        float(local_times[trough_idx]),
        float(local_rho[trough_idx]),
    )


def summarize(results: dict[str, dict[str, np.ndarray]]) -> list[dict[str, float | str]]:
    exact = results["rk4"]["exact_rho"]
    rows: list[dict[str, float | str]] = []
    for method, result in results.items():
        peak_t, peak, trough_t, trough = first_extrema(result["times"], result["rho"])
        rows.append(
            {
                "method": method,
                "max_abs_error": float(np.max(result["abs_error"])),
                "relative_l2_to_exact": relative_l2(result["rho"], exact),
                "final_rho": float(result["rho"][-1]),
                "first_peak_time": peak_t,
                "first_peak_rho": peak,
                "first_trough_time": trough_t,
                "first_trough_rho": trough,
            }
        )
    return rows


def write_outputs(out_dir: Path, results: dict[str, dict[str, np.ndarray]], rows: list[dict[str, float | str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "long_time_curve_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "method",
            "max_abs_error",
            "relative_l2_to_exact",
            "final_rho",
            "first_peak_time",
            "first_peak_rho",
            "first_trough_time",
            "first_trough_rho",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload: dict[str, np.ndarray] = {
        "times": results["rk4"]["times"],
        "exact_rho": results["rk4"]["exact_rho"],
        "exact_velocity": results["rk4"]["exact_velocity"],
        "rho_inf": np.array(CURVE_RHO_INF),
        "alpha": np.array(CURVE_ALPHA),
        "omega_d": np.array(CURVE_OMEGA_D),
        "rho0": np.array(CURVE_RHO0),
        "v0": np.array(CURVE_V0),
        "T": np.array(CURVE_T),
        "dt": np.array(CURVE_DT),
    }
    for method, result in results.items():
        for key in ("rho", "velocity", "abs_error"):
            payload[f"{method}_{key}"] = result[key]
    np.savez(out_dir / "long_time_curve_results.npz", **payload)


def save_plots(out_dir: Path, results: dict[str, dict[str, np.ndarray]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped plots.")
        return

    times = results["rk4"]["times"]
    exact = results["rk4"]["exact_rho"]
    plt.figure(figsize=(9.2, 4.8))
    plt.plot(times, exact, color="black", linewidth=2.4, linestyle="--", label="target exact")
    for method, result in results.items():
        plt.plot(result["times"], result["rho"], linewidth=1.5, alpha=0.9, label=method)
    plt.xlabel("t")
    plt.ylabel("rho(t)")
    plt.title("Long-time damped curve trend")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "long_time_curve_trend.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9.2, 4.3))
    for method, result in results.items():
        plt.semilogy(result["times"], np.maximum(result["abs_error"], 1.0e-14), label=method)
    plt.xlabel("t")
    plt.ylabel("|rho_num - rho_exact|")
    plt.title("Long-time curve absolute error")
    plt.grid(alpha=0.25, which="both")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "long_time_curve_error.png", dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Match the reference-style long-time rho(t) curve with FE/BE/trapezoidal/RK4.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "long_time_curve_trend")
    args = parser.parse_args()

    print("=== Long-time damped curve-trend benchmark ===")
    print(
        f"T={CURVE_T}, dt={CURVE_DT}, rho_inf={CURVE_RHO_INF}, "
        f"alpha={CURVE_ALPHA}, omega_d={CURVE_OMEGA_D}, rho0={CURVE_RHO0}, v0={CURVE_V0}"
    )
    results = run_methods()
    rows = summarize(results)
    write_outputs(args.output_dir, results, rows)
    save_plots(args.output_dir, results)
    for row in rows:
        print(
            f"{row['method']:>14s} max_abs_error={row['max_abs_error']:.3e} "
            f"relL2={row['relative_l2_to_exact']:.3e} "
            f"peak=({row['first_peak_time']:.2f}, {row['first_peak_rho']:.3f}) "
            f"trough=({row['first_trough_time']:.2f}, {row['first_trough_rho']:.3f})"
        )
    print(f"Saved curve-trend outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
