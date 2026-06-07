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

from fisher_kpp_rk4 import check_forward_euler_stability, check_rk4_stability, relative_l2, solve_1d_method
from fisher_kpp_rk4.config import (
    LONG_TIME_D,
    LONG_TIME_DT,
    LONG_TIME_DX,
    LONG_TIME_L,
    LONG_TIME_LEFT_BC,
    LONG_TIME_NT,
    LONG_TIME_NX,
    LONG_TIME_PROBE_X,
    LONG_TIME_R,
    LONG_TIME_RIGHT_BC,
    LONG_TIME_SAVE_INTERVAL,
    LONG_TIME_T,
    LONG_TIME_X,
    long_time_initial_condition,
)


METHODS = ("forward_euler", "backward_euler", "trapezoidal", "rk4")


def run_methods(*, save_interval: float = LONG_TIME_SAVE_INTERVAL) -> dict[str, dict[str, np.ndarray]]:
    return {
        method: solve_1d_method(
            method,
            x=LONG_TIME_X,
            dt=LONG_TIME_DT,
            Nt=LONG_TIME_NT,
            D=LONG_TIME_D,
            r=LONG_TIME_R,
            initial_condition=long_time_initial_condition,
            left_bc=LONG_TIME_LEFT_BC,
            right_bc=LONG_TIME_RIGHT_BC,
            save_interval=save_interval,
            probe_x=LONG_TIME_PROBE_X,
        )
        for method in METHODS
    }


def summarize(results: dict[str, dict[str, np.ndarray]]) -> list[dict[str, float | str | int]]:
    rk4_final = results["rk4"]["u_final"]
    rows: list[dict[str, float | str | int]] = []
    for method, result in results.items():
        newton_iters = result["newton_iterations"]
        newton_residual = result["newton_residual"]
        rows.append(
            {
                "method": method,
                "final_front": float(result["fronts"][-1]),
                "final_mass": float(result["mass"][-1]),
                "final_rho": float(result["rho"][-1]),
                "relative_l2_to_rk4_final": 0.0 if method == "rk4" else relative_l2(result["u_final"], rk4_final),
                "max_newton_iterations": int(newton_iters.max()) if len(newton_iters) else 0,
                "max_newton_residual": float(newton_residual.max()) if len(newton_residual) else 0.0,
            }
        )
    return rows


def write_outputs(out_dir: Path, results: dict[str, dict[str, np.ndarray]], rows: list[dict[str, float | str | int]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "long_time_method_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "final_front",
                "final_mass",
                "final_rho",
                "relative_l2_to_rk4_final",
                "max_newton_iterations",
                "max_newton_residual",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload: dict[str, np.ndarray] = {
        "x": LONG_TIME_X,
        "times": results["rk4"]["times"],
        "D": np.array(LONG_TIME_D),
        "r": np.array(LONG_TIME_R),
        "L": np.array(LONG_TIME_L),
        "T": np.array(LONG_TIME_T),
        "Nx": np.array(LONG_TIME_NX),
        "dt": np.array(LONG_TIME_DT),
        "dx": np.array(LONG_TIME_DX),
        "probe_x": np.array(LONG_TIME_PROBE_X),
    }
    for method, result in results.items():
        for key in ("snapshots", "fronts", "mass", "rho", "u_final", "newton_iterations", "newton_residual"):
            payload[f"{method}_{key}"] = result[key]
    np.savez(out_dir / "long_time_method_results.npz", **payload)


def save_plots(out_dir: Path, results: dict[str, dict[str, np.ndarray]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped plots.")
        return

    rk4 = results["rk4"]
    tt, xx = np.meshgrid(rk4["times"], rk4["x"], indexing="ij")
    fig = plt.figure(figsize=(9.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(tt, xx, rk4["snapshots"], cmap="viridis", linewidth=0.15, edgecolor="#343a40", alpha=0.92)
    ax.set_xlabel("time")
    ax.set_ylabel("space")
    ax.set_zlabel("u")
    ax.set_title("Adjusted RK4 long-time Fisher-KPP surface")
    fig.tight_layout()
    fig.savefig(out_dir / "adjusted_rk4_long_time_surface.png", dpi=180)
    plt.close(fig)

    plt.figure(figsize=(8.5, 5.2))
    for method, result in results.items():
        plt.plot(result["times"], result["rho"], label=method)
    plt.xlabel("time")
    plt.ylabel(f"rho(t)=u(x={LONG_TIME_PROBE_X:g}, t)")
    plt.title("Long-time probe trend under a fair parameter set")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "long_time_probe_rho.png", dpi=200)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for method, result in results.items():
        axes[0].plot(result["times"], result["fronts"], label=method)
        axes[1].plot(result["times"], result["mass"], label=method)
    axes[0].set_title("u=0.5 front position")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("x")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("mean density")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("mean u")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.savefig(out_dir / "long_time_front_mass.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(8.5, 5.2))
    for method, result in results.items():
        plt.plot(result["x"], result["u_final"], label=method)
    plt.xlabel("space")
    plt.ylabel("u(x,T)")
    plt.title("Final profiles at t=30")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "long_time_final_profiles.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FE, BE, trapezoidal, and RK4 on a long-time Fisher-KPP setup.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "long_time_methods")
    parser.add_argument("--save-interval", type=float, default=LONG_TIME_SAVE_INTERVAL)
    args = parser.parse_args()

    fe_info = check_forward_euler_stability(LONG_TIME_DX, LONG_TIME_DT, LONG_TIME_D, LONG_TIME_R, dim=1)
    rk4_info = check_rk4_stability(LONG_TIME_DX, LONG_TIME_DT, LONG_TIME_D, LONG_TIME_R, dim=1)
    print("=== Fair long-time Fisher-KPP parameter set ===")
    print(f"D={LONG_TIME_D}, r={LONG_TIME_R}, L={LONG_TIME_L}, T={LONG_TIME_T}")
    print(f"Nx={LONG_TIME_NX}, dx={LONG_TIME_DX:.6g}, Nt={LONG_TIME_NT}, dt={LONG_TIME_DT:.6g}")
    print(f"Forward Euler practical stability: {fe_info['is_practically_safe']} (limit={fe_info['dt_practical']:.6g})")
    print(f"RK4 practical stability: {rk4_info['is_practically_safe']} (limit={rk4_info['dt_practical']:.6g})")
    print("Note: Fisher-KPP has a maximum principle; long-time trends are front/mass/probe trends, not intrinsic oscillations.")

    results = run_methods(save_interval=args.save_interval)
    rows = summarize(results)
    write_outputs(args.output_dir, results, rows)
    save_plots(args.output_dir, results)
    for row in rows:
        print(
            f"{row['method']:>14s} final_front={row['final_front']:.4f} "
            f"final_mass={row['final_mass']:.4f} rho={row['final_rho']:.4f} "
            f"relL2_to_RK4={row['relative_l2_to_rk4_final']:.3e}"
        )
    print(f"Saved long-time comparison outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
