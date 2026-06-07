from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import check_rk4_stability, solve_1d_method
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the adjusted long-time RK4 Fisher-KPP configuration.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "long_time_rk4_adjusted")
    parser.add_argument("--save-interval", type=float, default=LONG_TIME_SAVE_INTERVAL)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    info = check_rk4_stability(LONG_TIME_DX, LONG_TIME_DT, LONG_TIME_D, LONG_TIME_R, dim=1)
    print("=== Adjusted long-time RK4 Fisher-KPP ===")
    print(f"D={LONG_TIME_D}, r={LONG_TIME_R}, L={LONG_TIME_L}, T={LONG_TIME_T}")
    print(f"Nx={LONG_TIME_NX}, dx={LONG_TIME_DX:.6g}, Nt={LONG_TIME_NT}, dt={LONG_TIME_DT:.6g}")
    print(f"RK4 practical stability: {info['is_practically_safe']} (limit={info['dt_practical']:.6g})")

    result = solve_1d_method(
        "rk4",
        x=LONG_TIME_X,
        dt=LONG_TIME_DT,
        Nt=LONG_TIME_NT,
        D=LONG_TIME_D,
        r=LONG_TIME_R,
        initial_condition=long_time_initial_condition,
        left_bc=LONG_TIME_LEFT_BC,
        right_bc=LONG_TIME_RIGHT_BC,
        save_interval=args.save_interval,
        probe_x=LONG_TIME_PROBE_X,
    )
    np.savez(
        args.output_dir / "adjusted_rk4_long_time_results.npz",
        **result,
        D=LONG_TIME_D,
        r=LONG_TIME_R,
        L=LONG_TIME_L,
        T=LONG_TIME_T,
        Nx=LONG_TIME_NX,
        dx=LONG_TIME_DX,
        dt=LONG_TIME_DT,
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped plots.")
        return

    tt, xx = np.meshgrid(result["times"], result["x"], indexing="ij")
    fig = plt.figure(figsize=(9.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(tt, xx, result["snapshots"], cmap="viridis", linewidth=0.15, edgecolor="#343a40", alpha=0.92)
    ax.set_xlabel("time")
    ax.set_ylabel("space")
    ax.set_zlabel("u")
    ax.set_title("Adjusted RK4 long-time Fisher-KPP surface")
    fig.tight_layout()
    fig.savefig(args.output_dir / "adjusted_rk4_long_time_surface.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    axes[0].plot(result["times"], result["rho"])
    axes[0].set_title(f"rho(t)=u(x={LONG_TIME_PROBE_X:g},t)")
    axes[0].set_xlabel("time")
    axes[0].grid(alpha=0.25)
    axes[1].plot(result["times"], result["fronts"])
    axes[1].set_title("front position")
    axes[1].set_xlabel("time")
    axes[1].grid(alpha=0.25)
    axes[2].plot(result["times"], result["mass"])
    axes[2].set_title("mean density")
    axes[2].set_xlabel("time")
    axes[2].grid(alpha=0.25)
    fig.savefig(args.output_dir / "adjusted_rk4_long_time_trends.png", dpi=200)
    plt.close(fig)

    print(f"Saved adjusted RK4 outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
