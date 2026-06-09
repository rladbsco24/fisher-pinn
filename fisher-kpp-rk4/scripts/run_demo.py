from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import check_rk4_stability, estimate_front_speed, solve_rk4, solve_rk4_2d
from fisher_kpp_rk4.config import (
    BOX_2D,
    D,
    D_2D,
    GRID_2D,
    L,
    Nt,
    Nt_2d,
    Nx,
    T,
    T_2D,
    ablowitz_zeppetella_exact,
    ablowitz_zeppetella_exact_2d,
    dt,
    dt_2d,
    dx,
    dx_2d,
    initial_condition,
    initial_condition_2d,
    left_bc,
    r,
    r_2D,
    right_bc,
    x,
    x_2d,
    y_2d,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run_1d() -> dict[str, np.ndarray]:
    info = check_rk4_stability(dx=dx, dt=dt, D=D, r=r, dim=1)
    print("=== 1D Fisher-KPP RK4 ===")
    print(f"D={D}, r={r}, L={L}, T={T}")
    print(f"Nx={Nx}, dx={dx:.6g}, Nt={Nt}, dt={dt:.6g}")
    print(f"Practical dt safe? {info['is_practically_safe']} (limit={info['dt_practical']:.6g})")

    result = solve_rk4(
        x=x,
        dt=dt,
        Nt=Nt,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=left_bc,
        right_bc=right_bc,
        save_interval=1.0,
        exact_solution=ablowitz_zeppetella_exact,
    )
    c_exact = 5.0 / np.sqrt(6.0)
    c_num = estimate_front_speed(result["times"], result["fronts"], t_min=1.0, x_max=x[-1])
    print(f"Ablowitz-Zeppetella exact speed c = {c_exact:.6g}")
    print(f"Estimated front speed = {c_num:.6g}")
    print(f"Final relative L2 vs exact = {float(result['relative_l2_final']):.3e}")

    np.savez(OUTPUT_DIR / "fisher_kpp_rk4_1d_results.npz", **result, D=D, r=r, L=L, T=T, dx=dx, dt=dt)
    return result


def run_2d() -> dict[str, np.ndarray]:
    info = check_rk4_stability(dx=dx_2d, dt=dt_2d, D=D_2D, r=r_2D, dim=2)
    print("\n=== 2D Fisher-KPP RK4 ===")
    print(f"D={D_2D}, r={r_2D}, box={BOX_2D}, T={T_2D}")
    print(f"grid={GRID_2D}x{GRID_2D}, dx={dx_2d:.6g}, Nt={Nt_2d}, dt={dt_2d:.6g}")
    print(f"Practical dt safe? {info['is_practically_safe']} (limit={info['dt_practical']:.6g})")

    result = solve_rk4_2d(
        x=x_2d,
        y=y_2d,
        dt=dt_2d,
        Nt=Nt_2d,
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=0.05,
        boundary_condition="dirichlet_exact",
        exact_solution=ablowitz_zeppetella_exact_2d,
    )
    print(f"Final mean mass = {result['mass'][-1]:.6g}")
    print(f"Final area u>=0.05 = {result['area_ge_0.05'][-1]:.6g}")
    print(f"Final area u>=0.10 = {result['area_ge_0.10'][-1]:.6g}")
    print(f"Final relative L2 vs exact = {float(result['relative_l2_final']):.3e}")

    np.savez(OUTPUT_DIR / "fisher_kpp_rk4_2d_results.npz", **result, D=D_2D, r=r_2D, box=BOX_2D, T=T_2D, dx=dx_2d, dt=dt_2d)
    return result


def save_plots(result_1d: dict[str, np.ndarray], result_2d: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipped plots.")
        return

    plt.figure(figsize=(8, 5))
    for t, u in zip(result_1d["times"], result_1d["snapshots"]):
        plt.plot(result_1d["x"], u, label=f"t={t:.0f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.ylim(-0.05, 1.05)
    plt.title("1D Fisher-KPP solved by MOL-FDM + RK4")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "snapshots_1d.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(result_1d["times"], result_1d["fronts"], marker="o")
    plt.xlabel("t")
    plt.ylabel("front position, u=0.5")
    plt.title("1D front propagation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "front_position_1d.png", dpi=200)
    plt.close()

    snapshots = result_2d["snapshots"]
    times = result_2d["times"]
    panel_idx = np.unique(np.linspace(0, len(times) - 1, 4, dtype=int))
    fig, axes = plt.subplots(1, len(panel_idx), figsize=(4.0 * len(panel_idx), 3.4), constrained_layout=True)
    if len(panel_idx) == 1:
        axes = [axes]
    for ax, idx in zip(axes, panel_idx):
        im = ax.imshow(
            snapshots[idx].T,
            origin="lower",
            extent=[result_2d["x"][0], result_2d["x"][-1], result_2d["y"][0], result_2d["y"][-1]],
            vmin=0.0,
            vmax=1.0,
            cmap="magma",
        )
        ax.set_title(f"t={times[idx]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.82)
    fig.suptitle("2D Fisher-KPP RK4 snapshots")
    fig.savefig(OUTPUT_DIR / "snapshots_2d.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(7, 4))
    plt.plot(result_2d["times"], result_2d["mass"], label="mean mass")
    plt.plot(result_2d["times"], result_2d["area_ge_0.05"], label="area u>=0.05")
    plt.plot(result_2d["times"], result_2d["area_ge_0.10"], label="area u>=0.10")
    plt.xlabel("t")
    plt.ylabel("fraction")
    plt.title("2D mass and front-area diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "front_area_mass_2d.png", dpi=200)
    plt.close()

    print("Saved PNGs under outputs/.")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    result_1d = run_1d()
    result_2d = run_2d()
    save_plots(result_1d, result_2d)
    print("Saved NPZ results under outputs/.")


if __name__ == "__main__":
    main()
