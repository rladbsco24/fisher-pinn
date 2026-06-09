from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import solve_rk4, solve_rk4_2d
from fisher_kpp_rk4.config import (
    D,
    D_2D,
    T,
    T_2D,
    ablowitz_zeppetella_exact,
    ablowitz_zeppetella_exact_2d,
    initial_condition,
    initial_condition_2d,
    left_bc,
    r,
    r_2D,
    right_bc,
    x_left,
    x_left_2d,
    x_right,
    x_right_2d,
    y_bottom_2d,
    y_top_2d,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run_case_1d(Nx: int, dt: float) -> dict[str, np.ndarray]:
    x = np.linspace(x_left, x_right, Nx)
    Nt = int(round(T / dt))
    dt = T / Nt
    return solve_rk4(
        x=x,
        dt=dt,
        Nt=Nt,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=left_bc,
        right_bc=right_bc,
        save_interval=T,
        exact_solution=ablowitz_zeppetella_exact,
    )


def run_case_2d(grid: int, steps: int) -> dict[str, np.ndarray]:
    x = np.linspace(x_left_2d, x_right_2d, grid)
    y = np.linspace(y_bottom_2d, y_top_2d, grid)
    dt = T_2D / steps
    return solve_rk4_2d(
        x=x,
        y=y,
        dt=dt,
        Nt=steps,
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=T,
        boundary_condition="dirichlet_exact",
        exact_solution=ablowitz_zeppetella_exact_2d,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows_1d: list[dict[str, float | int | str]] = []
    for Nx, dt in [(201, 0.005), (401, 0.0025), (801, 0.00125)]:
        sol = run_case_1d(Nx=Nx, dt=dt)
        err = float(sol["relative_l2_final"])
        row = {"dim": "1d", "grid": Nx, "dx": float(sol["x"][1] - sol["x"][0]), "dt": dt, "relative_l2": err}
        rows_1d.append(row)
        print(f"1D Nx={Nx:4d}, dx={row['dx']:.6g}, dt={dt:.6g}, relL2={err:.6e}")

    rows_2d: list[dict[str, float | int | str]] = []
    for grid, steps in [(41, 300), (61, 300), (81, 600)]:
        sol = run_case_2d(grid=grid, steps=steps)
        err = float(sol["relative_l2_final"])
        row = {"dim": "2d", "grid": grid, "dx": float(sol["x"][1] - sol["x"][0]), "dt": T_2D / steps, "relative_l2": err}
        rows_2d.append(row)
        print(f"2D grid={grid:3d}, dx={row['dx']:.6g}, dt={row['dt']:.6g}, relL2={err:.6e}")

    rows = rows_1d + rows_2d
    with open(OUTPUT_DIR / "convergence_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dim", "grid", "dx", "dt", "relative_l2"])
        writer.writeheader()
        writer.writerows(rows)

    print("Saved: outputs/convergence_summary.csv")


if __name__ == "__main__":
    main()
