from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import relative_l2, solve_rk4, solve_rk4_2d
from fisher_kpp_rk4.config import initial_condition, initial_condition_2d

OUTPUT_DIR = PROJECT_ROOT / "outputs"


def run_case_1d(Nx: int, dt: float) -> dict[str, np.ndarray]:
    D = 1.0
    r = 1.0
    L = 200.0
    T = 20.0
    x = np.linspace(0.0, L, Nx)
    Nt = int(round(T / dt))
    dt = T / Nt
    return solve_rk4(
        x=x,
        dt=dt,
        Nt=Nt,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=1.0,
        right_bc=0.0,
        save_interval=T,
    )


def run_case_2d(grid: int, steps: int) -> dict[str, np.ndarray]:
    D = 0.02
    r = 3.0
    box = 1.0
    T = 0.5
    x = np.linspace(0.0, box, grid)
    dt = T / steps
    return solve_rk4_2d(
        x=x,
        y=x,
        dt=dt,
        Nt=steps,
        D=D,
        r=r,
        initial_condition=initial_condition_2d,
        save_interval=T,
    )


def resample_square(field: np.ndarray, src_x: np.ndarray, dst_x: np.ndarray) -> np.ndarray:
    tmp = np.vstack([np.interp(dst_x, src_x, field[:, j]) for j in range(field.shape[1])]).T
    return np.vstack([np.interp(dst_x, src_x, tmp[i, :]) for i in range(tmp.shape[0])])


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows_1d: list[dict[str, float | int | str]] = []
    ref_1d = run_case_1d(Nx=1601, dt=0.00125)
    for Nx, dt in [(201, 0.01), (401, 0.01), (801, 0.005)]:
        sol = run_case_1d(Nx=Nx, dt=dt)
        u_ref_interp = np.interp(sol["x"], ref_1d["x"], ref_1d["u_final"])
        err = relative_l2(sol["u_final"], u_ref_interp)
        row = {"dim": "1d", "grid": Nx, "dx": 200.0 / (Nx - 1), "dt": dt, "relative_l2": err}
        rows_1d.append(row)
        print(f"1D Nx={Nx:4d}, dx={row['dx']:.6g}, dt={dt:.6g}, relL2={err:.6e}")

    rows_2d: list[dict[str, float | int | str]] = []
    ref_2d = run_case_2d(grid=81, steps=640)
    for grid, steps in [(31, 160), (41, 240), (51, 320)]:
        sol = run_case_2d(grid=grid, steps=steps)
        ref_on_grid = resample_square(ref_2d["u_final"], ref_2d["x"], sol["x"])
        err = relative_l2(sol["u_final"], ref_on_grid)
        row = {"dim": "2d", "grid": grid, "dx": 1.0 / (grid - 1), "dt": 0.5 / steps, "relative_l2": err}
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

