from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import check_rk4_stability, solve_rk4, solve_rk4_2d
from fisher_kpp_rk4.config import (
    D,
    D_2D,
    T,
    T_2D,
    ablowitz_zeppetella_exact,
    generalized_fisher_kpp_exact_2d,
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
TABLE_DIR = OUTPUT_DIR / "tables"
RK4_STAGES_PER_STEP = 4.0
REQUIRED_TABLE_PNGS = [
    "rk4_1d_spatial_comparison.png",
    "rk4_1d_time_comparison.png",
    "rk4_2d_spatial_comparison.png",
    "rk4_2d_time_comparison.png",
]

ONE_D_COLUMNS = [
    "dim",
    "method",
    "Nx",
    "dx",
    "dt",
    "Nt",
    "T",
    "runtime_sec",
    "rk4_stages_per_step",
    "stability_safe",
    "min_u",
    "max_u",
    "mean_u",
    "AZ_Relative_L2_Error",
]

TWO_D_COLUMNS = [
    "dim",
    "method",
    "Nx",
    "Ny",
    "dx",
    "dy",
    "dt",
    "Nt",
    "T",
    "runtime_sec",
    "rk4_stages_per_step",
    "stability_safe",
    "min_u",
    "max_u",
    "mean_u",
    "Exact_Relative_L2_Error",
]


def _round_dt(t_final: float, dt: float) -> tuple[int, float]:
    nt = int(round(t_final / dt))
    if nt <= 0:
        raise ValueError("time step produces a non-positive step count")
    return nt, float(t_final / nt)


def run_case_1d(nx: int, dt: float) -> dict[str, object]:
    x = np.linspace(x_left, x_right, nx)
    nt, dt_eff = _round_dt(T, dt)
    stability = check_rk4_stability(dx=float(x[1] - x[0]), dt=dt_eff, D=D, r=r, dim=1)
    start = time.perf_counter()
    sol = solve_rk4(
        x=x,
        dt=dt_eff,
        Nt=nt,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=left_bc,
        right_bc=right_bc,
        save_interval=T,
        exact_solution=ablowitz_zeppetella_exact,
    )
    runtime = time.perf_counter() - start
    final = np.asarray(sol["u_final"], dtype=np.float64)
    return {
        "dim": "1D",
        "method": "rk4",
        "Nx": int(nx),
        "dx": float(sol["x"][1] - sol["x"][0]),
        "dt": float(dt_eff),
        "Nt": int(nt),
        "T": float(T),
        "runtime_sec": float(runtime),
        "rk4_stages_per_step": RK4_STAGES_PER_STEP,
        "stability_safe": bool(stability["is_practically_safe"]),
        "min_u": float(final.min()),
        "max_u": float(final.max()),
        "mean_u": float(final.mean()),
        "AZ_Relative_L2_Error": float(sol["relative_l2_final"]),
    }


def run_case_2d(grid: int, dt: float) -> dict[str, object]:
    x = np.linspace(x_left_2d, x_right_2d, grid)
    y = np.linspace(y_bottom_2d, y_top_2d, grid)
    nt, dt_eff = _round_dt(T_2D, dt)
    dx = float(x[1] - x[0])
    stability = check_rk4_stability(dx=dx, dt=dt_eff, D=D_2D, r=r_2D, dim=2)
    start = time.perf_counter()
    sol = solve_rk4_2d(
        x=x,
        y=y,
        dt=dt_eff,
        Nt=nt,
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=T_2D,
        boundary_condition="dirichlet_exact",
        exact_solution=generalized_fisher_kpp_exact_2d,
    )
    runtime = time.perf_counter() - start
    final = np.asarray(sol["u_final"], dtype=np.float64)
    return {
        "dim": "2D",
        "method": "rk4",
        "Nx": int(grid),
        "Ny": int(grid),
        "dx": dx,
        "dy": float(y[1] - y[0]),
        "dt": float(dt_eff),
        "Nt": int(nt),
        "T": float(T_2D),
        "runtime_sec": float(runtime),
        "rk4_stages_per_step": RK4_STAGES_PER_STEP,
        "stability_safe": bool(stability["is_practically_safe"]),
        "min_u": float(final.min()),
        "max_u": float(final.max()),
        "mean_u": float(final.mean()),
        "Exact_Relative_L2_Error": float(sol["relative_l2_final"]),
    }


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if abs(number) >= 1.0e4 or (0.0 < abs(number) < 1.0e-4):
            return f"{number:.6e}"
        if abs(number) < 1.0:
            return f"{number:.6f}".rstrip("0").rstrip(".")
        return f"{number:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = path
    try:
        f = target.open("w", encoding="utf-8", newline="")
    except PermissionError:
        target = path.with_suffix(path.suffix + ".new")
        f = target.open("w", encoding="utf-8", newline="")
        print(f"Warning: {path} is locked; wrote {target} instead.")
    with f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(title: str, rows: list[dict[str, object]], columns: list[str]) -> str:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(column, "")) for column in columns) + " |")
    lines.append("")
    return "\n".join(lines)


def _write_markdown(path: Path, title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_markdown_table(title, rows, columns), encoding="utf-8")


def _print_console_table(title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    formatted_rows = [[_format_value(row.get(column, "")) for column in columns] for row in rows]
    widths = [
        max(len(column), *(len(row[idx]) for row in formatted_rows))
        for idx, column in enumerate(columns)
    ]
    line = " ".join("=" * width for width in widths)
    print(line)
    print(title)
    print(line)
    print(" ".join(column.rjust(widths[idx]) for idx, column in enumerate(columns)))
    for row in formatted_rows:
        print(" ".join(value.rjust(widths[idx]) for idx, value in enumerate(row)))
    print()


def _write_png_table(path: Path, title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    data = [[_format_value(row.get(column, "")) for column in columns] for row in rows]
    width = max(12.0, 0.70 * len(columns))
    height = max(2.4, 0.48 * (len(rows) + 2))
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#24272a")
    ax.set_facecolor("#24272a")
    ax.axis("off")
    ax.set_title(title, color="#f2f2f2", fontsize=13, fontfamily="monospace", pad=12, loc="left")
    table = ax.table(
        cellText=data,
        colLabels=columns,
        cellLoc="right",
        colLoc="right",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.42)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#24272a")
        cell.get_text().set_fontfamily("monospace")
        if row_idx == 0:
            cell.set_facecolor("#24272a")
            cell.get_text().set_color("#f2f2f2")
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor("#4a4a4a" if row_idx % 2 else "#282b2e")
            cell.get_text().set_color("#e5e5e5")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _write_table_bundle(name: str, title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    _write_csv(TABLE_DIR / f"{name}.csv", rows, columns)
    _write_markdown(TABLE_DIR / f"{name}.md", title, rows, columns)
    _write_png_table(TABLE_DIR / f"{name}.png", title, rows, columns)
    _print_console_table(title, rows, columns)


def _write_combined_markdown(tables: Iterable[tuple[str, list[dict[str, object]], list[str]]]) -> None:
    parts = ["# RK4 Grid Comparison Tables", ""]
    for title, rows, columns in tables:
        parts.append(_markdown_table(title, rows, columns))
    (TABLE_DIR / "rk4_grid_comparison_tables.md").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    one_d_spatial = [run_case_1d(nx=nx, dt=0.005) for nx in (101, 201, 401)]
    one_d_time = [run_case_1d(nx=201, dt=dt_value) for dt_value in (0.02, 0.01, 0.005, 0.0025)]
    two_d_spatial = [run_case_2d(grid=grid, dt=0.01) for grid in (41, 61, 81)]
    # With Nx=Ny=121, dt=0.04 is outside the explicit RK4 diffusion limit. The
    # time table therefore starts at dt=0.02, the largest practical stable value.
    two_d_time = [run_case_2d(grid=121, dt=dt_value) for dt_value in (0.02, 0.01, 0.005, 0.0025)]

    bundles = [
        ("rk4_1d_spatial_comparison", "1D RK4 spatial comparison", one_d_spatial, ONE_D_COLUMNS),
        ("rk4_1d_time_comparison", "1D RK4 time-step comparison", one_d_time, ONE_D_COLUMNS),
        ("rk4_2d_spatial_comparison", "2D RK4 spatial comparison", two_d_spatial, TWO_D_COLUMNS),
        ("rk4_2d_time_comparison", "2D RK4 time-step comparison", two_d_time, TWO_D_COLUMNS),
    ]
    for name, title, rows, columns in bundles:
        _write_table_bundle(name, title, rows, columns)

    _write_combined_markdown((title, rows, columns) for _, title, rows, columns in bundles)

    summary_rows: list[dict[str, object]] = []
    for rows in (one_d_spatial, one_d_time):
        for row in rows:
            summary_rows.append(
                {key: row[key] for key in ("dim", "method", "Nx", "dx", "dt", "Nt", "T", "AZ_Relative_L2_Error")}
            )
    for rows in (two_d_spatial, two_d_time):
        for row in rows:
            summary_rows.append(
                {
                    key: row[key]
                    for key in ("dim", "method", "Nx", "Ny", "dx", "dt", "Nt", "T", "Exact_Relative_L2_Error")
                }
            )
    _write_csv(OUTPUT_DIR / "convergence_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))

    print(f"Saved RK4 comparison tables under {TABLE_DIR}")
    print(f"Saved compact summary: {OUTPUT_DIR / 'convergence_summary.csv'}")


if __name__ == "__main__":
    main()
