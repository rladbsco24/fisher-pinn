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

from fisher_kpp_rk4 import solve_rk4, solve_rk4_2d
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
    "min_u",
    "max_u",
    "mean_u",
    "AZ_Relative_L2_Error",
]

TWO_D_COLUMNS = [
    "Nx",
    "Ny",
    "dx",
    "dy",
    "dt",
    "Nt",
    "T",
    "runtime_sec",
    "min_U",
    "max_U",
    "mean_U",
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
        "Nx": int(grid),
        "Ny": int(grid),
        "dx": dx,
        "dy": float(y[1] - y[0]),
        "dt": float(dt_eff),
        "Nt": int(nt),
        "T": float(T_2D),
        "runtime_sec": float(runtime),
        "min_U": float(final.min()),
        "max_U": float(final.max()),
        "mean_U": float(final.mean()),
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


def _decimal_places(value: float, *, max_places: int = 6) -> int:
    text = f"{float(value):.{max_places}f}".rstrip("0").rstrip(".")
    if "." not in text:
        return 0
    return len(text.split(".", 1)[1])


def _display_float(column: str, value: object, places_by_column: dict[str, int]) -> str:
    number = float(value)
    if column == "runtime_sec":
        return f"{number:.6f}"
    if column == "T":
        return f"{number:.1f}"
    if column in {"dx", "dy", "dt"}:
        return f"{number:.{places_by_column.get(column, 3)}f}"
    if column in {"min_u", "mean_u", "min_U", "max_U", "mean_U"}:
        return f"{number:.6f}"
    if column in {"max_u"}:
        if abs(number - 1.0) < 5.0e-6:
            return "1.0"
        return f"{number:.6f}"
    if column in {"AZ_Relative_L2_Error", "Exact_Relative_L2_Error"}:
        return f"{number:.6f}"
    return _format_value(number)


def _display_frame_rows(rows: list[dict[str, object]], columns: list[str]) -> list[dict[str, str]]:
    places_by_column: dict[str, int] = {}
    for column in ("dx", "dy", "dt"):
        values = [float(row[column]) for row in rows if column in row]
        if values:
            places_by_column[column] = max(1, max(_decimal_places(value, max_places=6) for value in values))
    formatted: list[dict[str, str]] = []
    for row in rows:
        formatted_row: dict[str, str] = {}
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, bool):
                formatted_row[column] = "True" if value else "False"
            elif isinstance(value, (int, np.integer)):
                formatted_row[column] = str(int(value))
            elif isinstance(value, (float, np.floating)):
                formatted_row[column] = _display_float(column, value, places_by_column)
            else:
                formatted_row[column] = str(value)
        formatted.append(formatted_row)
    return formatted


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


def _column_widths(frame, columns: list[str]) -> list[float]:
    weights: list[float] = []
    for column in columns:
        text_width = max(len(str(column)), *(len(str(value)) for value in frame[column].tolist()))
        weights.append(max(0.70, min(3.20, text_width / 8.2)))
    total = sum(weights)
    return [weight / total for weight in weights]


def _write_png_table(
    path: Path,
    title: str,
    rows: list[dict[str, object]],
    columns: list[str],
    *,
    show_index: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        return

    frame = pd.DataFrame(_display_frame_rows(rows, columns), columns=columns)
    display_columns = list(columns)
    if show_index:
        frame.insert(0, "", [str(index) for index in range(len(frame))])
        display_columns = [""] + display_columns

    col_widths = _column_widths(frame, display_columns)
    total_chars = sum(max(len(str(column)), *(len(str(value)) for value in frame[column].tolist())) for column in display_columns)
    width = max(12.0, min(24.0, total_chars * 0.155))
    height = max(1.55, 0.43 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("#24272a")
    ax.set_facecolor("#24272a")
    ax.axis("off")
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    table = ax.table(
        cellText=frame.to_numpy(),
        colLabels=display_columns,
        colWidths=col_widths,
        cellLoc="right",
        colLoc="right",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.PAD = 0.016
        cell.set_edgecolor("#24272a")
        cell.set_linewidth(0.0)
        cell.get_text().set_fontfamily("monospace")
        cell.get_text().set_clip_on(False)
        if row_idx == 0:
            cell.set_facecolor("#24272a")
            cell.get_text().set_color("#f2f2f2")
            cell.get_text().set_weight("bold")
            cell.get_text().set_fontsize(10.5)
            cell.get_text().set_ha("right")
        else:
            cell.set_facecolor("#4a4a4a" if row_idx % 2 else "#282b2e")
            cell.get_text().set_color("#e5e5e5")
            cell.get_text().set_fontsize(10.0)
            visible_column = display_columns[col_idx]
            if visible_column in {"dim", "method", "converged_all"}:
                cell.get_text().set_ha("center")
            else:
                cell.get_text().set_ha("right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def _write_table_bundle(
    name: str,
    title: str,
    rows: list[dict[str, object]],
    columns: list[str],
    *,
    show_index: bool,
) -> None:
    _write_csv(TABLE_DIR / f"{name}.csv", rows, columns)
    _write_markdown(TABLE_DIR / f"{name}.md", title, rows, columns)
    _write_png_table(TABLE_DIR / f"{name}.png", title, rows, columns, show_index=show_index)
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
    # time table therefore drops that unavailable row instead of replacing it.
    two_d_time = [run_case_2d(grid=121, dt=dt_value) for dt_value in (0.02, 0.01, 0.005)]

    bundles = [
        ("rk4_1d_spatial_comparison", "1D spatial comparison", one_d_spatial, ONE_D_COLUMNS, True),
        ("rk4_1d_time_comparison", "1D time comparison", one_d_time, ONE_D_COLUMNS, True),
        ("rk4_2d_spatial_comparison", "2D spatial comparison", two_d_spatial, TWO_D_COLUMNS, False),
        ("rk4_2d_time_comparison", "2D time comparison", two_d_time, TWO_D_COLUMNS, False),
    ]
    for name, title, rows, columns, show_index in bundles:
        _write_table_bundle(name, title, rows, columns, show_index=show_index)

    _write_combined_markdown((title, rows, columns) for _, title, rows, columns, _ in bundles)

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
                    "dim": "2D",
                    "method": "rk4",
                    "Nx": row["Nx"],
                    "Ny": row["Ny"],
                    "dx": row["dx"],
                    "dt": row["dt"],
                    "Nt": row["Nt"],
                    "T": row["T"],
                    "Exact_Relative_L2_Error": row["Exact_Relative_L2_Error"],
                }
            )
    _write_csv(OUTPUT_DIR / "convergence_summary.csv", summary_rows, sorted({key for row in summary_rows for key in row}))

    print(f"Saved RK4 comparison tables under {TABLE_DIR}")
    print(f"Saved compact summary: {OUTPUT_DIR / 'convergence_summary.csv'}")


if __name__ == "__main__":
    main()
