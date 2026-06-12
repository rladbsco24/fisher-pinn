from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RK4_ROOT = ROOT / "fisher-kpp-rk4"
RK4_SRC = RK4_ROOT / "src"
if str(RK4_SRC) not in sys.path:
    sys.path.insert(0, str(RK4_SRC))

from fisher_kpp_rk4 import relative_l2, solve_1d_method, solve_2d_method
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

ONE_D_REFERENCE_COLUMNS = [
    "dim",
    "method",
    "Nx",
    "dx",
    "dt",
    "dt_ref",
    "Nt",
    "T",
    "runtime_sec",
    "Reference_Relative_L2_Error",
    "Observed_Time_Order",
]

TWO_D_REFERENCE_COLUMNS = [
    "Nx",
    "Ny",
    "dx",
    "dy",
    "dt",
    "dt_ref",
    "Nt",
    "T",
    "runtime_sec",
    "Reference_Relative_L2_Error",
    "Observed_Time_Order",
]


def _round_dt(t_final: float, dt: float) -> tuple[int, float]:
    nt = int(round(t_final / dt))
    if nt <= 0:
        raise ValueError("time step produces a non-positive step count")
    return nt, float(t_final / nt)


def _method_label(method: str) -> str:
    normalized = method.lower().replace("-", "_").replace(" ", "_")
    if normalized in {"fe", "forward", "forward_euler"}:
        return "forward_euler"
    if normalized == "rk4":
        return "rk4"
    raise ValueError(f"Unsupported method {method!r}")


def run_case_1d(method: str, nx: int, dt: float) -> dict[str, object]:
    method_name = _method_label(method)
    x = np.linspace(x_left, x_right, nx)
    nt, dt_eff = _round_dt(T, dt)
    start = time.perf_counter()
    sol = solve_1d_method(
        method_name,
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
    dx = float(x[1] - x[0])
    return {
        "dim": "1D",
        "method": method_name,
        "Nx": int(nx),
        "dx": dx,
        "dt": float(dt_eff),
        "Nt": int(nt),
        "T": float(T),
        "runtime_sec": float(runtime),
        "min_u": float(final.min()),
        "max_u": float(final.max()),
        "mean_u": float(final.mean()),
        "AZ_Relative_L2_Error": float(sol["relative_l2_final"]),
    }


def run_case_2d(method: str, grid: int, dt: float) -> dict[str, object]:
    method_name = _method_label(method)
    x = np.linspace(x_left_2d, x_right_2d, grid)
    y = np.linspace(y_bottom_2d, y_top_2d, grid)
    nt, dt_eff = _round_dt(T_2D, dt)
    dx = float(x[1] - x[0])
    start = time.perf_counter()
    sol = solve_2d_method(
        method_name,
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


def _observed_orders(rows: list[dict[str, object]], *, error_key: str) -> list[dict[str, object]]:
    for idx, row in enumerate(rows):
        order = None
        if idx + 1 < len(rows):
            coarse_error = float(row[error_key])
            fine_error = float(rows[idx + 1][error_key])
            coarse_dt = float(row["dt"])
            fine_dt = float(rows[idx + 1]["dt"])
            if coarse_error > 0.0 and fine_error > 0.0 and coarse_dt > fine_dt:
                order = float(np.log(coarse_error / fine_error) / np.log(coarse_dt / fine_dt))
        row["Observed_Time_Order"] = order
    return rows


def run_temporal_reference_1d(
    method: str,
    *,
    nx: int = 201,
    dt_values: tuple[float, ...] = (0.02, 0.01, 0.005, 0.0025),
    dt_ref: float = 0.00125,
) -> list[dict[str, object]]:
    method_name = _method_label(method)
    x = np.linspace(x_left, x_right, nx)
    nt_ref, dt_ref_eff = _round_dt(T, dt_ref)
    ref = solve_1d_method(
        method_name,
        x=x,
        dt=dt_ref_eff,
        Nt=nt_ref,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=left_bc,
        right_bc=right_bc,
        save_interval=T,
        exact_solution=ablowitz_zeppetella_exact,
    )
    ref_final = np.asarray(ref["u_final"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for dt in dt_values:
        nt, dt_eff = _round_dt(T, dt)
        start = time.perf_counter()
        sol = solve_1d_method(
            method_name,
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
        rows.append(
            {
                "dim": "1D",
                "method": method_name,
                "Nx": int(nx),
                "dx": float(x[1] - x[0]),
                "dt": float(dt_eff),
                "dt_ref": float(dt_ref_eff),
                "Nt": int(nt),
                "T": float(T),
                "runtime_sec": float(runtime),
                "Reference_Relative_L2_Error": float(relative_l2(final, ref_final)),
                "Observed_Time_Order": None,
            }
        )
    return _observed_orders(rows, error_key="Reference_Relative_L2_Error")


def run_temporal_reference_2d(
    method: str,
    *,
    grid: int = 81,
    dt_values: tuple[float, ...] = (0.01, 0.005, 0.0025),
    dt_ref: float = 0.00125,
) -> list[dict[str, object]]:
    method_name = _method_label(method)
    x = np.linspace(x_left_2d, x_right_2d, grid)
    y = np.linspace(y_bottom_2d, y_top_2d, grid)
    nt_ref, dt_ref_eff = _round_dt(T_2D, dt_ref)
    ref = solve_2d_method(
        method_name,
        x=x,
        y=y,
        dt=dt_ref_eff,
        Nt=nt_ref,
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=T_2D,
        boundary_condition="dirichlet_exact",
        exact_solution=generalized_fisher_kpp_exact_2d,
    )
    ref_final = np.asarray(ref["u_final"], dtype=np.float64)
    rows: list[dict[str, object]] = []
    for dt in dt_values:
        nt, dt_eff = _round_dt(T_2D, dt)
        start = time.perf_counter()
        sol = solve_2d_method(
            method_name,
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
        rows.append(
            {
                "Nx": int(grid),
                "Ny": int(grid),
                "dx": float(x[1] - x[0]),
                "dy": float(y[1] - y[0]),
                "dt": float(dt_eff),
                "dt_ref": float(dt_ref_eff),
                "Nt": int(nt),
                "T": float(T_2D),
                "runtime_sec": float(runtime),
                "Reference_Relative_L2_Error": float(relative_l2(final, ref_final)),
                "Observed_Time_Order": None,
            }
        )
    return _observed_orders(rows, error_key="Reference_Relative_L2_Error")


def _format_value(column: str, value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if column in {"runtime_sec"}:
            return f"{number:.6f}"
        if column in {"T"}:
            return f"{number:.1f}"
        if column in {"dx", "dy"}:
            return f"{number:.3f}"
        if column in {"dt", "dt_ref"}:
            return f"{number:.4f}".rstrip("0").rstrip(".")
        if "Relative_L2_Error" in column:
            return f"{number:.6e}" if number < 1.0e-4 else f"{number:.6f}"
        if column == "Observed_Time_Order":
            return "" if not math.isfinite(number) else f"{number:.3f}"
        if column.startswith("min_") or column.startswith("max_") or column.startswith("mean_"):
            return f"{number:.6f}"
        return f"{number:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _display_frame(rows: list[dict[str, object]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{column: _format_value(column, row.get(column, "")) for column in columns} for row in rows], columns=columns)


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, title: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    frame = _display_frame(rows, columns)
    lines = [f"# {title}", "", "| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in frame.columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _column_widths(frame: pd.DataFrame, columns: list[str]) -> list[float]:
    weights: list[float] = []
    for column in columns:
        text_width = max(len(str(column)), *(len(str(value)) for value in frame[column].tolist()))
        weights.append(max(0.70, min(3.20, text_width / 8.2)))
    total = sum(weights)
    return [weight / total for weight in weights]


def _standard_table_png(path: Path, rows: list[dict[str, object]], columns: list[str], *, show_index: bool) -> None:
    frame = _display_frame(rows, columns)
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
    out_dir: Path,
    name: str,
    title: str,
    rows: list[dict[str, object]],
    columns: list[str],
    *,
    show_index: bool,
) -> Path:
    _write_csv(out_dir / f"{name}.csv", rows, columns)
    _write_markdown(out_dir / f"{name}.md", title, rows, columns)
    png = out_dir / f"{name}.png"
    _standard_table_png(png, rows, columns, show_index=show_index)
    return png


def _contact_sheet(paths: list[Path], out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    if not paths:
        return
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    thumbs = []
    for path in paths:
        with Image.open(path) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((420, 240), Image.LANCZOS)
            canvas = Image.new("RGB", (460, 300), "white")
            canvas.paste(thumb, ((460 - thumb.width) // 2, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 14), path.stem, fill=(20, 20, 20), font=font)
            thumbs.append(canvas)
    cols = 2
    rows = int(math.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * 460, rows * 300), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 460, (idx // cols) * 300))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=95)


def _method_tables(method: str, out_dir: Path) -> list[Path]:
    method_name = _method_label(method)
    method_dir = out_dir / method_name
    pngs: list[Path] = []

    one_d_spatial = [run_case_1d(method_name, nx=nx, dt=0.005) for nx in (101, 201, 401)]
    one_d_time = [run_case_1d(method_name, nx=201, dt=dt_value) for dt_value in (0.02, 0.01, 0.005, 0.0025)]
    one_d_ref = run_temporal_reference_1d(method_name)

    if method_name == "forward_euler":
        two_d_time_values = (0.01, 0.005, 0.0025, 0.00125)
        two_d_ref_values = (0.01, 0.005, 0.0025)
        two_d_ref_dt = 0.00125
    else:
        two_d_time_values = (0.02, 0.01, 0.005, 0.0025)
        two_d_ref_values = (0.02, 0.01, 0.005)
        two_d_ref_dt = 0.0025

    two_d_spatial = [run_case_2d(method_name, grid=grid, dt=0.01) for grid in (41, 61, 81)]
    two_d_time = [run_case_2d(method_name, grid=121, dt=dt_value) for dt_value in two_d_time_values]
    two_d_ref = run_temporal_reference_2d(method_name, grid=81, dt_values=two_d_ref_values, dt_ref=two_d_ref_dt)

    bundles = [
        ("1d_spatial_comparison", "1D spatial comparison", one_d_spatial, ONE_D_COLUMNS, True),
        ("1d_time_comparison", "1D time comparison", one_d_time, ONE_D_COLUMNS, True),
        ("1d_temporal_reference_convergence", "1D temporal reference convergence", one_d_ref, ONE_D_REFERENCE_COLUMNS, True),
        ("2d_spatial_comparison", "2D spatial comparison", two_d_spatial, TWO_D_COLUMNS, False),
        ("2d_time_comparison", "2D time comparison", two_d_time, TWO_D_COLUMNS, False),
        ("2d_temporal_reference_convergence", "2D temporal reference convergence", two_d_ref, TWO_D_REFERENCE_COLUMNS, False),
    ]
    for stem, title, rows, columns, show_index in bundles:
        pngs.append(_write_table_bundle(method_dir, f"{method_name}_{stem}", f"{method_name}: {title}", rows, columns, show_index=show_index))
    return pngs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPT-ready matched numerical method tables.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "matched_numerical_tables")
    parser.add_argument("--methods", nargs="+", default=["rk4", "forward_euler"], choices=["rk4", "forward_euler"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_pngs: list[Path] = []
    for method in args.methods:
        print(f"generating {method} tables", flush=True)
        all_pngs.extend(_method_tables(method, args.out_dir))

    _contact_sheet(all_pngs, args.out_dir / "all_tables_preview_contact_sheet.png")
    manifest = pd.DataFrame({"table_png": [str(path) for path in all_pngs]})
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    print(f"wrote {len(all_pngs)} table PNGs under {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
