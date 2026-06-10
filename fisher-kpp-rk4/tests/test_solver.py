from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fisher_kpp_rk4 import (
    check_forward_euler_stability,
    check_rk4_stability,
    relative_l2,
    solve_1d_method,
    solve_long_time_curve,
    solve_rk4,
    solve_rk4_2d,
)
from fisher_kpp_rk4.config import (
    CURVE_ALPHA,
    CURVE_DT,
    CURVE_NT,
    CURVE_OMEGA_D,
    CURVE_RHO0,
    CURVE_RHO_INF,
    CURVE_V0,
    Nt,
    T,
    ablowitz_zeppetella_exact,
    c,
    dt,
    generalized_fisher_kpp_exact_2d,
    left_bc,
    LONG_TIME_D,
    LONG_TIME_DT,
    LONG_TIME_DX,
    LONG_TIME_R,
    D,
    D_2D,
    Nt_2d,
    r,
    r_2D,
    right_bc,
    initial_condition,
    initial_condition_2d,
    x,
    x_2d,
    y_2d,
)


def test_1d_rk4_shapes_bounds_and_fronts() -> None:
    x = np.linspace(0.0, 20.0, 101)
    result = solve_rk4(
        x=x,
        dt=0.002,
        Nt=20,
        D=1.0,
        r=1.0,
        initial_condition=initial_condition,
        left_bc=1.0,
        right_bc=0.0,
        save_interval=0.02,
    )
    assert result["snapshots"].ndim == 2
    assert result["snapshots"].shape[1] == len(x)
    assert result["u_final"].shape == (len(x),)
    assert np.isfinite(result["snapshots"]).all()
    assert result["snapshots"].min() >= 0.0
    assert result["snapshots"].max() <= 1.0
    assert result["fronts"].shape == result["times"].shape


def test_ablowitz_zeppetella_1d_rk4_matches_exact_solution() -> None:
    result = solve_rk4(
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
    assert float(result["relative_l2_final"]) < 3.0e-3
    speed = result["fronts"][-1] - result["fronts"][0]
    assert abs(speed / T - c) < 0.1


def test_2d_rk4_shapes_bounds_and_front_areas() -> None:
    x = np.linspace(0.0, 1.0, 25)
    result = solve_rk4_2d(
        x=x,
        y=x,
        dt=0.001,
        Nt=8,
        D=0.02,
        r=3.0,
        initial_condition=initial_condition_2d,
        save_interval=0.004,
    )
    assert result["snapshots"].ndim == 3
    assert result["snapshots"].shape[1:] == (len(x), len(x))
    assert result["u_final"].shape == (len(x), len(x))
    assert np.isfinite(result["snapshots"]).all()
    assert result["snapshots"].min() >= 0.0
    assert result["snapshots"].max() <= 1.0
    assert result["area_ge_0.05"].shape == result["times"].shape
    assert result["area_ge_0.10"].shape == result["times"].shape
    assert result["mass"].shape == result["times"].shape


def test_generalized_fisher_kpp_2d_rk4_matches_exact_solution() -> None:
    result = solve_rk4_2d(
        x=x_2d,
        y=y_2d,
        dt=0.005,
        Nt=int(round(0.5 / 0.005)),
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=0.5,
        boundary_condition="dirichlet_exact",
        exact_solution=generalized_fisher_kpp_exact_2d,
    )
    assert float(result["relative_l2_final"]) < 3.0e-3


def test_stability_check_accepts_1d_and_2d() -> None:
    assert check_rk4_stability(dx=0.1, dt=0.001, D=1.0, r=1.0, dim=1)["is_practically_safe"]
    assert check_rk4_stability(dx=0.02, dt=0.003125, D=0.02, r=3.0, dim=2)["is_practically_safe"]


def test_long_time_fair_parameters_are_safe_for_explicit_methods() -> None:
    assert check_forward_euler_stability(LONG_TIME_DX, LONG_TIME_DT, LONG_TIME_D, LONG_TIME_R, dim=1)["is_practically_safe"]
    assert check_rk4_stability(LONG_TIME_DX, LONG_TIME_DT, LONG_TIME_D, LONG_TIME_R, dim=1)["is_practically_safe"]


def test_1d_method_comparison_runs_all_methods() -> None:
    x = np.linspace(0.0, 8.0, 81)

    def init(grid: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp((grid - 2.0) / 0.35))

    results = {
        method: solve_1d_method(
            method,
            x=x,
            dt=0.005,
            Nt=40,
            D=0.05,
            r=0.2,
            initial_condition=init,
            left_bc=1.0,
            right_bc=0.0,
            save_interval=0.05,
            probe_x=3.0,
        )
        for method in ("forward_euler", "backward_euler", "trapezoidal", "rk4")
    }

    rk4_final = results["rk4"]["u_final"]
    for result in results.values():
        assert result["snapshots"].shape[1] == len(x)
        assert result["fronts"].shape == result["times"].shape
        assert result["rho"].shape == result["times"].shape
        assert np.isfinite(result["snapshots"]).all()
        assert result["snapshots"].min() >= 0.0
        assert result["snapshots"].max() <= 1.0
    assert relative_l2(results["trapezoidal"]["u_final"], rk4_final) < 1.0e-3


def test_long_time_curve_trend_matches_reference_shape() -> None:
    result = solve_long_time_curve(
        "rk4",
        dt=CURVE_DT,
        Nt=CURVE_NT,
        rho_inf=CURVE_RHO_INF,
        alpha=CURVE_ALPHA,
        omega_d=CURVE_OMEGA_D,
        rho0=CURVE_RHO0,
        v0=CURVE_V0,
    )
    times = result["times"]
    rho = result["rho"]
    first_window = times <= 8.0
    peak_idx = int(np.argmax(rho[first_window]))
    trough_idx = int(np.argmin(rho[(times >= 3.0) & (times <= 7.0)]))
    trough_rho = rho[(times >= 3.0) & (times <= 7.0)][trough_idx]

    assert 0.68 <= rho[peak_idx] <= 0.76
    assert 1.5 <= times[peak_idx] <= 2.3
    assert 0.13 <= trough_rho <= 0.22
    assert 0.32 <= rho[-1] <= 0.36
    assert float(np.max(result["abs_error"])) < 1.0e-6


def test_long_time_curve_all_methods_stay_close_to_exact() -> None:
    for method, max_error in {
        "forward_euler": 3.5e-2,
        "backward_euler": 3.5e-2,
        "trapezoidal": 1.0e-3,
        "rk4": 1.0e-6,
    }.items():
        result = solve_long_time_curve(
            method,
            dt=CURVE_DT,
            Nt=CURVE_NT,
            rho_inf=CURVE_RHO_INF,
            alpha=CURVE_ALPHA,
            omega_d=CURVE_OMEGA_D,
            rho0=CURVE_RHO0,
            v0=CURVE_V0,
        )
        assert np.isfinite(result["rho"]).all()
        assert float(np.max(result["abs_error"])) < max_error


def test_notebook_code_cells_are_parseable() -> None:
    notebooks = [
        REPO_ROOT / "notebooks" / "fisher_kpp_rk4_demo.ipynb",
        REPO_ROOT / "notebooks" / "fisher_kpp_long_time_methods.ipynb",
    ]
    for notebook in notebooks:
        nb = json.loads(notebook.read_text(encoding="utf-8"))
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "\n".join(
                line
                for line in "".join(cell.get("source", [])).splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            ast.parse(source, filename=f"{notebook.name}:cell{idx}")


def test_required_2d_report_figures_are_wired_into_script_and_notebook() -> None:
    required_names = [
        "centerline_2d_exact_wave.png",
        "surface_2d_exact_wave_t00.png",
        "surface_2d_exact_wave_t02.png",
        "surface_2d_exact_wave_t04.png",
        "surface_2d_exact_wave_t06.png",
        "surface_2d_exact_wave_t08.png",
        "absolute_error_2d_surface_t00.png",
        "absolute_error_2d_surface_t02.png",
        "absolute_error_2d_surface_t04.png",
        "absolute_error_2d_surface_t06.png",
        "absolute_error_2d_surface_t08.png",
    ]
    script_source = (REPO_ROOT / "scripts" / "run_demo.py").read_text(encoding="utf-8")
    notebook_source = (REPO_ROOT / "notebooks" / "fisher_kpp_rk4_demo.ipynb").read_text(encoding="utf-8")
    preview_source = (REPO_ROOT.parent / "docs" / "figure_previews" / "README.md").read_text(encoding="utf-8")
    for source in (script_source, notebook_source):
        assert "REPORT_2D_TIMES" in source
        assert "surface_2d_exact_wave_" in source
        assert "absolute_error_2d_surface_" in source
        assert "centerline_2d_exact_wave.png" in source
    for name in required_names:
        assert f"rk4_report_{name}" in preview_source


def test_required_rk4_grid_tables_are_wired_into_script_and_notebook() -> None:
    required_names = [
        "rk4_1d_spatial_comparison.png",
        "rk4_1d_time_comparison.png",
        "rk4_1d_temporal_reference_convergence.png",
        "rk4_2d_spatial_comparison.png",
        "rk4_2d_time_comparison.png",
        "rk4_2d_temporal_reference_convergence.png",
    ]
    script_source = (REPO_ROOT / "scripts" / "run_convergence.py").read_text(encoding="utf-8")
    notebook_source = (REPO_ROOT / "notebooks" / "fisher_kpp_rk4_demo.ipynb").read_text(encoding="utf-8")
    preview_source = (REPO_ROOT.parent / "docs" / "figure_previews" / "README.md").read_text(encoding="utf-8")
    assert "avg_newton_iter" not in script_source
    assert "max_residual" not in script_source
    assert "rk4_stages_per_step" not in script_source
    assert "stability_safe" not in script_source
    assert '"min_U"' in script_source
    assert '"mean_U"' in script_source
    assert "Reference_Relative_L2_Error" in script_source
    assert "Observed_Time_Order" in script_source
    for name in required_names:
        assert name in script_source
        assert name in notebook_source
        assert f"rk4_table_{name.removeprefix('rk4_')}" in preview_source
