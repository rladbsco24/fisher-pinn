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

from fisher_kpp_rk4 import check_rk4_stability, solve_rk4, solve_rk4_2d
from fisher_kpp_rk4.config import initial_condition, initial_condition_2d


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


def test_stability_check_accepts_1d_and_2d() -> None:
    assert check_rk4_stability(dx=0.1, dt=0.001, D=1.0, r=1.0, dim=1)["is_practically_safe"]
    assert check_rk4_stability(dx=0.02, dt=0.003125, D=0.02, r=3.0, dim=2)["is_practically_safe"]


def test_notebook_code_cells_are_parseable() -> None:
    notebook = REPO_ROOT / "notebooks" / "fisher_kpp_rk4_demo.ipynb"
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

