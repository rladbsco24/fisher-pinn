from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_entrypoint_scripts_parse() -> None:
    for path in sorted((REPO_ROOT / "experiments").rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_canonical_experiment_readmes_exist() -> None:
    required = [
        REPO_ROOT / "experiments" / "README.md",
        REPO_ROOT / "experiments" / "basic_pinn" / "README.md",
        REPO_ROOT / "experiments" / "korea_pine_pinn" / "README.md",
        REPO_ROOT / "experiments" / "rk4_exact" / "README.md",
        REPO_ROOT / "docs" / "REPOSITORY_STRUCTURE.md",
    ]
    for path in required:
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()
