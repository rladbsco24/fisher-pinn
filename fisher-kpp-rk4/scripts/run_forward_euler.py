from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCHED_FIGURE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_matched_numerical_figures.py"
MATCHED_TABLE_SCRIPT = PROJECT_ROOT / "scripts" / "generate_matched_numerical_tables.py"


def _load_matched_figure_module():
    spec = importlib.util.spec_from_file_location("generate_matched_numerical_figures", MATCHED_FIGURE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MATCHED_FIGURE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified 1D/2D Forward Euler Fisher-KPP outputs.")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "matched_numerical_visualizations" / "forward_euler",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "matched_numerical_tables",
    )
    parser.add_argument("--skip-tables", action="store_true")
    args = parser.parse_args()

    module = _load_matched_figure_module()
    paths = module._forward_euler_figures(args.figure_dir)
    print(f"wrote {len(paths)} Forward Euler figures to {args.figure_dir}")

    if not args.skip_tables:
        subprocess.run(
            [
                sys.executable,
                str(MATCHED_TABLE_SCRIPT),
                "--out-dir",
                str(args.table_dir),
                "--methods",
                "forward_euler",
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
