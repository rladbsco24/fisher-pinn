from __future__ import annotations

import sys
from pathlib import Path

RK4_ROOT = Path(__file__).resolve().parents[2] / "fisher-kpp-rk4"
for path in (RK4_ROOT / "src", RK4_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_demo import OUTPUT_DIR, run_2d_report_visualization, save_2d_report_figures


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = run_2d_report_visualization()
    save_2d_report_figures(result)


if __name__ == "__main__":
    main()
