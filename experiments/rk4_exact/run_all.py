from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "fisher-kpp-rk4" / "scripts" / "run_demo.py"


def main() -> None:
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
