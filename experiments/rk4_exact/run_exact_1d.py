from __future__ import annotations

import sys
from pathlib import Path

RK4_ROOT = Path(__file__).resolve().parents[2] / "fisher-kpp-rk4"
for path in (RK4_ROOT / "src", RK4_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_demo import OUTPUT_DIR, run_1d


def _save_1d_plots(result: dict) -> None:
    import matplotlib.pyplot as plt

    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 5))
    for t, u in zip(result["times"], result["snapshots"]):
        plt.plot(result["x"], u, label=f"t={t:.0f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.ylim(-0.05, 1.05)
    plt.title("1D Ablowitz-Zeppetella Fisher-KPP solved by RK4")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exact_1d_snapshots.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(result["times"], result["fronts"], marker="o")
    plt.xlabel("t")
    plt.ylabel("front position, u=0.5")
    plt.title("1D exact-wave front propagation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exact_1d_front_position.png", dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = run_1d()
    _save_1d_plots(result)


if __name__ == "__main__":
    main()
