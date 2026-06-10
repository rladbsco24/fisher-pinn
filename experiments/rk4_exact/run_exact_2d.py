from __future__ import annotations

import sys
from pathlib import Path

RK4_ROOT = Path(__file__).resolve().parents[2] / "fisher-kpp-rk4"
for path in (RK4_ROOT / "src", RK4_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_demo import OUTPUT_DIR, run_2d


def _save_2d_plots(result: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    OUTPUT_DIR.mkdir(exist_ok=True)
    snapshots = result["snapshots"]
    times = result["times"]
    panel_idx = np.unique(np.linspace(0, len(times) - 1, 4, dtype=int))
    fig, axes = plt.subplots(1, len(panel_idx), figsize=(4.0 * len(panel_idx), 3.4), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, idx in zip(axes, panel_idx):
        im = ax.imshow(
            snapshots[idx].T,
            origin="lower",
            extent=[result["x"][0], result["x"][-1], result["y"][0], result["y"][-1]],
            vmin=0.0,
            vmax=1.0,
            cmap="cividis",
        )
        ax.set_title(f"t={times[idx]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.82)
    fig.suptitle("2D generalized Fisher-KPP exact-wave RK4 snapshots")
    fig.savefig(OUTPUT_DIR / "exact_2d_snapshots.png", dpi=200)
    plt.close(fig)

    plt.figure(figsize=(7, 4))
    plt.plot(result["times"], result["mass"], label="mean mass")
    plt.plot(result["times"], result["area_ge_0.05"], label="area u>=0.05")
    plt.plot(result["times"], result["area_ge_0.10"], label="area u>=0.10")
    plt.xlabel("t")
    plt.ylabel("fraction")
    plt.title("2D exact-wave mass and front-area diagnostics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exact_2d_front_area_mass.png", dpi=200)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    result = run_2d()
    _save_2d_plots(result)


if __name__ == "__main__":
    main()
