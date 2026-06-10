# Canonical Experiment Entrypoints

This directory is the organized front door for the repository. The numerical
and PINN implementations remain in their original packages so existing
notebooks keep working, but new runs should start from one of these entrypoints.

## Layout

| Area | Purpose | Main commands |
| --- | --- | --- |
| `basic_pinn/` | Synthetic Fisher-KPP PINN benchmarks | Zeppetella exact-wave 1D benchmark and Gaussian moving-front 2D benchmark |
| `korea_pine_pinn/` | Korea Forest Service pine-wilt forward PINN/RK4 comparison | Land-masked pine-wilt PINN with RK4 baseline |
| `rk4_exact/` | Exact-solution RK4 baselines | Ablowitz-Zeppetella 1D, generalized Fisher-KPP 2D, and 3D report surfaces |

Generated run outputs belong under `runs/` or `fisher-kpp-rk4/outputs/`. Those
directories are intentionally ignored by Git. Curated preview figures that are
useful to inspect on GitHub belong under `docs/figure_previews/`.

## Recommended Commands

```bash
python experiments/basic_pinn/run_zeppe_exact_1d.py --preset quick
python experiments/basic_pinn/run_gaussian_moving_front_2d.py --preset quick
python experiments/korea_pine_pinn/run_pine_pinn.py --preset quick
python experiments/rk4_exact/run_all.py
```

Use `--preset full` for publication-style runs. Full PINN runs are intentionally
much slower than the RK4 baselines.
