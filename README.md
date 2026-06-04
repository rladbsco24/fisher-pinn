# Fisher-KPP Origin Lab

Research-grade inverse origin inference for a 2D Fisher-KPP advection-reaction-diffusion
front. This is a cleaner and more honest replacement for a one-file PINN demo: it includes
synthetic data generation, modern PINN training stabilizers, interpretable source recovery,
stronger baselines, metrics, figures, and smoke tests.

## Problem

We observe only a late-time window of a spreading field

```text
u_t + v . grad(u) = D Laplacian(u) + r u (1 - u)
```

and infer where the initial localized seed was. Backward diffusion is ill-posed, so the
code treats this as a PDE-constrained inverse problem rather than a literal reverse-time
simulation.

## Method

The main estimator is `CaRA-gPINN-Seed`:

- Fourier-feature neural field for `u(x, y, t)`.
- Trainable Gaussian source head for an interpretable origin estimate.
- Hard initial-source envelope: the trainable source dominates at `t=0` and fades before
  the late observation window, so origin parameters are optimized through the PDE.
- Observation-only warm start from the late weighted centroid, optionally drift-corrected
  when advection is known. This same estimate is logged as a baseline.
- Synthetic observations mix uniform sensors with front-focused sensors so the inverse
  task is not dominated by near-zero background samples.
- Train/validation observation split, with held-out late-window MSE reported in
  `metrics.json`.
- Coarse differentiable source-shooting consistency: the trainable initial source is
  pushed through a finite-difference solver and compared directly against training
  observations. This gives the origin parameters a direct data gradient instead of relying
  only on the neural-field PDE residual.
- Causal time-bin weighting inspired by causal PINNs.
- Bounded residual-decay weighting inspired by recent adaptive weighting work.
- Residual-adaptive collocation points inspired by RAD/RAR-D sampling.
- Optional gradient-enhanced residual penalty for sharp fronts.
- Ensemble mode for epistemic uncertainty over origin estimates.

The novelty here is deliberately scoped: this repository combines modern PINN stabilizers
with a trainable source head and a differentiable finite-difference inverse baseline for
late-window Fisher-KPP origin inference. It does not claim that PINNs are the only possible
method.

## Install

```bash
cd C:\Users\yoonc\Downloads\fisher-kpp-origin-lab
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Run

```bash
python scripts\run_inverse_origin.py --quick --out-dir runs\quick
```

## Korea Pine-Wilt Compatibility

The referenced `korea-pine-wilt-pinn-main.zip` notebook uses a forward Fisher-KPP PINN
setup rather than this repository's default inverse-origin setting:

```text
u_t = D Laplacian(u) + r u (1 - u)
```

In that notebook's PINN loss there is no explicit boundary-condition penalty. The Korea
land mask is used for data construction/evaluation and later plotting; it is not enforced
as a neural Neumann or Dirichlet boundary loss. To match that problem setup, run:

```bash
python scripts\run_inverse_origin.py --quick --korea-pine-style --out-dir runs\korea_pine_style
```

This profile disables advection, learns `D` and `r`, disables the hard source envelope,
sets boundary/source/shooting losses to zero, and uses the notebook's density-weighted
data loss form `1 + 4u_obs`.

## Geo-Spectral Forward PINN

The improved forward profile keeps the same Fisher-KPP problem setup as the Korea
pine-wilt compatibility mode, but changes the model and training objective:

```bash
python scripts\run_inverse_origin.py --quick --geo-spectral-forward --out-dir runs\geo_spectral_forward
```

What changes:

- Spatial Fourier features are used as positional encoding, not as a periodic boundary
  assumption.
- Square-domain geo features encode boundary distance and simple interior geometry.
  The current synthetic problem treats the whole square as valid land; the sampler has a
  mask interface so a real land mask can replace `box` later.
- A Neumann boundary loss is enabled.
- PDE residuals are weighted toward infection-front regions using `u(1-u)` and
  `|grad u|`.
- Residual-adaptive collocation uses a combined residual/front score.
- Front-local gPINN loss penalizes residual gradients only near the active front.
- Last-layer L1 regularization discourages overusing the expanded spectral/geo feature
  basis.

## RK4 Same-Problem Baseline

The provided `fisher-kpp-rk4.zip` solver targets a different 1D Fisher-KPP front
problem with Dirichlet boundary conditions. This lab ports the RK4 idea to the same
problem used here: a 2D square-domain Fisher-KPP equation with the same Gaussian seed,
the same diffusion/reaction/advection settings, and no-flux Neumann boundaries.

Every experiment now computes an RK4 reference run after PINN training and reports:

- `pinn_final_time_relative_l2`: final-time PINN error against the synthetic reference.
- `rk4_final_time_relative_l2`: final-time RK4 error against the same synthetic reference.
- `pinn_vs_rk4_final_relative_l2`: final-time field difference between the two solvers.
- `train_observation_mse` / `validation_observation_mse`: PINN data fit.
- `rk4_train_observation_mse` / `rk4_validation_observation_mse`: RK4 data fit at the
  same observation coordinates.
- `rk4_runtime_sec`: wall-clock time for the RK4 baseline.

## Ablation Experiments

Run a small experiment matrix before making claims:

```bash
python scripts\run_ablation.py --preset quick --case-set core --seeds 7,8,9 --out-dir runs\ablations_core
```

For a fast wiring check:

```bash
python scripts\run_ablation.py --preset smoke --case-set anchor --seeds 7 --out-dir runs\ablations_smoke
```

Outputs:

- `manifest.json`: exact cases and seeds.
- `results.csv`: one row per case/seed run.
- `summary.json`: case-level mean and standard deviation.
- `summary.png`: origin-error bar plot.

Important ablation axes:

- `warm_start.mode`: `drift_corrected`, `centroid`, or `neutral`.
- `source_anchor`: checks whether the source estimate is being held near the warm start.
- `shooting`: checks whether the direct source-to-observation consistency loss matters.
  The default weight is conservative (`5`) because a coarse solver can improve held-out
  data fit while biasing the inferred origin. Strong shooting weights are best treated as
  an ablation, not an automatic improvement.
- `focus_fraction`: compares front-focused sensors against uniform-only sensors.
- `learn_drift`: tests harder settings where drift is not assumed in the model weights.

## Colab Notebook

Upload `fisher_kpp_origin_lab.ipynb` to Google Colab and run from the first cell.
The notebook is self-contained: if `fisher_origin_lab/` is not present, it rebuilds the
project package from an embedded source archive under `/content/fisher-kpp-origin-lab`.

No Drive mount or repository clone is required for the quick experiment.

For a more meaningful CPU run:

```bash
python scripts\run_inverse_origin.py --epochs 1200 --ensemble 3 --run-classical-baseline --out-dir runs\full
```

Outputs:

- `metrics.json`: origin errors, learned PDE coefficients, loss history.
- `observation_coverage.png`: train/validation spatial and temporal sample coverage.
- `reconstruction.png`: truth, PINN reconstruction, and absolute-error panels.
- `spacetime_error.png`: relative L2, mean density, and active-front coverage over time.
- `residual_front_diagnostics.png`: final-time residual, front indicator, and adaptive
  weighting maps.
- `pinn_vs_rk4_comparison.png`: final-time truth/PINN/RK4 fields, absolute-error maps,
  and a compact accuracy summary.
- `training_diagnostics.png`: loss components, learned `D/r`, front/sparsity diagnostics,
  and runtime trace.

## Why This Is Better Than The Original Demo

- The missing-notebook problem is gone; every referenced entry point exists.
- The README avoids "PINN-only" overclaiming.
- The source prior is explicit and trainable, not hidden inside an argmax at `t=0`.
- The inverse estimators start from a neutral source location, so the synthetic truth origin
  is used for scoring only, not for initialization.
- A differentiable finite-difference inverse solver is included as a strong classical
  baseline.
- A drift-corrected centroid baseline is included because known advection makes the
  original centroid-only comparison too weak.
- The training loop has causal weighting, residual-adaptive sampling, reproducible seeds,
  structured metrics, and tests.

## References

- Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", JCP 2019.
- Wang, Sankaran, Perdikaris, "Respecting causality is all you need for training
  physics-informed neural networks", arXiv:2203.07404.
- Wu, Zhu, Tan, Kartha, Lu, "A comprehensive study of non-adaptive and residual-based
  adaptive sampling for physics-informed neural networks", arXiv:2207.10289.
- Yu, Lu, Meng, Karniadakis, "Gradient-enhanced physics-informed neural networks for
  forward and inverse PDE problems", arXiv:2111.02801.
- Chen, Howard, Stinis, "Self-adaptive weights based on balanced residual decay rate for
  physics-informed neural networks and deep operator networks", arXiv:2407.01613.
- Chuprov, Derkach, Efremenko, Kychkin, "Application of Physics-Informed Neural Networks
  for Solving the Inverse Advection-Diffusion Problem to Localize Pollution Sources",
  arXiv:2503.18849.
