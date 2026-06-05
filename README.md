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
- Relative-progress adaptive loss balancing for active PINN terms, keeping data, PDE,
  known-IC, boundary, and front-gradient losses from drifting onto incompatible scales.
- Residual curriculum weighting that starts flatter and gradually emphasizes harder
  high-residual collocation points.
- Residual-adaptive collocation points inspired by RAD/RAR-D sampling.
- Optional gradient-enhanced residual penalty for sharp fronts.
- Moving-front validation metrics for mass, active-front area, and low-level front
  geometry instead of relying on final-time L2 alone.
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

- The default forward backbone is now a PirateNet-style residual architecture with
  near-identity adaptive skip connections and row-wise random weight factorization. This
  follows the moving-interface PINN/PirateNet literature, where dynamic fronts benefit
  more from architecture and training stabilization than from heavy geometric penalties.
- A NIF-style last-layer parameterized head is implemented as `architecture="nif_pirate"`.
  It separates a spatial ShapeNet from a time/physics ParameterNet, following Neural
  Implicit Flow's mesh-agnostic low-rank representation. It is exposed through the
  `geo_nif_front_area` ablation case rather than used as the default because a 60-epoch
  wiring check underperformed the current PirateNet/RWF profile on this Fisher-KPP front.
- Spatial Fourier features are used as positional encoding, not as a periodic boundary
  assumption. The forward preset now uses a lower Fourier scale because the target
  solution is a smooth diffusive Fisher-KPP front; high-frequency features caused
  sparse-observation blob artifacts.
- Square-domain geo features encode boundary distance and simple interior geometry.
  The current synthetic problem treats the whole square as valid land; the sampler has a
  mask interface so a real land mask can replace `box` later.
- A hard known-initial-condition ansatz makes `u(x,y,0)` exactly equal to the Gaussian
  seed. The known-IC loss remains in the objective and diagnostics, but for this preset it
  should be near numerical zero because the constraint is structural.
- Seed-centered radial/front features give the network coordinates aligned with the
  initial seed and expected Fisher-KPP front radius.
- Scaled traveling-wave features add a learned moving-frame coordinate based on
  `xi = (||x-x0|| - (3 sigma + 2 sqrt(D r) t)) / sqrt(D/r)`. This follows recent
  TW-PINN work and gives the network an explicit coordinate for translating Fisher-KPP
  fronts instead of requiring a generic MLP to rediscover that frame.
- A KPP front-speed envelope suppresses nonphysical low-amplitude background far outside
  the reachable front region.
- A Neumann boundary loss is enabled.
- PDE residuals are weighted toward infection-front regions using `u(1-u)` and
  `|grad u|`.
- RAR anchor refresh now uses a normalized front-aware monitor combining residual
  magnitude, `|grad u|`, and logistic activity `u(1-u)`. This follows adaptive sampling
  work for sharp PDE layers where residual-only refinement can miss moving fronts.
- A solver-assisted `geo_rk4_teacher_front_area` ablation adds RK4 pseudo-labels sampled
  across the full space-time domain, with extra probability mass near the low-density
  leading edge. This follows supervised/pretraining strategies for evolution-PDE PINNs.
  It is deliberately not the default pure-PINN profile because it uses a numerical
  teacher in addition to the PDE and sparse observations.
- A moving-front speed loss enforces the Fisher-KPP traveling-front relation
  `u_t + (2 sqrt(D r) + v.n) grad(u).n = 0` on active level-set bands. This makes the
  front move like a Fisher-KPP front instead of merely fitting late-time blobs. The
  implementation now ignores near-flat gradients so the speed diagnostic is not dominated
  by regions where normal speed is numerically ill-defined.
- A parabolic mass-balance loss enforces the no-flux Fisher-KPP integral identity
  `d mean(u)/dt = r mean(u(1-u))`. This adds a global growth check that the pointwise
  residual and sparse observations can miss.
- A leading-edge soft area constraint uses the known Gaussian initial condition and the
  linearized Fisher-KPP heat-kernel approximation to set analytic targets for low-level
  front areas (`u>0.05`, `u>0.10`). It combines soft area matching with a quantile hinge,
  so the metric is not satisfied merely by spreading tiny values everywhere.
- Expected-front PDE sampling and one-sided leading-edge floor losses are implemented as
  optional ablation knobs. Quick experiments showed they can improve apparent active
  area while degrading held-out MSE/mass, so they are not enabled in the default profile.
- Eikonal regularization from level-set moving-interface PINNs is not enabled here because
  Fisher-KPP `u` is not a signed-distance level-set field. The cited moving-interface
  paper also shows that Eikonal/mass penalties can hurt when their weights are not chosen
  carefully, so this lab treats them as diagnostics/ablations rather than default truth.
- Residual weighting follows an easy-to-hard curriculum: early epochs avoid overfitting
  residual outliers, then the exponent ramps toward the full adaptive residual weight.
- Adaptive relative loss balancing updates multipliers from each term's relative training
  progress. This is a lightweight Colab-friendly alternative to full per-term gradient
  surgery while targeting the same multi-objective imbalance issue.
- Already-satisfied hard constraints and sparse regularization are excluded from adaptive
  balancing so they cannot down-weight the active PDE/front losses.
- The training loop restores the best validation-observation checkpoint, which prevents
  longer runs from drifting after residual-adaptive refreshes.
- Residual-adaptive collocation uses a combined residual/front score.
- Front-local gPINN loss penalizes residual gradients only near the active front, while
  the moving-front speed loss checks the front's normal propagation speed.
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
- `front_area_005_mae` / `front_area_010_mae`: mean absolute error of the area where
  `u>0.05` and `u>0.10` across time.
- `active_front_area_mae`: mean absolute error of the `0.1<u<0.9` active-front band.
- `mass_mae`: mean absolute error of spatial mean density across time.
- `train_observation_mse` / `validation_observation_mse`: PINN data fit.
- `rk4_train_observation_mse` / `rk4_validation_observation_mse`: RK4 data fit at the
  same observation coordinates.
- `rk4_runtime_sec`: wall-clock time for the RK4 baseline.

## Ablation Experiments

For the original inverse-origin setup, run a source/warm-start matrix before making
origin-localization claims:

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

For the Korea-style forward Fisher-KPP setting, use the dedicated forward ablation
script. It scores the same problem by field error, RK4 comparison, low-level front
geometry, active-front coverage, and mass trajectory:

```bash
python scripts\run_forward_ablation.py --preset smoke --seeds 7 --out-dir runs\forward_ablation_smoke
python scripts\run_forward_ablation.py --preset quick --seeds 7,8,9 --out-dir runs\forward_ablation_quick
```

Forward ablation outputs:

- `results.csv`: one row per method/seed.
- `summary.json`: mean/std for final L2, validation MSE, front-area MAE, active-front
  MAE, and mass MAE.
- `summary.png`: side-by-side bars for final L2, `u>0.10` front-area MAE, and mass MAE.

Current 60-epoch quick check on one seed with the PirateNet/RWF default, scaled
traveling-wave features, and front-aware RAR gives `final_time_relative_l2 = 0.3600`,
`validation_observation_mse = 6.32e-4`, `front_area_010_mae = 0.0119`, and
`mass_mae = 0.0043`. Relative to the previous PirateNet/RWF quick check, field and
front metrics improved while the mass trajectory became slightly worse; treat this as a
wiring check, not a final benchmark.

The same 60-epoch sanity check with the new `nif_pirate` head ran successfully but was
weaker on this setup (`final_time_relative_l2 = 0.5401`,
`validation_observation_mse = 1.41e-3`, `front_area_010_mae = 0.0269`,
`mass_mae = 0.0040`), so it remains an explicit ablation until multi-seed tuning supports
promoting it.

The 60-epoch RK4-teacher sanity check (`geo_rk4_teacher_front_area` style,
`rk4_teacher=0.75`, 2048 teacher points in the quick config) gives
`final_time_relative_l2 = 0.3593`, `pinn_vs_rk4_final_relative_l2 = 0.3588`,
`validation_observation_mse = 6.31e-4`, `front_area_010_mae = 0.0120`, and
`mass_mae = 0.0042`. This is a small but consistent wiring improvement over the pure
60-epoch default in final L2, low-threshold front area, and mass, while RK4 itself remains
the much more accurate same-problem numerical baseline (`rk4_final_time_relative_l2 =
0.00464`).

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
- `spacetime_error.png`: relative L2, mean density, active-front coverage, and low-level
  front area over time.
- `residual_front_diagnostics.png`: final-time residual, front indicator, adaptive
  weighting maps, gradient-filtered normal front speed, and front-speed error.
- `pinn_vs_rk4_comparison.png`: final-time truth/PINN/RK4 fields, absolute-error maps,
  and a compact accuracy summary.
- `training_diagnostics.png`: loss components including known IC, moving-front speed,
  parabolic mass balance, learned `D/r`, residual curriculum, adaptive loss multipliers,
  and front/sparsity diagnostics.

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
- The training loop has causal weighting, residual curriculum, adaptive relative loss
  balancing, residual-adaptive sampling, reproducible seeds, structured metrics, and tests.

## References

- Raissi, Perdikaris, Karniadakis, "Physics-informed neural networks", JCP 2019.
- Fisher, "The wave of advance of advantageous genes", Annals of Eugenics 1937.
- Kolmogorov, Petrovskii, Piskunov, "A study of the diffusion equation with increase
  in the amount of substance, and its application to a biological problem", 1937.
- Wang, Sankaran, Perdikaris, "Respecting causality is all you need for training
  physics-informed neural networks", arXiv:2203.07404.
- Wu, Zhu, Tan, Kartha, Lu, "A comprehensive study of non-adaptive and residual-based
  adaptive sampling for physics-informed neural networks", arXiv:2207.10289.
- Mao, Meng, "Physics-informed neural networks with residual/gradient-based adaptive
  sampling methods for solving partial differential equations with sharp solutions",
  Applied Mathematics and Mechanics 2023.
- Hao et al., "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks
  for Solving PDEs", NeurIPS Datasets and Benchmarks 2024 / arXiv:2306.08827.
- Jagtap, Kharazmi, Karniadakis, "Conservative physics-informed neural networks on
  discrete domains for conservation laws", Computer Methods in Applied Mechanics and
  Engineering 2020.
- Yu, Lu, Meng, Karniadakis, "Gradient-enhanced physics-informed neural networks for
  forward and inverse PDE problems", arXiv:2111.02801.
- Wang, Li, Chen, Perdikaris, "PirateNets: Physics-informed Deep Learning with
  Residual Adaptive Networks", arXiv:2402.00326.
- Pan, Brunton, Kutz, "Neural Implicit Flow: a mesh-agnostic dimensionality reduction
  paradigm of spatio-temporal data", JMLR 2023.
- Rohrhofer, Posch, Gößnitzer, Geiger, "Approximating families of sharp solutions to
  Fisher's equation with physics-informed neural networks", Computer Physics
  Communications 2025.
- Han, Park, Gu, Jung, "A scaled TW-PINN: A physics-informed neural network for
  traveling wave solutions of reaction-diffusion equations with general coefficients",
  arXiv:2603.15331.
- Penwarden, Jagtap, Zhe, Karniadakis, Kirby, "A physics-informed neural network with
  supervised learning for evolution partial differential equations", Journal of
  Computational Physics 2023.
- Mullins, Kamil, Fahsi, Soulaïmani, "Physics-informed neural networks for solving
  moving interface flow problems using the level set approach", arXiv:2502.02440.
- Chen, Howard, Stinis, "Self-adaptive weights based on balanced residual decay rate for
  physics-informed neural networks and deep operator networks", arXiv:2407.01613.
- Bischof, Kraus, "Multi-Objective Loss Balancing for Physics-Informed Deep Learning",
  SSRN 2024. Introduces ReLoBRaLo-style relative loss balancing for PINNs.
- Liu, Chu, Thuerey, "ConFIG: Towards Conflict-free Training of Physics Informed Neural
  Networks", arXiv:2408.11104 / ICLR 2025.
- Yang, Wang, Li, Cao, Yan, Liu, "From Simple to Complex: Curriculum-Guided
  Physics-Informed Neural Networks via Gaussian Mixture Models", arXiv:2605.19263.
- Chuprov, Derkach, Efremenko, Kychkin, "Application of Physics-Informed Neural Networks
  for Solving the Inverse Advection-Diffusion Problem to Localize Pollution Sources",
  arXiv:2503.18849.
