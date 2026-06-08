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

Detailed method explanation, literature rationale, observations, and interpretation are
maintained in `docs/fisher_kpp_pinn_review_response.docx`.

## Figure Preview Gallery

Representative outputs from the current runnable scripts are committed in
`docs/figure_previews/` so GitHub readers can inspect the expected result shapes before
rerunning experiments:

```text
docs/figure_previews/README.md
```

The gallery includes RK4 1D/2D demos, long-time explicit/implicit integrator comparisons,
the damped long-time curve benchmark, Geo-Spectral PINN diagnostic figures, Korea
pine-wilt land-mask visualizations, GIF previews, and ablation summaries. The PINN and
Korea figures in that gallery are smoke/diagnostic previews generated with short runtimes;
use the longer commands below for publication-grade metrics.

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

## Korea Forest Service CSV Data And Notebook

The repository now includes a compact, GitHub-safe subset of the provided Korea Forest
Service pine-wilt CSV bundle:

- `data/korea_pine_wilt/processed/infected_points_2016_2023.csv.gz`: compact infected
  point observations with `x_5179`, `y_5179`, and `year`.
- `data/korea_pine_wilt/processed/infected_points_2016_2023.npz`: the same observations
  in faster NumPy form.
- `data/korea_pine_wilt/processed/manifest.json`: source-file sizes, year counts, and
  sha256 checksums for the original annual CSV files.
- `data/korea_pine_wilt/assets/skorea_provinces_2018.geojson`: the province geometry
  asset from the provided bundle.

The annual raw CSV files total roughly 1.17 GB and several individual files exceed
GitHub's normal 100 MB single-file limit, so they are not stored as ordinary Git blobs.
The committed compact files retain the infected-tree coordinates and years used by the
simulation code while keeping the repository cloneable without Git LFS quota pressure.
If the raw annual CSV files are available locally, rebuild the compact data with:

```bash
python scripts\build_korea_pine_wilt_compact_data.py --raw-dir C:\path\to\korea-pine-wilt-pinn-main\Data
```

The Korea forward baseline now enforces the committed province GeoJSON as a land mask.
Observation grids are zeroed outside land, RK4 uses masked no-flux diffusion at the
land/sea interface, metrics are evaluated on land cells, and the PINN baseline adds a
sea-exclusion penalty plus land-only PDE collocation. This prevents either baseline from
spreading pine-wilt density through the ocean cells of the rectangular computational grid.

Run the Korea pine-wilt 2D Fisher-KPP RK4 baseline:

```bash
python scripts\run_korea_pine_wilt_simulation.py --output-dir runs\korea_pine_wilt_csv_simulation
```

Outputs:

- `korea_pine_wilt_metrics.csv`: observed-year relative L2, correlation, and mean-density
  comparison for the RK4 fixed-solver baseline.
- `korea_pine_wilt_baseline_metrics.csv`: observed-year comparison for both RK4 and the
  repository PINN baseline.
- `observed_density_by_year.png`: compact CSV observations gridded by year.
- `rk4_forecast_timeline.png`: RK4 forward simulation from the 2016 observed density.
- `pinn_baseline_observed_years.png`: observed/PINN/error panels for selected observed
  years.
- `baseline_metric_comparison.png`: RK4 versus PINN observed-year relative L2 and
  correlation.
- `observed_vs_simulated_metrics.png`: observed-year metric trajectories.
- `korea_map_baselines.gif`: animated Korea province-map view of observed density,
  RK4, and the repository PINN baseline from 2016 through the configured forecast year.
- `korea_map_baselines_preview.png`: first-frame preview for environments that do not
  render GIFs inline.
- `korea_pine_wilt_summary.json`: run configuration and aggregate metrics.

The PINN baseline reuses this repository's PirateNet/Fourier/geo-feature PINN backbone
and fits the yearly gridded Korea observations with a weak Fisher-KPP PDE residual. It is
a diagnostic baseline, not a calibrated surveillance or intervention model. Use
`--skip-pinn` to run only the RK4 baseline, or increase `--pinn-epochs` for a stronger
PINN fit.

For an interactive Colab/Jupyter workflow, open:

```text
korea_pine_wilt_fisher_kpp_lab.ipynb
```

In Colab, the notebook clones this repository when the project files are not already
present, then reads the committed compact CSV/NPZ data.

## Geo-Spectral Forward PINN

The improved forward profile keeps the same Fisher-KPP problem setup as the Korea
pine-wilt compatibility mode, but changes the model and training objective. Detailed
method rationale and observations are kept in:

```text
docs/fisher_kpp_pinn_review_response.docx
```

Run the profile with:

```bash
python scripts\run_inverse_origin.py --quick --geo-spectral-forward --out-dir runs\geo_spectral_forward
```

For method ablations:

```bash
python scripts\run_forward_ablation.py --preset smoke --seeds 7 --out-dir runs\forward_ablation_smoke
```

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

## Long-Time `rho(t)` Curve PINN

The reference image's right panel is a damped scalar trend, not a monotone Fisher-KPP
front probe. To match that trend without changing the Fisher-KPP PDE interpretation,
the repository now includes a separate ODE-PINN benchmark:

```bash
python scripts\run_long_time_curve_pinn.py --quick
```

The fair long-time curve setting is:

```text
rho_inf=0.34, alpha=0.24, omega_d=1.0, rho(0)=0, rho_t(0)=0.60, T=30, dt=0.05
```

Forward Euler, backward Euler, trapezoidal, RK4, and the PINN all use this same target,
same time horizon, and same `dt`. The exact damped-oscillator solution is retained as
the reference, so the comparison isolates method/training error rather than mixing
different equations or time grids. The PINN uses a hard initial-condition ansatz, the
physics residual
`rho_tt + 2 alpha rho_t + (alpha^2 + omega_d^2)(rho - rho_inf) = 0`, and sparse exact
curve anchors to stabilize short Colab runs.

Outputs are written under `runs/long_time_curve_pinn/`:

- `pinn_long_time_curve_trend.png`
- `pinn_long_time_curve_diagnostics.png`
- `pinn_long_time_curve_results.npz`
- `metrics.json`

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

The current 60-epoch weak-RK4-teacher/front-profile sanity check
(`geo_spectral_forward().quick()`, `rk4_teacher=0.005`, 2048 teacher points,
batch 256) gives `final_time_relative_l2 = 0.2682`,
`pinn_vs_rk4_final_relative_l2 = 0.2693`,
`validation_observation_mse = 5.49e-4`, `front_area_005_mae = 0.0185`,
`front_area_010_mae = 0.0119`, and `mass_mae = 0.0025`. The new
front-normal profile loss targets the leading-edge profile around the expected
low-level front, which improves front-area and mass diagnostics relative to the
previous no-profile check (`5.57e-4`, `0.0195`, `0.0120`, `0.0026`) while leaving
final-field L2 in the same range. RK4 itself remains the much more accurate
same-problem numerical baseline (`rk4_final_time_relative_l2 = 0.00465`).

The first 60-epoch check of the cumulative `geo_levelset_time_slab` ablation
(`level_set_alignment=0.03`, 50% time-window collocation focus, 50% global replay,
weak RK4 teacher) gives `final_time_relative_l2 = 0.2830`,
`validation_observation_mse = 6.00e-4`, `front_area_005_mae = 0.0435`,
`front_area_010_mae = 0.0121`, and `mass_mae = 0.0037`. It is now stable, but it still
does not beat the tighter-envelope weak-RK4 case on all-purpose field/front/mass
accuracy. Treat it as a domain-decomposition/front-geometry ablation.

## Colab Notebook

Upload `fisher_kpp_origin_lab.ipynb` to Google Colab and run from the first cell.
The notebook is self-contained: if `fisher_origin_lab/` is not present, it rebuilds the
project package from an embedded source archive under `/content/fisher-kpp-origin-lab`.
In Colab, the bootstrap cell refreshes that embedded project directory on every run so
old `/content/fisher-kpp-origin-lab` files from a previous notebook cannot silently shadow
the updated front-profile and signed-error plotting code.

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
- `pinn_vs_rk4_comparison.png`: final-time truth/PINN/RK4 fields, signed and
  absolute PINN error, RK4 absolute error, truth/PINN front contours, and a compact
  accuracy summary.
- `pinn_evolution.gif`: animated truth/PINN/signed-error/absolute-error panels across
  the time horizon with per-frame relative L2 and run-level epoch/error caption. It is a
  diagnostic artifact, not an accuracy proof; low-epoch smoke runs and high-error runs
  are labeled directly in the GIF and in `metrics.json`.
- `training_diagnostics.png`: loss components including known IC, moving-front speed,
  parabolic mass balance, learned `D/r`, residual curriculum, adaptive loss multipliers,
  front-profile alignment, and front/sparsity diagnostics.

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

References and method rationale are maintained in
`docs/fisher_kpp_pinn_review_response.docx`.
