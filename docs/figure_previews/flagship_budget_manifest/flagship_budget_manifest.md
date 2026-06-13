# Flagship PINN Budget Manifest

Execution plan for high-budget PINN runs intended to close the accuracy gap to the same-problem RK4 references without RK4 teacher labels or postprocessing.

Default epochs: `20000`

The listed commands are prepared for execution but this script does not run them.

## basic_1d_az_flagship

Problem: Ablowitz-Zeppetella 1D traveling-front benchmark in the shared 2D PINN backbone

Command:

```powershell
python experiments/basic_pinn/run_zeppe_exact_1d.py --preset flagship --epochs 20000 --out-dir runs\basic_pinn\zeppe_exact_1d_flagship
```

Output directory: `runs\basic_pinn\zeppe_exact_1d_flagship`

Notes: Uses exact initial/boundary constraints and traveling-wave geometry, but no RK4 pseudo-labels or after-the-fact correction.

## basic_2d_gaussian_flagship

Problem: 2D Gaussian moving-front Fisher-KPP benchmark

Command:

```powershell
python experiments/basic_pinn/run_gaussian_moving_front_2d.py --preset flagship --epochs 20000 --run-classical-baseline --out-dir runs\basic_pinn\gaussian_moving_front_2d_flagship
```

Output directory: `runs\basic_pinn\gaussian_moving_front_2d_flagship`

Notes: Uses the same geo-spectral/front-aware model family and logs the RK4 reference as an evaluation baseline, not as supervised training labels.

## korea_pine_flagship

Problem: Korea pine-wilt land-mask reaction-diffusion PINN/RK4 comparison

Command:

```powershell
python experiments/korea_pine_pinn/run_pine_pinn.py --preset flagship --pinn-epochs 20000 --output-dir runs\korea_pine_pinn_flagship
```

Output directory: `runs\korea_pine_pinn_flagship`

Notes: Uses the Korea-specific land mask, sea exclusion, and observation timeline, while keeping the same phase-capable PINN backbone and explicit training budget scale as the synthetic flagship runs.
