# Fisher-KPP RK4

This repository solves Fisher-KPP reaction-diffusion equations with Method of
Lines, second-order central finite differences in space, and classical RK4 in time.
It includes both:

- **1D traveling-front Fisher-KPP** with Dirichlet boundaries.
- **2D square-domain Fisher-KPP** with no-flux boundaries and a Gaussian seed,
  matching the forward PINN/RK4 comparison problem used in the companion PINN lab.

## Equations

1D:

```text
u_t = D u_xx + r u(1-u)
```

2D:

```text
u_t = D (u_xx + u_yy) + r u(1-u)
```

## Project Structure

- `src/fisher_kpp_rk4/config.py`: 1D and 2D default PDE/domain/initial-condition settings.
- `src/fisher_kpp_rk4/solver.py`: 1D/2D RHS functions, RK4 steps, solvers, stability checks, and diagnostics.
- `scripts/run_demo.py`: runs both 1D and 2D examples and saves NPZ/PNG outputs.
- `scripts/run_convergence.py`: compact 1D and 2D convergence sanity check.
- `scripts/run_long_time_methods.py`: fair long-time 1D comparison of forward Euler, backward Euler, trapezoidal, and RK4.
- `scripts/run_long_time_rk4_adjusted.py`: adjusted RK4-only long-time run using the same fair parameter set.
- `notebooks/fisher_kpp_rk4_demo.ipynb`: notebook version of the 1D/2D demo.
- `notebooks/fisher_kpp_long_time_methods.ipynb`: Colab-aware long-time method-comparison notebook.
- `tests/test_solver.py`: smoke tests for solver shape, bounds, stability, and notebook integrity.

## Run as Python Script

```bash
cd fisher-kpp-rk4
pip install -r requirements.txt
python scripts/run_demo.py
```

Results are saved under `outputs/`:

- `fisher_kpp_rk4_1d_results.npz`
- `fisher_kpp_rk4_2d_results.npz`
- `snapshots_1d.png`
- `front_position_1d.png`
- `snapshots_2d.png`
- `front_area_mass_2d.png`

## Run as Notebook

Open `notebooks/fisher_kpp_rk4_demo.ipynb` in Jupyter, VS Code, or Colab and run cells sequentially.

## Optional Convergence Check

```bash
python scripts/run_convergence.py
```

This writes `outputs/convergence_summary.csv` with both 1D and 2D relative-L2 checks.

## Long-Time Method Comparison

The long-time setup uses one shared 1D Fisher-KPP parameter set for all four
time integrators:

```text
D=0.06, r=0.25, L=30, T=30, Nx=181, dt=0.05
left_bc=1, right_bc=0
```

This grid/time step is inside the practical explicit stability limits for both
forward Euler and RK4, while backward Euler and trapezoidal use Newton solves for
the same nonlinear semi-discrete system.

Run all four methods:

```bash
python scripts/run_long_time_methods.py
```

Run the adjusted RK4-only version:

```bash
python scripts/run_long_time_rk4_adjusted.py
```

Long-time outputs include:

- `long_time_method_summary.csv`
- `long_time_method_results.npz`
- `adjusted_rk4_long_time_surface.png`
- `long_time_probe_rho.png`
- `long_time_front_mass.png`
- `long_time_final_profiles.png`

The companion notebook is `notebooks/fisher_kpp_long_time_methods.ipynb`.
It is Colab-aware and will clone the GitHub repository under `/content` if it
is not already running from a local checkout.

## Tests

```bash
pip install -e .[demo]
pytest -q
```

The tests also parse the notebook code cells so broken markdown-string cells are caught
before release.

## Notes

- Classical RK4 is explicit. The solver reports a practical diffusion-reaction stability estimate.
- The 2D default uses `D=0.02`, `r=3.0`, `box=1.0`, `T=0.5`, `grid=51`, and `160` RK4 steps.
- Scalar Fisher-KPP with positive diffusion and logistic reaction follows a maximum-principle/front-propagation regime. The long-time plots therefore show surface, probe, front, and mass trends; they are not intended to reproduce intrinsic damped oscillations unless the PDE/model is changed.
- `tol` and `max_iter` are retained only for compatibility with implicit-solver experiments; RK4 does not use nonlinear iterations.
