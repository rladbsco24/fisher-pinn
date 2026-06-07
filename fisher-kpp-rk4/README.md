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
- `notebooks/fisher_kpp_rk4_demo.ipynb`: notebook version of the 1D/2D demo.
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
- `tol` and `max_iter` are retained only for compatibility with implicit-solver experiments; RK4 does not use nonlinear iterations.

