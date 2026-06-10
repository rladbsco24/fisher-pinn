# Fisher-KPP RK4

This repository solves Fisher-KPP reaction-diffusion equations with Method of
Lines, second-order central finite differences in space, and classical RK4 in time.
It includes both:

- **1D Ablowitz-Zeppetella traveling-wave Fisher-KPP** with time-dependent
  Dirichlet boundaries.
- **2D generalized Fisher-KPP traveling wave** with `p=1`, `phi=pi/4`,
  exact time-dependent Dirichlet boundaries, and relative-L2 checks against the
  analytic tanh solution.

## Equations

1D:

```text
u_t = D u_xx + r u(1-u)
```

2D:

```text
u_t = D (u_xx + u_yy) + r u(1-u)
```

The default demo uses `D=r=1`. In 2D this is the `p=1` member of the
generalized Fisher-KPP exact-wave family, so the reaction term is again
`u(1-u)`.

## Project Structure

- `src/fisher_kpp_rk4/config.py`: 1D and 2D default PDE/domain/initial-condition settings.
- `src/fisher_kpp_rk4/solver.py`: 1D/2D RHS functions, RK4 steps, solvers, stability checks, and diagnostics.
- `scripts/run_demo.py`: runs both 1D and 2D examples and saves NPZ/PNG outputs.
- `scripts/run_convergence.py`: compact 1D and 2D convergence sanity check.
- `scripts/run_long_time_methods.py`: fair long-time 1D comparison of forward Euler, backward Euler, trapezoidal, and RK4.
- `scripts/run_long_time_rk4_adjusted.py`: adjusted RK4-only long-time run using the same fair parameter set.
- `scripts/run_long_time_curve_trend.py`: reference-style damped `rho(t)` curve trend comparison for forward Euler, backward Euler, trapezoidal, and RK4.
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
- `fisher_kpp_rk4_2d_report_visualization.npz`
- `snapshots_1d.png`
- `front_position_1d.png`
- `snapshots_2d.png`
- `front_area_mass_2d.png`
- `centerline_2d_exact_wave.png`
- `surface_2d_exact_wave_t00.png`, `surface_2d_exact_wave_t02.png`, ..., `surface_2d_exact_wave_t08.png`
- `absolute_error_2d_surface_t00.png`, `absolute_error_2d_surface_t02.png`, ..., `absolute_error_2d_surface_t08.png`

## Run as Notebook

Open `notebooks/fisher_kpp_rk4_demo.ipynb` in Jupyter, VS Code, or Colab and run cells sequentially.

## Optional Convergence Check

```bash
python scripts/run_convergence.py
```

This writes RK4-based 1D/2D spatial-grid and time-step comparison tables under
`outputs/tables/` as CSV, Markdown, and PNG:

- `rk4_1d_spatial_comparison.{csv,md,png}`
- `rk4_1d_time_comparison.{csv,md,png}`
- `rk4_2d_spatial_comparison.{csv,md,png}`
- `rk4_2d_time_comparison.{csv,md,png}`
- `rk4_grid_comparison_tables.md`

The table columns mirror the report-style numerical-method tables while using
RK4-appropriate diagnostics. Because RK4 has no Newton iteration, the tables
report `rk4_stages_per_step = 4` and `stability_safe` instead of Newton
iteration and nonlinear residual columns. For the 2D `Nx=Ny=121` time-step
comparison, `dt=0.04` is excluded because it is outside the explicit RK4
diffusion stability estimate; the largest reported time step is `dt=0.02`.

The default short benchmark follows the exact traveling-wave regimes used for
direct numerical verification:

```text
1D:
u_t = u_xx + u(1-u)
c = 5/sqrt(6)
u(x,t) = (1 + exp((x - c t - x0)/sqrt(6)))^-2
x in [-20, 20], T=10, Nx=201, dt=0.005

2D:
u_t = u_xx + u_yy + u(1-u)
u(x,y,t) = [0.5 tanh((x+y)/(4 sqrt(3)) + 5t/12) + 0.5]^2
x,y in [-15, 15], T=3, grid=61x61, dt=0.01
```

Initial conditions and all Dirichlet boundary values are taken from the same
exact solution. The RK4 method, finite-difference stencil, and comparison
metrics are unchanged; only the benchmark problem definition is set to the
formal exact-wave regime above.

The script also exports the report-style 2D figures used for visual inspection:
the centerline \(u(x,0,t)\), five 3D solution surfaces, and five 3D
absolute-error surfaces at \(t=0,2,4,6,8\). This extended visualization uses the
same generalized Fisher-KPP equation, grid, time step, exact initial condition,
and exact Dirichlet boundaries as the 2D benchmark; it only extends the final
visualized time to \(T=8\) so the moving front and error surfaces match the
report figures.

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

### Long-Time Fair-Parameter Rationale

This section explains why the long-time Fisher-KPP parameters were selected for
the explicit/implicit method comparison. The goal is not to make one integrator
look best, and it is not to reproduce an oscillatory `rho(t)` curve. The goal is
to compare forward Euler, backward Euler, trapezoidal, and RK4 on the same
Fisher-KPP traveling-front regime under the same spatial grid, time step,
initial condition, boundary condition, output times, and metrics.

The common PDE is:

```text
u_t = D u_xx + r u(1-u)
```

The chosen regime is:

```text
D = 0.06
r = 0.25
L = 30
T = 30
Nx = 181
dx = L / (Nx - 1) = 1/6 = 0.166666...
dt = 0.05
Nt = 600
u(x,0) = 1 / (1 + exp((x - 7.0) / 0.9))
u(0,t) = 1
u(L,t) = 0
probe_x = 12
save_interval = 0.5
```

The diffusion and reaction values were chosen to keep the solution in a clear
Fisher-KPP pulled-front regime. The asymptotic minimal front speed is:

```text
c* = 2 sqrt(D r) = 2 sqrt(0.06 * 0.25) ~= 0.245
```

Over `T=30`, this speed moves the front by roughly:

```text
c* T ~= 7.35
```

The initial sigmoid front is centered at `x0=7`, so the final front is expected
near `x ~= 14.3`, well inside the domain `[0, 30]`. This is deliberate. If the
front reaches the right boundary, the method comparison becomes contaminated by
the Dirichlet boundary condition. With the present setting, the run is long
enough to show front propagation, mass growth, probe evolution, and numerical
diffusion, but short enough to avoid boundary-front interaction.

The natural Fisher-KPP front thickness scale is:

```text
sqrt(D / r) = sqrt(0.06 / 0.25) ~= 0.49
```

The initial sigmoid width is `0.9`, so the initial front is not a grid-scale
discontinuity. With `dx=1/6`, the front transition is represented by several grid
points. This matters because a very sharp initial jump would mostly test spatial
under-resolution and limiter behavior, not the time integrators. The selected
grid is therefore fine enough for a clean method-of-lines comparison while still
small enough for implicit Newton solves to run quickly in notebooks.

The time step `dt=0.05` was selected so that explicit methods can participate
fairly. For forward Euler, the 1D diffusion stability scale is:

```text
dt <= dx^2 / (2D)
   = (1/6)^2 / (2 * 0.06)
   ~= 0.231
```

The logistic reaction scale is:

```text
1 / r = 4.0
```

Diffusion is therefore the restrictive explicit scale. The actual time step is:

```text
dt = 0.05
D dt / dx^2 = 0.06 * 0.05 / (1/6)^2 ~= 0.108
```

So forward Euler is comfortably inside the diffusion stability limit. RK4 has a
larger practical stability allowance for the semi-discrete diffusion operator,
and this code reports that through `check_rk4_stability`. The important point is
that RK4 is not given a looser time step in the accuracy comparison. All methods
use the same `dt=0.05`; any difference in front position, profile smoothing, mass,
or probe value comes from the time-integration method, not from different
temporal resolution.

Backward Euler and trapezoidal are implicit. They can remain stable at much
larger time steps than forward Euler, but using a larger `dt` for them would turn
the experiment into an efficiency comparison rather than an accuracy comparison.
For this reason, the implicit methods also use `dt=0.05`. Their advantage is then
visible as different numerical damping and phase behavior, not as permission to
use a coarser time grid.

The implicit methods solve the nonlinear theta-method system with Newton
iterations. The tolerance and iteration cap are:

```text
tol = 1e-10
max_iter = 30
```

These values are intentionally tighter than the visual/metric scale of the
method comparison. The purpose is to keep nonlinear solve error below the
time-discretization and spatial-discretization errors. If the Newton tolerance
were loose, backward Euler or trapezoidal could look artificially diffusive or
inaccurate because the nonlinear solve stopped too early. With this setting, the
reported differences are dominated by the numerical time integrator itself.

The boundary conditions are fixed as:

```text
left_bc = 1
right_bc = 0
```

This creates a standard traveling-front benchmark: the left side is the invaded
state and the right side is the uninvaded state. It also makes the front position
metric well-defined. Since the front stays far from the right boundary during
`T=30`, the right boundary mostly stabilizes the far-field state rather than
driving the result.

The probe point is:

```text
probe_x = 12
```

This location is chosen ahead of the initial front center but behind the expected
final front. The probe therefore sees a meaningful transition from low density to
higher density during the run. A probe too far left would saturate near one; a
probe too far right would remain near zero. `probe_x=12` gives a useful scalar
trace for comparing numerical diffusion and front timing.

The comparison metrics are:

- final and time-dependent front position,
- mean mass,
- fixed-point probe trace `rho(t)=u(probe_x,t)`,
- final profile,
- final relative L2 against the RK4 reference in the all-method summary,
- Newton iteration diagnostics for the implicit methods.

The RK4 curve is used as the plotted reference in the method summary because it
is the highest-order method among the four at the same `dx` and `dt`. This does
not mean RK4 is treated as an exact analytical solution. It is a practical
same-grid reference for comparing lower-order explicit and implicit schemes.
For a formal convergence study, reduce `dx` and `dt` together and compare all
methods against a refined reference.

The fairness rule for this benchmark is:

```text
same PDE
same D and r
same initial condition
same boundary condition
same domain
same dx
same dt
same output times
same front/probe/mass/profile metrics
```

Changing `dt` independently for implicit methods is valid for a separate
cost-to-accuracy study, but it is not the experiment encoded here. This benchmark
answers: under one shared Fisher-KPP discretization, how do forward Euler,
backward Euler, trapezoidal, and RK4 differ in accuracy, damping, and front
timing?

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

## Reference-Style Long-Time Curve Trend

If the goal is the right-panel-style `rho(t)` curve rather than a Fisher-KPP
front probe, use the separate damped-oscillator benchmark:

```bash
python scripts/run_long_time_curve_trend.py
```

It uses:

```text
rho_inf=0.34, alpha=0.24, omega_d=1.0, rho(0)=0, rho_t(0)=0.60, T=30, dt=0.05
```

The exact curve rises sharply, overshoots, then decays toward the long-time
level with damped oscillations. The script overlays the exact curve and the
forward Euler, backward Euler, trapezoidal, and RK4 results in
`outputs/long_time_curve_trend/long_time_curve_trend.png`.

## Tests

```bash
pip install -e .[demo]
pytest -q
```

The tests also parse the notebook code cells so broken markdown-string cells are caught
before release.

## Notes

- Classical RK4 is explicit. The solver reports a practical diffusion-reaction stability estimate.
- The 2D default uses `D=1`, `r=1`, `x,y in [-15,15]`, `T=3`, `grid=61`, and exact time-dependent Dirichlet boundaries from the generalized Fisher-KPP tanh wave.
- Scalar Fisher-KPP with positive diffusion and logistic reaction follows a maximum-principle/front-propagation regime. The long-time plots therefore show surface, probe, front, and mass trends; they are not intended to reproduce intrinsic damped oscillations unless the PDE/model is changed.
- `tol` and `max_iter` are retained only for compatibility with implicit-solver experiments; RK4 does not use nonlinear iterations.
