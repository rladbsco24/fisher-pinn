# RK4 Exact-Solution Experiments

These entrypoints run the finite-difference method-of-lines RK4 baselines used
as deterministic references for the PINN work.

```bash
python experiments/rk4_exact/run_exact_1d.py
python experiments/rk4_exact/run_exact_2d.py
python experiments/rk4_exact/run_exact_3d_figures.py
python experiments/rk4_exact/run_all.py
```

The 1D benchmark is the Ablowitz-Zeppetella exact traveling wave for
`u_t = u_xx + u(1-u)`. The 2D benchmark is the generalized Fisher-KPP
traveling wave with exact time-dependent Dirichlet boundaries. The 3D script
does not solve a separate 3D PDE; it exports 3D surface and absolute-error
figures for the 2D exact-wave solution at `t = 0, 2, 4, 6, 8`, matching the
report figures requested for visual inspection.
