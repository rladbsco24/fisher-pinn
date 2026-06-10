# Basic PINN Experiments

The basic PINN family contains two synthetic Fisher-KPP benchmarks.

## 1D Ablowitz-Zeppetella Exact-Wave PINN

```bash
python experiments/basic_pinn/run_zeppe_exact_1d.py --preset full
```

This benchmark uses

```text
u_t = u_xx + u(1-u)
u(x,t) = (1 + exp((x - 5t/sqrt(6) - x0)/sqrt(6)))^-2
x in [-20, 20], T=10
```

The shared PINN backbone is two-dimensional, so the 1D exact wave is represented
as a `y`-invariant 2D field. Metrics are evaluated against the exact
Ablowitz-Zeppetella traveling wave.

## 2D Gaussian Moving-Front PINN

```bash
python experiments/basic_pinn/run_gaussian_moving_front_2d.py --preset full
```

This benchmark uses the repository's Geo-Spectral forward PINN configuration on
a 2D Gaussian seed moving-front Fisher-KPP problem. It includes known-initial
condition loss, front-aware sampling, level-set/front losses, time curriculum,
and the RK4 teacher/reference diagnostics already implemented in
`fisher_origin_lab`.
