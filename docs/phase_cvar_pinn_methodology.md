# Phase-CVaR PINN Addendum

This addendum documents two physics-only losses added to the baseline and Korea
pine-wilt PINN paths. They do not use RK4 solution labels and therefore remain
compatible with the non-teacher flagship protocol.

## Logit-Phase Fisher Residual

For the Fisher-KPP equation

```text
u_t + v . grad(u) - div(D grad(u)) - r u(1-u) = 0,
```

the standard residual contains the multiplicative factor `u(1-u)` in the
reaction and, after a logit transformation, in the entire equation. This is a
problem for moving fronts because the leading edge has small `u`: a model can
place the front incorrectly while the mean squared residual remains small.

Let

```text
q = logit(u) = log(u / (1-u)),     0 < u < 1.
```

Then

```text
u_t = u(1-u) q_t
grad(u) = u(1-u) grad(q)
Delta(u) = u(1-u) [Delta(q) + (1-2u)|grad(q)|^2].
```

For constant `D`, division by `u(1-u)` gives the equivalent phase equation

```text
q_t + v . grad(q) - D Delta(q) - D(1-2u)|grad(q)|^2 - r = 0.
```

For spatially varying diffusion, the additional exact term is
`- grad(D) . grad(q)`. The Korea pine-wilt path uses the same transformation
with separate `x` and `y` diffusion scales.

The implemented loss applies this residual only on the active range
`active_low < u < active_high` with a smooth mask and clipped residual. This
keeps the mathematical signal in the low-density leading edge while avoiding
singular behavior at exactly `u=0` or `u=1`.

## Residual CVaR / Soft L-infinity Tail

The second added loss is a conditional-value-at-risk proxy over the collocation
residual:

```text
CVaR_alpha(R^2) = mean of the largest alpha-fraction of residual squared values.
```

This complements the ordinary mean residual. The mean residual measures global
physics fit, while CVaR penalizes the worst local residual band. In a front
problem, the problematic set is geometrically small, so a pure L2 average can
dilute the exact region that determines front phase and speed. CVaR is a stable
finite-sample proxy for an L-infinity/adversarial residual objective.

## Implementation Locations

The base PINN path uses:

```text
fisher_origin_lab/losses.py::logit_phase_residual_loss
fisher_origin_lab/losses.py::residual_cvar_loss
fisher_origin_lab/train.py
```

The Korea pine-wilt path uses:

```text
fisher_origin_lab/korea_data.py::_korea_anisotropic_logit_phase_residual_loss
fisher_origin_lab/korea_data.py::fit_korea_pine_wilt_pinn
```

The default forward PINN preset enables `phase_pde=0.18` and
`residual_cvar=0.12`. The Ablowitz-Zeppetella benchmark enables
`phase_pde=0.25` and `residual_cvar=0.10`. The Korea path enables weaker
defaults because sparse observational support, sea exclusion, and mass
trajectory constraints remain the dominant data terms.

## Research Basis

The logit-phase residual is based on an exact Fisher-KPP variable transform and
the classical traveling-front structure of Fisher-KPP waves. It directly targets
the same front degeneracy described in moving-interface PINN work that uses
level-set/front-aware training.

The CVaR residual follows the same motivation as L-infinity/adversarial PINN
training: small average residual is not always a reliable proxy for solution
accuracy, especially when the PDE stability or the important physical structure
depends on localized regions. In practice it behaves like residual-adaptive
collocation but enters as a differentiable training objective on the already
sampled points.
