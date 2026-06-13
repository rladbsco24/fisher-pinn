# Korea Pine-Wilt PINN Experiment

This entrypoint runs the Korea Forest Service pine-wilt forward Fisher-KPP
comparison using the committed compact data and land mask.

```bash
python experiments/korea_pine_pinn/run_pine_pinn.py --preset full
```

For a high-budget run using the same family of training budget as the synthetic
flagship PINN runs:

```bash
python experiments/korea_pine_pinn/run_pine_pinn.py --preset flagship --output-dir runs/korea_pine_pinn_flagship
```

The run compares observed density, RK4 simulation, and the repository PINN on
the same initial density grid. The PINN now uses the same phase-capable
`OriginPINN + FrontPhaseHead` backbone as the synthetic flagship moving-front
experiments. For Korea pine-wilt the phase head is not supervised by an exact
front; instead it receives a weak pseudo-front anchor from the first observed
support contour and weak phase/field compatibility regularization. The Korea
loss stack also uses land-only PDE collocation, sea-exclusion loss,
known-initial-condition fitting, support Tversky plus support-area guards,
mass-trajectory guards, and learned effective diffusion/reaction parameters.

The `flagship` preset uses 20,000 PINN epochs, a 128 x 128 Korea grid, larger
land-only PDE/initial-condition/mass/phase batches, 120 RK4 steps per year,
and checkpoint/resume every 25 epochs.

Use `--time-axis pre_action_month --raw-csv-dir <Data>` when the original raw
Korea Forest Service CSV bundle is available and the comparison should be made
only up to the inferred large-scale control-action cutoff.
