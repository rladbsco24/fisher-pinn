# Korea Pine-Wilt PINN Experiment

This entrypoint runs the Korea Forest Service pine-wilt forward Fisher-KPP
comparison using the committed compact data and land mask.

```bash
python experiments/korea_pine_pinn/run_pine_pinn.py --preset full
```

The run compares observed density, RK4 simulation, and the repository PINN on
the same initial density grid. The PINN uses land-only PDE collocation,
sea-exclusion loss, known-initial-condition fitting, support/mass guards, and
learned effective diffusion/reaction parameters.

Use `--time-axis pre_action_month --raw-csv-dir <Data>` when the original raw
Korea Forest Service CSV bundle is available and the comparison should be made
only up to the inferred large-scale control-action cutoff.
