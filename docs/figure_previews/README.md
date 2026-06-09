# Figure Previews

These figures are committed previews from reproducible script runs. They show what each
code path produces before a user runs the notebooks or scripts locally. The
Geo-Spectral PINN and Korea pine-wilt PINN previews were refreshed from full
1200-epoch runs on 2026-06-09.

## RK4 1D And 2D Demos

![1D RK4 snapshots](rk4_demo_snapshots_1d.png)

![1D front position](rk4_demo_front_position_1d.png)

![2D RK4 snapshots](rk4_demo_snapshots_2d.png)

![2D front area and mass](rk4_demo_front_area_mass_2d.png)

## Long-Time Fisher-KPP Integrator Comparison

![Long-time RK4 surface](rk4_long_time_surface.png)

![Long-time probe rho](rk4_long_time_probe_rho.png)

![Long-time front and mass](rk4_long_time_front_mass.png)

![Long-time final profiles](rk4_long_time_final_profiles.png)

![Adjusted RK4 surface](rk4_adjusted_long_time_surface.png)

![Adjusted RK4 trends](rk4_adjusted_long_time_trends.png)

## Damped Long-Time Curve Benchmark

![FE BE trapezoidal RK4 damped curve trend](rk4_curve_trend.png)

![FE BE trapezoidal RK4 damped curve error](rk4_curve_error.png)

![PINN damped curve trend](pinn_curve_trend.png)

![PINN damped curve diagnostics](pinn_curve_diagnostics.png)

## Geo-Spectral PINN Diagnostic Preview

Run source: `python scripts\run_inverse_origin.py --geo-spectral-forward --epochs 1200 --ensemble 1 --run-classical-baseline --out-dir runs\figure_update_full_geo_spectral_forward`.
The refreshed run produced `pinn_final_time_relative_l2 = 0.9769`,
`pinn_final_time_max_abs_error = 0.1955`, `rk4_final_time_relative_l2 = 0.00147`,
`validation_observation_mse = 0.00578`, `front_area_010_mae = 0.01587`, and
`mass_mae = 0.00437`.

![PINN observation coverage](pinn_observation_coverage.png)

![PINN reconstruction](pinn_reconstruction.png)

![PINN spacetime error](pinn_spacetime_error.png)

![PINN residual front diagnostics](pinn_residual_front_diagnostics.png)

![PINN vs RK4 comparison](pinn_vs_rk4_comparison.png)

![PINN training diagnostics](pinn_training_diagnostics.png)

![PINN evolution GIF](pinn_evolution.gif)

## Korea Pine-Wilt Preview

Run source: `python scripts\run_korea_pine_wilt_simulation.py --output-dir runs\figure_update_korea_pine_wilt_full --pinn-epochs 1200`.
The refreshed run used grid size 96, 80 RK4 steps/year, and full 1200-epoch PINN
training. It produced PINN mean observed-year relative L2 `0.6883` and RK4 mean
observed-year relative L2 `2.5024` under the committed land-mask Fisher-KPP setup.

![Korea observed density by year](korea_observed_density_by_year.png)

![Korea RK4 forecast timeline](korea_rk4_forecast_timeline.png)

![Korea observed vs simulated metrics](korea_observed_vs_simulated_metrics.png)

![Korea RK4 vs PINN metric comparison](korea_baseline_metric_comparison.png)

![Korea PINN baseline observed years](korea_pinn_baseline_observed_years.png)

![Korea map baseline preview](korea_map_baselines_preview.png)

![Korea map baseline GIF](korea_map_baselines.gif)

![Korea error baseline preview](korea_error_baselines_preview.png)

![Korea error baseline GIF](korea_error_baselines.gif)

## Ablation Summaries

![Inverse ablation summary](inverse_ablation_summary.png)

![Forward ablation summary](forward_ablation_summary.png)
