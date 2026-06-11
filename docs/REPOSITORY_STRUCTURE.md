# Repository Structure

The repository now separates stable implementation packages, canonical
experiment entrypoints, notebooks, documents, and generated outputs.

## Stable Implementation Packages

`fisher_origin_lab/` contains the PINN implementation used by both the synthetic
Fisher-KPP experiments and the Korea pine-wilt experiment. It includes model
definitions, losses, samplers, RK4 references, plotting, Korea data processing,
and training loops.
The synthetic and Korea PINN tracks share the same forward backbone helpers;
Korea-specific code is limited to land-mask data handling, collocation masks,
sea exclusion, and observation-support losses.

`fisher-kpp-rk4/` contains the independent finite-difference RK4 solver package
for exact-solution numerical baselines. Its `src/` directory is importable as
`fisher_kpp_rk4` when `fisher-kpp-rk4/src` is on `PYTHONPATH`.

## Canonical Experiment Entrypoints

`experiments/basic_pinn/` is the official entrypoint for synthetic PINN runs.
It exposes the 1D Ablowitz-Zeppetella exact-wave benchmark and the 2D Gaussian
moving-front benchmark without requiring users to remember the lower-level
script flags. Both entrypoints support `smoke`, `quick`, `full`, and `flagship`
presets. `flagship` is the high-budget research run with 20,000 epochs,
large collocation/front batches, causal time slabs, L-BFGS polish, and resume
checkpoints.

`experiments/korea_pine_pinn/` is the official entrypoint for the Korea
Forest Service pine-wilt PINN/RK4 comparison. It also supports `flagship`,
using a larger Korea grid and the same 20,000-epoch PINN budget family.

`experiments/rk4_exact/` is the official entrypoint for RK4 exact-solution
baselines, including the required 1D, 2D, and 3D surface/error figure exports.

## Notebooks

The root notebooks remain in place for Colab compatibility:

```text
fisher_kpp_origin_lab.ipynb
fisher_kpp_origin_lab_colab.ipynb
korea_pine_wilt_fisher_kpp_lab.ipynb
```

The RK4 notebooks remain under `fisher-kpp-rk4/notebooks/`.

## Documentation And Preview Figures

Technical reports are stored in `docs/`. GitHub-readable figure previews are
stored in `docs/figure_previews/`.

## Generated Outputs

`runs/`, `fisher-kpp-rk4/outputs/`, caches, and temporary artifacts are generated
locally and should not be versioned. When a result is important for GitHub
inspection, copy a curated preview into `docs/figure_previews/` with a clear
filename and source description.
