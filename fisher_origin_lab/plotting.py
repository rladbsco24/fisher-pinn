from __future__ import annotations

from pathlib import Path
from typing import Iterable

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import DomainConfig, ObservationConfig, SeedConfig
from .losses import front_indicator_weights, front_speed_kinematics, pde_residual_terms
from .metrics import centroid_from_field, relative_l2
from .models import OriginPINN
from .simulate import ObservationData, TruthData, truth_field_at


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "blue": "#5477C4",
    "blue_light": "#CEDFFE",
    "gold": "#B8A037",
    "orange": "#CC6F47",
    "olive": "#71B436",
    "pink": "#BD569B",
    "rose": "#D95F5F",
    "teal": "#2A9D8F",
    "neutral": "#7A828F",
}


def _apply_chart_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "xtick.color": TOKENS["muted"],
            "ytick.color": TOKENS["muted"],
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
        }
    )


def _style_axis(ax: plt.Axes, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(labelsize=8)
    if grid:
        ax.grid(True, alpha=0.7)
    else:
        ax.grid(False)


def _write_title(fig: plt.Figure, title: str, subtitle: str | None = None, *, top: float = 0.88) -> None:
    fig.subplots_adjust(top=top)
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=13, fontweight="semibold", color=TOKENS["ink"])
    if subtitle:
        fig.text(0.01, 0.952, subtitle, ha="left", va="top", fontsize=9, color=TOKENS["muted"])


def _imshow(
    fig: plt.Figure,
    ax: plt.Axes,
    field: np.ndarray,
    domain: DomainConfig,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
) -> None:
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=[0, domain.box, 0, domain.box],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9, color=TOKENS["ink"])
    ax.set_xticks([])
    ax.set_yticks([])
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _as_history_arrays(history: list[dict[str, float]], key: str) -> tuple[np.ndarray, np.ndarray]:
    rows = [row for row in history if key in row and np.isfinite(row[key])]
    if not rows:
        return np.array([]), np.array([])
    return np.array([row["epoch"] for row in rows], dtype=np.float64), np.array([row[key] for row in rows], dtype=np.float64)


def predict_field(model: OriginPINN, time: float, n: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, model.domain.box, n)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    xy = torch.tensor(np.stack([x.ravel(), y.ravel()], axis=1), dtype=torch.float32, device=device)
    t = torch.full((n * n, 1), float(time), dtype=torch.float32, device=device)
    with torch.no_grad():
        field = model(xy, t).detach().cpu().numpy().reshape(n, n)
    return xs, field


def residual_front_maps(
    model: OriginPINN,
    time: float,
    n: int,
    device: torch.device,
    *,
    front_alpha: float = 0.0,
    front_gradient: float = 0.0,
    front_speed_min_grad: float = 1.0e-2,
) -> dict[str, np.ndarray]:
    xs = np.linspace(0.0, model.domain.box, n)
    x, y = np.meshgrid(xs, xs, indexing="ij")
    xy = torch.tensor(np.stack([x.ravel(), y.ravel()], axis=1), dtype=torch.float32, device=device)
    t = torch.full((n * n, 1), float(time), dtype=torch.float32, device=device)
    residual, u, u_xy, _, _ = pde_residual_terms(model, xy, t)
    grad_norm = torch.linalg.norm(u_xy, dim=-1, keepdim=True)
    weights = front_indicator_weights(u, u_xy, front_alpha, front_gradient)
    kin = front_speed_kinematics(model, xy, t)
    front_band = (
        (u.detach() > 0.05)
        & (u.detach() < 0.95)
        & (grad_norm.detach() > front_speed_min_grad)
    ).cpu().numpy().reshape(n, n)
    normal_speed = kin["observed_normal_speed"].detach().cpu().numpy().reshape(n, n)
    target_speed = kin["target_normal_speed"].detach().cpu().numpy().reshape(n, n)
    speed_error = np.abs(kin["speed_error"].detach().cpu().numpy().reshape(n, n))
    return {
        "u": u.detach().cpu().numpy().reshape(n, n),
        "residual_abs": residual.detach().abs().cpu().numpy().reshape(n, n),
        "front_indicator": (u.detach() * (1.0 - u.detach())).cpu().numpy().reshape(n, n),
        "grad_norm": grad_norm.detach().cpu().numpy().reshape(n, n),
        "front_weight": weights.detach().cpu().numpy().reshape(n, n),
        "normal_speed": np.where(front_band, normal_speed, np.nan),
        "target_speed": np.where(front_band, target_speed, np.nan),
        "front_speed_error": np.where(front_band, speed_error, np.nan),
    }


def save_reconstruction_figure(
    path: Path,
    truth: TruthData,
    model: OriginPINN,
    domain: DomainConfig,
    observations: ObservationConfig,
    seed: SeedConfig,
    device: torch.device,
    ensemble_centers: list[tuple[float, float]] | None = None,
) -> None:
    _apply_chart_theme()
    path.parent.mkdir(parents=True, exist_ok=True)
    times = [0.0, observations.start_time * 0.5, observations.start_time, domain.t_end]
    n = 96
    fig, axes = plt.subplots(3, len(times), figsize=(4.0 * len(times), 9.5), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.965, bottom=0.045, top=0.88, hspace=0.26, wspace=0.16)
    true_center = (seed.center_x, seed.center_y)
    xs_late, late = truth_field_at(truth, domain.t_end, n=n)
    centroid = centroid_from_field(xs_late, late)
    est_center = tuple(model.source.center().detach().cpu().numpy().tolist())
    show_source_estimate = bool(getattr(model, "use_source_envelope", False))

    for col, time_value in enumerate(times):
        _, true_field = truth_field_at(truth, time_value, n=n)
        _, pred = predict_field(model, time_value, n=n, device=device)
        err = np.abs(pred - true_field)
        for row, field, label, cmap, vmin, vmax in [
            (0, true_field, "truth", "magma", 0.0, 1.0),
            (1, pred, "pinn", "magma", 0.0, 1.0),
            (2, err, "absolute error", "viridis", 0.0, None),
        ]:
            ax = axes[row, col]
            _imshow(
                fig,
                ax,
                field,
                domain,
                title=f"{label} t={time_value:.2f}",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                colorbar=(col == len(times) - 1),
            )
            ax.plot(*true_center, marker="*", color="cyan", markersize=11, markeredgecolor="white", markeredgewidth=0.8)
            if row == 1 and show_source_estimate:
                ax.plot(*est_center, marker="+", color="lime", markersize=12, markeredgewidth=2.5)
                ax.plot(*centroid, marker="x", color="red", markersize=10, markeredgewidth=2.5)
                if ensemble_centers:
                    arr = np.array(ensemble_centers)
                    ax.scatter(arr[:, 0], arr[:, 1], s=18, c="white", edgecolors="black", linewidths=0.5)

    subtitle = "Rows show truth, prediction, and absolute error. Cyan marks the synthetic source."
    if show_source_estimate:
        subtitle += " Green is the trainable source estimate; red is the late weighted centroid."
    _write_title(fig, "Fisher-KPP field reconstruction", subtitle, top=0.88)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spacetime_error_figure(
    path: Path,
    truth: TruthData,
    model: OriginPINN,
    domain: DomainConfig,
    device: torch.device,
    *,
    n: int = 96,
    max_times: int = 14,
) -> None:
    _apply_chart_theme()
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(truth.times) <= max_times:
        times = truth.times
    else:
        idx = np.linspace(0, len(truth.times) - 1, max_times).astype(int)
        times = truth.times[idx]

    rel_l2 = []
    truth_mass = []
    pred_mass = []
    truth_front = []
    pred_front = []
    truth_level_005 = []
    pred_level_005 = []
    truth_level_010 = []
    pred_level_010 = []
    for time_value in times:
        _, true_field = truth_field_at(truth, float(time_value), n=n)
        _, pred = predict_field(model, float(time_value), n=n, device=device)
        rel_l2.append(relative_l2(pred, true_field))
        truth_mass.append(float(true_field.mean()))
        pred_mass.append(float(pred.mean()))
        truth_front.append(float(np.mean((true_field > 0.1) & (true_field < 0.9))))
        pred_front.append(float(np.mean((pred > 0.1) & (pred < 0.9))))
        truth_level_005.append(float(np.mean(true_field > 0.05)))
        pred_level_005.append(float(np.mean(pred > 0.05)))
        truth_level_010.append(float(np.mean(true_field > 0.10)))
        pred_level_010.append(float(np.mean(pred > 0.10)))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.08, top=0.84, hspace=0.34, wspace=0.24)
    axes[0, 0].plot(times, rel_l2, marker="o", color=COLORS["blue"], linewidth=1.4)
    axes[0, 0].set_ylabel("Relative L2")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_title("field error", fontsize=9)

    axes[0, 1].plot(times, truth_mass, marker="o", color=COLORS["neutral"], linewidth=1.2, label="truth")
    axes[0, 1].plot(times, pred_mass, marker="o", color=COLORS["olive"], linewidth=1.2, label="pinn")
    axes[0, 1].set_ylabel("Mean density")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_title("mass trajectory", fontsize=9)
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[1, 0].plot(times, truth_front, marker="o", color=COLORS["neutral"], linewidth=1.2, label="truth")
    axes[1, 0].plot(times, pred_front, marker="o", color=COLORS["orange"], linewidth=1.2, label="pinn")
    axes[1, 0].set_ylabel("Area fraction")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_title("active-front band", fontsize=9)
    axes[1, 0].legend(frameon=False, fontsize=8)

    axes[1, 1].plot(times, truth_level_005, marker="o", color=COLORS["neutral"], linewidth=1.2, label="truth u>0.05")
    axes[1, 1].plot(times, pred_level_005, marker="o", color=COLORS["teal"], linewidth=1.2, linestyle="--", label="pinn u>0.05")
    axes[1, 1].plot(times, truth_level_010, marker="s", color=COLORS["blue"], linewidth=1.2, label="truth u>0.10")
    axes[1, 1].plot(times, pred_level_010, marker="s", color=COLORS["pink"], linewidth=1.2, linestyle="--", label="pinn u>0.10")
    axes[1, 1].set_ylabel("Area fraction")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_title("low-level front area", fontsize=9)
    axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)

    for ax in axes.ravel():
        _style_axis(ax, grid=True)
    _write_title(
        fig,
        "Space-time prediction diagnostics",
        "Trend views summarize field error, mass, active front, and low-level moving-front area.",
        top=0.84,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_training_diagnostics_figure(
    path: Path,
    history: list[dict[str, float]],
    *,
    true_diffusion: float | None = None,
    true_reaction: float | None = None,
) -> None:
    _apply_chart_theme()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.08, top=0.84, hspace=0.34, wspace=0.24)

    for key, color in [
        ("total", COLORS["blue"]),
        ("data", COLORS["olive"]),
        ("validation_data", COLORS["blue_light"]),
        ("pde", COLORS["orange"]),
        ("ic", COLORS["gold"]),
        ("bc", COLORS["neutral"]),
        ("mass", COLORS["teal"]),
        ("expected_front_pde", COLORS["gold"]),
        ("leading_edge", COLORS["pink"]),
        ("leading_edge_area", COLORS["teal"]),
        ("front_contrast", COLORS["rose"]),
        ("front_profile", COLORS["blue"]),
        ("level_set", COLORS["gold"]),
        ("front_speed", COLORS["blue_light"]),
        ("front_grad", COLORS["pink"]),
        ("grad", COLORS["neutral"]),
    ]:
        epochs, values = _as_history_arrays(history, key)
        if len(values) and np.nanmax(values) > 0.0:
            axes[0, 0].plot(epochs, values, marker="o", linewidth=1.2, label=key, color=color)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("loss components", fontsize=9)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2)

    for key, color, true_value in [
        ("diffusion", COLORS["blue"], true_diffusion),
        ("reaction", COLORS["orange"], true_reaction),
    ]:
        epochs, values = _as_history_arrays(history, key)
        if len(values):
            axes[0, 1].plot(epochs, values, marker="o", linewidth=1.2, label=key, color=color)
            if true_value is not None:
                axes[0, 1].axhline(true_value, color=color, linestyle=":", linewidth=1.0)
    axes[0, 1].set_title("learned PDE coefficients", fontsize=9)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].legend(frameon=False, fontsize=8)

    for key, color in [
        ("front_weight_mean", COLORS["blue"]),
        ("residual_exponent", COLORS["gold"]),
        ("sparse", COLORS["orange"]),
        ("time_window_high", COLORS["pink"]),
        ("origin_error", COLORS["olive"]),
    ]:
        epochs, values = _as_history_arrays(history, key)
        if len(values) and np.nanmax(values) > 0.0:
            axes[1, 0].plot(epochs, values, marker="o", linewidth=1.2, label=key, color=color)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("auxiliary diagnostics", fontsize=9)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Value")
    axes[1, 0].legend(frameon=False, fontsize=8)

    adaptive_keys = [
        ("aw_data", COLORS["olive"]),
        ("aw_pde", COLORS["orange"]),
        ("aw_ic", COLORS["gold"]),
        ("aw_bc", COLORS["neutral"]),
        ("aw_mass", COLORS["teal"]),
        ("aw_expected_front_pde", COLORS["gold"]),
        ("aw_leading_edge", COLORS["pink"]),
        ("aw_front_contrast", COLORS["rose"]),
        ("aw_front_profile", COLORS["blue"]),
        ("aw_level_set", COLORS["gold"]),
        ("aw_front_speed", COLORS["blue_light"]),
        ("aw_front_grad", COLORS["pink"]),
    ]
    plotted_adaptive = False
    for key, color in adaptive_keys:
        epochs, values = _as_history_arrays(history, key)
        if len(values):
            axes[1, 1].plot(epochs, values, marker="o", linewidth=1.2, label=key.replace("aw_", ""), color=color)
            plotted_adaptive = True
    if plotted_adaptive:
        axes[1, 1].set_title("adaptive loss multipliers", fontsize=9)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Multiplier")
        axes[1, 1].legend(frameon=False, fontsize=7, ncol=2)
    else:
        epochs, elapsed = _as_history_arrays(history, "elapsed_sec")
        if len(elapsed):
            axes[1, 1].plot(epochs, elapsed, marker="o", linewidth=1.2, color=COLORS["neutral"])
        axes[1, 1].set_title("runtime accumulation", fontsize=9)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Seconds")

    for ax in axes.ravel():
        _style_axis(ax, grid=True)
    _write_title(
        fig,
        "Training diagnostics",
        "Loss including known IC, moving-front speed, mass balance, adaptive multipliers, and learned coefficients.",
        top=0.84,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_residual_front_diagnostics_figure(
    path: Path,
    truth: TruthData,
    model: OriginPINN,
    domain: DomainConfig,
    device: torch.device,
    *,
    time_value: float,
    front_alpha: float = 0.0,
    front_gradient: float = 0.0,
    n: int = 64,
) -> None:
    _apply_chart_theme()
    path.parent.mkdir(parents=True, exist_ok=True)
    _, true_field = truth_field_at(truth, time_value, n=n)
    maps = residual_front_maps(
        model,
        time_value,
        n=n,
        device=device,
        front_alpha=front_alpha,
        front_gradient=front_gradient,
    )
    abs_error = np.abs(maps["u"] - true_field)
    residual_log = np.log10(maps["residual_abs"] + 1.0e-8)

    panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = [
        ("truth", true_field, "magma", 0.0, 1.0),
        ("prediction", maps["u"], "magma", 0.0, 1.0),
        ("absolute error", abs_error, "viridis", 0.0, None),
        ("log10 |PDE residual|", residual_log, "cividis", None, None),
        ("u(1-u) front indicator", maps["front_indicator"], "plasma", 0.0, 0.25),
        ("front/adaptive weight", maps["front_weight"], "YlGnBu", None, None),
        ("normal front speed", maps["normal_speed"], "coolwarm", None, None),
        ("front-speed error", maps["front_speed_error"], "magma", 0.0, None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.4), constrained_layout=False)
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.06, top=0.84, hspace=0.30, wspace=0.20)
    for ax, (title, field, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        _imshow(fig, ax, field, domain, title=title, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=True)

    _write_title(
        fig,
        "Residual and active-front diagnostics",
        f"Maps at t={time_value:.2f}; residual, active-front weights, and normal speed explain moving-front behavior.",
        top=0.84,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pinn_rk4_comparison_figure(
    path: Path,
    truth: TruthData,
    rk4_truth: TruthData,
    model: OriginPINN,
    domain: DomainConfig,
    device: torch.device,
    metrics: dict[str, float | None],
    *,
    time_value: float,
    n: int = 96,
) -> None:
    _apply_chart_theme()
    path.parent.mkdir(parents=True, exist_ok=True)
    _, true_field = truth_field_at(truth, time_value, n=n)
    _, rk4_field = truth_field_at(rk4_truth, time_value, n=n)
    _, pinn_field = predict_field(model, time_value, n=n, device=device)
    signed_error = pinn_field - true_field
    signed_error_lim = float(np.nanmax(np.abs(signed_error)))
    if not np.isfinite(signed_error_lim) or signed_error_lim <= 0.0:
        signed_error_lim = 1.0e-12
    pinn_error = np.abs(pinn_field - true_field)
    rk4_error = np.abs(rk4_field - true_field)

    fig = plt.figure(figsize=(17.5, 7.6))
    fig.subplots_adjust(left=0.035, right=0.975, bottom=0.07, top=0.84, hspace=0.34, wspace=0.36)
    axes = np.array(
        [
            [
                fig.add_subplot(2, 4, 1),
                fig.add_subplot(2, 4, 2),
                fig.add_subplot(2, 4, 3),
                fig.add_subplot(2, 4, 4),
            ],
            [
                fig.add_subplot(2, 4, 5),
                fig.add_subplot(2, 4, 6),
                fig.add_subplot(2, 4, 7),
                fig.add_subplot(2, 4, 8),
            ],
        ]
    )

    for ax, title, field, cmap, vmin, vmax in [
        (axes[0, 0], "reference", true_field, "magma", 0.0, 1.0),
        (axes[0, 1], "PINN", pinn_field, "magma", 0.0, 1.0),
        (axes[0, 2], "RK4", rk4_field, "magma", 0.0, 1.0),
        (axes[1, 0], "PINN signed error", signed_error, "coolwarm", -signed_error_lim, signed_error_lim),
        (axes[1, 1], "PINN absolute error", pinn_error, "viridis", 0.0, None),
        (axes[1, 2], "RK4 absolute error", rk4_error, "viridis", 0.0, None),
    ]:
        _imshow(fig, ax, field, domain, title=title, cmap=cmap, vmin=vmin, vmax=vmax, colorbar=True)

    contour_ax = axes[0, 3]
    _imshow(fig, contour_ax, true_field, domain, title="front contours", cmap="Greys", vmin=0.0, vmax=max(0.2, float(true_field.max())), colorbar=False)
    xs = np.linspace(0.0, domain.box, n)
    levels = [0.05, 0.10]
    truth_cs = contour_ax.contour(xs, xs, true_field.T, levels=levels, colors=[COLORS["teal"], COLORS["blue"]], linewidths=1.2)
    pinn_cs = contour_ax.contour(xs, xs, pinn_field.T, levels=levels, colors=[COLORS["teal"], COLORS["blue"]], linewidths=1.2, linestyles="--")
    contour_ax.clabel(truth_cs, fmt=lambda value: f"T {value:.2f}", fontsize=6)
    contour_ax.clabel(pinn_cs, fmt=lambda value: f"P {value:.2f}", fontsize=6)
    contour_ax.plot([], [], color=COLORS["neutral"], linewidth=1.2, label="solid truth")
    contour_ax.plot([], [], color=COLORS["neutral"], linewidth=1.2, linestyle="--", label="dashed PINN")
    contour_ax.legend(frameon=False, fontsize=6, loc="upper right")

    ax = axes[1, 3]
    labels = ["PINN L2", "RK4 L2", "PINN/RK4 L2", "PINN val MSE", "RK4 val MSE"]
    values = [
        metrics.get("pinn_final_relative_l2"),
        metrics.get("rk4_final_relative_l2"),
        metrics.get("pinn_vs_rk4_final_relative_l2"),
        metrics.get("pinn_validation_mse"),
        metrics.get("rk4_validation_mse"),
    ]
    plot_labels = [label for label, value in zip(labels, values) if value is not None and np.isfinite(value)]
    plot_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["gold"], COLORS["blue_light"], "#FFBDA1"][: len(plot_values)]
    if plot_values:
        ax.set_title("accuracy summary", fontsize=9)
        ax.set_axis_off()
        ax.text(0.02, 0.92, "metric", transform=ax.transAxes, fontsize=8, color=TOKENS["muted"], ha="left")
        ax.text(0.98, 0.92, "value", transform=ax.transAxes, fontsize=8, color=TOKENS["muted"], ha="right")
        for idx, (label, value, color) in enumerate(zip(plot_labels, plot_values, colors)):
            y = 0.78 - idx * 0.14
            ax.scatter([0.035], [y], transform=ax.transAxes, s=36, color=color, edgecolor=TOKENS["ink"], linewidth=0.5)
            ax.text(0.09, y, label, transform=ax.transAxes, va="center", ha="left", fontsize=8, color=TOKENS["ink"])
            ax.text(0.98, y, f"{value:.3e}", transform=ax.transAxes, va="center", ha="right", fontsize=8, color=TOKENS["ink"])
    else:
        ax.text(0.5, 0.5, "No comparable metrics", ha="center", va="center", color=TOKENS["muted"])
        ax.set_xticks([])
        ax.set_yticks([])
        _style_axis(ax, grid=True)

    _write_title(
        fig,
        "PINN versus RK4 accuracy comparison",
        f"Both methods solve the same 2D Fisher-KPP problem; maps and metrics are evaluated at t={time_value:.2f}.",
        top=0.84,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_pinn_evolution_gif(
    path: Path,
    truth: TruthData,
    model: OriginPINN,
    domain: DomainConfig,
    device: torch.device,
    *,
    n: int = 72,
    max_frames: int = 24,
    fps: int = 5,
    caption: str | None = None,
) -> None:
    """Save a compact animated view of truth, PINN prediction, and error over time."""

    _apply_chart_theme()
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(truth.times) <= max_frames:
        times = truth.times
    else:
        idx = np.unique(np.linspace(0, len(truth.times) - 1, max_frames).astype(int))
        times = truth.times[idx]

    frames: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    error_vmax = 1.0e-12
    for time_value in times:
        _, true_field = truth_field_at(truth, float(time_value), n=n)
        _, pred = predict_field(model, float(time_value), n=n, device=device)
        err = np.abs(pred - true_field)
        error_vmax = max(error_vmax, float(np.nanmax(err)))
        frames.append((float(time_value), true_field, pred, err))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.97, bottom=0.08, top=0.78, wspace=0.18)
    panels = [
        ("reference", frames[0][1], "magma", 0.0, 1.0),
        ("PINN", frames[0][2], "magma", 0.0, 1.0),
        ("absolute error", frames[0][3], "viridis", 0.0, error_vmax),
    ]
    images = []
    for ax, (title, field, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(
            field.T,
            origin="lower",
            extent=[0, domain.box, 0, domain.box],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            animated=True,
        )
        ax.set_title(title, fontsize=9, color=TOKENS["ink"])
        ax.set_xticks([])
        ax.set_yticks([])
        images.append(im)
    fig.colorbar(images[2], ax=axes[2], fraction=0.046, pad=0.02)
    title = fig.text(
        0.01,
        0.965,
        "",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.01,
        0.915,
        caption or "Animated PINN reconstruction across the Fisher-KPP time horizon.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )

    def _update(frame_idx: int) -> list[object]:
        time_value, true_field, pred, err = frames[frame_idx]
        images[0].set_data(true_field.T)
        images[1].set_data(pred.T)
        images[2].set_data(err.T)
        title.set_text(f"PINN field evolution t={time_value:.3f}")
        return [*images, title]

    animation = FuncAnimation(fig, _update, frames=len(frames), interval=1000 / max(fps, 1), blit=True)
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_observation_coverage_figure(
    path: Path,
    train_observations: ObservationData,
    validation_observations: ObservationData,
    domain: DomainConfig,
    *,
    max_points: int = 2500,
) -> None:
    _apply_chart_theme()
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sample(obs: ObservationData) -> tuple[np.ndarray, np.ndarray]:
        if len(obs.xyt) <= max_points:
            return obs.xyt, obs.values[:, 0]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(obs.xyt), size=max_points, replace=False)
        return obs.xyt[idx], obs.values[idx, 0]

    train_xyt, train_values = _sample(train_observations)
    val_xyt, val_values = _sample(validation_observations)
    all_times = np.concatenate([train_observations.xyt[:, 2], validation_observations.xyt[:, 2]])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.13, top=0.78, wspace=0.28)
    sc = axes[0].scatter(
        train_xyt[:, 0],
        train_xyt[:, 1],
        c=train_values,
        s=8,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        alpha=0.75,
        linewidths=0.0,
    )
    axes[0].set_xlim(0.0, domain.box)
    axes[0].set_ylim(0.0, domain.box)
    axes[0].set_aspect("equal")
    axes[0].set_title("train samples", fontsize=9)
    fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.02)

    if len(val_xyt):
        axes[1].scatter(val_xyt[:, 0], val_xyt[:, 1], c=val_values, s=10, cmap="magma", vmin=0.0, vmax=1.0, alpha=0.75, linewidths=0.0)
    axes[1].set_xlim(0.0, domain.box)
    axes[1].set_ylim(0.0, domain.box)
    axes[1].set_aspect("equal")
    axes[1].set_title("validation samples", fontsize=9)

    bins = np.linspace(0.0, domain.t_end, min(12, max(4, len(np.unique(all_times)) + 1)))
    axes[2].hist(train_observations.xyt[:, 2], bins=bins, color=COLORS["blue"], alpha=0.65, label="train")
    if len(validation_observations.xyt):
        axes[2].hist(validation_observations.xyt[:, 2], bins=bins, color=COLORS["orange"], alpha=0.65, label="validation")
    axes[2].set_title("time coverage", fontsize=9)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Observation count")
    axes[2].legend(frameon=False, fontsize=8)

    for ax in axes:
        _style_axis(ax, grid=ax is axes[2])
    _write_title(
        fig,
        "Observation coverage",
        "Spatial samples are colored by normalized infection density; histogram shows train/validation time support.",
        top=0.78,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def generated_figure_names() -> list[str]:
    return [
        "observation_coverage.png",
        "reconstruction.png",
        "spacetime_error.png",
        "residual_front_diagnostics.png",
        "pinn_vs_rk4_comparison.png",
        "pinn_evolution.gif",
        "training_diagnostics.png",
    ]


def existing_figures(out_dir: Path) -> Iterable[Path]:
    for name in generated_figure_names():
        path = out_dir / name
        if path.exists():
            yield path
