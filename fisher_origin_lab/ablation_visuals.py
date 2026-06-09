from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageSequence

from .config import DomainConfig
from .plotting import COLORS, ERROR_CMAP, FIELD_CMAP, SIGNED_ERROR_CMAP, TOKENS, _add_colorbar, _apply_chart_theme, _save_figure


def _load_diagnostic_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ["truth_final", "pinn_final", "pinn_abs_error"]
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"{path} is missing diagnostic fields: {', '.join(missing)}")
        return {name: np.array(data[name]) for name in data.files}


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _imshow_field(
    fig: plt.Figure,
    ax: plt.Axes,
    field: np.ndarray,
    *,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    domain: DomainConfig | None = None,
) -> None:
    extent = None if domain is None else [0.0, domain.box, 0.0, domain.box]
    im = ax.imshow(
        field.T,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=9, color=TOKENS["ink"])
    ax.set_xticks([])
    ax.set_yticks([])
    _add_colorbar(fig, im, ax)


def save_feature_pair_error_map(
    path: Path,
    *,
    feature_name: str,
    without_label: str,
    with_label: str,
    without_fields: Path,
    with_fields: Path,
    without_metrics: dict[str, Any] | None = None,
    with_metrics: dict[str, Any] | None = None,
    domain: DomainConfig | None = None,
) -> dict[str, float | str]:
    """Save a final-time ON/OFF feature comparison using exact saved fields."""

    _apply_chart_theme()
    path.parent.mkdir(parents=True, exist_ok=True)
    without = _load_diagnostic_fields(without_fields)
    with_feature = _load_diagnostic_fields(with_fields)

    truth = with_feature["truth_final"]
    without_pred = without["pinn_final"]
    with_pred = with_feature["pinn_final"]
    without_error = without["pinn_abs_error"]
    with_error = with_feature["pinn_abs_error"]
    if truth.shape != without_pred.shape or truth.shape != with_pred.shape:
        raise ValueError("Feature comparison fields must share the same grid shape.")

    error_vmax = max(float(np.nanmax(without_error)), float(np.nanmax(with_error)), 1.0e-12)
    error_delta = with_error - without_error
    delta_lim = max(float(np.nanmax(np.abs(error_delta))), 1.0e-12)

    fig, axes = plt.subplots(2, 4, figsize=(18.8, 8.4), constrained_layout=False)
    fig.subplots_adjust(left=0.035, right=0.94, bottom=0.06, top=0.84, hspace=0.30, wspace=0.34)
    _imshow_field(fig, axes[0, 0], truth, title="reference", cmap=FIELD_CMAP, vmin=0.0, vmax=1.0, domain=domain)
    _imshow_field(fig, axes[0, 1], without_pred, title=f"without: {without_label}", cmap=FIELD_CMAP, vmin=0.0, vmax=1.0, domain=domain)
    _imshow_field(fig, axes[0, 2], without_error, title="without absolute error", cmap=ERROR_CMAP, vmin=0.0, vmax=error_vmax, domain=domain)
    _imshow_field(fig, axes[1, 0], truth, title="reference", cmap=FIELD_CMAP, vmin=0.0, vmax=1.0, domain=domain)
    _imshow_field(fig, axes[1, 1], with_pred, title=f"with: {with_label}", cmap=FIELD_CMAP, vmin=0.0, vmax=1.0, domain=domain)
    _imshow_field(fig, axes[1, 2], with_error, title="with absolute error", cmap=ERROR_CMAP, vmin=0.0, vmax=error_vmax, domain=domain)
    _imshow_field(
        fig,
        axes[1, 3],
        error_delta,
        title="error delta, with - without",
        cmap=SIGNED_ERROR_CMAP,
        vmin=-delta_lim,
        vmax=delta_lim,
        domain=domain,
    )

    ax = axes[0, 3]
    ax.set_axis_off()
    ax.set_title("validation metrics", fontsize=9, color=TOKENS["ink"])
    rows = [
        ("final rel L2", "final_time_relative_l2"),
        ("validation MSE", "validation_observation_mse"),
        ("front u>0.10 MAE", "front_area_010_mae"),
        ("active-front MAE", "active_front_area_mae"),
        ("mass MAE", "mass_mae"),
    ]
    without_metrics = without_metrics or {}
    with_metrics = with_metrics or {}
    ax.text(0.02, 0.90, "metric", transform=ax.transAxes, fontsize=8, color=TOKENS["muted"])
    ax.text(0.64, 0.90, "without", transform=ax.transAxes, fontsize=8, color=TOKENS["muted"], ha="right")
    ax.text(0.98, 0.90, "with", transform=ax.transAxes, fontsize=8, color=TOKENS["muted"], ha="right")
    for idx, (label, key) in enumerate(rows):
        y = 0.78 - idx * 0.14
        w0 = _metric(without_metrics, key)
        w1 = _metric(with_metrics, key)
        ax.text(0.02, y, label, transform=ax.transAxes, va="center", fontsize=8, color=TOKENS["ink"])
        ax.text(0.64, y, "-" if w0 is None else f"{w0:.3e}", transform=ax.transAxes, va="center", ha="right", fontsize=8, color=TOKENS["ink"])
        color = COLORS["teal"] if w0 is not None and w1 is not None and w1 <= w0 else COLORS["orange"]
        ax.text(0.98, y, "-" if w1 is None else f"{w1:.3e}", transform=ax.transAxes, va="center", ha="right", fontsize=8, color=color)

    fig.text(
        0.01,
        0.985,
        f"Feature validation: {feature_name}",
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.01,
        0.952,
        "Final-time maps compare the same Fisher-KPP setup with the selected model/training feature disabled and enabled.",
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    _save_figure(fig, path, dpi=150)
    plt.close(fig)
    return {
        "path": str(path),
        "without_max_abs_error": float(np.nanmax(without_error)),
        "with_max_abs_error": float(np.nanmax(with_error)),
        "mean_abs_error_delta": float(np.nanmean(error_delta)),
    }


def _gif_frames(path: Path) -> tuple[list[Image.Image], int]:
    with Image.open(path) as image:
        duration = int(image.info.get("duration", 200))
        frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(image)]
    if not frames:
        raise ValueError(f"{path} has no GIF frames.")
    return frames, duration


def _fit_frame(frame: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = ImageOps.contain(frame, size)
    canvas = Image.new("RGBA", size, "white")
    canvas.alpha_composite(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def save_feature_pair_evolution_gif(
    path: Path,
    *,
    feature_name: str,
    without_label: str,
    with_label: str,
    without_gif: Path,
    with_gif: Path,
) -> dict[str, int | str]:
    """Compose two case GIFs into a side-by-side feature validation GIF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    without_frames, duration0 = _gif_frames(without_gif)
    with_frames, duration1 = _gif_frames(with_gif)
    frame_count = max(len(without_frames), len(with_frames))
    width = max(max(frame.width for frame in without_frames), max(frame.width for frame in with_frames))
    height = max(max(frame.height for frame in without_frames), max(frame.height for frame in with_frames))
    gap = 16
    header = 74
    label_band = 26
    font = ImageFont.load_default()
    composed: list[Image.Image] = []
    for idx in range(frame_count):
        left_idx = round(idx * (len(without_frames) - 1) / max(frame_count - 1, 1))
        right_idx = round(idx * (len(with_frames) - 1) / max(frame_count - 1, 1))
        left = _fit_frame(without_frames[left_idx], (width, height))
        right = _fit_frame(with_frames[right_idx], (width, height))
        canvas = Image.new("RGBA", (width * 2 + gap, header + label_band + height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), f"Feature validation: {feature_name}", fill=(31, 36, 48), font=font)
        draw.text((10, 32), "Left disables the selected feature; right keeps it enabled under the same preset/seed.", fill=(111, 118, 138), font=font)
        draw.text((10, header), f"without: {without_label}", fill=(31, 36, 48), font=font)
        draw.text((width + gap + 10, header), f"with: {with_label}", fill=(31, 36, 48), font=font)
        canvas.alpha_composite(left, (0, header + label_band))
        canvas.alpha_composite(right, (width + gap, header + label_band))
        composed.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE))

    duration = max(80, int(round((duration0 + duration1) / 2)))
    composed[0].save(path, save_all=True, append_images=composed[1:], duration=duration, loop=0, disposal=2)
    preview = path.with_name(f"{path.stem}_preview.png")
    composed[-1].convert("RGB").save(preview)
    return {
        "path": str(path),
        "preview": str(preview),
        "frames": frame_count,
        "duration_ms": duration,
    }
