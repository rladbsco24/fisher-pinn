from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RK4_ROOT = ROOT / "fisher-kpp-rk4"
RK4_SRC = RK4_ROOT / "src"
if str(RK4_SRC) not in sys.path:
    sys.path.insert(0, str(RK4_SRC))


def _notebook_code(path: Path) -> str:
    nb = json.loads(path.read_text(encoding="utf-8"))
    parts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            parts.append("".join(source) if isinstance(source, list) else str(source))
    return "\n".join(parts)


def _load_trapezoidal_namespace(path: Path) -> dict[str, object]:
    src = _notebook_code(path)
    lines = src.splitlines()
    prefix = "\n".join(lines[:626])
    history_defs = "\n".join(lines[821:1036])
    ns: dict[str, object] = {}
    exec(prefix + "\n\n" + history_defs, ns)
    return ns


def _load_backward_euler_namespace(path: Path) -> dict[str, object]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    solver_cell = ""
    for cell in nb.get("cells", []):
        if cell.get("id") == "backward-euler-solvers":
            source = cell.get("source", "")
            solver_cell = "".join(source) if isinstance(source, list) else str(source)
            break
    if not solver_cell:
        raise RuntimeError(f"Could not find backward-euler-solvers cell in {path}")
    cut = solver_cell.index('print("=== 1D basic comparison ===")')
    ns: dict[str, object] = {}
    exec(solver_cell[:cut], ns)
    return ns


def _fig_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        return Image.fromarray(buf.copy(), mode="RGBA").convert("RGB")
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    rgba = buf[:, :, [1, 2, 3, 0]]
    return Image.fromarray(rgba.copy(), mode="RGBA").convert("RGB")


def _save_gif(
    *,
    path: Path,
    method_label: str,
    x: np.ndarray,
    y: np.ndarray,
    times: list[float],
    fields: list[np.ndarray],
    duration_ms: int = 180,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin = min(float(np.min(field)) for field in fields)
    vmax = max(float(np.max(field)) for field in fields)
    frames: list[Image.Image] = []
    for t, field in zip(times, fields):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(
            field.T,
            origin="lower",
            extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            aspect="equal",
        )
        ax.contour(x, y, field.T, levels=[0.1, 0.5], colors=["white", "black"], linewidths=[1.0, 0.9])
        ax.set_title(f"{method_label} 2D evolution, t={float(t):.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.82, label="u(x,y,t)")
        fig.tight_layout()
        frames.append(_fig_to_image(fig))
        plt.close(fig)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _save_surface_gif(
    *,
    path: Path,
    method_label: str,
    x: np.ndarray,
    y: np.ndarray,
    times: list[float],
    fields: list[np.ndarray],
    duration_ms: int = 180,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    zmin = min(float(np.min(field)) for field in fields)
    zmax = max(float(np.max(field)) for field in fields)
    frames: list[Image.Image] = []
    for t, field in zip(times, fields):
        fig = plt.figure(figsize=(6.2, 5.1))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            x_grid,
            y_grid,
            field,
            cmap="viridis",
            edgecolor="none",
            linewidth=0,
            antialiased=True,
            vmin=zmin,
            vmax=zmax,
        )
        ax.contour(x_grid, y_grid, field, levels=[0.1, 0.5], zdir="z", offset=zmin, colors=["white", "black"], linewidths=1.0)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        ax.set_zlim(zmin, zmax)
        ax.view_init(elev=30, azim=-60)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x,y,t)")
        ax.set_title(f"{method_label} 2D 3D-surface evolution, t={float(t):.2f}")
        fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, label="u(x,y,t)")
        fig.tight_layout()
        frames.append(_fig_to_image(fig))
        plt.close(fig)

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def _save_contact_sheet(root: Path, gif_paths: list[Path], output_name: str) -> Path:
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    thumbs: list[Image.Image] = []
    for gif in gif_paths:
        with Image.open(gif) as im:
            frame = im.convert("RGB")
            frame.thumbnail((320, 240), Image.LANCZOS)
            canvas = Image.new("RGB", (340, 280), "white")
            canvas.paste(frame, ((340 - frame.width) // 2, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 8), gif.parent.name, fill=(0, 0, 0), font=font)
            thumbs.append(canvas)
    sheet = Image.new("RGB", (340 * len(thumbs), 280), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (idx * 340, 0))
    out = root / output_name
    sheet.save(out, quality=95)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one matched 2D evolution GIF for each numerical method.")
    parser.add_argument(
        "--trapezoidal-notebook",
        type=Path,
        default=Path(r"C:\Users\yoonc\Downloads\KPP_Fisher_trapezoidal_AZ_exact (1).ipynb"),
    )
    parser.add_argument(
        "--backward-euler-notebook",
        type=Path,
        default=Path(r"C:\Users\yoonc\OneDrive\문서\카카오톡 받은 파일\KakaoTalk_Longtxt_20260612_1939_59_369.ipynb"),
    )
    parser.set_defaults(
        backward_euler_notebook=Path(r"C:\Users\yoonc\OneDrive\문서\카카오톡 받은 파일\KakaoTalk_Longtxt_20260612_1939_59_369.ipynb")
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "matched_numerical_visualizations")
    parser.add_argument("--t-end", type=float, default=8.0)
    parser.add_argument("--frames", type=int, default=33)
    args = parser.parse_args()

    save_times = [float(t) for t in np.linspace(0.0, args.t_end, int(args.frames))]
    gif_paths: list[Path] = []
    surface_gif_paths: list[Path] = []

    print("solving trapezoidal 2D for GIF", flush=True)
    trap = _load_trapezoidal_namespace(args.trapezoidal_notebook)
    trap_history = trap["solve_trapezoidal_2d_newton_history"](
        Nx=61, dt=0.01, T=args.t_end, L=30.0, save_times=save_times
    )
    path = args.out_dir / "trapezoidal_newton" / "2d_evolution.gif"
    _save_gif(
        path=path,
        method_label="Trapezoidal-Newton",
        x=trap_history["x"],
        y=trap_history["y"],
        times=list(trap_history["history"].keys()),
        fields=list(trap_history["history"].values()),
    )
    gif_paths.append(path)
    print(f"wrote {path}", flush=True)
    surface_path = args.out_dir / "trapezoidal_newton" / "2d_surface_evolution.gif"
    _save_surface_gif(
        path=surface_path,
        method_label="Trapezoidal-Newton",
        x=trap_history["x"],
        y=trap_history["y"],
        times=list(trap_history["history"].keys()),
        fields=list(trap_history["history"].values()),
    )
    surface_gif_paths.append(surface_path)
    print(f"wrote {surface_path}", flush=True)

    print("solving backward Euler 2D for GIF", flush=True)
    be = _load_backward_euler_namespace(args.backward_euler_notebook)
    be_history = be["solve_2d"](Nx=61, Ny=61, dt=0.01, T=args.t_end, save_times=save_times)
    path = args.out_dir / "backward_euler_newton" / "2d_evolution.gif"
    _save_gif(
        path=path,
        method_label="Backward Euler-Newton",
        x=be_history["x"],
        y=be_history["y"],
        times=list(be_history["history"].keys()),
        fields=list(be_history["history"].values()),
    )
    gif_paths.append(path)
    print(f"wrote {path}", flush=True)
    surface_path = args.out_dir / "backward_euler_newton" / "2d_surface_evolution.gif"
    _save_surface_gif(
        path=surface_path,
        method_label="Backward Euler-Newton",
        x=be_history["x"],
        y=be_history["y"],
        times=list(be_history["history"].keys()),
        fields=list(be_history["history"].values()),
    )
    surface_gif_paths.append(surface_path)
    print(f"wrote {surface_path}", flush=True)

    print("solving Forward Euler 2D for GIF", flush=True)
    from fisher_kpp_rk4 import solve_2d_method
    from fisher_kpp_rk4.config import (
        D_2D,
        generalized_fisher_kpp_exact_2d,
        initial_condition_2d,
        r_2D,
        x_left_2d,
        x_right_2d,
        y_bottom_2d,
        y_top_2d,
    )

    x = np.linspace(x_left_2d, x_right_2d, 61)
    y = np.linspace(y_bottom_2d, y_top_2d, 61)
    dt = 0.01
    fe = solve_2d_method(
        "forward_euler",
        x=x,
        y=y,
        dt=dt,
        Nt=int(round(args.t_end / dt)),
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=args.t_end / (int(args.frames) - 1),
        boundary_condition="dirichlet_exact",
        exact_solution=generalized_fisher_kpp_exact_2d,
    )
    path = args.out_dir / "forward_euler" / "2d_evolution.gif"
    _save_gif(
        path=path,
        method_label="Forward Euler",
        x=x,
        y=y,
        times=[float(t) for t in fe["times"]],
        fields=[field for field in fe["snapshots"]],
    )
    gif_paths.append(path)
    print(f"wrote {path}", flush=True)
    surface_path = args.out_dir / "forward_euler" / "2d_surface_evolution.gif"
    _save_surface_gif(
        path=surface_path,
        method_label="Forward Euler",
        x=x,
        y=y,
        times=[float(t) for t in fe["times"]],
        fields=[field for field in fe["snapshots"]],
    )
    surface_gif_paths.append(surface_path)
    print(f"wrote {surface_path}", flush=True)

    print("solving RK4 2D for GIF", flush=True)
    from fisher_kpp_rk4 import solve_rk4_2d
    from fisher_kpp_rk4.config import (
        D_2D,
        generalized_fisher_kpp_exact_2d,
        initial_condition_2d,
        r_2D,
        x_left_2d,
        x_right_2d,
        y_bottom_2d,
        y_top_2d,
    )

    x = np.linspace(x_left_2d, x_right_2d, 61)
    y = np.linspace(y_bottom_2d, y_top_2d, 61)
    dt = 0.01
    rk4 = solve_rk4_2d(
        x=x,
        y=y,
        dt=dt,
        Nt=int(round(args.t_end / dt)),
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=args.t_end / (int(args.frames) - 1),
        boundary_condition="dirichlet_exact",
        exact_solution=generalized_fisher_kpp_exact_2d,
    )
    path = args.out_dir / "rk4" / "2d_evolution.gif"
    _save_gif(
        path=path,
        method_label="RK4",
        x=x,
        y=y,
        times=[float(t) for t in rk4["times"]],
        fields=[field for field in rk4["snapshots"]],
    )
    gif_paths.append(path)
    print(f"wrote {path}", flush=True)
    surface_path = args.out_dir / "rk4" / "2d_surface_evolution.gif"
    _save_surface_gif(
        path=surface_path,
        method_label="RK4",
        x=x,
        y=y,
        times=[float(t) for t in rk4["times"]],
        fields=[field for field in rk4["snapshots"]],
    )
    surface_gif_paths.append(surface_path)
    print(f"wrote {surface_path}", flush=True)

    preview = _save_contact_sheet(args.out_dir, gif_paths, "all_methods_2d_gif_preview.png")
    surface_preview = _save_contact_sheet(args.out_dir, surface_gif_paths, "all_methods_2d_surface_gif_preview.png")
    manifest_path = args.out_dir / "gif_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "t_end": args.t_end,
                "requested_frames": args.frames,
                "gifs": [str(path) for path in gif_paths],
                "surface_gifs": [str(path) for path in surface_gif_paths],
                "preview": str(preview),
                "surface_preview": str(surface_preview),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"wrote {preview}", flush=True)
    print(f"wrote {surface_preview}", flush=True)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
