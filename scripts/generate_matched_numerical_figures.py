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


def _safe_time_label(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _save_1d(method_label: str, out_dir: Path, x: np.ndarray, history: dict[float, np.ndarray], exact_fn) -> list[Path]:
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for t, u in history.items():
        ax.plot(x, u, label=f"t = {float(t):g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"1D KPP-Fisher equation using {method_label} method")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "1d_profiles.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    final_t = max(float(t) for t in history)
    u_num = history[final_t]
    u_exact = exact_fn(x, final_t)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, np.abs(u_num - u_exact), color="tab:red")
    ax.set_xlabel("x")
    ax.set_ylabel("absolute error")
    ax.set_title(f"1D absolute error at t = {final_t:g}")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    path = out_dir / "1d_absolute_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def _save_2d(
    method_label: str,
    out_dir: Path,
    x: np.ndarray,
    y: np.ndarray,
    history: dict[float, np.ndarray],
    exact_fn,
) -> list[Path]:
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
    center_j = len(y) // 2

    fig, ax = plt.subplots(figsize=(8, 5))
    for t, field in history.items():
        ax.plot(x, field[:, center_j], label=f"t = {float(t):g}")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,0,t)")
    ax.set_title("2D generalized Fisher-KPP centerline over time")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "2d_centerline.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    vmin = min(float(np.min(field)) for field in history.values())
    vmax = max(float(np.max(field)) for field in history.values())
    times = list(history.keys())
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4), constrained_layout=True)
    for ax, t in zip(np.ravel(axes), times):
        im = ax.imshow(
            history[t].T,
            origin="lower",
            extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            aspect="equal",
        )
        ax.set_title(f"t={float(t):g}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=np.ravel(axes).tolist(), shrink=0.75, label="u(x,y,t)")
    fig.suptitle(f"2D generalized Fisher-KPP heatmap snapshots ({method_label})", y=1.02)
    path = out_dir / "2d_heatmap_snapshots.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    for t, field in history.items():
        t_float = float(t)
        label = _safe_time_label(t_float)
        exact = exact_fn(x_grid, y_grid, t_float)
        abs_error = np.abs(field - exact)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(x_grid, y_grid, field, cmap="viridis", edgecolor="none", alpha=0.96)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x,y,t)")
        ax.view_init(elev=30, azim=45)
        ax.set_title(f"2D generalized Fisher-KPP 3D Surface at t = {t_float:g}")
        fig.colorbar(surf, ax=ax, shrink=0.65, label="u(x,y,t)")
        fig.tight_layout()
        path = out_dir / f"2d_surface_t{label}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        for title, data, cmap, zlabel, stem in [
            ("Numerical", field, "viridis", "u_num(x,y,t)", "2d_numerical"),
            ("Exact", exact, "viridis", "u_exact(x,y,t)", "2d_exact"),
            ("Absolute Error", abs_error, "viridis", "|u_num - u_exact|", "2d_absolute_error"),
        ]:
            fig = plt.figure(figsize=(9, 6))
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(x_grid, y_grid, data, cmap=cmap, edgecolor="none")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel(zlabel)
            ax.view_init(elev=30, azim=45)
            ax.set_title(f"{title} 3D Surface at t = {t_float:g}")
            colorbar_label = "absolute error" if title == "Absolute Error" else stem.removeprefix("2d_")
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label=colorbar_label)
            fig.tight_layout()
            path = out_dir / f"{stem}_t{label}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def _load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _thumbnail(path: Path, *, size: tuple[int, int] = (360, 235)) -> Image.Image:
    with Image.open(path) as im:
        image = im.convert("RGB")
    image.thumbnail(size, Image.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def _save_method_contact_sheet(method_dir: Path) -> Path:
    items = [
        ("1D profiles", method_dir / "1d_profiles.png"),
        ("2D heatmaps", method_dir / "2d_heatmap_snapshots.png"),
        ("Final 3D error", method_dir / "2d_absolute_error_t8p0.png"),
    ]
    font_title = _load_font(18)
    font_small = _load_font(14)
    cell_w, cell_h = 390, 300
    header_h = 52
    sheet = Image.new("RGB", (cell_w * len(items), header_h + cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 14), method_dir.name, fill=(20, 20, 20), font=font_title)
    for col, (label, path) in enumerate(items):
        x0 = col * cell_w
        draw.text((x0 + 16, header_h - 22), label, fill=(60, 60, 60), font=font_small)
        if path.exists():
            sheet.paste(_thumbnail(path), (x0 + 15, header_h + 25))
    out = method_dir.parent / f"{method_dir.name}_preview_contact_sheet.png"
    sheet.save(out, quality=95)
    return out


def _save_all_methods_contact_sheet(root: Path, method_dirs: list[Path]) -> Path:
    method_previews = [_save_method_contact_sheet(method_dir) for method_dir in method_dirs]
    font_title = _load_font(22)
    font_small = _load_font(14)
    row_w, row_h = 1180, 315
    header_h = 54
    sheet = Image.new("RGB", (row_w, header_h + row_h * len(method_previews)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 14), "Matched Fisher-KPP numerical visualizations", fill=(15, 15, 15), font=font_title)
    for row, preview in enumerate(method_previews):
        y0 = header_h + row * row_h
        draw.text((18, y0 + 12), preview.stem.replace("_preview_contact_sheet", ""), fill=(60, 60, 60), font=font_small)
        thumb = _thumbnail(preview, size=(1120, 280))
        sheet.paste(thumb, (40, y0 + 30))
    out = root / "all_methods_preview_contact_sheet.png"
    sheet.save(out, quality=95)
    return out


def _rk4_figures(out_dir: Path) -> list[Path]:
    from fisher_kpp_rk4 import solve_rk4, solve_rk4_2d
    from fisher_kpp_rk4.config import (
        D,
        D_2D,
        Nt,
        ablowitz_zeppetella_exact,
        dt,
        generalized_fisher_kpp_exact_2d,
        initial_condition,
        initial_condition_2d,
        left_bc,
        r,
        r_2D,
        right_bc,
        x,
        x_left_2d,
        x_right_2d,
        y_bottom_2d,
        y_top_2d,
    )

    result_1d = solve_rk4(
        x=x,
        dt=dt,
        Nt=Nt,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=left_bc,
        right_bc=right_bc,
        save_interval=2.0,
        exact_solution=ablowitz_zeppetella_exact,
    )
    history_1d = {float(t): field for t, field in zip(result_1d["times"], result_1d["snapshots"])}
    paths = _save_1d("RK4", out_dir, result_1d["x"], history_1d, ablowitz_zeppetella_exact)

    x2 = np.linspace(x_left_2d, x_right_2d, 61)
    y2 = np.linspace(y_bottom_2d, y_top_2d, 61)
    result_2d = solve_rk4_2d(
        x=x2,
        y=y2,
        dt=0.01,
        Nt=800,
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=2.0,
        boundary_condition="dirichlet_exact",
        exact_solution=generalized_fisher_kpp_exact_2d,
    )
    history_2d = {float(t): field for t, field in zip(result_2d["times"], result_2d["snapshots"])}
    paths.extend(_save_2d("RK4", out_dir, x2, y2, history_2d, generalized_fisher_kpp_exact_2d))
    return paths


def _forward_euler_figures(out_dir: Path) -> list[Path]:
    from fisher_kpp_rk4 import solve_1d_method, solve_2d_method
    from fisher_kpp_rk4.config import (
        D,
        D_2D,
        Nt,
        ablowitz_zeppetella_exact,
        dt,
        generalized_fisher_kpp_exact_2d,
        initial_condition,
        initial_condition_2d,
        left_bc,
        r,
        r_2D,
        right_bc,
        x,
        x_left_2d,
        x_right_2d,
        y_bottom_2d,
        y_top_2d,
    )

    result_1d = solve_1d_method(
        "forward_euler",
        x=x,
        dt=dt,
        Nt=Nt,
        D=D,
        r=r,
        initial_condition=initial_condition,
        left_bc=left_bc,
        right_bc=right_bc,
        save_interval=2.0,
        exact_solution=ablowitz_zeppetella_exact,
    )
    history_1d = {float(t): field for t, field in zip(result_1d["times"], result_1d["snapshots"])}
    paths = _save_1d("Forward Euler", out_dir, result_1d["x"], history_1d, ablowitz_zeppetella_exact)

    x2 = np.linspace(x_left_2d, x_right_2d, 61)
    y2 = np.linspace(y_bottom_2d, y_top_2d, 61)
    result_2d = solve_2d_method(
        "forward_euler",
        x=x2,
        y=y2,
        dt=0.01,
        Nt=800,
        D=D_2D,
        r=r_2D,
        initial_condition=initial_condition_2d,
        save_interval=2.0,
        boundary_condition="dirichlet_exact",
        exact_solution=generalized_fisher_kpp_exact_2d,
    )
    history_2d = {float(t): field for t, field in zip(result_2d["times"], result_2d["snapshots"])}
    paths.extend(_save_2d("Forward Euler", out_dir, x2, y2, history_2d, generalized_fisher_kpp_exact_2d))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched numerical Fisher-KPP figures.")
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
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading trapezoidal notebook functions", flush=True)
    trap = _load_trapezoidal_namespace(args.trapezoidal_notebook)
    print("solving trapezoidal 1D", flush=True)
    trap_1d = trap["solve_trapezoidal_1d_newton_history"](
        Nx=201, dt=0.005, T=10.0, L=40.0, save_times=[0, 2, 4, 6, 8, 10]
    )
    print("solving trapezoidal 2D", flush=True)
    trap_2d = trap["solve_trapezoidal_2d_newton_history"](
        Nx=61, dt=0.01, T=8.0, L=30.0, save_times=[0, 2, 4, 6, 8]
    )
    trap_dir = args.out_dir / "trapezoidal_newton"
    paths = _save_1d("Trapezoidal-Newton", trap_dir, trap_1d["x"], trap_1d["history"], trap["exact_solution_1d"])
    paths.extend(_save_2d("Trapezoidal-Newton", trap_dir, trap_2d["x"], trap_2d["y"], trap_2d["history"], trap["exact_solution_2d"]))
    print(f"wrote {len(paths)} trapezoidal figures to {trap_dir}", flush=True)

    print("loading backward Euler notebook functions", flush=True)
    be = _load_backward_euler_namespace(args.backward_euler_notebook)
    print("solving backward Euler 1D", flush=True)
    be_1d = be["solve_1d"](Nx=201, dt=0.005, T=10.0, save_times=[0, 2, 4, 6, 8, 10])
    print("solving backward Euler 2D", flush=True)
    be_2d = be["solve_2d"](Nx=61, Ny=61, dt=0.01, T=8.0, save_times=[0, 2, 4, 6, 8])
    be_dir = args.out_dir / "backward_euler_newton"
    paths = _save_1d("Backward Euler-Newton", be_dir, be_1d["x"], be_1d["history"], be["u_exact_1d"])
    paths.extend(_save_2d("Backward Euler-Newton", be_dir, be_2d["x"], be_2d["y"], be_2d["history"], be["u_exact_2d"]))
    print(f"wrote {len(paths)} backward Euler figures to {be_dir}", flush=True)

    print("solving RK4 1D/2D", flush=True)
    rk4_dir = args.out_dir / "rk4"
    paths = _rk4_figures(rk4_dir)
    print(f"wrote {len(paths)} RK4 figures to {rk4_dir}", flush=True)

    print("solving Forward Euler 1D/2D", flush=True)
    forward_dir = args.out_dir / "forward_euler"
    paths = _forward_euler_figures(forward_dir)
    print(f"wrote {len(paths)} Forward Euler figures to {forward_dir}", flush=True)

    preview = _save_all_methods_contact_sheet(args.out_dir, [trap_dir, be_dir, rk4_dir, forward_dir])
    print(f"wrote {preview}", flush=True)

    manifest = {
        "output_dir": str(args.out_dir),
        "methods": {
            "trapezoidal_newton": str(trap_dir),
            "backward_euler_newton": str(be_dir),
            "rk4": str(rk4_dir),
            "forward_euler": str(forward_dir),
        },
        "figure_count": len(list(args.out_dir.rglob("*.png"))),
        "preview": str(preview),
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote manifest {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
