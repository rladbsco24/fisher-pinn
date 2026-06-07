from __future__ import annotations

import numpy as np


# 1D traveling-front setup -------------------------------------------------------
D = 1.0
r = 1.0
L = 200.0
T = 60.0
Nx = 401
dx = L / (Nx - 1)
x = np.linspace(0.0, L, Nx)
dt = 0.01
Nt = int(round(T / dt))
dt = T / Nt
left_bc = 1.0
right_bc = 0.0


def initial_condition(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((x - 50.0) / 5.0))


# 2D square-domain setup matching the PINN/RK4 comparison problem ----------------
D_2D = 0.02
r_2D = 3.0
BOX_2D = 1.0
T_2D = 0.5
GRID_2D = 51
TRUTH_STEPS_2D = 160
x_2d = np.linspace(0.0, BOX_2D, GRID_2D)
y_2d = np.linspace(0.0, BOX_2D, GRID_2D)
dx_2d = BOX_2D / (GRID_2D - 1)
dt_2d = T_2D / TRUTH_STEPS_2D
Nt_2d = TRUTH_STEPS_2D
seed_center_x = 0.32
seed_center_y = 0.68
seed_sigma = 0.07
seed_amplitude = 0.35


def initial_condition_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dist2 = (x - seed_center_x) ** 2 + (y - seed_center_y) ** 2
    return seed_amplitude * np.exp(-dist2 / (2.0 * seed_sigma**2))


# Kept for compatibility with implicit solvers; RK4 does not use them.
tol = 1.0e-10
max_iter = 30

