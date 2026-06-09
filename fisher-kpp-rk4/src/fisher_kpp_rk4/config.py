from __future__ import annotations

import numpy as np


# Ablowitz-Zeppetella exact traveling-front setup ------------------------------
D = 1.0
r = 1.0
c = 5.0 / np.sqrt(6.0)
alpha = 1.0 / np.sqrt(6.0)
L = 40.0
x_left = -20.0
x_right = 20.0
T = 10.0
Nx = 201
dx = L / (Nx - 1)
x = np.linspace(x_left, x_right, Nx)
dt = 0.005
Nt = int(round(T / dt))
dt = T / Nt
x0 = 0.0


def ablowitz_zeppetella_exact(x: np.ndarray, t: float | np.ndarray, x0_value: float = x0) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - c * np.asarray(t, dtype=np.float64) - float(x0_value)) / np.sqrt(6.0)
    return np.power(1.0 + np.exp(np.clip(z, -80.0, 80.0)), -2.0)


def initial_condition(x: np.ndarray) -> np.ndarray:
    return ablowitz_zeppetella_exact(x, 0.0)


def left_bc(t: float) -> float:
    return float(ablowitz_zeppetella_exact(np.asarray([x_left]), t)[0])


def right_bc(t: float) -> float:
    return float(ablowitz_zeppetella_exact(np.asarray([x_right]), t)[0])


# 2D generalized Fisher-KPP exact traveling-front setup ------------------------
D_2D = 1.0
r_2D = 1.0
P_2D = 1.0
PHI_2D = np.pi / 4.0
C_2D = 0.0
L_2D = 30.0
BOX_2D = L_2D
x_left_2d = -15.0
x_right_2d = 15.0
y_bottom_2d = -15.0
y_top_2d = 15.0
T_2D = 3.0
GRID_2D = 61
TRUTH_STEPS_2D = 300
x_2d = np.linspace(x_left_2d, x_right_2d, GRID_2D)
y_2d = np.linspace(y_bottom_2d, y_top_2d, GRID_2D)
dx_2d = L_2D / (GRID_2D - 1)
dt_2d = T_2D / TRUTH_STEPS_2D
Nt_2d = TRUTH_STEPS_2D


def generalized_fisher_kpp_exact_2d(
    x: np.ndarray,
    y: np.ndarray,
    t: float | np.ndarray,
    *,
    p: float = P_2D,
    phi: float = PHI_2D,
    c0: float = C_2D,
) -> np.ndarray:
    if p <= 0.0:
        raise ValueError("p must be positive for the generalized Fisher-KPP exact solution.")
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    spatial = x_arr * np.sin(phi) + y_arr * np.cos(phi)
    psi = (p / (2.0 * np.sqrt(2.0 * p + 4.0))) * spatial
    psi = psi + (p * (p + 4.0) / (4.0 * (p + 2.0))) * t_arr + c0
    base = 0.5 * np.tanh(psi) + 0.5
    return np.power(np.clip(base, 0.0, 1.0), 2.0 / p)


def initial_condition_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return generalized_fisher_kpp_exact_2d(x, y, 0.0)


def ablowitz_zeppetella_exact_2d(x: np.ndarray, y: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    return generalized_fisher_kpp_exact_2d(x, y, t)


# Kept for compatibility with implicit solvers; RK4 does not use them.
tol = 1.0e-10
max_iter = 30


# Long-time, fair-method setup --------------------------------------------------
#
# These values are chosen so forward Euler, backward Euler, trapezoidal, and RK4
# can all use the same grid and time step through t=30 without boundary-front
# interaction. Fisher-KPP is not oscillatory under this setup; the long-time
# diagnostics are front position, mean density, and a probe trace rho(t).
LONG_TIME_D = 0.06
LONG_TIME_R = 0.25
LONG_TIME_L = 30.0
LONG_TIME_T = 30.0
LONG_TIME_NX = 181
LONG_TIME_X0 = 7.0
LONG_TIME_WIDTH = 0.9
LONG_TIME_PROBE_X = 12.0
LONG_TIME_SAVE_INTERVAL = 0.5
LONG_TIME_LEFT_BC = 1.0
LONG_TIME_RIGHT_BC = 0.0
LONG_TIME_X = np.linspace(0.0, LONG_TIME_L, LONG_TIME_NX)
LONG_TIME_DX = LONG_TIME_L / (LONG_TIME_NX - 1)
LONG_TIME_DT = 0.05
LONG_TIME_NT = int(round(LONG_TIME_T / LONG_TIME_DT))
LONG_TIME_DT = LONG_TIME_T / LONG_TIME_NT


def long_time_initial_condition(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((x - LONG_TIME_X0) / LONG_TIME_WIDTH))


# Long-time curve-trend benchmark ---------------------------------------------
#
# The right panel in the reference image is a damped oscillatory scalar trend,
# not a scalar Fisher-KPP probe trajectory. These parameters define a separate
# fair ODE benchmark for matching that visual trend with FE/BE/trapezoidal/RK4.
CURVE_T = 30.0
CURVE_DT = 0.05
CURVE_NT = int(round(CURVE_T / CURVE_DT))
CURVE_DT = CURVE_T / CURVE_NT
CURVE_RHO_INF = 0.34
CURVE_ALPHA = 0.24
CURVE_OMEGA_D = 1.0
CURVE_RHO0 = 0.0
CURVE_V0 = 0.60
