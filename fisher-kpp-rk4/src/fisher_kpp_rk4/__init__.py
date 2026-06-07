"""Utilities for solving 1D and 2D Fisher-KPP equations with RK4."""

from .solver import (
    apply_dirichlet_bc,
    apply_neumann_bc_2d,
    area_fraction_2d,
    check_rk4_stability,
    estimate_front_speed,
    fisher_kpp_rhs,
    fisher_kpp_rhs_2d,
    front_position,
    mean_mass,
    relative_l2,
    rk4_step,
    rk4_step_2d,
    solve_rk4,
    solve_rk4_2d,
)

__all__ = [
    "apply_dirichlet_bc",
    "apply_neumann_bc_2d",
    "area_fraction_2d",
    "check_rk4_stability",
    "estimate_front_speed",
    "fisher_kpp_rhs",
    "fisher_kpp_rhs_2d",
    "front_position",
    "mean_mass",
    "relative_l2",
    "rk4_step",
    "rk4_step_2d",
    "solve_rk4",
    "solve_rk4_2d",
]
