"""
PyroBot: Physics-Informed Kinetics Simulator (TG/DTG Solver)
===========================================================
This module solves the solid-state kinetics ordinary differential equations (ODE)
to simulate thermogravimetric (TG) and derivative thermogravimetric (DTG) curves
under arbitrary heating rates, performing thermal stability & slagging safety checks.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def solve_pyrolysis_kinetics_ode(
    ea_kjmol: float,
    beta_kmin: float,
    ash_pct: float,
    pre_exponential_a: float = 2.4e10,  # Standard catalytic frequency factor in min^-1
    reaction_order_n: float = 1.0,
    temp_start_c: float = 100.0,
    temp_end_c: float = 900.0,
    steps: int = 800,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Numerically solves the solid-state reaction rate equation:
    d(alpha)/dT = (A / beta) * exp(-Ea / (R * T)) * (1 - alpha)^n
    
    Returns:
        temperatures_c: Temperature array in Celsius
        tg_curve: Mass remaining percentage (%)
        dtg_curve: Rate of mass loss with respect to temperature (%/Celsius)
    """
    R = 8.314e-3  # Universal gas constant in kJ/(mol * K)
    
    # Boundary conversion to Kelvin
    T_start = temp_start_c + 273.15
    T_end = temp_end_c + 273.15
    T_span = (T_start, T_end)
    
    # ODE rate equation
    def dalpha_dT(T: float, alpha: float) -> float:
        # Prevent conversion index overshoot
        if alpha >= 1.0:
            return 0.0
            
        rate_const = pre_exponential_a / beta_kmin
        arrhenius = np.exp(-ea_kjmol / (R * T))
        kinetics_f = (1.0 - alpha) ** reaction_order_n
        
        return float(rate_const * arrhenius * kinetics_f)

    # Initial condition: conversion alpha = 0.0 at T_start
    t_eval = np.linspace(T_start, T_end, steps)
    sol = solve_ivp(
        fun=lambda T, y: [dalpha_dT(T, y[0])],
        t_span=T_span,
        y0=[0.0],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )
    
    # Extract solution
    T_kelvin = sol.t
    alpha_vals = sol.y[0]
    
    # Convert back to physical measurements
    temperatures_c = T_kelvin - 273.15
    
    # TG curve represents mass remaining: Combustibles * (1 - alpha) + Ash
    combustible_pct = 100.0 - ash_pct
    tg_curve = combustible_pct * (1.0 - alpha_vals) + ash_pct
    
    # DTG curve is the derivative: -d(TG)/dT = Combustibles * d(alpha)/dT
    dtg_curve = np.zeros(len(T_kelvin))
    for idx, (T, alpha) in enumerate(zip(T_kelvin, alpha_vals)):
        dtg_curve[idx] = combustible_pct * dalpha_dT(T, alpha)
        
    return temperatures_c, tg_curve, dtg_curve


# -----------------------------------------------------------------------------
# Thermal Stability & Slagging Safeguards Checker
# -----------------------------------------------------------------------------
def run_recipe_safety_audit(
    temperatures_c: np.ndarray,
    tg_curve: np.ndarray,
    dtg_curve: np.ndarray,
    oxides_dict: Dict[str, float],
) -> Dict[str, any]:
    """
    Perform physics-informed safety audit on the candidate co-pyrolysis recipe.
    
    Returns:
        Dictionary containing safety metrics (slagging risk, runaway risk, audit report)
    """
    # 1. runaway risk: Check maximum DTG mass loss rate
    max_dtg_rate = float(np.max(dtg_curve))
    peak_idx = int(np.argmax(dtg_curve))
    peak_temp_c = float(temperatures_c[peak_idx])
    
    # Active pyrolysis window indices
    # T_10% conversion to T_90% conversion
    total_loss = tg_curve[0] - tg_curve[-1]
    active_mask = (tg_curve[0] - tg_curve) >= 0.10 * total_loss
    active_temps = temperatures_c[active_mask]
    
    active_window_start = float(active_temps[0]) if len(active_temps) > 0 else 250.0
    active_window_end = float(active_temps[-1]) if len(active_temps) > 0 else 550.0
    
    runaway_risk = "LOW"
    if max_dtg_rate > 0.8:  # excessive reaction rate
        runaway_risk = "HIGH"
    elif max_dtg_rate > 0.4:
        runaway_risk = "MEDIUM"
        
    # 2. Slagging Risk: Calculate basicity ratio on oxide compositions
    # Slagging index B/A = (Fe2O3 + CaO + MgO + Na2O + K2O) / (SiO2 + Al2O3)
    try:
        sio2 = oxides_dict.get("Ash_SiO2", 1.0)
        al2o3 = oxides_dict.get("Ash_Al2O3", 1.0)
        fe2o3 = oxides_dict.get("Ash_Fe2O3", 0.0)
        cao = oxides_dict.get("Ash_CaO", 0.0)
        mgo = oxides_dict.get("Ash_MgO", 0.0)
        na2o = oxides_dict.get("Ash_Na2O", 0.0)
        k2o = oxides_dict.get("Ash_K2O", 0.0)
        
        acid = sio2 + al2o3
        basic = fe2o3 + cao + mgo + na2o + k2o
        slagging_index = basic / acid if acid > 0 else 0.0
    except Exception:
        slagging_index = 0.0
        
    slagging_risk = "LOW"
    if slagging_index > 1.2:
        slagging_risk = "HIGH"  # Severe fluid-bed sticking/slagging risk
    elif slagging_index > 0.6:
        slagging_risk = "MEDIUM"
        
    passed = (runaway_risk != "HIGH") and (slagging_risk != "HIGH")
    
    report = (
        f"Kinetic Peak Temp: {peak_temp_c:.1f}°C, "
        f"Max DTG Loss Rate: {max_dtg_rate:.3f}%/°C. "
        f"Alkali Basicity Index: {slagging_index:.2f}."
    )
    
    return {
        "passed": passed,
        "peak_temp_c": peak_temp_c,
        "max_dtg_rate": max_dtg_rate,
        "slagging_index": slagging_index,
        "runaway_risk": runaway_risk,
        "slagging_risk": slagging_risk,
        "active_window": (active_window_start, active_window_end),
        "report": report
    }


# -----------------------------------------------------------------------------
# Curve Plotting Engine
# -----------------------------------------------------------------------------
def plot_tg_dtg_simulations(
    temperatures_c: np.ndarray,
    tg_curve: np.ndarray,
    dtg_curve: np.ndarray,
    output_path: Path,
    title: str = "Co-Pyrolysis Virtual TG/DTG Simulation",
) -> None:
    """Save publication-quality TG and DTG curve figures (PNG and SVG)."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Primary axis for TG curve
    color = 'tab:blue'
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Mass Remaining (%)', color=color)
    ax1.plot(temperatures_c, tg_curve, color=color, linewidth=2, label='TG (Mass Remaining)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Secondary axis for DTG curve
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Mass Loss Rate (%/°C)', color=color)
    ax2.plot(temperatures_c, dtg_curve, color=color, linewidth=1.8, linestyle='--', label='DTG (Rate)')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(title, fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    # Save both vector and raster formats
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".svg"), format="svg")
    plt.close(fig)
