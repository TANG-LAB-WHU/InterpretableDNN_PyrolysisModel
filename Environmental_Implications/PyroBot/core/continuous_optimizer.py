"""
PyroBot: Mathematical Continuous Optimization Layer
===================================================
This module provides continuous optimization solvers (Brent scalar minimization
and SLSQP constrained multi-variable optimization) to identify optimal co-pyrolysis
feedstock recipe simplexes.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .dnn_surrogates import MatlabNeuralNetworkWrapper, scale_features, unscale_outputs

# -----------------------------------------------------------------------------
# Feedstock Database Names Loader
# -----------------------------------------------------------------------------
def load_feedstock_names(reference_xlsx: Path) -> Dict[int, str]:
    """Read feedstock ID to physical name mapping from reference Excel."""
    try:
        df = pd.read_excel(reference_xlsx)
        first_col = df.columns[0]
        return {i + 1: str(name).strip() for i, name in enumerate(df[first_col])}
    except Exception as e:
        print(f"Warning: Failed to load feedstock names: {e}. Fallback mapping activated.")
        # Fallback dictionary for basic robustness
        return {
            1: "Alum sludge", 2: "Anaerobic sewage sludge", 3: "Paper mill sludge",
            6: "Sewage sludge", 8: "Textile dyeing sludge", 14: "Corn stover",
            41: "Walnut shell", 45: "Wheat straw", 70: "Pine wood"
        }


# -----------------------------------------------------------------------------
# Unified Multi-Objective Objective Function
# -----------------------------------------------------------------------------
def evaluate_blend_properties(
    blend_vector: np.ndarray,  # mixing ratio of additives (excludes locked sludge)
    sludge_ratio: float,
    active_additive_ids: list[int],
    baseline_vector: np.ndarray,
    feature_names: list[str],
    ea_wrapper: MatlabNeuralNetworkWrapper,
    ea_train_X: np.ndarray,
    ea_train_y: np.ndarray,
    yield_wrapper: MatlabNeuralNetworkWrapper,
    yield_train_X: np.ndarray,
    yield_train_y: np.ndarray,
) -> Tuple[float, float, float, float]:
    """
    Predict physical properties for a candidate blending recipe.
    
    Returns:
        Predicted apparent Ea (kJ/mol), Biochar Yield (%), Bioliquid Yield (%), Biogas Yield (%)
    """
    # 1. Reconstruct full 24-feature (or co-pyrolysis feature) vector
    x_raw = baseline_vector.copy()
    
    # Map locked sludge (Feedstock ID = 6 usually)
    # Ratios and IDs are mapped into FeedstockType_i and MixingRatio_i columns
    # Find positions
    ft_cols = [i for i, name in enumerate(feature_names) if name.startswith("FeedstockType_")]
    mr_cols = [i for i, name in enumerate(feature_names) if name.startswith("MixingRatio_")]
    
    # Reset all co-pyrolysis additive slots first to prevent residual mapping leak
    for col_idx in ft_cols + mr_cols:
        x_raw[col_idx] = 0.0
        
    # The first co-pyrolysis component slot is reserved for locked Sewage Sludge (ID = 6)
    x_raw[ft_cols[0]] = 6.0
    x_raw[mr_cols[0]] = sludge_ratio
    
    # Fill remaining slots with active candidate additives
    for slot_idx, (add_id, ratio) in enumerate(zip(active_additive_ids, blend_vector), start=1):
        if slot_idx < len(ft_cols):
            x_raw[ft_cols[slot_idx]] = float(add_id)
            x_raw[mr_cols[slot_idx]] = float(ratio)
            
    # Calculate derived kinetic attributes: ReactionTime = TargetTemperature / HeatingRate + deltaTime
    tt_idx = feature_names.index("TargetTemperature/Celsius")
    hr_idx = feature_names.index("HeatingRate/(K/min)") if "HeatingRate/(K/min)" in feature_names else feature_names.index("Heating rate")
    rt_idx = feature_names.index("ReactionTime/min") if "ReactionTime/min" in feature_names else feature_names.index("Reaction time")
    
    if x_raw[hr_idx] > 0:
        x_raw[rt_idx] = x_raw[tt_idx] / x_raw[hr_idx]
        
    # 2. apparent Ea forward prediction
    x_scaled_ea = scale_features(x_raw.reshape(1, -1), ea_train_X)
    ea_pred_norm = ea_wrapper.predict(x_scaled_ea)[0]
    ea_val = unscale_outputs(ea_pred_norm, ea_train_y)
    
    # 3. Product yields forward prediction (multi-output)
    x_scaled_yield = scale_features(x_raw.reshape(1, -1), yield_train_X)
    
    # Yield outputs unscaling index loop (Biochar=0, Bioliquid=1, Biogas=2)
    yields = []
    for out_idx in range(3):
        yield_wrapper.target_idx = out_idx
        y_pred_norm = yield_wrapper.predict(x_scaled_yield)[0]
        y_val = unscale_outputs(y_pred_norm, yield_train_y, out_idx)
        yields.append(y_val)
        
    return float(ea_val), float(yields[0]), float(yields[1]), float(yields[2])


# -----------------------------------------------------------------------------
# Brent Single-Additive Continuous Optimizer
# -----------------------------------------------------------------------------
def optimize_single_additive_blend(
    additive_id: int,
    sludge_ratio: float,
    max_additive_ratio: float,
    baseline_vector: np.ndarray,
    feature_names: list[str],
    ea_wrapper: MatlabNeuralNetworkWrapper,
    ea_train_X: np.ndarray,
    ea_train_y: np.ndarray,
    yield_wrapper: MatlabNeuralNetworkWrapper,
    yield_train_X: np.ndarray,
    yield_train_y: np.ndarray,
    objective: str = "ea_reduction",  # options: 'ea_reduction', 'char_maximization', 'pareto'
) -> Tuple[float, Dict[str, float]]:
    """
    Run continuous Bounded Brent's scalar minimization to find the optimal ratio
    for a single additive.
    
    Returns:
        Best Ratio, Dictionary of predicted properties at best ratio
    """
    def target_fun(ratio: float) -> float:
        ea, char, liq, gas = evaluate_blend_properties(
            blend_vector=np.array([ratio]),
            sludge_ratio=sludge_ratio,
            active_additive_ids=[additive_id],
            baseline_vector=baseline_vector,
            feature_names=feature_names,
            ea_wrapper=ea_wrapper,
            ea_train_X=ea_train_X,
            ea_train_y=ea_train_y,
            yield_wrapper=yield_wrapper,
            yield_train_X=yield_train_X,
            yield_train_y=yield_train_y
        )
        if objective == "ea_reduction":
            return ea  # Minimize apparent Ea
        elif objective == "char_maximization":
            return -char  # Maximize Biochar
        else:
            # Multi-objective Pareto: Minimize Ea while maximizing Biochar
            return ea - 10.0 * char
            
    # Bounded Brent's scalar search
    res = minimize_scalar(target_fun, bounds=(0.0, max_additive_ratio), method='bounded')
    best_ratio = float(res.x)
    
    # Calculate properties at the best ratio
    ea, char, liq, gas = evaluate_blend_properties(
        blend_vector=np.array([best_ratio]),
        sludge_ratio=sludge_ratio,
        active_additive_ids=[additive_id],
        baseline_vector=baseline_vector,
        feature_names=feature_names,
        ea_wrapper=ea_wrapper,
        ea_train_X=ea_train_X,
        ea_train_y=ea_train_y,
        yield_wrapper=yield_wrapper,
        yield_train_X=yield_train_X,
        yield_train_y=yield_train_y
    )
    
    stats = {"Ea": ea, "Char_Yield": char, "Liquid_Yield": liq, "Gas_Yield": gas}
    return best_ratio, stats


# -----------------------------------------------------------------------------
# SLSQP Constrained Multi-Variable Recipe Optimizer
# -----------------------------------------------------------------------------
def optimize_multi_feedstock_blend(
    active_additive_ids: list[int],
    sludge_ratio: float,
    max_individual_ratio: float,
    baseline_vector: np.ndarray,
    feature_names: list[str],
    ea_wrapper: MatlabNeuralNetworkWrapper,
    ea_train_X: np.ndarray,
    ea_train_y: np.ndarray,
    yield_wrapper: MatlabNeuralNetworkWrapper,
    yield_train_X: np.ndarray,
    yield_train_y: np.ndarray,
    objective: str = "ea_reduction",
    sparsity_threshold: float = 0.001,  # 0.1% Sparsity limit
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run constrained Sequential Least Squares Programming (SLSQP) minimization
    to discover optimal co-pyrolysis multi-component recipes.
    
    Returns:
        Array of optimal ratios, Dictionary of properties at optimal ratios
    """
    k = len(active_additive_ids)
    
    # 1. Simplex Sum-Equality Constraint: sum(r_i) = 1.0 - sludge_ratio
    total_additive_space = 1.0 - sludge_ratio
    constraints = [
        {"type": "eq", "fun": lambda r: np.sum(r) - total_additive_space}
    ]
    
    # 2. Individual additive bounds: 0 <= r_i <= max_individual_ratio
    bounds = [(0.0, max_individual_ratio) for _ in range(k)]
    
    # 3. Initial guestimate (equal ratios satisfying capacity)
    r0 = np.ones(k) * (total_additive_space / k)
    
    def target_fun(r: np.ndarray) -> float:
        ea, char, liq, gas = evaluate_blend_properties(
            blend_vector=r,
            sludge_ratio=sludge_ratio,
            active_additive_ids=active_additive_ids,
            baseline_vector=baseline_vector,
            feature_names=feature_names,
            ea_wrapper=ea_wrapper,
            ea_train_X=ea_train_X,
            ea_train_y=ea_train_y,
            yield_wrapper=yield_wrapper,
            yield_train_X=yield_train_X,
            yield_train_y=yield_train_y
        )
        if objective == "ea_reduction":
            return ea
        elif objective == "char_maximization":
            return -char
        else:
            # Balanced joint weights
            return ea - 5.0 * char
            
    # Continuous SLSQP simplex solver
    res = minimize(target_fun, r0, method='SLSQP', bounds=bounds, constraints=constraints)
    best_r = np.clip(res.x, 0.0, max_individual_ratio)
    
    # 4. Apply Sparsity Filter to prevent mathematical degeneracies (clip ratios < 0.1%)
    best_r[best_r < sparsity_threshold] = 0.0
    
    # Normalize back to preserve capacity constraint after clipping
    if np.sum(best_r) > 0:
        best_r = (best_r / np.sum(best_r)) * total_additive_space
    else:
        best_r = r0  # fallback to initial uniform
        
    # Re-evaluate final clean statistics
    ea, char, liq, gas = evaluate_blend_properties(
        blend_vector=best_r,
        sludge_ratio=sludge_ratio,
        active_additive_ids=active_additive_ids,
        baseline_vector=baseline_vector,
        feature_names=feature_names,
        ea_wrapper=ea_wrapper,
        ea_train_X=ea_train_X,
        ea_train_y=ea_train_y,
        yield_wrapper=yield_wrapper,
        yield_train_X=yield_train_X,
        yield_train_y=yield_train_y
    )
    
    stats = {"Ea": ea, "Char_Yield": char, "Liquid_Yield": liq, "Gas_Yield": gas}
    return best_r, stats


def main():
    print("Continuous optimizer library loaded successfully.")


if __name__ == "__main__":
    main()

