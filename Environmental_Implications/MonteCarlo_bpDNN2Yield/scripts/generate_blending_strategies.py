"""
Feedstock Blending Strategy Optimizer for Co-Pyrolysis Product Yields
======================================================================
This utility discovers and optimizes multi-feedstock co-pyrolysis blending recipes
designed to maximize Biochar, Bioliquid, and Biogas product yields when mixed
with U.S. sewage sludge.

Key Capabilities:
-----------------
1. Parses SHAP attribution values to pre-screen and rank the most promising additives.
2. Integrates continuous mathematical optimization (Scipy):
   - Brent's Bounded method (minimize_scalar) for single-additive optimization.
   - Constrained SLSQP method (minimize) for joint multi-additive optimizations under Sum-Equality capacity locks.
3. Implements mathematical sparsity filters (ratios >= 0.1%) to protect against degenerate,
   collapsed combination representations (guaranteeing k-component co-pyrolysis).
4. Leverages parallel multiprocessing (multiprocessing.Pool) to scale combination sweeps.
5. Injects the comprehensive global FEEDSTOCK_NAMES physical mappings (1 to 118).
6. Automatically formats publication-ready graphs and datasets featuring ID (Physical Name).
"""

from __future__ import annotations

# Set threading environment variables before importing numpy to prevent OpenBLAS/MKL thread thrashing
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import multiprocessing
import logging
import itertools
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import minimize_scalar, minimize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Global Mappings and Configurations
# -----------------------------------------------------------------------------

TARGET_DIR_MAP = {
    "Biochar": "01_Biochar",
    "Bioliquid": "01_Bioliquid",
    "Biogas": "01_Biogas",
}

OUTPUT_IDX = {"Biochar": 0, "Bioliquid": 1, "Biogas": 2}

FT_PREFIX = "FeedstockType_"
MR_PREFIX = "MixingRatio_"

# Physical feedstock name mapping compiled from literature database (1 to 118)
FEEDSTOCK_NAMES = {
    1: "Alum sludge",
    2: "Anaerobic sewage sludge",
    3: "Paper mill sludge",
    4: "Pharmaceutical sludge",
    5: "Pulp and paper industry wastewater sludge",
    6: "Sewage sludge",
    7: "Sewage sludge anaerobically digested",
    8: "Textile dyeing sludge",
    9: "Activated sludge",
    10: "Primary sludge",
    11: "Almond shell",
    12: "Amaranthus retroflexus L. biomass",
    13: "Bambara groundnut shell",
    14: "Corn stover",
    15: "Corncob",
    16: "Cornelian cherry stones",
    17: "Cotton stalk",
    18: "Dried distillers grains with solubles",
    19: "Garlic biomass",
    20: "Grape biomass",
    21: "Hazelnut kernel husk",
    22: "Lemon peel",
    23: "Banana residues (peal and leaves)",
    24: "Oat straw",
    25: "Olive waste",
    26: "Orange peel and pomace",
    27: "Oreganum stalk",
    28: "Palm kernel shell",
    29: "Peanut shell biomass",
    30: "Pepper stem",
    31: "Pistachio shell",
    32: "Rice husk",
    33: "Rice straw",
    34: "Saffron petals",
    35: "Sugarcane biomass",
    36: "Sunflower shell biomass",
    37: "Tobacco leaf",
    38: "Tobacco stalk",
    39: "Ugu plant",
    40: "Vine pruning biomass",
    41: "Walnut shell",
    42: "Waste cereals",
    43: "Waste nuts and shells",
    44: "Watermelon rind",
    45: "Wheat straw",
    46: "Oilseed rape straw",
    47: "Waste tire",
    48: "Chicken litter",
    49: "Chicken bedding materials",
    50: "Cattle manure",
    51: "Goat manure",
    52: "Camel manure",
    53: "Swine manure",
    54: "Poultry manure",
    55: "Horse manure",
    56: "Turkey litter",
    57: "Water buffalo manure",
    58: "Cladophora sp.",
    59: "Lyngbya sp.",
    60: "Ulva lactuca aquatic biomass",
    61: "Beech wood",
    62: "Coconut",
    63: "Empty fruit bunch",
    64: "Eucalyptus biomass",
    65: "Hazelnut shell",
    66: "Maesopsis eminii wood",
    67: "Mesocarp fiber",
    68: "Bamboo biomass",
    69: "Palm shell",
    70: "Pine wood",
    71: "Rubber wood",
    72: "Spruce wood",
    73: "Willow",
    74: "Wood sawdust",
    75: "Bone residues",
    76: "Food waste anaerobically digested",
    77: "Meat and bone meal",
    78: "Raw food waste",
    79: "Waste plastic mixture",
    80: "Polyethylene",
    81: "Polypropylene",
    82: "Polystyrene",
    83: "Cellulose",
    84: "Hemicellulose",
    85: "Lignin",
    86: "Xylan",
    87: "Humic acid",
    88: "Fulvic acid",
    89: "Humin",
    90: "Al2O3",
    91: "Ca(OH)2",
    92: "Ca-bentonite",
    93: "CaO",
    94: "K2CO3",
    95: "Kaolin",
    96: "MgO",
    97: "Algal biomass",
    98: "Sewage sludge aerobically digested",
    99: "Poplar wood",
    100: "Polyethylene terephthalate",
    101: "Aquatic plant biomass",
    102: "HZSM-5",
    103: "Potato peel",
    104: "Co(NO3)2",
    105: "Ni(NO3)2",
    106: "Fe(NO3)3",
    107: "Durian waste",
    108: "Mango waste",
    109: "Tomato peel",
    110: "Herbaceous plant biomass",
    111: "Mesquite tree",
    112: "Shrubby plant biomass",
    113: "Arbor tree plant biomass",
    114: "Coffee husk",
    115: "Refuse-derived fuel",
    116: "Low-density polyethylene",
    117: "Macroalgae",
    118: "High-density polyethylene"
}

# -----------------------------------------------------------------------------
# Add script parent to system path for imports
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from shap_analysis_yield import (  # type: ignore
    load_matlab_data,
    extract_neural_network_data,
    generate_feature_names,
    MatlabNeuralNetworkWrapper,
)

# -----------------------------------------------------------------------------
# Module level helper functions
# -----------------------------------------------------------------------------


def _load_feature_names(feature_file: str) -> List[str]:
    """Read feature names file and extract features."""
    names = []
    with open(feature_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if ". " in line:
                try:
                    _, feat = line.strip().split(". ", 1)
                    names.append(feat)
                except ValueError:
                    continue
    return names


def summarise_feedstock_shap(shap_vals: np.ndarray, feat_names: List[str]) -> pd.DataFrame:
    """Return DataFrame with mean SHAP statistics for each feedstock."""
    shap_arr = np.asarray(shap_vals, dtype=float)
    if shap_arr.ndim != 2 or shap_arr.shape[1] != len(feat_names):
        raise ValueError("SHAP array shape and feature list length mismatch.")

    # Identify feedstock & mixing-ratio indices
    ft_indices: dict[int, int] = {}
    mr_indices: dict[int, int] = {}
    for idx, fname in enumerate(feat_names):
        if fname.startswith(FT_PREFIX):
            ft_idx = int(fname.replace(FT_PREFIX, ""))
            ft_indices[ft_idx] = idx
        elif fname.startswith(MR_PREFIX):
            mr_idx = int(fname.replace(MR_PREFIX, ""))
            mr_indices[mr_idx] = idx

    global _ORIG_FEATURE_MATRIX
    if _ORIG_FEATURE_MATRIX is None:
        raise RuntimeError("Original feature matrix has not been initialised.")

    X_arr = _ORIG_FEATURE_MATRIX

    records = []
    for i in range(1, 119):
        ft_col = ft_indices.get(i)
        mr_col = mr_indices.get(i)
        if ft_col is None or mr_col is None:
            continue

        ft_shap = shap_arr[:, ft_col]
        mr_shap = shap_arr[:, mr_col]

        # Derive statistics from original inputs
        ratio_values = X_arr[:, mr_col]
        present_mask = ratio_values > 0

        if present_mask.any():
            mean_ratio = ratio_values[present_mask].mean()
            median_ratio = np.median(ratio_values[present_mask])
            max_ratio = ratio_values[present_mask].max()
            sample_count = int(present_mask.sum())
        else:
            mean_ratio = median_ratio = max_ratio = 0.0
            sample_count = 0

        rec = {
            "Feedstock_ID": i,
            "Samples_used": sample_count,
            "Mean_ratio": round(float(mean_ratio), 4),
            "Median_ratio": round(float(median_ratio), 4),
            "Max_ratio": round(float(max_ratio), 4),
            "FeedstockType_meanSHAP": ft_shap.mean(),
            "FeedstockType_posMean": ft_shap[ft_shap > 0].mean() if (ft_shap > 0).any() else 0.0,
            "MixingRatio_meanSHAP": mr_shap.mean(),
            "MixingRatio_posMean": mr_shap[mr_shap > 0].mean() if (mr_shap > 0).any() else 0.0,
        }
        rec["Combined_score"] = rec["FeedstockType_posMean"] + rec["MixingRatio_posMean"]
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values("Combined_score", ascending=False).reset_index(drop=True)
    return df


# -----------------------------------------------------------------------------
# Parallel Worker for Continuous Joint Blending Recipe Optimization
# -----------------------------------------------------------------------------


def _optimize_combo_worker(args: tuple) -> dict | None:
    """Optimize a single feedstock combination using SLSQP in parallel."""
    (
        combo,
        feat_names_net,
        baseline_vec,
        train_min,
        train_range,
        y_train_min,
        y_range,
        tgt_01,
        leftover_ratio,
        combo_max_ratio,
        baseline_yield,
        mat_file,
        product_name,
        target_idx,
    ) = args

    # Local imports inside child processes to ensure independence
    import numpy as np
    from scipy.optimize import minimize
    from shap_analysis_yield import (
        load_matlab_data,
        extract_neural_network_data,
        MatlabNeuralNetworkWrapper,
    )

    try:
        # Load network locally in worker to avoid pickling/sharing issues
        mat_data = load_matlab_data(mat_file)
        _, _, net_struct = extract_neural_network_data(mat_data)
        wrapper = MatlabNeuralNetworkWrapper(net_struct, target_idx=target_idx)
    except Exception:
        return None

    def combo_obj_fun(ratios):
        x = baseline_vec.copy()
        for fid, ratio_val in zip(combo, ratios):
            ft_col = feat_names_net.index(f"FeedstockType_{fid}")
            mr_col = feat_names_net.index(f"MixingRatio_{fid}")
            x[ft_col] = 1.0
            x[mr_col] = ratio_val

        # Scale input
        x_scaled = np.clip((x - train_min) / train_range, 0.0, 1.0)
        y_pred_scaled = wrapper.predict(x_scaled.reshape(1, -1))[0]
        if tgt_01:
            y_val = y_pred_scaled * y_range + y_train_min
        else:
            y_val = ((y_pred_scaled + 1.0) / 2.0) * y_range + y_train_min
        
        # We want to MAXIMIZE target yield, so return negative
        return -float(y_val)

    # SLSQP constraints: sum(ratios) == leftover_ratio
    cons = {"type": "eq", "fun": lambda r: np.sum(r) - leftover_ratio}
    bounds = [(0.0, combo_max_ratio) for _ in range(len(combo))]
    # Initial guess
    x0 = np.minimum(np.array([leftover_ratio / len(combo)] * len(combo)), combo_max_ratio)

    res = minimize(combo_obj_fun, x0, method="SLSQP", bounds=bounds, constraints=cons)

    if not res.success:
        return None

    best_yield_val = -float(res.fun)
    best_ratio_assignment = res.x

    # Sparsity protection: all ratios >= 0.1% (non-degenerate true k-component co-pyrolysis)
    if np.any(best_ratio_assignment < 0.001):
        return None

    actual_sum = float(np.sum(best_ratio_assignment))

    combo_names = " + ".join(FEEDSTOCK_NAMES.get(fid, f"ID_{fid}") for fid in combo)
    mixing_ratios_names = " + ".join(
        f"{FEEDSTOCK_NAMES.get(fid, f'ID_{fid}')}:{ratio_val:.4f}" for fid, ratio_val in zip(combo, best_ratio_assignment)
    )

    return {
        "Combo": "-".join(map(str, combo)),
        "Combo_Names": combo_names,
        "Size": len(combo),
        "Pred_yield_%": best_yield_val,
        "Delta_yield_%": best_yield_val - baseline_yield,
        "MixingRatios": "-".join(
            f"{fid}:{ratio_val:.4f}" for fid, ratio_val in zip(combo, best_ratio_assignment)
        ),
        "MixingRatios_Names": mixing_ratios_names,
        "TotalFeedRatio": round(actual_sum, 4),
        "SludgeRatio": round(1.0 - actual_sum, 4),
    }


# -----------------------------------------------------------------------------
# Core Product processing workflow
# -----------------------------------------------------------------------------


def process_product(
    product_name: str,
    shap_root: str,
    out_root: str,
    top_n: int,
    simulate: bool = False,
    mat_file: str | None = None,
    mc_csv: str | None = None,
    ratio_grid: np.ndarray | None = None,
    temp_values: np.ndarray | None = None,
    combo_size: int = 2,
    combo_max_ratio: float = 0.25,
    sludge_ratio: float = 0.50,
    method: str = "scipy",
    cores: int = 1,
    max_single_ratio: float = 0.50,
):
    """Analyze feedstock contributions and optimize blending recipes for a product."""
    logger.info("=======================================================================")
    logger.info("Processing target product: %s", product_name)
    logger.info("=======================================================================")

    subdir = TARGET_DIR_MAP[product_name]
    target_path = Path(shap_root) / subdir
    shap_file = target_path / "shap_values.npy"
    feat_file = Path(shap_root) / "00_feature_names_used.txt"

    if not shap_file.exists():
        logger.error("SHAP values file not found for %s: %s", product_name, shap_file)
        return

    shap_values = np.load(str(shap_file))
    feat_names_net = _load_feature_names(str(feat_file))

    df_summary = summarise_feedstock_shap(shap_values, feat_names_net)
    df_top = df_summary.head(top_n).copy()
    # Insert physical feedstock names
    df_top.insert(1, "Feedstock_Name", df_top["Feedstock_ID"].map(FEEDSTOCK_NAMES))

    global _ORIG_FEATURE_MATRIX, _NN_WRAPPER
    global _TRAIN_MIN, _TRAIN_RANGE, _TARGET_MIN, _TARGET_RANGE, _TARGET_IN_0_1, _BASELINE_YIELDS

    baseline_yield = _BASELINE_YIELDS.get(product_name, 0.0)

    if simulate:
        if mat_file is None or mc_csv is None:
            raise ValueError("Simulation requires --mat_file and --mc_csv arguments")

        mc_df = pd.read_csv(mc_csv)
        
        # Build baseline input vector
        baseline_vec = np.zeros(len(feat_names_net), dtype=float)
        for i, feat in enumerate(feat_names_net):
            if feat in mc_df.columns:
                baseline_vec[i] = mc_df[feat].median()
            else:
                baseline_vec[i] = 0.0
                if feat == "ReactorType":
                    if "ReactorType" in mc_df.columns and not mc_df["ReactorType"].empty:
                        baseline_vec[i] = mc_df["ReactorType"].mode().iloc[0]

        train_min = _TRAIN_MIN
        train_range = _TRAIN_RANGE
        y_train_min = _TARGET_MIN[OUTPUT_IDX[product_name]]
        y_range = _TARGET_RANGE[OUTPUT_IDX[product_name]]
        tgt_01 = _TARGET_IN_0_1

        best_ratios = []
        pred_yields = []
        delta_yields = []

        logger.info("Optimizing single blending ratios for top %d feedstocks (method: %s) ...", len(df_top), method)
        for row in df_top.itertuples():
            feed_id = int(row.Feedstock_ID)
            ft_col = feat_names_net.index(f"{FT_PREFIX}{feed_id}")
            mr_col = feat_names_net.index(f"{MR_PREFIX}{feed_id}")

            if method == "scipy":
                # Continuous parabolic optimization using Brent's method
                def single_obj(ratio):
                    x = baseline_vec.copy()
                    x[ft_col] = 1.0
                    x[mr_col] = ratio
                    x_scaled = np.clip((x - train_min) / train_range, 0.0, 1.0)
                    _NN_WRAPPER.target_idx = OUTPUT_IDX[product_name]
                    pred_scaled = _NN_WRAPPER.predict(x_scaled.reshape(1, -1))[0]
                    if tgt_01:
                        y_val = pred_scaled * y_range + y_train_min
                    else:
                        y_val = ((pred_scaled + 1.0) / 2.0) * y_range + y_train_min
                    return -float(y_val)  # minimize negative to maximize

                res = minimize_scalar(single_obj, bounds=(0.0, max_single_ratio), method="bounded")
                best_yield_val = -float(res.fun)
                best_ratio = float(res.x)
            else:
                # Traditional discrete grid search
                best_yield_val = -np.inf
                best_ratio = None
                active_grid = ratio_grid[ratio_grid <= max_single_ratio] if len(ratio_grid) > 0 else ratio_grid
                if len(active_grid) == 0:
                    active_grid = np.array([max_single_ratio])
                
                for ratio in active_grid:
                    x = baseline_vec.copy()
                    x[ft_col] = 1.0
                    x[mr_col] = ratio
                    x_scaled = np.clip((x - train_min) / train_range, 0.0, 1.0)
                    _NN_WRAPPER.target_idx = OUTPUT_IDX[product_name]
                    pred_scaled = _NN_WRAPPER.predict(x_scaled.reshape(1, -1))[0]
                    if tgt_01:
                        y_val = pred_scaled * y_range + y_train_min
                    else:
                        y_val = ((pred_scaled + 1.0) / 2.0) * y_range + y_train_min
                    
                    if y_val > best_yield_val:
                        best_yield_val = y_val
                        best_ratio = ratio

            best_ratios.append(best_ratio)
            pred_yields.append(best_yield_val)
            delta_yields.append(best_yield_val - baseline_yield)

        df_top["Best_ratio"] = best_ratios
        df_top["Pred_yield_%"] = pred_yields
        df_top["Delta_yield_%"] = delta_yields

        # Re-rank by largest positive yield gain
        df_top.sort_values("Delta_yield_%", ascending=False, inplace=True, ignore_index=True)

    # Establish output path
    product_out = Path(out_root) / product_name
    product_out.mkdir(parents=True, exist_ok=True)

    csv_path = product_out / f"top_{top_n}_feedstocks_yield.csv"
    df_top.to_csv(csv_path, index=False)
    logger.info("Saved top feedstock ranking CSV for %s → %s", product_name, csv_path)

    # ------------------------------------------------------------------
    # Temperature (Conversion) dependence plots
    # ------------------------------------------------------------------
    if simulate and temp_values is not None and len(temp_values) > 1:
        temp_idx = feat_names_net.index("TargetTemperature/Celsius") if "TargetTemperature/Celsius" in feat_names_net else None
        if temp_idx is not None:
            pos_df = df_top[df_top["Delta_yield_%"] > 0].copy()
            if not pos_df.empty:
                fig_temp, ax_temp = plt.subplots(figsize=(8, 5))
                temp_data = {}
                for row in pos_df.itertuples():
                    feed_id = int(row.Feedstock_ID)
                    ft_col = feat_names_net.index(f"{FT_PREFIX}{feed_id}")
                    mr_col = feat_names_net.index(f"{MR_PREFIX}{feed_id}")
                    ratio_use = row.Best_ratio if not pd.isna(row.Best_ratio) else 0.1
                    deltas_series = []
                    for temp in temp_values:
                        x = baseline_vec.copy()
                        x[temp_idx] = temp
                        x[ft_col] = 1.0
                        x[mr_col] = ratio_use
                        x_scaled = np.clip((x - train_min) / train_range, 0.0, 1.0)
                        _NN_WRAPPER.target_idx = OUTPUT_IDX[product_name]
                        pred_scaled = _NN_WRAPPER.predict(x_scaled.reshape(1, -1))[0]
                        if tgt_01:
                            pred_real = pred_scaled * y_range + y_train_min
                        else:
                            pred_real = ((pred_scaled + 1.0) / 2.0) * y_range + y_train_min
                        deltas_series.append(pred_real - baseline_yield)
                    
                    feed_name = FEEDSTOCK_NAMES.get(feed_id, "")
                    label_str = f"ID {feed_id} ({feed_name})" if feed_name else f"ID {feed_id}"
                    ax_temp.plot(temp_values, deltas_series, label=label_str)
                    # Label columns strictly in format "ID (Physical Name)" to elevate academic reading
                    temp_data[f"{feed_id} ({feed_name})"] = deltas_series

                ax_temp.set_xlabel("TargetTemperature/Celsius")
                ax_temp.set_ylabel("Δ Yield (%) vs baseline")
                ax_temp.set_title(f"ΔYield vs Temperature for Positive-Gain Candidates ({product_name})")
                ax_temp.legend(fontsize=6, ncol=2)
                fig_temp.tight_layout()
                
                fig_temp.savefig(product_out / "conversion_dependence.png", dpi=300)
                fig_temp.savefig(product_out / "conversion_dependence.svg", format="svg")
                plt.close(fig_temp)
                logger.info("Saved temperature-conversion dependence plot to %s", product_out / "conversion_dependence.png")

                temp_df = pd.DataFrame(temp_data, index=temp_values)
                temp_df.index.name = "TargetTemperature/Celsius"
                conv_csv_path = product_out / "conversion_dependence_data.csv"
                temp_df.to_csv(conv_csv_path)
                logger.info("Saved temperature conversion dependency data to %s", conv_csv_path)

    # ------------------------------------------------------------------
    # Joint Combination Blending Recipe Search
    # ------------------------------------------------------------------
    if simulate and combo_size >= 2:
        pos_ids = df_top[df_top["Delta_yield_%"] > 0]["Feedstock_ID"].astype(int).tolist()
        logger.info("Searching joint combinations among %d positive-gain candidates ...", len(pos_ids))

        leftover_ratio = max(0.0, 1.0 - sludge_ratio)

        if method == "scipy":
            combos = []
            for r in range(2, combo_size + 1):
                for combo in itertools.combinations(pos_ids, r):
                    if leftover_ratio > r * combo_max_ratio + 1e-8:
                        continue
                    combos.append(combo)

            if combos:
                tasks = [
                    (
                        combo,
                        feat_names_net,
                        baseline_vec,
                        train_min,
                        train_range,
                        _TARGET_MIN[OUTPUT_IDX[product_name]],
                        _TARGET_RANGE[OUTPUT_IDX[product_name]],
                        _TARGET_IN_0_1,
                        leftover_ratio,
                        combo_max_ratio,
                        baseline_yield,
                        mat_file,
                        product_name,
                        OUTPUT_IDX[product_name],
                    )
                    for combo in combos
                ]

                logger.info("Running parallel SLSQP optimization on %d combinations with %d core(s) ...", len(combos), cores)
                if cores > 1:
                    with multiprocessing.Pool(processes=cores) as pool:
                        combo_results = pool.map(_optimize_combo_worker, tasks)
                else:
                    combo_results = [_optimize_combo_worker(t) for t in tasks]

                combo_records = [r for r in combo_results if r is not None]
            else:
                combo_records = []

            if combo_records:
                combo_df = pd.DataFrame(combo_records)
                combo_df.sort_values("Delta_yield_%", ascending=False, inplace=True, ignore_index=True)
                combo_df["Delta_yield_positive"] = combo_df["Delta_yield_%"] > 0
            else:
                combo_df = pd.DataFrame(columns=["Combo", "Combo_Names", "Size", "Pred_yield_%", "Delta_yield_%", "MixingRatios", "MixingRatios_Names", "TotalFeedRatio", "SludgeRatio", "Delta_yield_positive"])
        else:
            # Discrete Grid Search Fallback
            combo_records = []
            for r in range(2, combo_size + 1):
                for combo in itertools.combinations(pos_ids, r):
                    for ratios in itertools.product(ratio_grid, repeat=r):
                        if any(np.isclose(ratio, 0.0, atol=1e-8) for ratio in ratios):
                            continue
                        total_ratio = sum(ratios)
                        if total_ratio <= 0 or total_ratio > leftover_ratio:
                            continue

                        x = baseline_vec.copy()
                        for fid, ratio in zip(combo, ratios):
                            ft_col = feat_names_net.index(f"{FT_PREFIX}{fid}")
                            mr_col = feat_names_net.index(f"{MR_PREFIX}{fid}")
                            x[ft_col] = 1.0
                            x[mr_col] = ratio

                        x_scaled = np.clip((x - train_min) / train_range, 0.0, 1.0)
                        _NN_WRAPPER.target_idx = OUTPUT_IDX[product_name]
                        pred_scaled = _NN_WRAPPER.predict(x_scaled.reshape(1, -1))[0]
                        if _TARGET_IN_0_1:
                            pred_real = pred_scaled * y_range + y_train_min
                        else:
                            pred_real = ((pred_scaled + 1.0) / 2.0) * y_range + y_train_min

                        delta = pred_real - baseline_yield

                        combo_names = " + ".join(FEEDSTOCK_NAMES.get(fid, f"ID_{fid}") for fid in combo)
                        mixing_ratios_names = " + ".join(
                            f"{FEEDSTOCK_NAMES.get(fid, f'ID_{fid}')}:{ratio_val:.4f}" for fid, ratio_val in zip(combo, ratios)
                        )

                        combo_records.append({
                            "Combo": "-".join(map(str, combo)),
                            "Combo_Names": combo_names,
                            "Size": r,
                            "Pred_yield_%": pred_real,
                            "Delta_yield_%": delta,
                            "MixingRatios": "-".join(f"{fid}:{ratio_val:.4f}" for fid, ratio_val in zip(combo, ratios)),
                            "MixingRatios_Names": mixing_ratios_names,
                            "TotalFeedRatio": round(total_ratio, 4),
                            "SludgeRatio": round(1.0 - total_ratio, 4),
                            "Delta_yield_positive": delta > 0,
                        })

            if combo_records:
                combo_df = pd.DataFrame(combo_records)
                combo_df.sort_values("Delta_yield_%", ascending=False, inplace=True, ignore_index=True)
            else:
                combo_df = pd.DataFrame(columns=["Combo", "Combo_Names", "Size", "Pred_yield_%", "Delta_yield_%", "MixingRatios", "MixingRatios_Names", "TotalFeedRatio", "SludgeRatio", "Delta_yield_positive"])

        # Save combinations
        combo_csv = product_out / "combo_results.csv"
        combo_df.to_csv(combo_csv, index=False)
        logger.info("Saved joint combinations recipe ranking CSV → %s", combo_csv)

        # Plot best combinations by size
        if not combo_df.empty and combo_df["Delta_yield_positive"].any():
            best_per_size = combo_df[combo_df["Delta_yield_positive"]].groupby("Size").first().reset_index()
            best_per_size.to_csv(product_out / "combo_best_data.csv", index=False)
            logger.info("Saved best combination per size CSV → %s", product_out / "combo_best_data.csv")

            fig_c, ax_c = plt.subplots(figsize=(8, 5))
            ax_c.bar(best_per_size["Size"].astype(str), best_per_size["Delta_yield_%"], color="purple")
            ax_c.set_xlabel("Number of feedstocks in combo")
            ax_c.set_ylabel("Best ΔYield (%) vs baseline")
            plt.title(f"Best combo ΔYield by size ({product_name})")

            for idx, row in best_per_size.iterrows():
                x = str(row["Size"])
                y = row["Delta_yield_%"]
                combo = row["Combo_Names"]
                ratios = row["MixingRatios_Names"]
                if len(combo) > 30:
                    combo = combo[:27] + "..."
                label = f"{combo}\n{ratios}"
                ax_c.annotate(label, xy=(idx, y), xytext=(0, 5), textcoords='offset points',
                              ha='center', va='bottom', fontsize=7, rotation=90)

            fig_c.tight_layout()
            combo_png = product_out / "combo_best.png"
            combo_svg = product_out / "combo_best.svg"
            fig_c.savefig(combo_png, dpi=300)
            fig_c.savefig(combo_svg, format="svg")
            plt.close(fig_c)
            logger.info("Saved combination best-of-size plot → %s", combo_png)

    # ------------------------------------------------------------------
    # Visualizations: Single Feedstock Contributions
    # ------------------------------------------------------------------
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ids = df_top["Feedstock_ID"].astype(str)
        score_vals = df_top["Combined_score"]
        mean_ratios = df_top["Mean_ratio"]

        # Overlay physical feedstock name under x-ticks if space permits
        ticks = [f"ID {fid}\n({FEEDSTOCK_NAMES.get(int(fid), '')[:15]})" for fid in ids]

        ax1.bar(ids, score_vals, color="skyblue", label="Combined positive SHAP score")
        ax1.set_xlabel("Feedstock Type")
        ax1.set_ylabel("Combined SHAP score", color="skyblue")
        ax1.tick_params(axis="y", labelcolor="skyblue")
        ax1.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)

        ax2 = ax1.twinx()
        ax2.plot(ids, mean_ratios, color="orangered", marker="o", label="Mean mixing ratio (0–1)")
        ax2.set_ylabel("Mean mixing ratio (0–1)", color="orangered")
        ax2.tick_params(axis="y", labelcolor="orangered")

        plt.title(f"Top {top_n} Feedstocks – {product_name}")
        fig.tight_layout()

        fig.savefig(product_out / f"top_{top_n}_feedstocks.png", dpi=300)
        fig.savefig(product_out / f"top_{top_n}_feedstocks.svg", format="svg")
        plt.close(fig)

        logger.info("Saved contribution plots to %s", product_out / f"top_{top_n}_feedstocks.png")

        # Yield Delta Gain plot
        if simulate and "Delta_yield_%" in df_top.columns:
            try:
                fig2, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(ids, df_top["Delta_yield_%"], color="seagreen")

                for bar, ratio_val in zip(bars, df_top["Best_ratio"]):
                    height = bar.get_height()
                    offset = 1.5 if height >= 0 else -1.5
                    va = "bottom" if height >= 0 else "top"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + offset,
                        f"{ratio_val:.2f}",
                        ha="center",
                        va=va,
                        fontsize=8,
                        color="black",
                        rotation=90,
                    )

                ax.set_xlabel("Feedstock Type")
                ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=7)
                ax.set_ylabel("Δ Yield (%) vs baseline", color="seagreen")
                plt.title(f"Predicted Yield Gain – {product_name} (Baseline {baseline_yield:.2f} %)")
                fig2.tight_layout()
                
                fig2.savefig(product_out / f"top_{top_n}_delta_gain.png", dpi=300)
                fig2.savefig(product_out / f"top_{top_n}_delta_gain.svg", format="svg")
                plt.close(fig2)
                logger.info("Saved delta yield gain plots to %s", product_out / f"top_{top_n}_delta_gain.png")
            except Exception as e:
                logger.warning("Failed to create delta-yield plot: %s", e)
    except Exception as e:
        logger.warning("Failed to generate contribution charts: %s", e)


# -----------------------------------------------------------------------------
# Main CLI Command Line Parser
# -----------------------------------------------------------------------------


def main():
    # Dynamically locate the latest SHAP results folder if available
    shap_outputs_dir = PROJECT_ROOT / "results" / "shap_outputs"
    latest_shap_dir = ""
    if shap_outputs_dir.exists():
        subdirs = [d for d in shap_outputs_dir.iterdir() if d.is_dir() and d.name.startswith("SHAP_Analysis_Results_")]
        if subdirs:
            latest_shap_dir = str(max(subdirs, key=lambda d: d.name))
    if not latest_shap_dir:
        latest_shap_dir = str(shap_outputs_dir)

    parser = argparse.ArgumentParser(
        description="Optimize co-pyrolysis blending strategies for product yields.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--shap_dir", type=str, default=latest_shap_dir, help="Path to SHAP analysis results")
    parser.add_argument("--output_dir", type=str, default="../results/blending_outputs", help="Directory where product subfolders are saved")
    parser.add_argument("--top", type=int, default=15, help="Number of top feedstock candidates to analyze")
    parser.add_argument("--mat_file", type=str, default="../bpDNN4PyroProd_modelfiles/Results_trained.mat", help="Path to pre-trained MATLAB .mat weights")
    parser.add_argument("--simulate", action="store_true", help="Execute neural-network forward optimizations")
    parser.add_argument("--mc_csv", type=str, default="../results/mc_outputs/mc_us_sludge_predictions.csv", help="Baseline sludge prediction CSV")
    parser.add_argument("--test_ratio", type=float, default=None, help="Force a static mixing ratio (0-1), ignoring optimization bounds")
    parser.add_argument("--ratio_search", type=str, default="0.05:0.95:0.05", help="Discrete grid specification 'start:end:step' for grid method")
    parser.add_argument("--combo_size", type=int, default=3, help="Maximum combo size k of additives to evaluate")
    parser.add_argument("--combo_ratio_limit", type=float, default=None, help="Maximum total additive ratio limit (defaults to 1.0 - sludge_ratio)")
    parser.add_argument("--sludge_ratio", type=float, default=0.50, help="Sludge fraction locked in final co-pyrolysis blend")
    parser.add_argument("--max-single-ratio", type=float, default=0.50, help="Single additive mixing ratio cap")
    parser.add_argument("--combo_max_ratio", type=float, default=0.25, help="Single additive cap inside multi-additive combos")
    parser.add_argument("--method", choices=["scipy", "grid"], default="scipy", help="Optimization solver framework")
    default_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    parser.add_argument("--cores", type=int, default=default_cores, help="Multiprocessing Pool processes cap")
    parser.add_argument("--temp_grid", type=str, default="300:700:10", help="Temperature grid specification 'start:end:step' for dependence charts")

    args = parser.parse_args()

    shap_dir = Path(args.shap_dir)
    # Auto-resolve to latest subfolder if passed as parent directory
    if shap_dir.exists() and not (shap_dir / "00_feature_names_used.txt").exists():
        subdirs = [d for d in shap_dir.iterdir() if d.is_dir() and d.name.startswith("SHAP_Analysis_Results_")]
        if subdirs:
            latest_sub = max(subdirs, key=lambda d: d.name)
            logger.info("Provided shap_dir is a parent folder; auto-resolving to latest run: %s", latest_sub)
            shap_dir = latest_sub
    out_dir = Path(args.output_dir)
    mat_file = Path(args.mat_file)

    if not mat_file.exists():
        raise FileNotFoundError(f"MATLAB model not found at {mat_file}")

    # ------------------------------------------------------------------
    # Load and cache neural network training parameters globally
    # ------------------------------------------------------------------
    global _ORIG_FEATURE_MATRIX, _ORIG_FEATURE_NAMES, _TRAIN_MIN, _TRAIN_RANGE
    global _TARGET_MIN, _TARGET_MAX, _TARGET_RANGE, _TARGET_IN_0_1, _NN_WRAPPER, _BASELINE_YIELDS

    logger.info("Global loading MATLAB training weights ...")
    mat_data = load_matlab_data(str(mat_file))
    X_mat, y_mat, net_struct = extract_neural_network_data(mat_data)
    _ORIG_FEATURE_MATRIX = X_mat
    _ORIG_FEATURE_NAMES = generate_feature_names(X_mat, mat_data)

    _TRAIN_MIN = X_mat.min(axis=0)
    _TRAIN_MAX = X_mat.max(axis=0)
    _TRAIN_RANGE = _TRAIN_MAX - _TRAIN_MIN
    _TRAIN_RANGE[_TRAIN_RANGE == 0] = 1.0

    y_train_dummy = y_mat.astype(float)
    _TARGET_MIN = y_train_dummy.min(axis=0)
    _TARGET_MAX = y_train_dummy.max(axis=0)
    _TARGET_RANGE = _TARGET_MAX - _TARGET_MIN
    _TARGET_RANGE[_TARGET_RANGE == 0] = 1.0
    _TARGET_IN_0_1 = np.all(_TARGET_MIN >= -1e-6) and np.all(_TARGET_MAX <= 1.0 + 1e-6)

    _NN_WRAPPER = MatlabNeuralNetworkWrapper(net_struct, target_idx=0)

    # Resolve and parse temperature grid for graphs
    try:
        t_start, t_end, t_step = map(float, args.temp_grid.split(":"))
        temp_values = np.arange(t_start, t_end + 1e-8, t_step)
    except ValueError:
        raise ValueError("temp_grid must be in 'start:end:step' format")

    # Parsing discrete legacy ratio grid fallback
    global _RATIO_GRID
    if args.test_ratio is None:
        try:
            r_start, r_end, r_step = map(float, args.ratio_search.split(":"))
        except ValueError:
            raise ValueError("ratio_search must be in 'start:end:step' format")
        _RATIO_GRID = np.round(np.arange(r_start, r_end + 1e-8, r_step), 4)
        _RATIO_GRID = _RATIO_GRID[_RATIO_GRID > 1e-8]
    else:
        _RATIO_GRID = np.array([args.test_ratio])

    # Cache baseline yield values
    _BASELINE_YIELDS = {"Biochar": 0.0, "Bioliquid": 0.0, "Biogas": 0.0}
    if args.simulate:
        mc_csv_path = Path(args.mc_csv)
        if mc_csv_path.exists():
            mc_df = pd.read_csv(mc_csv_path)
            for prod in _BASELINE_YIELDS.keys():
                if prod in mc_df.columns:
                    _BASELINE_YIELDS[prod] = float(mc_df[prod].median())
        logger.info("Baseline yields parsed from MC predictions: %s", _BASELINE_YIELDS)

    combo_ratio_limit = args.combo_ratio_limit
    if combo_ratio_limit is None:
        combo_ratio_limit = max(0.0, 1.0 - args.sludge_ratio)

    # Process all three products sequentially
    for product in TARGET_DIR_MAP.keys():
        process_product(
            product_name=product,
            shap_root=str(shap_dir),
            out_root=str(out_dir),
            top_n=args.top,
            simulate=args.simulate,
            mat_file=str(mat_file),
            mc_csv=str(args.mc_csv),
            ratio_grid=_RATIO_GRID,
            temp_values=temp_values,
            combo_size=args.combo_size,
            combo_max_ratio=args.combo_max_ratio,
            sludge_ratio=args.sludge_ratio,
            method=args.method,
            cores=args.cores,
            max_single_ratio=args.max_single_ratio,
        )

    logger.info("All yield co-pyrolysis blending strategy optimizations finalized.")


# -----------------------------------------------------------------------------
# Module level caches for clean imports and subprocess execution
# -----------------------------------------------------------------------------
_ORIG_FEATURE_MATRIX: np.ndarray | None = None
_ORIG_FEATURE_NAMES: List[str] | None = None
_TRAIN_MIN: np.ndarray | None = None
_TRAIN_RANGE: np.ndarray | None = None
_TARGET_MIN: np.ndarray | None = None
_TARGET_MAX: np.ndarray | None = None
_TARGET_RANGE: np.ndarray | None = None
_TARGET_IN_0_1: bool = True
_NN_WRAPPER: MatlabNeuralNetworkWrapper | None = None
_BASELINE_YIELDS: dict | None = None
_RATIO_GRID: np.ndarray | None = None


if __name__ == "__main__":
    main()