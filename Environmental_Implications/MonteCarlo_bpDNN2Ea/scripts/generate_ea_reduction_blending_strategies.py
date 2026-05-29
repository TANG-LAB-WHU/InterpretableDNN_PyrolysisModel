"""
Optimization of Feedstock Blending Strategies for Minimizing Activation Energy (Ea)
==================================================================================
This script analyzes SHAP attribution values of the Ea neural network model to identify
optimal blending strategies. It features advanced continuous numerical optimization
to break the discrete ratio limitation and supports 192-core HPC parallelization.

Workflow
--------
1.  Load SHAP values to identify feedstocks whose presence/mixing ratio most strongly
    reduce Ea (highest negative mean SHAP contribution).
2.  Rank the 118 candidate feedstock types (additives) based on their combined negative SHAP score (Type + Ratio).
3.  Perform forward neural-network simulations for the top candidates:
    - Single feedstock optimization using Brent's Bounded method (minimize_scalar)
      to find the exact mathematical optimal mixing ratio.
    - Exhaustive combination search among negative-gain feedstocks.
    - Joint feedstock ratio optimization under bounds and equality constraints
      (sum of ratios = 1.0 - sludge_ratio) using SLSQP (minimize).
4.  Optionally compute and plot ΔEa over a specified Degree_conversion grid.
5.  Save results to:
    - ``top_20_feedstocks_ea.csv`` - optimized single-feedstock ratios and ΔEa.
    - ``combo_results.csv`` - optimized multi-feedstock joint mixtures.
    - Figures (Violin/Bar/Line plots) for delta gain and conversion dependence.

Usage:
---
    python generate_ea_reduction_blending_strategies.py --simulate --search_combos --method scipy

"""

import os
# Set threading environment variables before importing numpy to prevent OpenBLAS/MKL thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import argparse
import logging
from typing import List
import multiprocessing
import warnings

import numpy as np
import pandas as pd
import matplotlib
import itertools
from scipy.optimize import minimize_scalar, minimize

# Suppress duplicate variable name warnings from SciPy MATLAB loading
warnings.filterwarnings("ignore", message="Duplicate variable name")

matplotlib.use("Agg")  # Safe for headless execution
import matplotlib.pyplot as plt



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
#  Shared constants
# -----------------------------------------------------------------------------

FT_PREFIX = "FeedstockType_"
MR_PREFIX = "MixingRatio_"

_RATIO_GRID_DEFAULT = (0.05, 0.50, 0.05)  # Default search grid if --simulate

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
#  Helper utilities (imported from shap_analysis_ea.py residing at project root)
# -----------------------------------------------------------------------------

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

try:
    from shap_analysis_ea import (
        load_matlab_data,
        extract_neural_network_data,
        generate_feature_names,
        MatlabNeuralNetworkWrapper,
    )
except Exception as imp_err:
    raise ImportError(
        "Unable to import helper utilities from shap_analysis_ea.py – check project structure."
    ) from imp_err


# -----------------------------------------------------------------------------
#  SHAP-related helpers
# -----------------------------------------------------------------------------

def _load_feature_names(name_file: str) -> List[str]:
    """Read *00_feature_names_used.txt* produced by SHAP analysis."""
    names: List[str] = []
    with open(name_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if ". " in line:
                try:
                    _idx, feat = line.strip().split(". ", 1)
                    names.append(feat)
                except ValueError:
                    continue
    return names


def summarise_feedstock_shap(shap_vals: np.ndarray, feat_names: List[str]) -> pd.DataFrame:
    """Return DataFrame ranking feedstocks by negative SHAP (Ea-reducing) impact."""
    shap_arr = np.asarray(shap_vals, dtype=float)
    if shap_arr.ndim != 2 or shap_arr.shape[1] != len(feat_names):
        raise ValueError("SHAP array shape and feature list length mismatch.")

    # Build index maps
    ft_idx: dict[int, int] = {}
    mr_idx: dict[int, int] = {}
    for i, name in enumerate(feat_names):
        if name.startswith(FT_PREFIX):
            ft_id = int(name.replace(FT_PREFIX, ""))
            ft_idx[ft_id] = i
        elif name.startswith(MR_PREFIX):
            mr_id = int(name.replace(MR_PREFIX, ""))
            mr_idx[mr_id] = i

    records = []
    for feed_id in range(1, 119):  # 1 … 118 inclusive
        if feed_id not in ft_idx or feed_id not in mr_idx:
            continue  # skip if any part missing
        ft_shap = shap_arr[:, ft_idx[feed_id]]
        mr_shap = shap_arr[:, mr_idx[feed_id]]

        ft_neg_mean = ft_shap[ft_shap < 0].mean() if (ft_shap < 0).any() else 0.0
        mr_neg_mean = mr_shap[mr_shap < 0].mean() if (mr_shap < 0).any() else 0.0

        record = {
            "Feedstock_ID": feed_id,
            "FeedstockType_negMean": ft_neg_mean,
            "MixingRatio_negMean": mr_neg_mean,
        }
        # Positive magnitude of combined negative effect (bigger → more Ea reduction)
        record["Combined_score"] = abs(ft_neg_mean) + abs(mr_neg_mean)
        records.append(record)

    df = pd.DataFrame(records)
    df.sort_values("Combined_score", ascending=False, inplace=True, ignore_index=True)
    return df


# -----------------------------------------------------------------------------
#  Forward simulation helpers (optional)
# -----------------------------------------------------------------------------

def prepare_network(mat_path: str):
    """Load MATLAB .mat network + training data for scaling."""
    mat_data = load_matlab_data(mat_path)
    X_train, y_train, net_struct = extract_neural_network_data(mat_data)
    feature_names = generate_feature_names(X_train, mat_data)

    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    train_range = train_max - train_min
    train_range[train_range == 0] = 1.0

    y_train_min = y_train.min()
    y_train_max = y_train.max()
    y_range = y_train_max - y_train_min if y_train_max != y_train_min else 1.0
    target_in_0_1 = 0.0 <= y_train_min <= 1.0 and 0.0 <= y_train_max <= 1.0

    wrapper = MatlabNeuralNetworkWrapper(net_struct)

    return wrapper, feature_names, train_min, train_range, y_train_min, y_range, target_in_0_1


def predict_ea(
    wrapper: "MatlabNeuralNetworkWrapper",
    x_unscaled: np.ndarray,
    train_min: np.ndarray,
    train_range: np.ndarray,
    y_train_min: float,
    y_range: float,
    target_in_0_1: bool,
):
    """Scale *x_unscaled*, predict Ea (kJ/mol) and inverse-transform."""
    x_scaled = np.clip((x_unscaled - train_min) / train_range, 0.0, 1.0)
    y_pred_scaled = wrapper.predict(x_scaled.reshape(1, -1))[0]
    if target_in_0_1:
        ea = y_pred_scaled * y_range + y_train_min
    else:
        ea = ((y_pred_scaled + 1.0) / 2.0) * y_range + y_train_min
    return float(ea)

# -----------------------------------------------------------------------------
#  Combination ratio search helper
# -----------------------------------------------------------------------------

def generate_ratio_vectors(num_feeds: int, total_ratio: float, grid: np.ndarray, max_ratio: float, tol: float = 1e-6):
    """Generate all length-*num_feeds* vectors from *grid* whose elements sum to *total_ratio*.

    Each element must be <= *max_ratio*.  Returns list of tuples.
    The search is depth-first and prunes infeasible paths early for efficiency.
    """

    # Clean grid: positive values <= max_ratio and not greater than total_ratio
    grid_vals = sorted({float(round(v, 8)) for v in grid if (v > 0) and (v <= max_ratio + tol) and (v <= total_ratio + tol)})
    if not grid_vals:
        return []

    vectors: list[tuple[float, ...]] = []

    def _dfs(depth: int, current: list[float], remaining: float):
        if depth == num_feeds - 1:
            # Last element must take the remaining amount
            if abs(remaining) <= tol and (not current):
                return  # skip empty
            if abs(remaining) <= tol:
                return  # rounding artifacts leading to zero remainder
            if remaining < -tol or remaining > max_ratio + tol:
                return
            # Check if remaining value is in grid (within tol)
            for gv in grid_vals:
                if abs(gv - remaining) <= tol:
                    vectors.append(tuple(current + [gv]))
                    break
            return

        # Choose next value
        for v in grid_vals:
            if v > remaining + tol:
                break  # grid sorted ascending; rest will be larger
            _dfs(depth + 1, current + [v], remaining - v)

    _dfs(0, [], round(total_ratio, 8))
    return vectors


# -----------------------------------------------------------------------------
#  Main processing routine
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
        baseline_ea,
        mat_file,
    ) = args

    # Local imports inside child processes to ensure independence
    import numpy as np
    from scipy.optimize import minimize
    from shap_analysis_ea import (
        load_matlab_data,
        extract_neural_network_data,
        MatlabNeuralNetworkWrapper,
    )

    try:
        # Load network locally in worker to avoid ctypes/MATLAB pickling issues
        mat_data = load_matlab_data(mat_file)
        _, _, net_struct = extract_neural_network_data(mat_data)
        wrapper = MatlabNeuralNetworkWrapper(net_struct)
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
            ea = y_pred_scaled * y_range + y_train_min
        else:
            ea = ((y_pred_scaled + 1.0) / 2.0) * y_range + y_train_min
        return float(ea)

    # SLSQP Setup: Sum of auxiliary ratios must EXACTLY equal leftover_ratio (1.0 - sludge_ratio)
    # This enforces a strictly fixed sludge ratio across all combinations, maintaining rigorous scientific control.
    cons = {"type": "eq", "fun": lambda r: np.sum(r) - leftover_ratio}
    bounds = [(0.0, combo_max_ratio) for _ in range(len(combo))]
    # Initialize guess, clipping to combo_max_ratio to stay within bounds
    x0 = np.minimum(np.array([leftover_ratio / len(combo)] * len(combo)), combo_max_ratio)

    res = minimize(combo_obj_fun, x0, method="SLSQP", bounds=bounds, constraints=cons)

    if not res.success:
        return None

    best_ea_val = float(res.fun)
    best_ratio_assignment = res.x

    # Sparsity & Physical Consistency Check:
    # Enforce that all selected feedstocks in a Size=k combination are genuinely active (ratio >= 0.1%).
    # This mathematically prevents degenerate representations where a larger size combo collapses 
    # to a smaller one by setting redundant variables to exactly 0.00%.
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
        "Pred_Ea_kJmol": best_ea_val,
        "Delta_Ea_kJmol": best_ea_val - baseline_ea,
        "MixingRatios": "-".join(
            f"{fid}:{ratio_val:.4f}" for fid, ratio_val in zip(combo, best_ratio_assignment)
        ),
        "MixingRatios_Names": mixing_ratios_names,
        "TotalFeedRatio": round(actual_sum, 4),
        "SludgeRatio": round(1.0 - actual_sum, 4),
    }


# -----------------------------------------------------------------------------
#  Main processing routine
# -----------------------------------------------------------------------------

def process_ea(
    shap_dir: str,
    out_dir: str,
    top_n: int,
    simulate: bool,
    mat_file: str | None = None,
    mc_csv: str | None = None,
    ratio_grid: np.ndarray | None = None,
    conversion_values: np.ndarray | None = None,
    search_combos: bool = False,
    combo_max_ratio: float = 0.5,
    sludge_ratio: float = 0.5,
    method: str = "scipy",
    cores: int = 1,
    max_single_ratio: float = 0.20,
):
    # Ensure output directory exists early for any plots
    os.makedirs(out_dir, exist_ok=True)

    shap_file = os.path.join(shap_dir, "shap_values.npy")
    feat_file = os.path.join(shap_dir, "00_feature_names_used.txt")

    if not os.path.exists(shap_file):
        raise FileNotFoundError(f"SHAP values not found: {shap_file}")

    shap_values = np.load(shap_file)
    feature_names = _load_feature_names(feat_file)

    df_rank = summarise_feedstock_shap(shap_values, feature_names)
    df_top = df_rank.head(top_n).copy()
    # Insert physical feedstock names as a separate column for direct read
    df_top.insert(1, "Feedstock_Name", df_top["Feedstock_ID"].map(FEEDSTOCK_NAMES))

    # Optional forward simulation ------------------------------------------------
    if simulate:
        if mat_file is None or mc_csv is None:
            raise ValueError("--simulate requires --mat_file and --mc_csv arguments")

        wrapper, feat_names_net, train_min, train_range, y_train_min, y_range, tgt_01 = prepare_network(mat_file)

        # Baseline Ea from Monte-Carlo predictions
        mc_df = pd.read_csv(mc_csv)
        baseline_ea = mc_df["Ea_kJmol"].median()
        logger.info("Baseline median Ea: %.3f kJ/mol", baseline_ea)

        # Build baseline input vector (median of numeric cols or 0)
        baseline_vec = np.zeros(len(feat_names_net), dtype=float)
        for i, feat in enumerate(feat_names_net):
            if feat in mc_df.columns:
                baseline_vec[i] = mc_df[feat].median()
            else:
                baseline_vec[i] = 0.0

        best_ratios = []
        pred_eas = []
        delta_eas = []

        logger.info("Finding optimal mixing ratios for top %d feedstocks (method: %s) ...", len(df_top), method)
        for row in df_top.itertuples():
            feed_id = int(row.Feedstock_ID)
            ft_col = feat_names_net.index(f"{FT_PREFIX}{feed_id}")
            mr_col = feat_names_net.index(f"{MR_PREFIX}{feed_id}")

            if method == "scipy":
                # Continuous numerical optimization using minimize_scalar (Brent's Bounded method)
                def single_obj(ratio):
                    x = baseline_vec.copy()
                    x[ft_col] = 1.0
                    x[mr_col] = ratio
                    return predict_ea(wrapper, x, train_min, train_range, y_train_min, y_range, tgt_01)

                res = minimize_scalar(single_obj, bounds=(0.0, max_single_ratio), method="bounded")
                best_ea_val = float(res.fun)
                best_ratio = float(res.x)
            else:
                # Traditional discrete grid search
                best_ea_val = np.inf
                best_ratio = None
                # Filter ratio grid to respect max_single_ratio constraint
                active_grid = ratio_grid[ratio_grid <= max_single_ratio] if len(ratio_grid) > 0 else ratio_grid
                if len(active_grid) == 0:
                    active_grid = np.array([max_single_ratio])
                for ratio in active_grid:
                    x = baseline_vec.copy()
                    x[ft_col] = 1.0
                    x[mr_col] = ratio
                    ea_val = predict_ea(
                        wrapper,
                        x,
                        train_min,
                        train_range,
                        y_train_min,
                        y_range,
                        tgt_01,
                    )
                    if ea_val < best_ea_val:
                        best_ea_val = ea_val
                        best_ratio = ratio

            best_ratios.append(best_ratio)
            pred_eas.append(best_ea_val)
            # Negative ΔEa means activation energy is reduced relative to baseline
            delta_eas.append(best_ea_val - baseline_ea)

        df_top["Best_ratio"] = best_ratios
        df_top["Pred_Ea_kJmol"] = pred_eas
        df_top["Delta_Ea_kJmol"] = delta_eas

        # Re-rank by most negative ΔEa (largest Ea reduction)
        df_top.sort_values("Delta_Ea_kJmol", ascending=True, inplace=True, ignore_index=True)

        # ------------------------------------------------------------------
        # Degree_conversion dependence plot (optional)
        # ------------------------------------------------------------------
        if conversion_values is not None and len(conversion_values) > 1:
            conv_idx = feat_names_net.index("Degree_conversion") if "Degree_conversion" in feat_names_net else None
            if conv_idx is not None:
                neg_df = df_top[df_top["Delta_Ea_kJmol"] < 0].copy()
                if not neg_df.empty:
                    fig_conv, ax_conv = plt.subplots(figsize=(8, 5))
                    conv_data = {}
                    for row in neg_df.itertuples():
                        feed_id = int(row.Feedstock_ID)
                        ft_col = feat_names_net.index(f"{FT_PREFIX}{feed_id}")
                        mr_col = feat_names_net.index(f"{MR_PREFIX}{feed_id}")
                        ratio_use = row.Best_ratio if not pd.isna(row.Best_ratio) else 0.1
                        deltas_series = []
                        for conv in conversion_values:
                            x = baseline_vec.copy()
                            x[conv_idx] = conv
                            x[ft_col] = 1.0
                            x[mr_col] = ratio_use
                            ea_val = predict_ea(wrapper, x, train_min, train_range, y_train_min, y_range, tgt_01)
                            deltas_series.append(ea_val - baseline_ea)
                        feed_name = FEEDSTOCK_NAMES.get(feed_id, "")
                        label_str = f"ID {feed_id} ({feed_name})" if feed_name else f"ID {feed_id}"
                        ax_conv.plot(conversion_values, deltas_series, label=label_str)
                        conv_data[f"ID_{feed_id} ({feed_name})"] = deltas_series
                    ax_conv.set_xlabel("Degree_conversion")
                    ax_conv.set_ylabel("Δ Ea (kJ/mol) vs baseline")
                    ax_conv.set_title("ΔEa vs Conversion for Negative-Gain Candidates")
                    ax_conv.legend(fontsize=6, ncol=2)
                    fig_conv.tight_layout()
                    fig_conv.savefig(os.path.join(out_dir, f"conversion_dependence.png"))
                    fig_conv.savefig(os.path.join(out_dir, f"conversion_dependence.svg"), format="svg")
                    plt.close(fig_conv)
                    logger.info("Saved conversion-dependence plot → %s", os.path.join(out_dir, "conversion_dependence.png"))

                    # Save raw data to CSV
                    conv_df = pd.DataFrame(conv_data, index=conversion_values)
                    conv_df.index.name = "Degree_conversion"
                    conv_csv_path = os.path.join(out_dir, "conversion_dependence_data.csv")
                    conv_df.to_csv(conv_csv_path)
                    logger.info("Saved conversion data → %s", conv_csv_path)

        # ------------------------------------------------------------------
        # Combination search across all negative-gain candidates (optional)
        # ------------------------------------------------------------------
        if simulate and search_combos:
            neg_ids = df_top[df_top["Delta_Ea_kJmol"] < 0]["Feedstock_ID"].astype(int).tolist()
            logger.info("Searching combinations among %d negative-gain feedstocks …", len(neg_ids))

            max_feed_ratio = combo_max_ratio
            leftover_ratio = max(0.0, 1.0 - sludge_ratio)

            if method == "scipy":
                # Continuous optimization with multi-core parallelization
                combos = []
                for r in range(2, len(neg_ids) + 1):
                    for combo in itertools.combinations(neg_ids, r):
                        # Skip early if bounds are mathematically infeasible
                        if leftover_ratio > r * max_feed_ratio + 1e-8:
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
                            y_train_min,
                            y_range,
                            tgt_01,
                            leftover_ratio,
                            max_feed_ratio,
                            baseline_ea,
                            mat_file,
                        )
                        for combo in combos
                    ]
                    logger.info("Launching parallel SLSQP optimization across %d cores for %d combinations ...", cores, len(combos))
                    with multiprocessing.Pool(processes=cores) as pool:
                        results = pool.map(_optimize_combo_worker, tasks)
                    combo_records = [r for r in results if r is not None]
                else:
                    combo_records = []
            else:
                # Traditional discrete grid search (single-threaded)
                combo_records = []
                for r in range(2, len(neg_ids) + 1):
                    for combo in itertools.combinations(neg_ids, r):
                        # If per-feed cap already prevents feasible allocation, skip early
                        if leftover_ratio > r * max_feed_ratio + 1e-8:
                            continue

                        # Generate candidate ratio vectors (order-sensitive permutations considered later)
                        ratio_vectors = generate_ratio_vectors(r, leftover_ratio, ratio_grid, max_feed_ratio)
                        if not ratio_vectors:
                            continue

                        best_ea_val = np.inf
                        best_ratio_assignment = None

                        # Test each ratio vector and its permutations (if r > 1)
                        for vec in ratio_vectors:
                            perms = [vec] if r == 1 else set(itertools.permutations(vec))
                            for perm in perms:
                                x = baseline_vec.copy()
                                for fid, ratio_val in zip(combo, perm):
                                    ft_col = feat_names_net.index(f"{FT_PREFIX}{fid}")
                                    mr_col = feat_names_net.index(f"{MR_PREFIX}{fid}")
                                    x[ft_col] = 1.0
                                    x[mr_col] = ratio_val
                                ea_val = predict_ea(wrapper, x, train_min, train_range, y_train_min, y_range, tgt_01)
                                if ea_val < best_ea_val:
                                    best_ea_val = ea_val
                                    best_ratio_assignment = perm

                        if best_ratio_assignment is None:
                            continue

                        combo_names = " + ".join(FEEDSTOCK_NAMES.get(fid, f"ID_{fid}") for fid in combo)
                        mixing_ratios_names = " + ".join(
                            f"{FEEDSTOCK_NAMES.get(fid, f'ID_{fid}')}:{ratio_val:.3f}" for fid, ratio_val in zip(combo, best_ratio_assignment)
                        )
                        combo_records.append({
                            "Combo": "-".join(map(str, combo)),
                            "Combo_Names": combo_names,
                            "Size": r,
                            "Pred_Ea_kJmol": best_ea_val,
                            "Delta_Ea_kJmol": best_ea_val - baseline_ea,
                            "MixingRatios": "-".join(
                                f"{fid}:{ratio_val:.3f}" for fid, ratio_val in zip(combo, best_ratio_assignment)
                            ),
                            "MixingRatios_Names": mixing_ratios_names,
                            "TotalFeedRatio": leftover_ratio,
                            "SludgeRatio": sludge_ratio,
                        })

            if combo_records:
                df_combo = pd.DataFrame(combo_records)
                df_combo.sort_values("Delta_Ea_kJmol", inplace=True)
                combo_csv = os.path.join(out_dir, "combo_results.csv")
                df_combo.to_csv(combo_csv, index=False)
                logger.info("Saved combo results → %s (best ΔEa %.2f)", combo_csv, df_combo["Delta_Ea_kJmol"].min())

                # Plot best combos per size
                best_per_size = df_combo.groupby("Size").first().reset_index()
                fig_c, ax_c = plt.subplots(figsize=(8,5))
                ax_c.bar(best_per_size["Size"].astype(str), best_per_size["Delta_Ea_kJmol"], color="purple")
                ax_c.set_xlabel("Number of feedstocks in combo")
                ax_c.set_ylabel("Best ΔEa (kJ/mol) vs baseline")
                ax_c.set_title("Best combo ΔEa by size")
                fig_c.tight_layout()
                combo_png = os.path.join(out_dir, "combo_best.png")

                # Annotate each bar with combo and ratios
                for idx, row in best_per_size.iterrows():
                    x = str(row["Size"])
                    y = row["Delta_Ea_kJmol"]
                    combo = row["Combo"]
                    ratios = row["MixingRatios"]
                    label = f"{combo}\n{ratios}"
                    ax_c.annotate(label, xy=(idx, y), xytext=(0, 5), textcoords='offset points',
                                  ha='center', va='bottom', fontsize=7, rotation=90)

                fig_c.savefig(combo_png, dpi=300)
                fig_c.savefig(combo_png.replace(".png", ".svg"), format="svg")
                plt.close(fig_c)
                logger.info("Saved combo plot → %s", combo_png)

                # Export plot data to CSV
                best_per_size.to_csv(os.path.join(out_dir, "combo_best_data.csv"), index=False)

    # ----------------------------------------------------------------------
    #  Output: CSV + basic plots
    # ----------------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"top_{top_n}_feedstocks_ea.csv")
    df_top.to_csv(csv_path, index=False)
    logger.info("Saved ranking → %s", csv_path)

    # Combined-score bar plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        ids = df_top["Feedstock_ID"].astype(str)
        scores = df_top["Combined_score"]
        ax.bar(ids, scores, color="steelblue")
        ax.set_xlabel("Feedstock ID")
        ax.set_ylabel("Combined |neg SHAP| score (higher → lower Ea)")
        plt.title(f"Top {top_n} Ea-reducing Feedstocks")
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"top_{top_n}_feedstocks.png")
        svg_path = png_path.replace(".png", ".svg")
        fig.savefig(png_path, dpi=300)
        fig.savefig(svg_path, format="svg")
        plt.close(fig)
        logger.info("Saved plot → %s", png_path)
    except Exception as e:
        logger.warning("Failed to create bar plot: %s", e)

    # Delta-Ea plot
    if simulate:
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ids = df_top["Feedstock_ID"].astype(str)
            deltas = df_top["Delta_Ea_kJmol"].fillna(0.0)
            bars = ax2.bar(ids, deltas, color="seagreen")
            ax2.set_xlabel("Feedstock ID")
            ax2.set_ylabel("Δ Ea (kJ/mol) vs baseline (negative = improvement)")
            plt.title(f"Predicted Ea Gain – Baseline {baseline_ea:.2f} kJ/mol")

            # annotate ratio on bars
            for bar, ratio in zip(bars, df_top["Best_ratio"]):
                if ratio is None or np.isnan(ratio):
                    continue
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5 if height >= 0 else height - 0.5,
                    f"{height:.2f}\n(r={ratio:.2f})",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=8,
                    rotation=90,
                )
            fig2.tight_layout()
            delta_png = os.path.join(out_dir, f"top_{top_n}_delta_gain.png")
            delta_svg = delta_png.replace(".png", ".svg")
            fig2.savefig(delta_png, dpi=300)
            fig2.savefig(delta_svg, format="svg")
            plt.close(fig2)
            logger.info("Saved ΔEa values to plot (%s) – min %.2f, max %.2f", delta_png, deltas.min(), deltas.max())
        except Exception as e:
            logger.warning("Failed to create ΔEa plot: %s", e)


def main():
    from pathlib import Path
    
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Dynamically locate the latest SHAP results folder if available
    shap_outputs_dir = project_root / "results" / "shap_outputs"
    latest_shap_dir = ""
    if shap_outputs_dir.exists():
        subdirs = [d for d in shap_outputs_dir.iterdir() if d.is_dir() and d.name.startswith("SHAP_Analysis_Ea_Results_")]
        if subdirs:
            latest_shap_dir = str(max(subdirs, key=lambda d: d.name))
    if not latest_shap_dir:
        latest_shap_dir = str(shap_outputs_dir)
        
    default_output_dir = project_root / "results" / "blending_outputs"
    default_mat_file = project_root / "bpDNN4Ea_modelfiles" / "Results_trained.mat"
    default_mc_csv = project_root / "results" / "mc_outputs" / "mc_ea_predictions.csv"

    parser = argparse.ArgumentParser(
        description="Generate feedstock blending strategies that reduce activation energy (Ea).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shap_dir",
        type=str,
        default=latest_shap_dir,
        help="Directory containing SHAP results for Ea (shap_values.npy & 00_feature_names_used.txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(default_output_dir),
        help="Directory to write CSV and plots",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top feedstocks to keep",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Enable forward simulation with neural network to estimate absolute Ea",
    )
    parser.add_argument(
        "--mat_file",
        type=str,
        default=str(default_mat_file),
        help="Path to MATLAB .mat file with trained neural-network (required for --simulate)",
    )
    parser.add_argument(
        "--mc_csv",
        type=str,
        default=str(default_mc_csv),
        help="Monte-Carlo CSV containing baseline predictions (required for --simulate)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=None,
        help="If set, test only this mixing-ratio value for each feedstock",
    )
    parser.add_argument(
        "--ratio_search",
        type=str,
        default="0.05:0.5:0.05",
        help="Ratio grid 'start:end:step' used when searching best ratio (ignored if --test_ratio)",
    )
    parser.add_argument(
        "--conversion_grid",
        type=str,
        default=None,
        help="If set, e.g. '0.1:1.0:0.1', plot ΔEa over Degree_conversion grid.",
    )
    parser.add_argument(
        "--search_combos",
        action="store_true",
        help="Enable exhaustive combination search among negative-gain feedstocks.",
    )
    parser.add_argument(
        "--combo_max_ratio",
        type=float,
        default=0.20,
        help="Maximum ratio per feedstock when searching combinations.",
    )
    parser.add_argument(
        "--sludge_ratio",
        type=float,
        default=0.5,
        help="Fraction of sludge in the final blend. Feedstock ratios in each combo will sum to (1 - sludge_ratio).",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["scipy", "grid"],
        default="scipy",
        help="Optimization method: 'scipy' for continuous optimization, 'grid' for discrete scan.",
    )
    
    # Added --cores CLI option, defaulting to SLURM_CPUS_PER_TASK for HPC scaling
    default_cores = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    parser.add_argument(
        "--cores",
        type=int,
        default=default_cores,
        help="Number of CPU cores for parallel combination search.",
    )
    
    # Added --max-single-ratio CLI option to limit the single-feedstock replacement ratio and respect waste disposal constraints
    parser.add_argument(
        "--max-single-ratio",
        type=float,
        default=0.20,
        help="Maximum mixing ratio for a single feedstock optimization (e.g. 0.20 for 20%% addition limit, keeping sludge at >=80%%).",
    )
    args = parser.parse_args()

    # Build ratio grid
    if args.test_ratio is not None:
        ratio_grid = np.array([round(float(args.test_ratio), 4)])
    else:
        try:
            r_start, r_end, r_step = map(float, args.ratio_search.split(":"))
            if not (0.0 <= r_start < r_end <= 1.0 and r_step > 0):
                raise ValueError
        except ValueError:
            raise ValueError("--ratio_search must be 'start:end:step' within 0–1")
        ratio_grid = np.round(np.arange(r_start, r_end + 1e-8, r_step), 4)

    # Build conversion values array if requested
    conv_values = None
    if args.conversion_grid:
        try:
            c_start, c_end, c_step = map(float, args.conversion_grid.split(":"))
            conv_values = np.round(np.arange(c_start, c_end + 1e-8, c_step), 4)
        except ValueError:
            raise ValueError("--conversion_grid must be 'start:end:step'")

    # Process
    shap_dir = os.path.abspath(args.shap_dir)
    out_dir = os.path.abspath(args.output_dir)

    logger.info("Using SHAP dir: %s", shap_dir)
    logger.info("Writing output to: %s", out_dir)

    process_ea(
        shap_dir=shap_dir,
        out_dir=out_dir,
        top_n=args.top,
        simulate=args.simulate,
        mat_file=os.path.abspath(args.mat_file) if args.simulate else None,
        mc_csv=os.path.abspath(args.mc_csv) if args.simulate else None,
        ratio_grid=ratio_grid,
        conversion_values=conv_values,
        search_combos=args.search_combos,
        combo_max_ratio=args.combo_max_ratio,
        sludge_ratio=args.sludge_ratio,
        method=args.method,
        cores=args.cores,
        max_single_ratio=args.max_single_ratio,
    )

    logger.info("Analysis completed.")


if __name__ == "__main__":
    main()