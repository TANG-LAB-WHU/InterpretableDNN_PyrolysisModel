import os
import numpy as np
import pandas as pd
import argparse
import logging
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for safe script execution
import matplotlib.pyplot as plt
from typing import List
import itertools  # Added for feedstock combination search

# -----------------------------------------------------------------------------
#  Feedstock Blending Strategy Generator
# -----------------------------------------------------------------------------
#  This utility analyses previously-computed SHAP values to discover which
#  feedstock types and mixing ratios are most beneficial for improving the
#  yields of Biochar, Bioliquid and Biogas during co-pyrolysis with U.S. sewage
#  sludge.  The script ranks the 118 available feedstock types according to
#  their average positive SHAP contribution and writes the top-N candidates
#  to separate CSV files inside product-specific sub-folders.
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
#  Utility helpers
# -----------------------------------------------------------------------------

TARGET_DIR_MAP = {
    "Biochar": "01_Char",
    "Bioliquid": "01_Liquid",
    "Biogas": "01_Gas",
}

# Mapping from product name to neural-network output index used in MatlabNeuralNetworkWrapper
OUTPUT_IDX = {"Biochar": 0, "Bioliquid": 1, "Biogas": 2}

# Columns prefixes in the feature list
FT_PREFIX = "FeedstockType_"
MR_PREFIX = "MixingRatio_"

# Default grid for ratio search (start, end, step). Updated via CLI.
_RATIO_GRID_DEFAULT = (0.05, 0.5, 0.05)  # 5 %–50 % by 5 % steps

# Attempt to import helper utilities from sibling directory (shap_analysis.py)
# This avoids code duplication and leverages the existing MATLAB-loader helpers.
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
    if PROJECT_ROOT not in os.sys.path:
        os.sys.path.append(PROJECT_ROOT)

    from shap_analysis import (
        load_matlab_data,
        extract_neural_network_data,
        generate_feature_names,
        MatlabNeuralNetworkWrapper,
    )
except Exception as _imp_err:
    raise ImportError(
        "Unable to import utilities from shap_analysis.py – ensure the project "
        "structure is intact. Original error: {}".format(_imp_err)
    )

def load_feature_names(feature_file: str) -> List[str]:
    """Read *00_feature_names_used.txt* and return the ordered list of names."""
    names = []
    with open(feature_file, "r", encoding="utf-8") as fh:
        for line in fh:
            # Lines are of form "123. FeatureName" – split on the first dot
            if ". " in line:
                try:
                    _, feat = line.strip().split(". ", 1)
                    names.append(feat)
                except ValueError:
                    continue
    return names


def summarise_feedstock_shap(shap_vals: np.ndarray, feat_names: List[str]) -> pd.DataFrame:
    """Return a DataFrame with mean SHAP statistics for each feedstock type."""
    shap_arr = np.asarray(shap_vals, dtype=float)
    if shap_arr.ndim != 2 or shap_arr.shape[1] != len(feat_names):
        raise ValueError("SHAP array shape and feature list length mismatch.")

    # Identify feedstock & mixing-ratio column indices
    ft_indices: dict[int, int] = {}
    mr_indices: dict[int, int] = {}
    for idx, fname in enumerate(feat_names):
        if fname.startswith(FT_PREFIX):
            ft_idx = int(fname.replace(FT_PREFIX, ""))  # 1…118
            ft_indices[ft_idx] = idx
        elif fname.startswith(MR_PREFIX):
            mr_idx = int(fname.replace(MR_PREFIX, ""))
            mr_indices[mr_idx] = idx

    # ------------------------------------------------------------------
    #  Load original feature matrix (to derive mixing-ratio statistics)
    # ------------------------------------------------------------------
    global _ORIG_FEATURE_MATRIX, _ORIG_FEATURE_NAMES
    if _ORIG_FEATURE_MATRIX is None:
        raise RuntimeError("Original feature matrix has not been initialised.")

    X_arr = _ORIG_FEATURE_MATRIX

    records = []
    for i in range(1, 119):  # 1 … 118 inclusive
        ft_col = ft_indices.get(i)
        mr_col = mr_indices.get(i)
        if ft_col is None or mr_col is None:
            # Skip if any part is missing – should not happen with full model
            continue

        ft_shap = shap_arr[:, ft_col]
        mr_shap = shap_arr[:, mr_col]

        # Derive statistics from original input data
        ratio_values = X_arr[:, mr_col]
        # Presence mask: mixing ratio strictly > 0
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
        # Combined score emphasises positive contributions from both terms
        rec["Combined_score"] = rec["FeedstockType_posMean"] + rec["MixingRatio_posMean"]
        records.append(rec)

    df = pd.DataFrame(records)
    df = df.sort_values("Combined_score", ascending=False).reset_index(drop=True)
    return df

# -----------------------------------------------------------------------------
#  Combination search helper
# -----------------------------------------------------------------------------

def search_best_combinations(
    product_name: str,
    candidate_ids: list[int],
    feat_names: List[str],
    max_combo_size: int,
    ratio_vals: np.ndarray,
    ratio_limit: float,
):
    """Exhaustively search feedstock combinations (size 2‒*max_combo_size*) that
    improve the target product yield compared with the sludge-only baseline.

    All combinations whose total mixing-ratio does not exceed *ratio_limit*
    are evaluated. Results are returned for both positive and negative Δ-yield (%),
    sorted descending by that gain. A column Delta_yield_positive is added for clarity.
    """
    global _BASELINE_INPUT_VEC, _NN_WRAPPER, _TRAIN_MIN, _TRAIN_RANGE
    global _TARGET_MIN, _TARGET_RANGE, _TARGET_IN_0_1, _BASELINE_YIELDS

    records: list[dict] = []
    total_combinations = 0
    positive_combinations = 0

    ft_cols = {fid: feat_names.index(f"{FT_PREFIX}{fid}") for fid in candidate_ids}
    mr_cols = {fid: feat_names.index(f"{MR_PREFIX}{fid}") for fid in candidate_ids}

    for r in range(2, max_combo_size + 1):
        for combo in itertools.combinations(candidate_ids, r):
            for ratios in itertools.product(ratio_vals, repeat=r):
                if any(np.isclose(ratio, 0.0, atol=1e-8) for ratio in ratios):
                    continue
                total_ratio = sum(ratios)
                if total_ratio <= 0 or total_ratio > ratio_limit:
                    continue
                inp = _BASELINE_INPUT_VEC.copy()
                for fid, ratio in zip(combo, ratios):
                    inp[ft_cols[fid]] = 1.0
                    inp[mr_cols[fid]] = ratio
                inp_scaled = np.clip((inp - _TRAIN_MIN) / _TRAIN_RANGE, 0.0, 1.0)
                _NN_WRAPPER.target_idx = OUTPUT_IDX[product_name]
                pred_scaled = _NN_WRAPPER.predict(inp_scaled.reshape(1, -1))[0]
                tgt_min = _TARGET_MIN[OUTPUT_IDX[product_name]]
                tgt_rng = _TARGET_RANGE[OUTPUT_IDX[product_name]]
                pred_real = (
                    pred_scaled * tgt_rng + tgt_min
                    if _TARGET_IN_0_1
                    else ((pred_scaled + 1.0) / 2.0) * tgt_rng + tgt_min
                )
                delta = pred_real - _BASELINE_YIELDS[product_name]
                total_combinations += 1
                if delta > 0:
                    positive_combinations += 1
                records.append(
                    {
                        "Feedstock_IDs": "+".join(map(str, combo)),
                        "Size": r,
                        "Ratios": "+".join(f"{rv:.2f}" for rv in ratios),
                        "Total_ratio": total_ratio,
                        "Pred_yield_%": pred_real,
                        "Delta_yield_%": delta,
                        "Delta_yield_positive": delta > 0,
                    }
                )
    logger.info(f"Total combinations evaluated: {total_combinations}, Positive combinations: {positive_combinations}")
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df.sort_values("Delta_yield_%", ascending=False, inplace=True, ignore_index=True)
    return df


def process_product(product_name: str, shap_root: str, out_root: str, top_n: int):
    """Analyse one product target and write ranking CSV to *out_root*."""
    subdir = TARGET_DIR_MAP[product_name]
    target_path = os.path.join(shap_root, subdir)
    shap_file = os.path.join(target_path, "shap_values.npy")
    feat_file = os.path.join(shap_root, "00_feature_names_used.txt")

    if not os.path.exists(shap_file):
        logger.error("SHAP values not found: %s", shap_file)
        return
    shap_values = np.load(shap_file)
    feat_names = load_feature_names(feat_file)

    df_summary = summarise_feedstock_shap(shap_values, feat_names)
    # Make an explicit copy to avoid SettingWithCopyWarning when adding columns later
    df_top = df_summary.head(top_n).copy()

    # --------------------------------------------------------------
    #  Optional forward simulation to get predicted Δ-yield vs baseline
    # --------------------------------------------------------------
    global _BASELINE_INPUT_VEC, _BASELINE_YIELDS, _TRAIN_MIN, _TRAIN_RANGE, _NN_WRAPPER, _RATIO_GRID
    if _SIMULATE:
        delta_vals = []
        predicted_vals = []
        best_ratios = []

        for row in df_top.itertuples():
            feed_id = int(row.Feedstock_ID)
            # Column indices for current feedstock
            ft_col = feat_names.index(f"{FT_PREFIX}{feed_id}")
            mr_col = feat_names.index(f"{MR_PREFIX}{feed_id}")

            # Determine candidate ratio values
            if _TEST_RATIO is not None:
                ratio_candidates = [_TEST_RATIO]
            else:
                ratio_candidates = _RATIO_GRID

            best_delta = -np.inf
            best_pred = None
            best_ratio = None

            for ratio_val in ratio_candidates:
                # build new input vector (unscaled)
                inp = _BASELINE_INPUT_VEC.copy()
                inp[ft_col] = 1.0
                inp[mr_col] = ratio_val

                # scale 0-1
                inp_scaled = np.clip((inp - _TRAIN_MIN) / _TRAIN_RANGE, 0.0, 1.0)

                # predict for this product
                _NN_WRAPPER.target_idx = OUTPUT_IDX[product_name]
                pred = _NN_WRAPPER.predict(inp_scaled.reshape(1, -1))[0]

                # inverse scaling of target (same logic as mc_us_sludge_prediction.py)
                tgt_min = _TARGET_MIN[OUTPUT_IDX[product_name]]
                tgt_range = _TARGET_RANGE[OUTPUT_IDX[product_name]]

                if _TARGET_IN_0_1:
                    pred_real = pred * tgt_range + tgt_min
                else:
                    pred_real = ((pred + 1.0) / 2.0) * tgt_range + tgt_min

                delta = pred_real - _BASELINE_YIELDS[product_name]

                # keep best (largest) delta
                if delta > best_delta:
                    best_delta = delta
                    best_pred = pred_real
                    best_ratio = ratio_val

            delta_vals.append(best_delta)
            predicted_vals.append(best_pred)
            best_ratios.append(best_ratio)
    else:
        delta_vals = None
        predicted_vals = None

    # Ensure output directory exists
    product_out = os.path.join(out_root, product_name)
    os.makedirs(product_out, exist_ok=True)

    csv_path = os.path.join(product_out, "top_{}_feedstocks.csv".format(top_n))

    # Add simulation results to dataframe if available
    if _SIMULATE and delta_vals is not None:
        df_top["Best_ratio"] = best_ratios
        df_top["Pred_yield_%"] = predicted_vals
        df_top["Delta_yield_%"] = delta_vals
        # Sort by yield gain (descending) before saving/plotting
        df_top = df_top.sort_values("Delta_yield_%", ascending=False).reset_index(drop=True)

    df_top.to_csv(csv_path, index=False)
    logger.info("Saved top-%d feedstock ranking for %s → %s", top_n, product_name, csv_path)

    # ------------------------------------------------------------------
    #  Search for multi-feedstock combination strategies (requires simulation)
    # ------------------------------------------------------------------
    if _SIMULATE and _COMBO_SIZE >= 2:
        candidate_ids = df_top["Feedstock_ID"].tolist()
        logger.info(f"Bioliquid candidate_ids: {candidate_ids}")
        actual_combo_size = _COMBO_SIZE
        found_combo = False
        while actual_combo_size >= 2 and not found_combo:
            if len(candidate_ids) < actual_combo_size:
                logger.info(f"Not enough candidates (%d) for combo size %d – trying smaller size", len(candidate_ids), actual_combo_size)
                actual_combo_size -= 1
                continue
            combo_df = search_best_combinations(
                product_name,
                candidate_ids,
                feat_names,
                actual_combo_size,
                _RATIO_GRID,
                _COMBO_RATIO_LIMIT,
            )
            logger.info(f"Bioliquid combo_df shape: {combo_df.shape}")
            if combo_df is not None and not combo_df.empty:
                # Remove any combinations where any ratio is zero (as string, e.g. "0.00")
                combo_df = combo_df[~combo_df["Ratios"].str.contains(r"\b0\.00\b")]
                logger.info(f"Bioliquid combo_df after removing zero ratios: {combo_df.shape}")
            
            if combo_df is not None and not combo_df.empty:
                combo_df["Delta_yield_positive"] = combo_df["Delta_yield_%"] > 0
            else:
                combo_df = pd.DataFrame(columns=["Feedstock_IDs","Size","Ratios","Total_ratio","Pred_yield_%","Delta_yield_%","Delta_yield_positive"])
            combo_csv = os.path.join(product_out, "combination_strategies.csv")
            combo_df.to_csv(combo_csv, index=False)
            logger.info(f"Bioliquid combination_strategies.csv written: {combo_csv}, shape: {combo_df.shape}")
            
            if combo_df["Delta_yield_positive"].any():
                best_per_size = combo_df[combo_df["Delta_yield_positive"]].groupby("Size").first().reset_index()
                import matplotlib.pyplot as plt
                fig_c, ax_c = plt.subplots(figsize=(8, 5))
                ax_c.bar(best_per_size["Size"].astype(str), best_per_size["Delta_yield_%"], color="purple")
                ax_c.set_xlabel("Number of feedstocks in combo")
                ax_c.set_ylabel("Best ΔYield (%) vs baseline")
                plt.title(f"Best combo ΔYield by size ({product_name})")
                fig_c.tight_layout()
                combo_png = os.path.join(product_out, "combo_best.png")
                combo_svg = os.path.join(product_out, "combo_best.svg")
                for idx, row in best_per_size.iterrows():
                    x = str(row["Size"])
                    y = row["Delta_yield_%"]
                    combo = row["Feedstock_IDs"]
                    ratios = row["Ratios"]
                    label = f"{combo}\n{ratios}"
                    ax_c.annotate(label, xy=(idx, y), xytext=(0, 5), textcoords='offset points',
                                  ha='center', va='bottom', fontsize=7, rotation=90)
                fig_c.savefig(combo_png, dpi=300)
                fig_c.savefig(combo_svg, format="svg")
                plt.close(fig_c)
                logger.info(f"Saved combo plot for {product_name} → %s", combo_png)
                best_per_size.to_csv(os.path.join(product_out, "combo_best_data.csv"), index=False)
            else:
                logger.info(f"No positive-gain combinations found for {product_name} at combo size {actual_combo_size}")
            found_combo = True
        if not found_combo:
            logger.info(f"No valid combination strategies found for {product_name} (even at smallest combo size)")

    # ------------------------------------------------------------------
    #  Create bar plot visualising Combined_score and Mean_ratio
    # ------------------------------------------------------------------
    try:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ids = df_top["Feedstock_ID"].astype(str)
        score_vals = df_top["Combined_score"]
        ratio_vals = df_top["Mean_ratio"]

        bar = ax1.bar(ids, score_vals, color="skyblue", label="Combined positive SHAP score")
        ax1.set_xlabel("Feedstock ID")
        ax1.set_ylabel("Combined SHAP score", color="skyblue")
        ax1.tick_params(axis="y", labelcolor="skyblue")

        # Secondary axis for mean ratio values
        ax2 = ax1.twinx()
        ax2.plot(ids, ratio_vals, color="orangered", marker="o", label="Mean mixing ratio (0–1)")
        ax2.set_ylabel("Mean mixing ratio (0–1)", color="orangered")
        ax2.tick_params(axis="y", labelcolor="orangered")

        plt.title(f"Top {top_n} Feedstocks – {product_name}")
        fig.tight_layout()

        plot_png = os.path.join(product_out, f"top_{top_n}_feedstocks.png")
        plot_svg = os.path.join(product_out, f"top_{top_n}_feedstocks.svg")
        fig.savefig(plot_png, dpi=300)
        fig.savefig(plot_svg, format="svg")
        plt.close(fig)

        logger.info("Saved visualisation → %s (and .svg)", plot_png)

        # additional figure showing delta yields if simulation enabled
        if _SIMULATE and "Delta_yield_%" in df_top.columns:
            try:
                fig2, ax = plt.subplots(figsize=(10, 6))

                bars = ax.bar(ids, df_top["Delta_yield_%"], color="seagreen")

                # Annotate each bar with the optimal ratio value
                for bar, ratio_val in zip(bars, df_top["Best_ratio"]):
                    height = bar.get_height()
                    # Place text slightly above or below bar depending on sign
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

                ax.set_xlabel("Feedstock ID")
                ax.set_ylabel("Δ Yield (%) vs baseline", color="seagreen")
                plt.title(
                    f"Predicted Yield Gain – {product_name} (Baseline {_BASELINE_YIELDS.get(product_name, 0):.2f} %)"
                )
                fig2.tight_layout()
                delta_png = os.path.join(product_out, f"top_{top_n}_delta_gain.png")
                delta_svg = os.path.join(product_out, f"top_{top_n}_delta_gain.svg")
                fig2.savefig(delta_png, dpi=300)
                fig2.savefig(delta_svg, format="svg")
                plt.close(fig2)
                logger.info("Saved delta-yield plot → %s", delta_png)
            except Exception as e:
                logger.warning("Failed to create delta-yield plot: %s", e)
    except Exception as plot_err:
        logger.warning("Failed to create plot for %s: %s", product_name, plot_err)


# -----------------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate feedstock blending strategies based on SHAP values.")
    parser.add_argument(
        "--shap_dir",
        type=str,
        default="../../SHAP_Analysis_Results_20250630_123705",
        help="Path to the SHAP analysis results directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Output directory where Biochar/Bioliquid/Biogas sub-folders will be created (default: current directory)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top feedstock candidates to keep for each product (default: %(default)s)",
    )
    parser.add_argument(
        "--mat_file",
        type=str,
        default="../../GPM_SHAP_matlab/Results/Training/Results_trained.mat",
        help="Path to the MATLAB training .mat file containing the original feature matrix (default: %(default)s)",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=0,
        help="Skip feedstocks that appear in fewer than this number of samples (default: %(default)s)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run forward simulation with the neural network to estimate absolute yield gain",
    )
    parser.add_argument(
        "--mc_csv",
        type=str,
        default="../mc_us_sludge_predictions.csv",
        help="Monte-Carlo CSV containing baseline sludge predictions (needed for --simulate)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=None,
        help="Override mixing ratio to test (0–1). If set, disables ratio grid search.",
    )
    parser.add_argument(
        "--ratio_search",
        type=str,
        default="0.05:0.95:0.05",
        help=(
            "Ratio grid specification 'start:end:step' within 0–1 used when searching "
            "for the optimal mixing ratio of each feedstock (default: %(default)s). "
            "Ignored if --test_ratio is provided."
        ),
    )
    # Combination-search specific parameters
    parser.add_argument(
        "--combo_size",
        type=int,
        default=2,
        help="Maximum number of feedstocks to include in each combination strategy (default: %(default)s). All sizes from 2 up to this value will be evaluated.",
    )
    # Removed --top_combos and --combo_samples to always enumerate the full combination space
    parser.add_argument(
        "--combo_ratio_limit",
        type=float,
        default=0.5,
        help="Maximum total mixing-ratio sum allowed for a combination (default: %(default)s)",
    )
    parser.add_argument(
        "--sludge_ratio",
        type=float,
        default=0.5,
        help="Fraction of sludge in the final blend (default: %(default)s). The maximum total mixing ratio for all other feedstocks in a combination will be set to 1 - sludge_ratio unless --combo_ratio_limit is explicitly set.",
    )
    args = parser.parse_args()

    shap_dir = os.path.abspath(args.shap_dir)
    out_dir = os.path.abspath(args.output_dir)

    # ------------------------------------------------------------------
    #  Load original feature matrix once and cache for global access
    # ------------------------------------------------------------------
    global _ORIG_FEATURE_MATRIX, _ORIG_FEATURE_NAMES

    mat_path = os.path.abspath(args.mat_file)
    logger.info("Loading MATLAB training data from: %s", mat_path)

    mat_data = load_matlab_data(mat_path)
    X_mat, _y_dummy, _net_struct = extract_neural_network_data(mat_data)
    _ORIG_FEATURE_NAMES = generate_feature_names(X_mat, mat_data)
    _ORIG_FEATURE_MATRIX = X_mat  # numpy array

    logger.info("Original feature matrix loaded: %s", _ORIG_FEATURE_MATRIX.shape)

    # prepare additional globals for simulation if required
    global _SIMULATE, _TEST_RATIO, _BASELINE_INPUT_VEC, _BASELINE_YIELDS, _NN_WRAPPER
    global _TRAIN_MIN, _TRAIN_RANGE, _TARGET_MIN, _TARGET_RANGE, _TARGET_IN_0_1

    _SIMULATE = args.simulate
    _TEST_RATIO = args.test_ratio

    # training min/max for scaling
    _TRAIN_MIN = _ORIG_FEATURE_MATRIX.min(axis=0)
    _TRAIN_MAX = _ORIG_FEATURE_MATRIX.max(axis=0)
    _TRAIN_RANGE = _TRAIN_MAX - _TRAIN_MIN
    _TRAIN_RANGE[_TRAIN_RANGE == 0] = 1.0

    # build neural-network wrapper once
    _NN_WRAPPER = MatlabNeuralNetworkWrapper(_net_struct, target_idx=0)

    # target min/max for inverse scaling
    y_train_dummy = _y_dummy.astype(float)
    _TARGET_MIN = y_train_dummy.min(axis=0)
    _TARGET_MAX = y_train_dummy.max(axis=0)
    _TARGET_RANGE = _TARGET_MAX - _TARGET_MIN
    _TARGET_RANGE[_TARGET_RANGE == 0] = 1.0
    _TARGET_IN_0_1 = np.all(_TARGET_MIN >= -1e-6) and np.all(_TARGET_MAX <= 1.0 + 1e-6)

    # build global ratio grid for search
    global _RATIO_GRID
    if _TEST_RATIO is None:
        try:
            r_start, r_end, r_step = map(float, args.ratio_search.split(":"))
            if not (0.0 < r_start < r_end <= 1.0 and r_step > 0):
                raise ValueError
        except ValueError:
            raise ValueError("--ratio_search must be in 'start:end:step' format with 0 < start < end <= 1 and start > 0 to avoid zero ratios.")
        _RATIO_GRID = np.round(np.arange(r_start, r_end + 1e-8, r_step), 4)
        # Strictly filter out zero values from ratio grid
        _RATIO_GRID = _RATIO_GRID[_RATIO_GRID > 1e-8]
    else:
        _RATIO_GRID = np.array([_TEST_RATIO])

    if _SIMULATE:
        # load baseline MC csv
        mc_csv_path = os.path.abspath(args.mc_csv)
        logger.info("Loading Monte-Carlo baseline from: %s", mc_csv_path)
        mc_df = pd.read_csv(mc_csv_path)

        # build baseline vector aligned to feature list, fill missing with 0
        baseline_vals = []
        for feat in _ORIG_FEATURE_NAMES:
            if feat in mc_df.columns:
                baseline_vals.append(mc_df[feat].median())
            else:
                # default: 0 for mixing ratios / indicator, or overall median 0
                baseline_vals.append(0.0)
                if feat == "ReactorType":
                    # choose most frequent reactor type if available
                    if "ReactorType" in mc_df.columns and not mc_df["ReactorType"].empty:
                        baseline_vals[-1] = mc_df["ReactorType"].mode().iloc[0]
        _BASELINE_INPUT_VEC = np.asarray(baseline_vals, dtype=float)

        # assume product column names match product list
        _BASELINE_YIELDS = {
            "Biochar": mc_df["Biochar"].median(),
            "Bioliquid": mc_df["Bioliquid"].median() if "Bioliquid" in mc_df.columns else mc_df["Bioliquid"].median(),
            "Biogas": mc_df["Biogas"].median() if "Biogas" in mc_df.columns else mc_df["Biogas"].median(),
        }

        logger.info("Baseline median yields: %s", _BASELINE_YIELDS)

    # combination search globals
    global _COMBO_SIZE, _COMBO_RATIO_LIMIT
    _COMBO_SIZE = max(2, args.combo_size)
    # If combo_ratio_limit is not set by user (left as default 0.5), use 1 - sludge_ratio
    if "--combo_ratio_limit" in os.sys.argv:
        _COMBO_RATIO_LIMIT = min(1.0, max(0.0, args.combo_ratio_limit))
    else:
        _COMBO_RATIO_LIMIT = min(1.0, max(0.0, 1.0 - args.sludge_ratio))

    logger.info("Using SHAP directory: %s", shap_dir)
    logger.info("Writing results to: %s", out_dir)

    for product in TARGET_DIR_MAP.keys():
        process_product(product, shap_dir, out_dir, args.top)

    logger.info("All product analyses completed.")


if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
#  Module-level cache for original training features (initialised in main()).
# -----------------------------------------------------------------------------

_ORIG_FEATURE_MATRIX: np.ndarray | None = None
_ORIG_FEATURE_NAMES: List[str] | None = None

# globals for simulation
_SIMULATE: bool = False
_TEST_RATIO: float | None = None
_BASELINE_INPUT_VEC: np.ndarray | None = None
_BASELINE_YIELDS: dict | None = None
_TRAIN_MIN: np.ndarray | None = None
_TRAIN_RANGE: np.ndarray | None = None
_TARGET_MIN: np.ndarray | None = None
_TARGET_RANGE: np.ndarray | None = None
_TARGET_IN_0_1: bool = True
_NN_WRAPPER: MatlabNeuralNetworkWrapper | None = None

# ratio grid used for optimisation
_RATIO_GRID: np.ndarray | None = None 