import os
import argparse
import logging
from typing import List

import numpy as np
import pandas as pd
import matplotlib
import itertools

matplotlib.use("Agg")  # Safe for headless execution
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
#  Feedstock Blending Strategy Generator for LOWERING Activation Energy (Ea)
# -----------------------------------------------------------------------------
# 1. Analyse SHAP values of an Ea neural-network model to find feedstocks whose
#    presence / mixing ratio most strongly DECREASE Ea (i.e. negative SHAP).
# 2. Rank the 118 candidate feedstock types by their combined negative SHAP
#    contribution (FeedstockType + MixingRatio).
# 3. Optionally run a forward simulation with the trained MATLAB neural network
#    to estimate the absolute Ea that can be achieved at various mixing ratios
#    for the top-N candidates.
# 4. Write the ranked list (and plots) to the user-specified output directory.
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
#  Helper utilities (imported from shap_analysis_ea.py residing at project root)
# -----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.append(PROJECT_ROOT)

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

        for row in df_top.itertuples():
            feed_id = int(row.Feedstock_ID)
            ft_col = feat_names_net.index(f"{FT_PREFIX}{feed_id}")
            mr_col = feat_names_net.index(f"{MR_PREFIX}{feed_id}")

            best_ea_val = np.inf
            best_ratio = None

            for ratio in ratio_grid:
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
                        ax_conv.plot(conversion_values, deltas_series, label=f"ID {feed_id}")
                        conv_data[f"ID_{feed_id}"] = deltas_series
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

            combo_records = []
            max_feed_ratio = combo_max_ratio
            leftover_ratio = max(0.0, 1.0 - sludge_ratio)
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
                    best_ratio_assignment: list[float] | None = None

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

                    combo_records.append({
                        "Combo": "-".join(map(str, combo)),
                        "Size": r,
                        "Pred_Ea_kJmol": best_ea_val,
                        "Delta_Ea_kJmol": best_ea_val - baseline_ea,
                        "MixingRatios": "-".join(
                            f"{fid}:{ratio_val:.3f}" for fid, ratio_val in zip(combo, best_ratio_assignment)
                        ),
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


# -----------------------------------------------------------------------------
#  Entry-point
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate feedstock blending strategies that reduce activation energy (Ea).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shap_dir",
        type=str,
        default="../../SHAP_Analysis_Ea_Results_20250701_000529",
        help="Directory containing SHAP results for Ea (shap_values.npy & 00_feature_names_used.txt)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
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
        default="../../Results_trained.mat",
        help="Path to MATLAB .mat file with trained neural-network (required for --simulate)",
    )
    parser.add_argument(
        "--mc_csv",
        type=str,
        default="../mc_ea_predictions.csv",
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
        default=0.5,
        help="Maximum ratio per feedstock when searching combinations.",
    )
    parser.add_argument(
        "--sludge_ratio",
        type=float,
        default=0.5,
        help="Fraction of sludge in the final blend. Feedstock ratios in each combo will sum to (1 - sludge_ratio).",
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
    )

    logger.info("Analysis completed.")


if __name__ == "__main__":
    main() 