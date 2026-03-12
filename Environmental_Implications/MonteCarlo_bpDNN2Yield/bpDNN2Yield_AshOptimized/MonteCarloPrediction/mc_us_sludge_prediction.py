import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
#  Monte Carlo Prediction of Pyrolysis Products for U.S. Sewage Sludge
# -----------------------------------------------------------------------------
# 1. Sample input features from given ranges (Municipal_Sludge_Data_cleaned_mean)
# 2. Use fallback constants from US_SewageSludge.xlsx for unspecified ranges
# 3. Load a pre-trained neural network (MATLAB .mat) via helper utilities
# 4. Predict Char / Gas / Liquid yields for thousands of Monte Carlo samples
# 5. Visualise uncertainty with violin plots and export CSV of raw predictions
# -----------------------------------------------------------------------------

# Add project root to PYTHONPATH so we can import shap_analysis utilities
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from shap_analysis import (
    load_matlab_data,
    extract_neural_network_data,
    generate_feature_names,
    MatlabNeuralNetworkWrapper,
    convert_yield_to_percentage,
    BASE_FEATURE_NAMES,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def read_parameter_ranges(range_path: str) -> pd.DataFrame:
    """Read an Excel file that contains min/max (or mean) for each feature.

    The function tries to be robust to different layouts:
    1. If the sheet has explicit columns named "min" / "max" it will use them.
    2. If the sheet contains rows labelled "min" / "max" it will use them.
    3. Otherwise it will compute column-wise min / max from the numeric data.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by feature name with columns ["min", "max"].
    """
    df_raw = pd.read_excel(range_path, header=0)
    df = df_raw.copy()

    # ------------------------------------------------------------------
    # Helper: locate columns whose names include certain keywords
    # ------------------------------------------------------------------
    def _find_col(keywords):
        for col in df.columns:
            cl = str(col).lower()
            if any(k in cl for k in keywords):
                return col
        return None

    min_col = _find_col(["min", "minimum", "lower", "low"])
    max_col = _find_col(["max", "maximum", "upper", "high"])
    feat_col = df.columns[0]  # assume first column lists feature names

    if min_col and max_col:
        ranges = df[[feat_col, min_col, max_col]].copy()
        ranges.columns = ["feature", "min", "max"]
        # Ensure numeric
        ranges["min"] = pd.to_numeric(ranges["min"], errors="coerce")
        ranges["max"] = pd.to_numeric(ranges["max"], errors="coerce")
        ranges.set_index("feature", inplace=True)
    else:
        # Case 2: rows named min/max (same as before)
        first_col_lower = df.iloc[:, 0].astype(str).str.lower()
        if (first_col_lower == "min").any() and (first_col_lower == "max").any():
            min_row = df[first_col_lower == "min"].iloc[0]
            max_row = df[first_col_lower == "max"].iloc[0]
            features = df.columns[1:]
            ranges = pd.DataFrame({
                "min": pd.to_numeric(min_row[1:], errors="coerce"),
                "max": pd.to_numeric(max_row[1:], errors="coerce"),
            }, index=features)
        else:
            # Case 3: derive from numeric data across rows (fallback)
            numeric_df = df.select_dtypes(include=[np.number])
            ranges = pd.DataFrame({
                "min": numeric_df.min(),
                "max": numeric_df.max(),
            })

    # ------------------------------------------------------------------
    # Canonicalise feature names (remove spaces, add suffixes, etc.)
    # ------------------------------------------------------------------
    ranges.index = [canonicalize_feature_name(idx) for idx in ranges.index]
    # Drop potential duplicates by keeping first occurrence
    ranges = ranges[~ranges.index.duplicated(keep="first")]
    return ranges


# Mapping of alternative column names to canonical model feature names
CANONICAL_NAME_MAP = {
    "volatilematter": "VolatileMatters/%",
    "vm": "VolatileMatters/%",
    "fixedcarbon": "FixedCarbon/%",
    "fc": "FixedCarbon/%",
    "ash": "Ash/%",
    "c": "C/%",
    "h": "H/%",
    "o": "O/%",
    "n": "N/%",
    "s": "S/%",
    "sio2": "Ash_SiO2",
    "na2o": "Ash_Na2O",
    "mgo": "Ash_MgO",
    "al2o3": "Ash_Al2O3",
    "k2o": "Ash_K2O",
    "cao": "Ash_CaO",
    "p2o5": "Ash_P2O5",
    "cuo": "Ash_CuO",
    "zno": "Ash_ZnO",
    "fe2o3": "Ash_Fe2O3",
    "targettemperature": "TargetTemperature/Celsius",
}


def canonicalize_feature_name(name: str) -> str:
    """Return canonical model feature name given a raw column header."""
    if not isinstance(name, str):
        return name
    key = name.strip().lower().replace("%", "").replace("/", " ")
    key = key.replace("ash_", "")  # will add Ash_ back for oxides via mapping
    key = key.replace("ash ", "")
    key = key.replace(" ", "")
    return CANONICAL_NAME_MAP.get(key, name)


def load_constant_features(constant_path: str) -> pd.Series:
    """Read Excel file containing constant feature values (single row)."""
    df = pd.read_excel(constant_path, header=0)
    if len(df) == 1:
        const_series = df.iloc[0]
    else:
        const_series = df.mean()

    # Canonicalise index names
    const_series.index = [canonicalize_feature_name(idx) for idx in const_series.index]
    # Drop duplicates if any
    const_series = const_series[~const_series.index.duplicated(keep="first")]
    return const_series


def build_monte_carlo_samples(ranges: pd.DataFrame, constants: pd.Series, n: int) -> pd.DataFrame:
    """Generate *n* Monte Carlo samples that satisfy domain mass-balance constraints.

    The generated samples obey:
    1. VolatileMatters/% + FixedCarbon/% + Ash/% == 100
    2. Ash/% + C/% + H/% + O/% + N/% + S/% == 100
    3. Sum of oxide components (SiO2 … Fe2O3) <= Ash/%

    Ranges for each feature are taken from *ranges*.  If a feature is missing
    from *ranges*, a fallback constant value is taken from *constants*.
    """

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    VM_KEY = "VolatileMatters/%"
    FC_KEY = "FixedCarbon/%"
    ASH_KEY = "Ash/%"
    C_KEY = "C/%"
    H_KEY = "H/%"
    O_KEY = "O/%"
    N_KEY = "N/%"
    S_KEY = "S/%"

    OXIDE_KEYS = [
        "Ash_SiO2",
        "Ash_Na2O",
        "Ash_MgO",
        "Ash_Al2O3",
        "Ash_K2O",
        "Ash_CaO",
        "Ash_P2O5",
        "Ash_CuO",
        "Ash_ZnO",
        "Ash_Fe2O3",
    ]

    def get_range(feature: str):
        """Return (lo, hi) tuple for *feature* from *ranges* or constants."""
        if feature in ranges.index:
            lo, hi = ranges.loc[feature, ["min", "max"]].values.tolist()
        else:
            # Fall-back: treat constant as both min and max
            val = constants.get(feature, np.nan)
            lo, hi = val, val
        return lo, hi

    def random_in_range(lo: float, hi: float):
        r"""Uniform random float in [lo, hi]. Keeps constants when bounds equal."""
        if np.isnan(lo) or np.isnan(hi) or np.isclose(lo, hi):
            return lo if not np.isnan(lo) else hi
        return np.random.uniform(lo, hi)

    def random_int_in_range(lo: float, hi: float):
        r"""Random integer in the closed interval [lo, hi] handling NaNs."""
        if np.isnan(lo) or np.isnan(hi):
            return np.nan
        lo_i, hi_i = int(round(lo)), int(round(hi))
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        if lo_i == hi_i:
            return lo_i
        # NumPy randint upper bound is exclusive → add 1 to include hi_i
        return np.random.randint(lo_i, hi_i + 1)

    def generate_single_sample(max_attempts: int = 5000):
        """Generate one sample satisfying all constraints (with retries)."""
        for _ in range(max_attempts):
            row = {}
            # ---------------- Proximate analysis ----------------
            # Sample Volatile Matter and Ash directly; compute Fixed Carbon
            vm = random_in_range(*get_range(VM_KEY))
            ash = random_in_range(*get_range(ASH_KEY))
            fc = 100.0 - vm - ash

            # Validate Fixed Carbon lies within its permissible range
            fc_lo, fc_hi = get_range(FC_KEY)
            if fc < fc_lo or fc > fc_hi:
                continue  # resample

            # Basic non-negativity guard
            if fc < 0:
                continue

            row[VM_KEY] = vm
            row[FC_KEY] = fc
            row[ASH_KEY] = ash

            # ---------------- Ultimate analysis -----------------
            c = random_in_range(*get_range(C_KEY))
            h = random_in_range(*get_range(H_KEY))
            n = random_in_range(*get_range(N_KEY))
            s = random_in_range(*get_range(S_KEY))
            o = 100.0 - (ash + c + h + n + s)
            o_lo, o_hi = get_range(O_KEY)
            if o < o_lo or o > o_hi:
                continue  # invalid, resample
            row.update({C_KEY: c, H_KEY: h, N_KEY: n, S_KEY: s, O_KEY: o})

            # ---------------- Oxide composition -----------------
            oxide_sum = 0.0
            valid_oxides = True
            for oxide in OXIDE_KEYS:
                ox_lo, ox_hi = get_range(oxide)
                val = random_in_range(ox_lo, ox_hi)
                if not np.isnan(val):
                    oxide_sum += val
                    row[oxide] = val  # only keep if a real number
            # The oxide values are given in mg/g whereas Ash/% is wt%.
            # Direct comparison is therefore meaningless; we skip the strict
            # inequality check to avoid over-rejecting otherwise valid samples.
            # If unit conversion is later clarified, this can be reinstated.

            # ---------------- Remaining features ---------------
            for feature in ranges.index:
                if feature in row:
                    continue  # already assigned
                lo, hi = get_range(feature)
                canon_feat = _canonical_no_space(feature)
                if canon_feat in INT_SAMPLED_FEATURES:
                    val = random_int_in_range(lo, hi)
                else:
                    val = random_in_range(lo, hi)
                if not np.isnan(val):
                    row[feature] = val  # skip NaNs – will fill from constants

            # Fill any missing or NaN features from constants
            for feature, value in constants.items():
                if feature not in row or np.isnan(row.get(feature, np.nan)):
                    row[feature] = value

            return row  # success
        raise RuntimeError("Unable to generate a valid sample within max_attempts")

    # ------------------------------------------------------------------
    # Generate all samples
    # ------------------------------------------------------------------
    sample_rows = [generate_single_sample() for _ in range(n)]
    return pd.DataFrame(sample_rows)


def load_trained_model(mat_path: str):
    # Try direct load; if it fails (e.g., due to Windows MAX_PATH issues),
    # copy to a temporary short path and retry.
    mat_data = load_matlab_data(mat_path)
    if not mat_data:
        import tempfile
        import shutil

        print("Direct load failed—retrying with shortened temporary path …")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_mat_path = os.path.join(tmpdir, "model.mat")
            shutil.copy2(mat_path, tmp_mat_path)
            mat_data = load_matlab_data(tmp_mat_path)

    if not mat_data:
        raise RuntimeError("Failed to load MATLAB model data. Ensure the .mat file is valid and not v7.3 HDF5 format.")

    # Extract both inputs and targets to recover original target scaling later
    X_dummy, y_dummy, net_struct = extract_neural_network_data(mat_data)

    wrapper = MatlabNeuralNetworkWrapper(net_struct)

    feature_names = generate_feature_names(X_dummy, mat_data)

    # Return both X_dummy (for input scaling) *and* y_dummy (for output inverse-scaling)
    return wrapper, feature_names, X_dummy, y_dummy


def align_sample_features(df_samples: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Align DataFrame columns to model expected order and return ndarray."""
    aligned = pd.DataFrame(index=df_samples.index)
    for feat in feature_names:
        if feat in df_samples.columns:
            aligned[feat] = df_samples[feat]
        else:
            aligned[feat] = 0.0  # default zero if feature missing
    return aligned.values


def plot_uncertainty(predictions_pct: pd.DataFrame, output_path: str):
    """Create violin plot and save both PNG and SVG for Illustrator editing."""
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    melted = predictions_pct.melt(var_name="Product", value_name="Yield (%)")
    ax = sns.violinplot(
        data=melted,
        x="Product",
        y="Yield (%)",
        hue="Product",
        inner="quartile",
        palette="Set2",
    )
    # Remove redundant legend produced by hue duplication
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.title("Monte Carlo Prediction Uncertainty – U.S. Sewage Sludge")
    plt.tight_layout()
    # Save PNG
    plt.savefig(output_path, dpi=300)
    # Save SVG alongside for Adobe Illustrator
    svg_path = os.path.splitext(output_path)[0] + ".svg"
    plt.savefig(svg_path, format="svg")
    plt.close()


def expand_single_point_ranges(ranges: pd.DataFrame, rel_variation: float = 0.1) -> pd.DataFrame:
    """Expand rows where min == max (or only one value) by ±rel_variation.

    This is useful when the *Municipal_Sludge_Data_cleaned_mean.xlsx* file only
    provides mean values (no explicit min/max).  In that case Monte Carlo
    sampling would otherwise produce constant values and thus no sensitivity.

    Parameters
    ----------
    ranges : pd.DataFrame
        DataFrame with columns ["min", "max"]. It will be *modified in place*.
    rel_variation : float, optional (default=0.10)
        Relative half-range used to construct the interval, e.g. 0.1 → ±10 %.

    Returns
    -------
    pd.DataFrame
        The same DataFrame instance (for convenience).
    """

    for feat in ranges.index:
        lo = ranges.at[feat, "min"]
        hi = ranges.at[feat, "max"]

        # Harmonise NaNs – treat single available value as both bounds
        if pd.isna(lo) and not pd.isna(hi):
            lo = hi
        if pd.isna(hi) and not pd.isna(lo):
            hi = lo

        if pd.isna(lo) or pd.isna(hi):
            # Still unresolved → skip expansion (will later fall back to constant)
            continue

        if np.isclose(lo, hi):
            mid = lo
            if np.isfinite(mid) and mid != 0:
                delta = abs(mid) * rel_variation
                ranges.at[feat, "min"] = mid - delta
                ranges.at[feat, "max"] = mid + delta
            else:
                # For true zeros keep as a single-point range
                pass

    return ranges


# -----------------------------------------------------------------------------
# Global helper – canonicalise by removing spaces, slashes, percent signs
# ------------------------------------------------------------------


def _canonical_no_space(name: str) -> str:
    """Lower-case *name* and strip whitespace, slash and percent symbols."""
    return (
        str(name).lower().replace(" ", "").replace("/", "").replace("%", "")
    )


# Features that should be *sampled as integers* rather than floats.  The names
# are stored in canonicalised (space-free, lower-case) form so we can match
# robustly against spreadsheet headers.

INT_SAMPLED_FEATURES = {"reactortype"}


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo prediction for U.S. sewage sludge pyrolysis products.")
    parser.add_argument("--samples", type=int, default=10000, help="Number of Monte Carlo samples (default: 10000)")
    parser.add_argument("--variation", type=float, default=0.10, help="Relative variation (e.g., 0.1 for ±10%) when only mean values are available (default: 0.10)")
    parser.add_argument("--seed", type=int, default=500, help="Random seed for reproducibility (default: 500)")
    args = parser.parse_args()

    n_samples = args.samples

    # ------------------------------------------------------------------
    # Reproducibility – set global NumPy seed early
    # ------------------------------------------------------------------
    np.random.seed(args.seed)

    # ------------------------------------------------------------------
    # Locate required data files (try project folder first, then workspace)
    # ------------------------------------------------------------------

    # .mat model is relative to project structure and should exist
    mat_model_path = os.path.join(
        PROJECT_ROOT,
        "GPM_SHAP_matlab",
        "Results",
        "Training",
        "Results_trained.mat",
    )

    # Excel files may live one or two levels above; search flexibly
    candidate_dirs = [
        PROJECT_ROOT,
        os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir)),  # one level up
        os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir, os.pardir)),  # two levels up
    ]

    def find_file(filename: str) -> str:
        for d in candidate_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"{filename} not found in {candidate_dirs}")

    range_path = find_file("Municipal_Sludge_Data_cleaned_mean.xlsx")
    constant_path = find_file("US_SewageSludge.xlsx")

    print("Reading parameter ranges …")
    ranges = read_parameter_ranges(range_path)

    # Expand single-point ranges based on --variation argument
    expand_single_point_ranges(ranges, rel_variation=args.variation)

    # ------------------------------------------------------------------
    # EXCLUDE categorical / ratio features that must remain constant
    # ------------------------------------------------------------------
    NON_SAMPLED_PREFIXES = ("feedstocktype", "mixingratio")  # case-insensitive
    # Additional single-feature names that must stay constant (case-insensitive, space-agnostic)
    # (Heating rate is now sampled; Reaction time is derived later and therefore NOT listed here.)
    NON_SAMPLED_NAMES = {"location"}

    to_drop = [
        idx
        for idx in ranges.index
        if str(idx).lower().startswith(NON_SAMPLED_PREFIXES)
        or _canonical_no_space(idx) in NON_SAMPLED_NAMES
        # INT_SAMPLED_FEATURES are variable – do NOT drop them
    ]
    if to_drop:
        ranges.drop(index=to_drop, inplace=True)
        print(f"Excluded {len(to_drop)} non-sampled features (kept constant from spreadsheet): {to_drop}")

    print(f"Loaded/expanded ranges for {len(ranges)} features")

    print("Reading constant feature values …")
    constants = load_constant_features(constant_path)

    print(f"Generating {n_samples} Monte Carlo samples …")
    df_samples = build_monte_carlo_samples(ranges, constants, n_samples)

    # ------------------------------------------------------------------
    # Derive Reaction time = Target temperature / Heating rate + deltaTime (unit consistency
    # assumed: °C and °C/min → min). Apply before any further NaN handling.
    # ------------------------------------------------------------------
    tt_col = "TargetTemperature/Celsius"
    hr_col = "Heating rate"
    rt_col = "Reaction time"
    delta_col = "deltaTime"

    if tt_col in df_samples.columns and hr_col in df_samples.columns:
        # Ensure Heating rate is strictly positive.
        if (df_samples[hr_col] == 0).any():
            raise ValueError("Heating rate contains zero values; cannot compute Reaction time.")

        # Use deltaTime if present, else assume zero offset.
        if delta_col not in df_samples.columns:
            # Create deltaTime with zeros if not sampled from the spreadsheet
            df_samples[delta_col] = 0.0
        delta_series = df_samples[delta_col]
        df_samples[rt_col] = df_samples[tt_col] / df_samples[hr_col] + delta_series
    else:
        print("Warning: could not derive Reaction time – columns missing.")

    # NEW: Replace any remaining NaN with zeros to prevent NaN propagation in predictions
    nan_count = df_samples.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values found in generated samples; replacing with 0.")
        df_samples = df_samples.fillna(0.0)

    # ------------------------------------------------------------------
    # Overwrite all non-sampled features with their spreadsheet constants
    # to guarantee they remain unchanged across Monte Carlo draws.
    # ------------------------------------------------------------------
    const_update = {
        feat: val
        for feat, val in constants.items()
        if str(feat).lower().startswith(NON_SAMPLED_PREFIXES)
        or _canonical_no_space(feat) in NON_SAMPLED_NAMES
    }
    if const_update:
        df_samples = df_samples.assign(**const_update)

    # NEW ──────────────────────────────────────────────────────────────
    # If any column shows zero variance (all identical values), add a
    # small relative jitter (±5 %) to avoid degenerate predictions that
    # collapse the violin plot into a line. This keeps physical meaning
    # while providing enough variability for visualisation.
    # ------------------------------------------------------------------
    const_cols = [
        c for c in df_samples.columns
        if df_samples[c].std() == 0
        and not str(c).lower().startswith(("feedstocktype", "mixingratio"))
        and _canonical_no_space(c) not in NON_SAMPLED_NAMES
        and _canonical_no_space(c) not in INT_SAMPLED_FEATURES
    ]

    if const_cols:
        rng = np.random.default_rng(args.seed)
        jitter_update = {}
        for col in const_cols:
            base_val = df_samples[col].iloc[0]
            if np.isfinite(base_val) and base_val != 0:
                noise = rng.uniform(-0.05, 0.05, size=len(df_samples)) * base_val
                jitter_update[col] = np.clip(base_val + noise, 0, None)
            else:
                # For true zeros (e.g., missing oxide components) add tiny noise
                jitter_update[col] = rng.uniform(0, 0.05, size=len(df_samples))

        df_samples = df_samples.assign(**jitter_update)
        print(f"Added jitter to {len(const_cols)} constant feature(s) to enable meaningful violin plots.")

    # ------------------------------------------------------------------
    # rename columns so they match the feature names used during training
    # ------------------------------------------------------------------
    RENAME_FOR_MODEL = {
        "Heating rate": "HeatingRate/(K/min)",
        "Reaction time": "ReactionTime/min",
    }
    df_samples = df_samples.rename(columns=RENAME_FOR_MODEL)

    # Optional: defragment for faster column operations
    df_samples = df_samples.copy()

    print("Loading trained neural network model …")
    wrapper, model_feature_names, X_train, y_train = load_trained_model(mat_model_path)

    # Align sample feature order and SCALE to training range (0-1)
    X_raw = align_sample_features(df_samples, model_feature_names)

    # Compute min-max from training data to replicate MATLAB preprocessing
    train_min = X_train.min(axis=0)
    train_max = X_train.max(axis=0)
    denom = train_max - train_min
    denom[denom == 0] = 1.0           # avoid divide-by-zero for constant cols

    # ALWAYS scale the Monte-Carlo samples exactly like the training data
    X_input = np.clip((X_raw - train_min) / denom, 0.0, 1.0)

    print("Predicting yields …")
    # The MatlabNeuralNetworkWrapper returns a single target at a time. Predict
    # all three products (Biochar, Biogas, Bioliquid) by switching the target index.
    product_names = ["Biochar", "Biogas", "Bioliquid"]
    y_pred_all = []
    for idx in range(3):
        wrapper.target_idx = idx  # choose output neuron
        y_pred_all.append(wrapper.predict(X_input))

    # Stack to shape (n_samples, 3)
    y_pred_matrix = np.column_stack(y_pred_all)

    # ------------------------------------------------------------------
    # Inverse-transform model outputs back to original units (wt % or similar)
    # ------------------------------------------------------------------
    y_train_arr = np.asarray(y_train, dtype=float)
    tgt_min = y_train_arr.min(axis=0)
    tgt_max = y_train_arr.max(axis=0)
    tgt_range = tgt_max - tgt_min
    tgt_range[tgt_range == 0] = 1.0  # avoid division by zero

    # Heuristic: 0-1 scaling vs. ‑1…1 scaling
    if np.all(tgt_min >= -1e-6) and np.all(tgt_max <= 1.0 + 1e-6):
        y_real = y_pred_matrix * tgt_range + tgt_min
    else:
        y_real = ((y_pred_matrix + 1.0) / 2.0) * tgt_range + tgt_min

    predictions_pct = pd.DataFrame(y_real, columns=product_names, dtype=float)

    # ------------------------------------------------------------------
    # Remove samples with non-physical negative yields
    # ------------------------------------------------------------------
    neg_mask = (predictions_pct < 0).any(axis=1)
    if neg_mask.any():
        removed = neg_mask.sum()
        print(f"Removing {removed} sample(s) with negative yields to retain physical plausibility …")
        predictions_pct = predictions_pct.loc[~neg_mask].reset_index(drop=True)
        df_samples = df_samples.loc[~neg_mask].reset_index(drop=True)

    # Save predictions together with corresponding input features
    output_dir = SCRIPT_DIR
    os.makedirs(output_dir, exist_ok=True)
    combined_df = pd.concat([df_samples, predictions_pct], axis=1)
    csv_path = os.path.join(output_dir, "mc_us_sludge_predictions.csv")
    combined_df.to_csv(csv_path, index=False)
    print(f"Saved predictions and input features to {csv_path}")

    # Plot uncertainty
    fig_path = os.path.join(output_dir, "mc_us_sludge_uncertainty.png")
    plot_uncertainty(predictions_pct, fig_path)
    print(f"Saved uncertainty plot to {fig_path}")

    # Print summary statistics to console
    summary = predictions_pct.describe(percentiles=[0.05, 0.5, 0.95]).loc[["mean", "std", "5%", "95%"]]
    print("\nSummary statistics (percentage):")
    print(summary)


if __name__ == "__main__":
    main() 